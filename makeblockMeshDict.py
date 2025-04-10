import textwrap, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data_frame_columns = ['id', 'x', 'y', 'z', 'geometry']
df = pd.DataFrame(columns=data_frame_columns)
global_points_id = 0


def generate_FOAM_head():
    head = textwrap.dedent("""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\      /  F ield         | foam-extend: Open Source CFD                    |
|  \\    /   O peration     | Version:     4.0                                |
|   \\  /    A nd           | Web:         http://www.foam-extend.org         |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 0.001;
    """)
    return head

def bristle_points(x_center, y_center, radius):
    """
    points[0] 为左下， 1 为右下， 2 为右上， 3 为左上
    """
    angles = np.radians([-135, -45, 45, 135])
    points = [[x_center + radius * np.cos(a), y_center + radius * np.sin(a)] for a in angles]
    return points

def format_number(num):
    """ 格式化数字，保留4位小数，但如果是整数则不保留小数 """
    if isinstance(num, float):
        return "{:.4f}".format(num).rstrip('0').rstrip('.')
    return str(num)

class VertexManager:
    def __init__(self):
        self.global_points_id = 0
        self.id_to_vertex = {}  # ID -> 坐标
        self.vertex_to_id = {}  # 坐标 -> ID（用于反向查询）
        self.output_list = ["vertices\n(\n"]

    def add_vertices(self, points):
        """添加多个顶点，并自动分配ID"""
        start_id = self.global_points_id
        for i, point in enumerate(points):
            point_tuple = tuple(point)  # 转换为元组，方便作为键
            self.id_to_vertex[start_id + i] = point_tuple  # ID -> 坐标
            self.vertex_to_id[point_tuple] = start_id + i  # 坐标 -> ID
            formatted_point = " ".join(format_number(p) for p in point)
            self.output_list.append(f"\t({formatted_point})      //{start_id + i}\n")
        self.global_points_id += len(points)  # 更新全局ID
        self.output_list.append("\n")
    
    def get_vertex(self, point_id):
        """根据ID获取点坐标"""
        return self.id_to_vertex.get(point_id, None)

    def get_id_by_xy(self, x, y):
        """ 通过 (X, Y) 查询匹配的 (Z, ID) 列表 """
        results = []
        for (vx, vy, vz), vid in self.vertex_to_id.items():
            if np.isclose(vx, x) and np.isclose(vy, y):  # 避免浮点数误差
                results.append((vz, vid))
        return results if results else None  # 若无匹配返回 None

    def get_output(self):
        """返回 blockMeshDict 格式的字符串"""
        self.output_list.append(");\n\n")
        return "".join(self.output_list)


def generate_vertices(cubic_size, radius, bristle_length):
    center_of_bristle = [cubic_size / 2, cubic_size / 2]
    inner_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius / 2)
    out_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius)
    vertices_manager = VertexManager()
    
    # === 1. 生成 root_vertices（Z = 0，不包含 inner_circle_points） ===
    root_vertices = [
        [0, 0, 0],
        [out_circle_points[0][0], 0, 0],
        [out_circle_points[1][0], 0, 0],
        [cubic_size, 0, 0],
        [0, out_circle_points[0][1], 0],
        [out_circle_points[0][0], out_circle_points[0][1], 0],
        [out_circle_points[1][0], out_circle_points[1][1], 0],
        [cubic_size, out_circle_points[1][1], 0],
        [0, out_circle_points[2][1], 0],
        [out_circle_points[3][0], out_circle_points[3][1], 0],
        [out_circle_points[2][0], out_circle_points[2][1], 0],
        [cubic_size, out_circle_points[3][1], 0],
        [0, cubic_size, 0],
        [out_circle_points[3][0], cubic_size, 0],
        [out_circle_points[2][0], cubic_size, 0],
        [cubic_size, cubic_size, 0]
    ]
    
    vertices_manager.add_vertices(root_vertices)

    # === 2. 生成 bristle_end_vertices（Z = bristle_length，包含 inner_circle_points） ===
    bristle_end_vertices = [[x, y, bristle_length] for x, y, _ in root_vertices]
    bristle_end_vertices += [[x, y, bristle_length] for x, y in inner_circle_points]  # 这里添加 inner_circle_points

    # 按 X 方向排序
    # bristle_end_vertices.sort(key=lambda v: (v[0], v[1]))

    vertices_manager.add_vertices(bristle_end_vertices)

    # === 3. 生成 roof_vertices（Z = cubic_size，包含 inner_circle_points） ===
    roof_vertices = [[x, y, bristle_length*4/3] for x, y, _ in root_vertices]
    roof_vertices += [[x, y, bristle_length*4/3] for x, y in inner_circle_points]  # 这里添加 inner_circle_points

    # 按 Y 方向排序
    # roof_vertices.sort(key=lambda v: (v[0], v[1]))

    vertices_manager.add_vertices(roof_vertices)

    return vertices_manager, inner_circle_points, out_circle_points

def generate_solid_vertices(cubic_size, radius, bristle_length):
    center_of_bristle = [cubic_size / 2, cubic_size / 2]
    inner_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius / 2)
    out_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius)
    vertices_manager = VertexManager()
    
    # === 1. 生成 root_vertices（Z = 0，不包含 inner_circle_points） ===
    root_vertices = [[x, y, 0] for x, y in out_circle_points] + [[x, y, 0] for x, y in inner_circle_points]
    
    vertices_manager.add_vertices(root_vertices)

    # === 2. 生成 bristle_end_vertices（Z = bristle_length，包含 inner_circle_points） ===
    bristle_end_vertices = [[x, y, bristle_length] for x, y, _ in root_vertices]
    vertices_manager.add_vertices(bristle_end_vertices)

    return vertices_manager, inner_circle_points, out_circle_points

def find_left_bottom_vertices(vertex_manager, target_z, inner_circle_points, out_circle_points):
    filtered_points = []
    for point_id, coords in vertex_manager.id_to_vertex.items():
        x, y, z = coords
        if np.isclose(z, target_z):
            filtered_points.append((x, y, point_id))
    num_of_points = len(filtered_points)
    
    if not filtered_points:
        return []
    
    # 2.1 找出 X 最大的点和 Y 最大的点
    max_x_point = max(filtered_points, key=lambda item: item[0])
    max_y_point = max(filtered_points, key=lambda item: item[1])
    
    # 2.2 获取 out_circle_points[3]（左下点）的 ID
    out_lower_left = out_circle_points[0]  # [x, y]
    out_lower_left_id = None
    for x, y, point_id in filtered_points:
        if np.isclose(x, out_lower_left[0]) and np.isclose(y, out_lower_left[1]):
            out_lower_left_id = point_id
            break
    
    # 2.3 获取 inner_circle_points 所有 4 个点的 ID
    inner_point_ids = []
    for inner_point in inner_circle_points:  # 遍历所有 4 个内圈点
        for x, y, point_id in filtered_points:
            if np.isclose(x, inner_point[0]) and np.isclose(y, inner_point[1]):
                inner_point_ids.append(point_id)
                break
    
    # 2.4 构造最终列表
    result_ids = []
    for x, y, point_id in filtered_points:
        # 排除 X 和 Y 最大的点
        if np.isclose(x, max_x_point[0]) or np.isclose(y, max_y_point[1]):
            continue
        # 排除 out_circle_points[3]
        if point_id == out_lower_left_id:
            continue
        # 排除所有 inner_circle_points 的点
        if point_id in inner_point_ids:
            continue
        result_ids.append(point_id)
    
    return result_ids, num_of_points

def find_left_bottom_vertices_simple(vertex_manager, target, XYZ="Z"):
    filtered_points = []
    for point_id, coords in vertex_manager.id_to_vertex.items():
        x, y, z = coords
        if XYZ=="Z":
            if np.isclose(z, target):
                filtered_points.append((x, y, point_id))
        elif XYZ == "Y":
            if np.isclose(y, target):
                filtered_points.append((x, z, point_id))
        elif XYZ == "X":
            if np.isclose(x, target):
                filtered_points.append((y, z, point_id))
    if not filtered_points:
        return []
    
    # 找到最左（X 最小）且最下（Y 最小）的点
    max_x = max(filtered_points, key=lambda item: item[0])[0]
    max_y = max(filtered_points, key=lambda item: item[1])[1]
    
    result_ids = []
    for x,y,point_id in filtered_points:
        if np.isclose(x, max_x) or  np.isclose(y, max_y):
            continue
        result_ids.append(point_id)
    print(result_ids)
    return result_ids

def get_specific_circle_points(vertex_manager, target_z, inner_circle_points, out_circle_points):
    # 1. 筛选出所有Z坐标为target_z的点
    filtered_points = []
    for point_id, coords in vertex_manager.id_to_vertex.items():
        x, y, z = coords
        if np.isclose(z, target_z):
            filtered_points.append((x, y, point_id))
    
    if not filtered_points:
        return []
    
    # 2. 获取目标点的ID
    result_ids = []
    
    # 2.3 获取 out_circle_points[3]（左下外圈点），并重复一次
    out_bottom_left = out_circle_points[0]
    for x, y, point_id in filtered_points:
        if np.isclose(x, out_bottom_left[0]) and np.isclose(y, out_bottom_left[1]):
            result_ids.append(point_id)
            result_ids.append(point_id)  # 重复一次
            break
    
    # 2.2 获取 inner_circle_points[1]（右下内圈点）
    inner_bottom_right = inner_circle_points[1]
    for x, y, point_id in filtered_points:
        if np.isclose(x, inner_bottom_right[0]) and np.isclose(y, inner_bottom_right[1]):
            result_ids.append(point_id)
            break
    
    # 2.1 获取 inner_circle_points[2]（左上内圈点）
    inner_top_left = inner_circle_points[3]
    for x, y, point_id in filtered_points:
        if np.isclose(x, inner_top_left[0]) and np.isclose(y, inner_top_left[1]):
            result_ids.append(point_id)
            break
    
    return result_ids

def get_all_circle_points_separately(vertex_manager, target_z, inner_circle_points, out_circle_points):
    # 1. 筛选出所有Z坐标为target_z的点
    filtered_points = []
    for point_id, coords in vertex_manager.id_to_vertex.items():
        x, y, z = coords
        if np.isclose(z, target_z):
            filtered_points.append((x, y, point_id))
    
    # 2. 初始化存储列表
    out_circle_ids = []
    inner_circle_ids = []
    
    # 3. 查找外圈4个点的ID（按原始顺序）
    for out_point in out_circle_points:
        found = False
        for x, y, point_id in filtered_points:
            if np.isclose(x, out_point[0]) and np.isclose(y, out_point[1]):
                out_circle_ids.append(point_id)
                found = True
                break
        if not found:
            out_circle_ids.append(None)  # 如果没找到，用None占位
    
    # 4. 查找内圈4个点的ID（按原始顺序）
    for inner_point in inner_circle_points:
        found = False
        for x, y, point_id in filtered_points:
            if np.isclose(x, inner_point[0]) and np.isclose(y, inner_point[1]):
                inner_circle_ids.append(point_id)
                found = True
                break
        if not found:
            inner_circle_ids.append(None)  # 如果没找到，用None占位
    
    return out_circle_ids, inner_circle_ids

def surrounding_blocks(start_id, length_each_layer, partition_XY, partition_Z):
    block_line = f"\thex ({start_id} {start_id+1} {start_id+5} {start_id+4} {start_id+length_each_layer} {start_id+length_each_layer+1} {start_id+length_each_layer+5} {start_id+length_each_layer+4}) ({partition_XY} {partition_XY} {partition_Z}) simpleGrading (1 1 1)\n"
    return block_line

def generate_blocks(vertices, bristle_length, inner_circle_points, out_circle_points, partition_XY, partition_Z):
    output_blocks = ["blocks\n(\n"]
    root_left_bottom_points, root_points_num = find_left_bottom_vertices(vertices, 0, inner_circle_points, out_circle_points)
    bristle_top_left_bottom_points, bristle_top_points_num = find_left_bottom_vertices(vertices, bristle_length, inner_circle_points, out_circle_points)
    
    for id in root_left_bottom_points:
        output_blocks.append(surrounding_blocks(id, root_points_num, partition_XY, partition_Z))
    
    for id in bristle_top_left_bottom_points:
        output_blocks.append(surrounding_blocks(id, bristle_top_points_num, partition_XY, int(partition_Z/3)))
        
    quad_groups = [
    [out_circle_points[0], inner_circle_points[0], inner_circle_points[3], out_circle_points[3]],
    [out_circle_points[0], out_circle_points[1], inner_circle_points[1], inner_circle_points[0]],
    [inner_circle_points[1], out_circle_points[1], out_circle_points[2], inner_circle_points[2]],
    [inner_circle_points[3], inner_circle_points[2], out_circle_points[2], out_circle_points[3]]
    ]
    
    top_patches = []
    bristle_top_patches = []
    for quad in quad_groups:
        # 通过 (x, y) 查找 ID，并提取 ID 值
        ids = []  # Z = bristle_length
        ids_upper = []  # Z = cubic_size
        
        
        for x, y in quad:
            matched_points = vertices.get_id_by_xy(x, y)  # 获取 (Z, ID) 列表
            
            # 用字典查找 Z 值
            z_id_map = {z: vid for z, vid in matched_points}
            
            if bristle_length in z_id_map and bristle_length*4/3 in z_id_map:
                ids.append(z_id_map[bristle_length])  # 取 Z = bristle_length 的 ID
                ids_upper.append(z_id_map[bristle_length*4/3])  # 取 Z = bristle_length*4/3 的 ID
            else:
                raise ValueError(f"无法找到点 ({x}, {y}) 在 Z = {bristle_length} 或 Z = {bristle_length*4/3} 的 ID")
        
        # 生成 hex 结构的文本
        hex_line = f"\thex ({ids[0]} {ids[1]} {ids[2]} {ids[3]} {ids_upper[0]} {ids_upper[1]} {ids_upper[2]} {ids_upper[3]}) ({partition_XY} {partition_XY} {int(partition_Z/3)}) simpleGrading (1 1 1)\n"
        output_blocks.append(hex_line)
        top_patches.append(ids_upper)
        bristle_top_patches.append(ids)
    
    bristle_out_circle_ids, bristle_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length, inner_circle_points, out_circle_points)
    top_out_circle_ids, top_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length*4/3, inner_circle_points, out_circle_points)
    brislte_center_hex = f"\thex ({bristle_inner_circle_ids[0]} {bristle_inner_circle_ids[1]} {bristle_inner_circle_ids[2]} {bristle_inner_circle_ids[3]} {top_inner_circle_ids[0]} {top_inner_circle_ids[1]} {top_inner_circle_ids[2]} {top_inner_circle_ids[3]}) ({partition_XY} {partition_XY} {int(partition_Z/3)}) simpleGrading (1 1 1)\n"
    output_blocks.append(brislte_center_hex)
    output_blocks.append("\n")
    output_blocks.append(");\n\n")
    return output_blocks, top_patches, bristle_top_patches
    
def generate_solid_blocks(vertices, bristle_length, inner_circle_points, out_circle_points, partition_XY, partition_Z):
    output_blocks = ["blocks\n(\n"]
        
    quad_groups = [
    [out_circle_points[0], inner_circle_points[0], inner_circle_points[3], out_circle_points[3]],
    [out_circle_points[0], out_circle_points[1], inner_circle_points[1], inner_circle_points[0]],
    [inner_circle_points[1], out_circle_points[1], out_circle_points[2], inner_circle_points[2]],
    [inner_circle_points[3], inner_circle_points[2], out_circle_points[2], out_circle_points[3]]
    ]
    
    bottom_patches = []
    bristle_top_patches = []
    for quad in quad_groups:
        # 通过 (x, y) 查找 ID，并提取 ID 值
        ids = []  # Z = bristle_length
        ids_upper = []  # Z = cubic_size
        
        for x, y in quad:
            matched_points = vertices.get_id_by_xy(x, y)  # 获取 (Z, ID) 列表
            
            # 用字典查找 Z 值
            z_id_map = {z: vid for z, vid in matched_points}
            
            if 0 in z_id_map and bristle_length in z_id_map:
                ids.append(z_id_map[0])  # 取 Z = bristle_length 的 ID
                ids_upper.append(z_id_map[bristle_length])  # 取 Z = bristle_length*4/3 的 ID
            else:
                raise ValueError(f"无法找到点 ({x}, {y}) 在 Z = {0} 或 Z = {bristle_length} 的 ID")
        
        # 生成 hex 结构的文本
        hex_line = f"\thex ({ids[0]} {ids[1]} {ids[2]} {ids[3]} {ids_upper[0]} {ids_upper[1]} {ids_upper[2]} {ids_upper[3]}) ({partition_XY} {partition_XY} {partition_Z}) simpleGrading (1 1 1)\n"
        output_blocks.append(hex_line)
        bottom_patches.append(ids)
        bristle_top_patches.append(ids_upper)
    
    bristle_out_circle_ids, bristle_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length, inner_circle_points, out_circle_points)
    root_out_circle_ids, root_inner_circle_ids = get_all_circle_points_separately(vertices, 0, inner_circle_points, out_circle_points)
    brislte_center_hex = f"\thex ({root_inner_circle_ids[0]} {root_inner_circle_ids[1]} {root_inner_circle_ids[2]} {root_inner_circle_ids[3]} {bristle_inner_circle_ids[0]} {bristle_inner_circle_ids[1]} {bristle_inner_circle_ids[2]} {bristle_inner_circle_ids[3]}) ({partition_XY} {partition_XY} {partition_Z}) simpleGrading (1 1 1)\n"
    output_blocks.append(brislte_center_hex)
    output_blocks.append("\n")
    output_blocks.append(");\n\n")
    return output_blocks, bottom_patches, bristle_top_patches
    
def generate_edges(vertices, cubic_size, bristle_length, inner_circle_points, out_circle_points):
    output_edges = ["edges\n(\n"]
    root_out_circle_ids, root_inner_circle_ids = get_all_circle_points_separately(vertices, 0, inner_circle_points, out_circle_points)
    bristle_out_circle_ids, bristle_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length, inner_circle_points, out_circle_points)
    top_out_circle_ids, top_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length*4/3, inner_circle_points, out_circle_points)

    def edge_generation(ids, z):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_size/2+radius*np.sin(alpha)} {cubic_size/2-radius*np.cos(beta)} {z})\n"
            alpha += np.pi/2
            beta += np.pi/2
            output_edges.append(edge_line)
    
    edge_generation(root_out_circle_ids, 0)
    edge_generation(bristle_out_circle_ids, bristle_length)
    edge_generation(top_out_circle_ids, bristle_length*4/3)
    
    output_edges.append(");\n\n")
    return output_edges

def generate_solid_edges(vertices, cubic_size, bristle_length, inner_circle_points, out_circle_points):
    output_edges = ["edges\n(\n"]
    root_out_circle_ids, root_inner_circle_ids = get_all_circle_points_separately(vertices, 0, inner_circle_points, out_circle_points)
    bristle_out_circle_ids, bristle_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length, inner_circle_points, out_circle_points)

    def edge_generation(ids, z):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_size/2+radius*np.sin(alpha)} {cubic_size/2-radius*np.cos(beta)} {z})\n"
            alpha += np.pi/2
            beta += np.pi/2
            output_edges.append(edge_line)
    
    edge_generation(root_out_circle_ids, 0)
    edge_generation(bristle_out_circle_ids, bristle_length)
    
    output_edges.append(");\n\n")
    return output_edges

def generate_patches(vertices, inner_circle_points, out_circle_points, top_patches, bristle_top_patches, cubic_size):
    output_patches = ["patches\n(\n"]
    output_patches.append("\tpatch bottom\n")
    output_patches.append("\t(\n")
    bottom_ids, root_points_num=find_left_bottom_vertices(vertices, 0, inner_circle_points, out_circle_points)
    for id in bottom_ids:
        output_patches.append(f"\t\t({id} {id+1} {id+5} {id+4})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch top\n")
    output_patches.append("\t(\n")
    top_left_bottom_points, top_points_num = find_left_bottom_vertices(vertices, bristle_length*4/3, inner_circle_points, out_circle_points)
    top_out_circle_ids, top_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length*4/3, inner_circle_points, out_circle_points)
    for id in top_left_bottom_points:
        output_patches.append(f"\t\t({id} {id+1} {id+5} {id+4})\n")
    for i in range(len(top_patches)):
        output_patches.append(f"\t\t({top_patches[i][0]} {top_patches[i][1]} {top_patches[i][2]} {top_patches[i][3]})\n")
    output_patches.append(f"\t\t({top_inner_circle_ids[0]} {top_inner_circle_ids[1]} {top_inner_circle_ids[2]} {top_inner_circle_ids[3]})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch inlet\n")
    output_patches.append("\t(\n")
    inlet_left_coner_ids = find_left_bottom_vertices_simple(vertices, 0, "X")
    for id in inlet_left_coner_ids:
        if id < (max(inlet_left_coner_ids) + min(inlet_left_coner_ids))/2:
            output_patches.append(f"\t\t({id} {id+4} {id+root_points_num+4} {id+root_points_num})\n")
        else:
            output_patches.append(f"\t\t({id} {id+4} {id+top_points_num+4} {id+top_points_num})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch outlet\n")
    output_patches.append("\t(\n")
    outlet_left_coner_ids = find_left_bottom_vertices_simple(vertices, cubic_size, "X")
    for id in outlet_left_coner_ids:
        if id < (max(outlet_left_coner_ids) + min(outlet_left_coner_ids))/2:
            output_patches.append(f"\t\t({id} {id+4} {id+root_points_num+4} {id+root_points_num})\n")
        else:
            output_patches.append(f"\t\t({id} {id+4} {id+top_points_num+4} {id+top_points_num})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch bristle\n")
    output_patches.append("\t(\n")
    root_out_circle_ids, root_inner_circle_ids = get_all_circle_points_separately(vertices, 0, inner_circle_points, out_circle_points)
    bristle_out_circle_ids, bristle_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length, inner_circle_points, out_circle_points)
    output_patches.append(f"\t\t({bristle_inner_circle_ids[0]} {bristle_inner_circle_ids[1]} {bristle_inner_circle_ids[2]} {bristle_inner_circle_ids[3]})\n")
    for i in range(len(bristle_top_patches)):
        output_patches.append(f"\t\t({bristle_top_patches[i][0]} {bristle_top_patches[i][1]} {bristle_top_patches[i][2]} {bristle_top_patches[i][3]})\n")
    for i in range(len(bristle_out_circle_ids)):
        output_patches.append(f"\t\t({root_out_circle_ids[i]} {root_out_circle_ids[(i+1) % len(root_out_circle_ids)]} {bristle_out_circle_ids[(i+1) % len(bristle_out_circle_ids)]} {bristle_out_circle_ids[i]})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch frontAndBackPlanes\n")
    output_patches.append("\t(\n")
    front_left_coner_ids = find_left_bottom_vertices_simple(vertices, 0, "Y")
    for id in front_left_coner_ids:
        if id < (max(front_left_coner_ids) + min(front_left_coner_ids))/2:
            output_patches.append(f"\t\t({id} {id+1} {id+root_points_num+1} {id+root_points_num})\n")
        else:
            output_patches.append(f"\t\t({id} {id+1} {id+top_points_num+1} {id+top_points_num})\n")
    back_left_coner_ids = find_left_bottom_vertices_simple(vertices, cubic_size, "Y")
    for id in back_left_coner_ids:
        if id < (max(back_left_coner_ids) + min(back_left_coner_ids))/2:
            output_patches.append(f"\t\t({id} {id+1} {id+root_points_num+1} {id+root_points_num})\n")
        else:
            output_patches.append(f"\t\t({id} {id+1} {id+top_points_num+1} {id+top_points_num})\n")
    output_patches.append("\t)\n")
    output_patches.append(");\n\n")
    return output_patches

def generate_solid_patches(vertices, inner_circle_points, out_circle_points, root_patches, bristle_top_patches):
    output_patches = ["patches\n(\n"]
    
    output_patches.append("\tpatch bristle\n")
    output_patches.append("\t(\n")
    root_out_circle_ids, root_inner_circle_ids = get_all_circle_points_separately(vertices, 0, inner_circle_points, out_circle_points)
    bristle_out_circle_ids, bristle_inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length, inner_circle_points, out_circle_points)
    output_patches.append(f"\t\t({bristle_inner_circle_ids[0]} {bristle_inner_circle_ids[1]} {bristle_inner_circle_ids[2]} {bristle_inner_circle_ids[3]})\n")
    for i in range(len(bristle_top_patches)):
        output_patches.append(f"\t\t({bristle_top_patches[i][0]} {bristle_top_patches[i][1]} {bristle_top_patches[i][2]} {bristle_top_patches[i][3]})\n")
    for i in range(len(bristle_out_circle_ids)):
        output_patches.append(f"\t\t({root_out_circle_ids[i]} {root_out_circle_ids[(i+1) % len(root_out_circle_ids)]} {bristle_out_circle_ids[(i+1) % len(bristle_out_circle_ids)]} {bristle_out_circle_ids[i]})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch plateFix\n")
    output_patches.append("\t(\n")
    output_patches.append(f"\t\t({root_inner_circle_ids[0]} {root_inner_circle_ids[1]} {root_inner_circle_ids[2]} {root_inner_circle_ids[3]})\n")
    for i in range(len(root_patches)):
        output_patches.append(f"\t\t({root_patches[i][0]} {root_patches[i][1]} {root_patches[i][2]} {root_patches[i][3]})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append(");\n\n")
    return output_patches


def generate_ends():
    end = textwrap.dedent("""\
mergePatchPairs
(
);


// ************************************************************************* //

    """)
    return end

def extract_vertices(vertices_manager):
    """ 从 blockMeshDict 解析出顶点坐标 """
    vertices = []
    pattern = r"\(\s*([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s*\)"  # 正则匹配 ( x y z )

    for line in vertices_manager.get_output().split("\n"):  # 修正：确保传入的是字符串
        match = re.search(pattern, line)
        if match:
            x, y, z = map(float, match.groups())
            vertices.append((x, y, z))

    return np.array(vertices)

# 生成 blockMeshDict 文件
fluid_mesh = r"fluid\constant\polyMesh\blockMeshDict"
head = generate_FOAM_head()
cubic_size = 2000
bristle_length = 1500
radius = 250
partition_XY = 5
partition_Z = 10

vertices, inner_circle_points, out_circle_points = generate_vertices(cubic_size, radius, bristle_length)
blocks, top_patches, bristle_top_patches = generate_blocks(vertices, bristle_length, inner_circle_points, out_circle_points, partition_XY, partition_Z)
edges = generate_edges(vertices, cubic_size, bristle_length, inner_circle_points, out_circle_points)
patches = generate_patches(vertices, inner_circle_points, out_circle_points, top_patches, bristle_top_patches, cubic_size)
end = generate_ends()
# **修正写入文件的方式**
with open(fluid_mesh, 'w') as file:
    file.write(head)
    file.write(vertices.get_output())  # **修正点**
    file.write("".join(blocks))
    file.write("".join(edges))
    file.write("".join(patches))
    file.write("".join(end))

solid_mesh = r"solid\constant\polyMesh\blockMeshDict"#"blockMeshDict.solid"
solid_partition_XY = 5
solid_partition_Z = 10
solid_vertices, solid_inner_circle_points, solid_out_circle_points = generate_solid_vertices(cubic_size, radius, bristle_length)
solid_blocks, solid_root_patches, solid_bristle_top_patches = generate_solid_blocks(solid_vertices, bristle_length, solid_inner_circle_points, solid_out_circle_points, solid_partition_XY, solid_partition_Z)
solid_edges = generate_solid_edges(solid_vertices, cubic_size, bristle_length, solid_inner_circle_points, solid_out_circle_points)
solid_patches = generate_solid_patches(solid_vertices, solid_inner_circle_points, solid_out_circle_points, solid_root_patches, solid_bristle_top_patches)
with open(solid_mesh, 'w') as file:
    file.write(head)
    file.write(solid_vertices.get_output())
    file.write("".join(solid_blocks))
    file.write("".join(solid_edges))
    file.write("".join(solid_patches))
    file.write("".join(end))



# 提取顶点数据
# vertices = extract_vertices(vertices)

# # 绘制3D散点图
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(projection='3d')

# ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o', s=5)

# # 设置坐标轴标签
# ax.set_xlabel("X Axis")
# ax.set_ylabel("Y Axis")
# ax.set_zlabel("Z Axis")
# ax.set_title("3D Scatter Plot of Extracted Vertices")

# plt.show()