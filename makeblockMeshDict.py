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
    
    def sort_vertices_by_zyx(self):
        """按 Z, Y, X 排序所有点并重新分配 ID"""
        # 提取所有点并排序
        sorted_points = sorted(self.id_to_vertex.values(), key=lambda p: (p[2], p[1], p[0]))

        # 清空原有数据
        self.id_to_vertex.clear()
        self.vertex_to_id.clear()
        self.output_list = ["vertices\n(\n"]

        # 重新分配ID并更新结构
        for new_id, point in enumerate(sorted_points):
            point_tuple = tuple(point)
            self.id_to_vertex[new_id] = point_tuple
            self.vertex_to_id[point_tuple] = new_id
            formatted_point = " ".join(format_number(p) for p in point)
            self.output_list.append(f"\t({formatted_point})      //{new_id}\n")

        self.global_points_id = len(sorted_points)
        self.output_list.append("\n")

def generate_solid_vertices(root_width, root_length, root_hight, radius, bristle_length, num_bristles, G_over_D):
    
    vertices_manager = VertexManager()
    
    bottom_vertices = [
        [0, root_length, 0],
        [root_width, root_length, 0],
        [root_width, 0, 0],
        [0, 0, 0]
    ]
    
    for i in range(num_bristles-1):
        bottom_G_over_D_sectors = [
            [0, (i + 1) * (G_over_D + radius * 2), 0],
            [root_width, (i + 1) * (G_over_D + radius * 2), 0]
        ]
        bottom_vertices.extend(bottom_G_over_D_sectors)
        
    for i in range(num_bristles):
        bottom_middle_sectors = [
            [0, G_over_D / 2 + radius - root_width / 2 + i * (G_over_D + radius * 2), 0],
            [0, G_over_D / 2 + radius + root_width / 2 + i * (G_over_D + radius * 2), 0],
            [root_width, G_over_D / 2 + radius + root_width / 2 + i * (G_over_D + radius * 2), 0],
            [root_width, G_over_D / 2 + radius - root_width / 2 + i * (G_over_D + radius * 2), 0]
        ]
        bottom_vertices.extend(bottom_middle_sectors)
    
    
    root_vertices = [[x, y, root_hight] for x,y,_ in bottom_vertices]
    vertices_manager.add_vertices(root_vertices)
    
    bristle_vertices = []
    for i in range(num_bristles):
        center_of_bristle = [root_width / 2, G_over_D / 2 + radius + i * (G_over_D + radius * 2)]
        
        out_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius)
        bristle_circle_vertices = [[x, y, 0] for x, y in out_circle_points]
        bottom_vertices.extend(bristle_circle_vertices)
        bristle_vertices.extend(bristle_circle_vertices)
        
        inner_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius/2)
        bristle_inner_circle_vertices = [[x, y, 0] for x, y in inner_circle_points]
        bristle_vertices.extend(bristle_inner_circle_vertices)
        bottom_vertices.extend(bristle_inner_circle_vertices)
    
    vertices_manager.add_vertices(bottom_vertices)
        
    bristle_inner_circle_root_vertices = [[x, y, root_hight] for x,y,_ in bristle_vertices]
    vertices_manager.add_vertices(bristle_inner_circle_root_vertices)
    
    bristle_inner_circle_top_vertices = [[x, y, root_hight+bristle_length] for x,y,_ in bristle_vertices]
    vertices_manager.add_vertices(bristle_inner_circle_top_vertices)
    
    vertices_manager.sort_vertices_by_zyx()
    
    return vertices_manager

def sort_ids_by_axis(vertex_manager, id_list, axis='z'):
    """
    根据指定 axis 对给定 ID 列表排序，并返回排序后的 ID 列表。
    支持 list、set、tuple 输入。
    """
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError("axis 参数必须为 'x'、'y' 或 'z'")

    axis_idx = axis_map[axis]
    return sorted(id_list, key=lambda vid: vertex_manager.get_vertex(vid)[axis_idx])

def find_left_bottom_vertices(vertex_manager, target_z, root_bristle_bit_group):
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
    
    # 2.4 构造最终列表
    result_ids = []
    for x, y, point_id in filtered_points:
        # 排除 X 和 Y 最大的点
        if np.isclose(x, max_x_point[0]) or np.isclose(y, max_y_point[1]):
            continue
        if point_id in root_bristle_bit_group:
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
    num_of_points = len(filtered_points)
    
    # 找到最左（X 最小）且最下（Y 最小）的点
    max_x = max(filtered_points, key=lambda item: item[0])[0]
    max_y = max(filtered_points, key=lambda item: item[1])[1]
    
    result_ids = []
    for x,y,point_id in filtered_points:
        if np.isclose(x, max_x) or  np.isclose(y, max_y):
            continue
        result_ids.append(point_id)
    # print(result_ids)
    return result_ids, num_of_points

def find_vertices(vertex_manager, target, XYZ="Z"):
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
    num_of_points = len(filtered_points)
    
    result_ids = []
    for x,y,point_id in filtered_points:
        result_ids.append(point_id)
    # print(result_ids)
    return result_ids, num_of_points

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

def remove_middle(arr):
    result = []
    for i in range(0, len(arr), 3):
        group = arr[i:i+3]
        if len(group) >= 1:
            result.append(group[0])
        if len(group) == 3:
            result.append(group[2])
        # 如果不足三个元素则按需要处理

    return result

def generate_solid_blocks(vertices, cubic_width, root_block_hight, bristle_length, radius, partition_Y, partition_Z, partition_G):
    output_blocks = ["blocks\n(\n"]

    bottom_ids, bottom_points_num = find_vertices(vertices, 0, XYZ="Z")
    root_ids, root_points_num = find_vertices(vertices, root_block_hight, XYZ="Z")
    top_ids, top_points_num = find_vertices(vertices, root_block_hight+bristle_length, XYZ="Z")
    #底层外框
    bristle_left_ids, bristle_left_points_num = find_vertices(vertices, 0, XYZ="X")
    bottom_left_vertices_ids = set(bottom_ids) & set(bristle_left_ids)
    bottom_left_vertices_ids_sorted = sort_ids_by_axis(vertices, bottom_left_vertices_ids, axis='y')
    bottom_left_vertices_ids_sorted = remove_middle(bottom_left_vertices_ids_sorted)
    bottom_left_blocks_ids_sorted = bottom_left_vertices_ids_sorted[::2]
    # 底层毛
    cylinder_left_ids, cylinder_left_points_num = find_vertices(vertices, cubic_width/2-radius/(2**(0.5)), XYZ="X")
    cylinder_left_vertices_ids = set(bottom_ids) & set(cylinder_left_ids)
    cylinder_left_ids_sorted = sort_ids_by_axis(vertices, cylinder_left_vertices_ids, axis='y')
    cylinder_left_ids_sorted = cylinder_left_ids_sorted[::2]
    # 顶层毛
    cylinder_top_left_vertices_ids = set(top_ids) & set(cylinder_left_ids)
    cylinder_top_left_ids_sorted = sort_ids_by_axis(vertices, cylinder_top_left_vertices_ids, axis='y')
    cylinder_top_left_ids_sorted = cylinder_top_left_ids_sorted[::2]
    # 底层毛内框
    cylinder_inner_left_ids, cylinder_inner_left_points_num = find_vertices(vertices, cubic_width/2-radius/2/(2**(0.5)), XYZ="X")
    cylinder_inner_left_vertices_ids = set(bottom_ids) & set(cylinder_inner_left_ids)
    cylinder_inner_left_ids_sorted = sort_ids_by_axis(vertices, cylinder_inner_left_vertices_ids, axis='y')
    cylinder_inner_left_ids_sorted = cylinder_inner_left_ids_sorted[::2]
    # 顶层毛内框
    cylinder_top_inner_left_vertices_ids = set(top_ids) & set(cylinder_inner_left_ids)
    cylinder_top_inner_left_ids_sorted = sort_ids_by_axis(vertices, cylinder_top_inner_left_vertices_ids, axis='y')
    cylinder_top_inner_left_ids_sorted = cylinder_top_inner_left_ids_sorted[::2]
    
    #底层基座上下
    for index, id in enumerate(bottom_left_vertices_ids_sorted[:-1]):
        hex_line = (f"\thex ({id} {id+1} {id+3} {id+2} {id+bottom_points_num} {id+1+bottom_points_num} {id+3+bottom_points_num} {id+2+bottom_points_num})"
        f"({partition_Y} {partition_G} {partition_Y}) simpleGrading (1 1 1)\n")
        output_blocks.append(hex_line)
    
    for index, id in enumerate(cylinder_inner_left_ids_sorted):
        #底层鬃毛内圈
        hex_line = f"\thex ({id} {id+1} {id+3} {id+2} {id+bottom_points_num} {id+1+bottom_points_num} {id+3+bottom_points_num} {id+2+bottom_points_num}) ({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
        output_blocks.append(hex_line)
        #底层鬃毛外圈
        cylinder_hex_line = (
                f"\thex ({cylinder_left_ids_sorted[index]} {id} {id+2} {cylinder_left_ids_sorted[index]+6} " 
                f"{cylinder_left_ids_sorted[index]+bottom_points_num} {id+bottom_points_num} {id+2+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num}) "
                f"({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
                
                f"\thex ({cylinder_left_ids_sorted[index]+1} {id+1} {id} {cylinder_left_ids_sorted[index]} " 
                f"{cylinder_left_ids_sorted[index]+1+bottom_points_num} {id+1+bottom_points_num} {id+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num}) "
                f"({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
                
                f"\thex ({cylinder_left_ids_sorted[index]+7} {id+3} {id+1} {cylinder_left_ids_sorted[index]+1} " 
                f"{cylinder_left_ids_sorted[index]+7+bottom_points_num} {id+3+bottom_points_num} {id+1+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num}) "
                f"({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
                
                f"\thex ({cylinder_left_ids_sorted[index]+6} {id+2} {id+3} {cylinder_left_ids_sorted[index]+7} " 
                f"{cylinder_left_ids_sorted[index]+6+bottom_points_num} {id+2+bottom_points_num} {id+3+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num}) "
                f"({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
            )
        output_blocks.append(cylinder_hex_line)
        
        #底层基座中心
        cylinder_out_hex_line = (
            f"\thex ({bottom_left_blocks_ids_sorted[index]+2} {cylinder_left_ids_sorted[index]} {cylinder_left_ids_sorted[index]+6} {bottom_left_blocks_ids_sorted[index]+12} " 
            f"{bottom_left_blocks_ids_sorted[index]+2+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num} {bottom_left_blocks_ids_sorted[index]+12+bottom_points_num}) "
            f"({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
            
            f"\thex ({bottom_left_blocks_ids_sorted[index]+3} {cylinder_left_ids_sorted[index]+1} {cylinder_left_ids_sorted[index]} {bottom_left_blocks_ids_sorted[index]+2} " 
            f"{bottom_left_blocks_ids_sorted[index]+3+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num} {bottom_left_blocks_ids_sorted[index]+2+bottom_points_num}) "
            f"({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
            
            f"\thex ({bottom_left_blocks_ids_sorted[index]+13} {cylinder_left_ids_sorted[index]+7} {cylinder_left_ids_sorted[index]+1} {bottom_left_blocks_ids_sorted[index]+3} " 
            f"{bottom_left_blocks_ids_sorted[index]+13+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num} {bottom_left_blocks_ids_sorted[index]+3+bottom_points_num}) "
            f"({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
            
            f"\thex ({bottom_left_blocks_ids_sorted[index]+12} {cylinder_left_ids_sorted[index]+6} {cylinder_left_ids_sorted[index]+7} {bottom_left_blocks_ids_sorted[index]+13} " 
            f"{bottom_left_blocks_ids_sorted[index]+12+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num} {bottom_left_blocks_ids_sorted[index]+13+bottom_points_num}) "
            f"({partition_Y} {partition_Y} {partition_Y}) simpleGrading (1 1 1)\n"
        )
            
        output_blocks.append(cylinder_out_hex_line)
        #鬃毛内层
        hex_line = (
            f"\thex ({id+bottom_points_num} {id+1+bottom_points_num} {id+3+bottom_points_num} {id+2+bottom_points_num} "
            f"{cylinder_top_inner_left_ids_sorted[index]} {1+cylinder_top_inner_left_ids_sorted[index]} {cylinder_top_inner_left_ids_sorted[index]+3} {cylinder_top_inner_left_ids_sorted[index]+2}) "
            f"({partition_Y} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
        )
        output_blocks.append(hex_line)
        #鬃毛外圈
        bristle_hex_line = (
            f"\thex ({cylinder_left_ids_sorted[index]+bottom_points_num} {id+bottom_points_num} {id+2+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num} " 
            f"{cylinder_top_left_ids_sorted[index]} {cylinder_top_inner_left_ids_sorted[index]} {cylinder_top_inner_left_ids_sorted[index]+2} {cylinder_top_left_ids_sorted[index]+6}) "
            f"({partition_Y} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+1+bottom_points_num} {id+1+bottom_points_num} {id+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num} " 
            f"{cylinder_top_left_ids_sorted[index]+1} {cylinder_top_inner_left_ids_sorted[index]+1} {cylinder_top_inner_left_ids_sorted[index]} {cylinder_top_left_ids_sorted[index]}) "
            f"({partition_Y} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+7+bottom_points_num} {id+3+bottom_points_num} {id+1+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num} " 
            f"{cylinder_top_left_ids_sorted[index]+7} {cylinder_top_inner_left_ids_sorted[index]+3} {cylinder_top_inner_left_ids_sorted[index]+1} {cylinder_top_left_ids_sorted[index]+1}) "
            f"({partition_Y} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+6+bottom_points_num} {id+2+bottom_points_num} {id+3+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num} " 
            f"{cylinder_top_left_ids_sorted[index]+6} {cylinder_top_inner_left_ids_sorted[index]+2} {cylinder_top_inner_left_ids_sorted[index]+3} {cylinder_top_left_ids_sorted[index]+7}) "
            f"({partition_Y} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
        )
        output_blocks.append(bristle_hex_line)
    
    output_blocks.append("\n")
    output_blocks.append(");\n\n")
    return output_blocks, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num, bottom_left_vertices_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted

def generate_solid_edges(cubic_width, G_over_D, radius, bristle_length, root_block_hight, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num):
    output_edges = ["edges\n(\n"]
    
    def edge_generation(ids, index, z):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_width/2+radius*np.sin(alpha)} {G_over_D/2 + radius + index*(G_over_D+radius*2)-radius*np.cos(beta)} {z})\n"
            alpha += np.pi/2
            beta += np.pi/2
            output_edges.append(edge_line)

    for index, id in enumerate(cylinder_left_ids_sorted):
        bottom_out_circle_ids = [id, id+1, id+7, id+6]
        root_out_circle_ids = [i+bottom_points_num for i in bottom_out_circle_ids]
        edge_generation(bottom_out_circle_ids, index, 0)
        edge_generation(root_out_circle_ids, index, root_block_hight)
    for index, id in enumerate(cylinder_top_left_ids_sorted):
        root_out_circle_ids = [id, id+1, id+7, id+6]
        edge_generation(root_out_circle_ids, index, root_block_hight+bristle_length)

    
    output_edges.append(");\n\n")
    return output_edges

def generate_solid_patches(bottom_left_vertices_ids_sorted, bottom_points_num, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted):    
    output_patches = ["patches\n(\n"]
    output_patches.append("\tpatch bristle_interFSI\n")
    output_patches.append("\t(\n")
    
    for index, id in enumerate(bottom_left_vertices_ids_sorted[::2]):
        if index < len(bottom_left_vertices_ids_sorted[::2]) - 1:
            #入口侧
            output_patches.append(f"\t\t({id} {id+2} {id+2+bottom_points_num} {id+bottom_points_num})\n")
            output_patches.append(f"\t\t({id+2} {id+12} {id+12+bottom_points_num} {id+2+bottom_points_num})\n")
            output_patches.append(f"\t\t({id+12} {id+14} {id+14+bottom_points_num} {id+12+bottom_points_num})\n")
            #出口侧
            output_patches.append(f"\t\t({id+1} {id+1+bottom_points_num} {id+3+bottom_points_num} {id+3})\n")
            output_patches.append(f"\t\t({id+3} {id+3+bottom_points_num} {id+13+bottom_points_num} {id+13})\n")
            output_patches.append(f"\t\t({id+13} {id+13+bottom_points_num} {id+15+bottom_points_num} {id+15})\n")
            #上盖
            output_patches.append(f"\t\t({id+bottom_points_num} {id+2+bottom_points_num} {id+3+bottom_points_num} {id+1+bottom_points_num})\n")
            output_patches.append(f"\t\t({id+12+bottom_points_num} {id+14+bottom_points_num} {id+15+bottom_points_num} {id+13+bottom_points_num})\n")
            
            bristle_root_patch = (
                f"\t\t({id+12+bottom_points_num} {id+10+bottom_points_num} "
                f"{id+4+bottom_points_num} {id+2+bottom_points_num})\n"
                
                f"\t\t({id+2+bottom_points_num} {id+4+bottom_points_num} "
                f"{id+5+bottom_points_num} {id+3+bottom_points_num})\n"
                
                f"\t\t({id+3+bottom_points_num} {id+5+bottom_points_num} "
                f"{id+11+bottom_points_num} {id+13+bottom_points_num})\n"
                
                f"\t\t({id+13+bottom_points_num} {id+11+bottom_points_num} "
                f"{id+10+bottom_points_num} {id+12+bottom_points_num})\n"
            )
            output_patches.append(bristle_root_patch)
    
    output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[0]+1} {bottom_left_vertices_ids_sorted[0]} {bottom_left_vertices_ids_sorted[0]+bottom_points_num} {bottom_left_vertices_ids_sorted[0]+bottom_points_num+1})\n")
    
    output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[-1]} {bottom_left_vertices_ids_sorted[-1]+1} {bottom_left_vertices_ids_sorted[-1]+bottom_points_num+1} {bottom_left_vertices_ids_sorted[-1]+bottom_points_num})\n")
    
    for index, id in enumerate(cylinder_left_ids_sorted):
        bristle_cylinder_patch = (
            f"\t\t({id+bottom_points_num} {cylinder_top_left_ids_sorted[index]} "
            f"{cylinder_top_left_ids_sorted[index]+6} {id+6+bottom_points_num})\n"
            
            f"\t\t({id+1+bottom_points_num} {cylinder_top_left_ids_sorted[index]+1} "
            f"{cylinder_top_left_ids_sorted[index]} {id+bottom_points_num})\n"
            
            f"\t\t({id+7+bottom_points_num} {cylinder_top_left_ids_sorted[index]+7} "
            f"{cylinder_top_left_ids_sorted[index]+1} {id+1+bottom_points_num})\n"
            
            f"\t\t({id+6+bottom_points_num} {cylinder_top_left_ids_sorted[index]+6} "
            f"{cylinder_top_left_ids_sorted[index]+7} {id+7+bottom_points_num})\n"
        )
        output_patches.append(bristle_cylinder_patch)
        bristle_top_patch = (
            f"\t\t({cylinder_top_inner_left_ids_sorted[index]} {cylinder_top_inner_left_ids_sorted[index]+1} "
            f"{cylinder_top_inner_left_ids_sorted[index]+3} {cylinder_top_inner_left_ids_sorted[index]+2})\n"
            
            f"\t\t({cylinder_top_left_ids_sorted[index]} {cylinder_top_inner_left_ids_sorted[index]} "
            f"{cylinder_top_inner_left_ids_sorted[index]+2} {cylinder_top_left_ids_sorted[index]+6})\n"
            
            f"\t\t({cylinder_top_left_ids_sorted[index]+1} {cylinder_top_inner_left_ids_sorted[index]+1} "
            f"{cylinder_top_inner_left_ids_sorted[index]} {cylinder_top_left_ids_sorted[index]})\n"
            
            f"\t\t({cylinder_top_left_ids_sorted[index]+7} {cylinder_top_inner_left_ids_sorted[index]+3} "
            f"{cylinder_top_inner_left_ids_sorted[index]+1} {cylinder_top_left_ids_sorted[index]+1})\n"
            
            f"\t\t({cylinder_top_left_ids_sorted[index]+6} {cylinder_top_inner_left_ids_sorted[index]+2} "
            f"{cylinder_top_inner_left_ids_sorted[index]+3} {cylinder_top_left_ids_sorted[index]+7})\n"
            
        )
        output_patches.append(bristle_top_patch)
        
    # output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[0]} {bottom_left_vertices_ids_sorted[0]+1} {bottom_left_vertices_ids_sorted[0]+1+bottom_points_num} {bottom_left_vertices_ids_sorted[0]+bottom_points_num})\n")
    # output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[-1]+1} {bottom_left_vertices_ids_sorted[-1]} {bottom_left_vertices_ids_sorted[-1]+bottom_points_num} {bottom_left_vertices_ids_sorted[-1]+1+bottom_points_num})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch plateFix\n")
    output_patches.append("\t(\n")
    for index, id in enumerate(bottom_left_vertices_ids_sorted[::2]):
        if index < len(bottom_left_vertices_ids_sorted[::2]) - 1:
            bottom_patch = (
                f"\t\t({id} {id+1} {id+3} {id +2})\n"
                f"\t\t({id+2} {id+4} {id+10} {id+12})\n"
                f"\t\t({id+3} {id+5} {id+4} {id+2} )\n"
                f"\t\t({id+13} {id+11} {id+5} {id+3})\n"
                f"\t\t({id+12} {id+10} {id+11} {id+13})\n"
                f"\t\t({id+12} {id+13} {id+15} {id+14})\n"
            )
            output_patches.append(bottom_patch)
    
    for index, id in enumerate(cylinder_left_ids_sorted):

        bottom_cylinder_patch = (
            f"\t\t({cylinder_inner_left_ids_sorted[index]+2} {cylinder_inner_left_ids_sorted[index]+3} "
            f"{cylinder_inner_left_ids_sorted[index]+1} {cylinder_inner_left_ids_sorted[index]})\n"
            
            f"\t\t({id+6} {cylinder_inner_left_ids_sorted[index]+2} "
            f"{cylinder_inner_left_ids_sorted[index]} {id})\n"
            
            f"\t\t({id} {cylinder_inner_left_ids_sorted[index]} "
            f"{cylinder_inner_left_ids_sorted[index]+1} {id+1})\n"
            
            f"\t\t({id+1} {cylinder_inner_left_ids_sorted[index]+1} "
            f"{cylinder_inner_left_ids_sorted[index]+3} {id+7})\n"
            
            f"\t\t({id+7} {cylinder_inner_left_ids_sorted[index]+3} "
            f"{cylinder_inner_left_ids_sorted[index]+2} {id+6})\n"
        )
        output_patches.append(bottom_cylinder_patch)
    
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
head = generate_FOAM_head()

G_over_D = 15
bristle_length = 100
radius = 1.4 * bristle_length / 140 / 2 # 140是翅尖部分的bristle长度，1.4是实际bristle的直径
num_bristles = 7
bristle_gap = radius * 2 * G_over_D # 这个数字是 gap/diameter
partition_G = 8

partition_Z = 200 # int(bristle_length / mesh_size)
root_width, root_length, root_hight = 2, num_bristles * (radius*2+bristle_gap), 2 

end = generate_ends()

solid_partition_XY = 3
solid_mesh = "solid/constant/polyMesh/blockMeshDict"#"blockMeshDict.solid"
solid_vertices = generate_solid_vertices(root_width, root_length, root_hight, radius, bristle_length, num_bristles, G_over_D)
solid_blocks, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num, bottom_left_vertices_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted = generate_solid_blocks(solid_vertices, root_width, root_hight, bristle_length, radius, solid_partition_XY, partition_Z, partition_G)
solid_edges = generate_solid_edges(root_width, G_over_D, radius, bristle_length, root_hight, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num)
solid_patches = generate_solid_patches(bottom_left_vertices_ids_sorted, bottom_points_num, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted)
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