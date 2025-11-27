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

def generate_vertices(cubic_width, cubic_length, radius, bristle_length, num_bristles, bristle_gap, root_block_hight, root_block_length, root_block_width):
    vertices_manager = VertexManager()
    solid_blocks_xy_vertices = []
    bottom_vertices = [
        [0, 0, 0],
        [cubic_width/2-root_block_width/2, 0, 0],
        [cubic_width/2+root_block_width/2, 0, 0],
        [cubic_width, 0, 0]
    ]
    for i in range(num_bristles):
        bottom_middle_sector = [
            [0, cubic_length/2-root_block_length/2 + i * root_block_length / num_bristles, 0],
            [cubic_width/2-root_block_width/2, cubic_length/2-root_block_length/2 + i * root_block_length / num_bristles, 0],
            [cubic_width/2+root_block_width/2, cubic_length/2-root_block_length/2 + i * root_block_length / num_bristles, 0],
            [cubic_width, cubic_length/2-root_block_length/2 + i * root_block_length / num_bristles, 0]
        ]
        solid_blocks_xy = [
            [cubic_width/2-root_block_width/2, cubic_length/2-root_block_length/2 + i * root_block_length / num_bristles, 0],
            [cubic_width/2+root_block_width/2, cubic_length/2-root_block_length/2 + i * root_block_length / num_bristles, 0]
        ]
        solid_blocks_xy_vertices.extend(solid_blocks_xy)
        bottom_vertices.extend(bottom_middle_sector)
    bottom_top_sector = [
        [0, cubic_length/2+root_block_length/2, 0],
        [cubic_width/2-root_block_width/2, cubic_length/2+root_block_length/2, 0],
        [cubic_width/2+root_block_width/2, cubic_length/2+root_block_length/2, 0],
        [cubic_width, cubic_length/2+root_block_length/2, 0],
        [0, cubic_length, 0],
        [cubic_width/2-root_block_width/2, cubic_length, 0],
        [cubic_width/2+root_block_width/2, cubic_length, 0],
        [cubic_width, cubic_length, 0]
    ]
    solid_blocks_xy = [
        [cubic_width/2-root_block_width/2, cubic_length/2+root_block_length/2, 0],
        [cubic_width/2+root_block_width/2, cubic_length/2+root_block_length/2, 0]
    ]
    solid_blocks_xy_vertices.extend(solid_blocks_xy)
    bottom_vertices.extend(bottom_top_sector)
    vertices_manager.add_vertices(bottom_vertices)
    # === 1. 生成 root_vertices（Z = 0，不包含 inner_circle_points） ===
    root_vertices = [[x, y, root_block_hight] for x, y, _ in bottom_vertices]
    center_of_bristle = [cubic_width / 2, cubic_length / 2]
    out_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius)
    bristle_bit_vertices = [] 
    add_middle_vertices = [
        [out_circle_points[0][0], out_circle_points[0][1], root_block_hight],
        [out_circle_points[1][0], out_circle_points[1][1], root_block_hight],
        [out_circle_points[3][0], out_circle_points[3][1], root_block_hight],
        [out_circle_points[2][0], out_circle_points[2][1], root_block_hight]
    ]
    root_vertices.extend(add_middle_vertices)
    bristle_bit_vertices.extend(add_middle_vertices)
    count = (num_bristles-1)/2
    for i in range(1, num_bristles):
        # 奇数往下，偶数往上
        offset = count * (bristle_gap + radius * 2)
        direction = np.cos(i * np.pi)
        center_of_bristle = [
            cubic_width / 2,
            cubic_length / 2 + direction * offset
        ]
        out_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius)
        add_rest_vertices = [
            [out_circle_points[0][0], out_circle_points[0][1], root_block_hight],
            [out_circle_points[1][0], out_circle_points[1][1], root_block_hight],
            [out_circle_points[3][0], out_circle_points[3][1], root_block_hight],
            [out_circle_points[2][0], out_circle_points[2][1], root_block_hight]
        ]

        bristle_bit_vertices.extend(add_rest_vertices)
        root_vertices.extend(add_rest_vertices)

        if i % 2 == 0 and i != 0:
            count -= 1
    solid_blocks_xy_vertices.extend(bristle_bit_vertices)
    vertices_manager.add_vertices(root_vertices)

    # === 2. 生成 bristle_end_vertices（Z = bristle_length，包含 inner_circle_points） ===
    bristle_end_vertices = [[x, y, bristle_length + root_block_hight] for x, y, _ in root_vertices]
    center_of_bristle = [cubic_width / 2, cubic_length / 2]
    inner_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius / 2)
    bristle_end_vertices += [[x, y, bristle_length + root_block_hight] for x, y in inner_circle_points]  # 这里添加 inner_circle_points
    bristle_inner_vertices = [[x, y, bristle_length] for x, y in inner_circle_points]
    count = (num_bristles-1)/2
    for i in range(1, num_bristles, 1):
        # 奇数往下，偶数往上
        offset = count * (bristle_gap + radius * 2)
        direction = np.cos(i * np.pi)
        center_of_bristle = [
            cubic_width / 2,
            cubic_length / 2 + direction * offset
        ]
        inner_circle_points = bristle_points(center_of_bristle[0], center_of_bristle[1], radius / 2)
        bristle_end_vertices += [[x, y, bristle_length + root_block_hight] for x, y in inner_circle_points]
        bristle_inner_vertices += [[x, y, bristle_length] for x, y in inner_circle_points]
        if i % 2 == 0 and i != 0:
            count -= 1
    solid_blocks_xy_vertices.extend(bristle_inner_vertices)
    vertices_manager.add_vertices(bristle_end_vertices)

    # === 3. 生成 roof_vertices（Z = cubic_size，包含 inner_circle_points） ===
    roof_vertices = [[x, y, bristle_length*1.5 + root_block_hight] for x, y, _ in bristle_end_vertices]
    vertices_manager.add_vertices(roof_vertices)
    vertices_manager.sort_vertices_by_zyx()

    return vertices_manager, solid_blocks_xy_vertices

def generate_solid_vertices(solid_blocks_xy_vertices, root_block_hight, bristle_length, root_block_width):
    
    vertices_manager = VertexManager()
    
    # === 1. 生成 root_vertices（Z = 0，不包含 inner_circle_points） ===
    root_vertices = [[x, y, 0] for x, y, _ in solid_blocks_xy_vertices]
    vertices_manager.add_vertices(root_vertices)

    root_vertices = [[x, y, root_block_hight] for x, y, _ in solid_blocks_xy_vertices]
    vertices_manager.add_vertices(root_vertices)
    
    # === 2. 生成 bristle_end_vertices（Z = bristle_length，包含 inner_circle_points） ===
    bristle_end_vertices = [[x, y, root_block_hight+bristle_length]
                            for x, y, _ in solid_blocks_xy_vertices
                            if not np.isclose(x, root_block_width) and not np.isclose(x, root_block_width * 2)
                            ]
    vertices_manager.add_vertices(bristle_end_vertices)
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

def generate_blocks(vertices, bristle_length, partition_X, partition_Y_outside, partition_Z, root_block_hight, root_block_width, cubic_length, radius, number_of_bristles, partition_Z_top):
    output_blocks = ["blocks\n(\n"]
    
    bottom_ids, bottom_points_num = find_left_bottom_vertices_simple(vertices, 0, XYZ="Z")
    root_ids, root_points_num = find_left_bottom_vertices_simple(vertices, root_block_hight, XYZ="Z")
    bristle_top_ids, bristle_top_points_num = find_left_bottom_vertices_simple(vertices, root_block_hight+bristle_length, XYZ="Z")
    
    stream_right_wall, stream_right_points_num = find_left_bottom_vertices_simple(vertices, 0, XYZ="Y")
    stream_left_wall, stream_left_points_num = find_left_bottom_vertices_simple(vertices, cubic_length, XYZ="Y")
    
    bristle_left_ids, bristle_left_points_num = find_vertices(vertices, cubic_width/2-root_block_width/2, XYZ="X")
    bristle_left_ids_not_full = [id for id in bristle_left_ids if id < bottom_points_num]
    bottom_ids_left_corner = set(bottom_ids)-set(bristle_left_ids_not_full[1:-2])
    bristle_left_ids_not_full_root = [id for id in bristle_left_ids if id < root_points_num+bottom_points_num and id > bottom_points_num]
    bristle_left_ids_not_full_top = [id for id in bristle_left_ids if id > root_points_num+bottom_points_num and id < root_points_num+bottom_points_num+bristle_top_points_num]
    root_ids_left_corner = set(root_ids)-set(bristle_left_ids_not_full_root[1:-2])
    
    bottom_ids_all, bottom_points_num = find_vertices(vertices, 0, XYZ="Z")
    inlet_ids, inlet_points_num = find_vertices(vertices, 0, XYZ="X")
    bristle_right_ids, bristle_right_points_num = find_vertices(vertices, cubic_width/2+root_block_width/2, XYZ="X")
    bottom_ids_left_corner_left_row = sorted(list(set(bottom_ids_all) & set(inlet_ids)))
    bottom_ids_left_corner_right_row = sorted(list(set(bottom_ids_all) & set(bristle_right_ids)))
    
    root_ids_all, root_points_num = find_vertices(vertices, root_block_hight, XYZ="Z")
    root_left_bottom_points_left_row = sorted(list(set(root_ids_all) & set(inlet_ids)))
    root_left_bottom_points_right_row = sorted(list(set(root_ids_all) & set(bristle_right_ids)))
    
    bristle_top_ids_all, bristle_top_points_num = find_vertices(vertices, root_block_hight+bristle_length, XYZ="Z")
    bristle_top_left_bottom_points_left_row = sorted(list(set(bristle_top_ids_all) & set(inlet_ids)))
    bristle_top_left_bottom_points_right_row = sorted(list(set(bristle_top_ids_all) & set(bristle_right_ids)))
    
    bristle_ids, bristle_points_num = find_vertices(vertices, cubic_width/2-radius/(2**(0.5)), XYZ="X")
    bristle_ids_right, bristle_points_num = find_vertices(vertices, cubic_width/2+radius/(2**(0.5)), XYZ="X")
    root_ids_left_corner = sorted(list(set(root_ids)-set(bristle_left_ids_not_full_root[1:-2])-set(root_left_bottom_points_right_row)
                                       -set(root_left_bottom_points_left_row)-set(bristle_ids_right)-set(bristle_ids)))
    
    inner_bristle_ids_left, bristle_points_num = find_vertices(vertices, cubic_width/2-radius/2/(2**(0.5)), XYZ="X")
    inner_bristle_ids_right, bristle_points_num = find_vertices(vertices, cubic_width/2+radius/2/(2**(0.5)), XYZ="X")
    top_ids_left_corner = sorted(list(set(bristle_top_ids)-set(bristle_left_ids_not_full_top[1:-2])
                                       -set(bristle_top_left_bottom_points_right_row)-set(bristle_top_left_bottom_points_left_row)
                                       -set(bristle_ids_right)-set(bristle_ids)-set(inner_bristle_ids_left)-set(inner_bristle_ids_right)))
    #入口处网格，翅膀根
    for index, id in enumerate(bottom_ids_left_corner_left_row[0:-1]):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {bottom_ids_left_corner_left_row[index+1]+1} {bottom_ids_left_corner_left_row[index+1]} "
                f"{root_left_bottom_points_left_row[index]} {root_left_bottom_points_left_row[index]+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]}) "
                f"({partition_X} {number_of_bristles} {8*G_D}) simpleGrading (1 1 1)\n"
            )
        elif index == len(bottom_ids_left_corner_left_row[0:-1])-1:
            block_line = (
                f"\thex ({id} {id+1} {bottom_ids_left_corner_left_row[index+1]+1} {bottom_ids_left_corner_left_row[index+1]} "
                f"{root_left_bottom_points_left_row[index]} {root_left_bottom_points_left_row[index]+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]}) "
                f"({partition_X} {number_of_bristles} {8*G_D}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
                f"\thex ({id} {id+1} {bottom_ids_left_corner_left_row[index+1]+1} {bottom_ids_left_corner_left_row[index+1]} "
                f"{root_left_bottom_points_left_row[index]} {root_left_bottom_points_left_row[index]+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]}) "
                f"({partition_X} {partition_Y_outside} {8*G_D}) simpleGrading (1 1 1)\n"
            )
        output_blocks.append(block_line)
    output_blocks.append("\n")
    #出口网格，翅膀根
    for index, id in enumerate(bottom_ids_left_corner_right_row[0:-1]):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {bottom_ids_left_corner_right_row[index+1]+1} {bottom_ids_left_corner_right_row[index+1]} "
                f"{root_left_bottom_points_right_row[index]} {root_left_bottom_points_right_row[index]+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]}) "
                f"({partition_X} {number_of_bristles} {8*G_D}) simpleGrading (1 1 1)\n"
            )
        elif index == len(bottom_ids_left_corner_right_row[0:-1])-1:
            block_line = (
                f"\thex ({id} {id+1} {bottom_ids_left_corner_right_row[index+1]+1} {bottom_ids_left_corner_right_row[index+1]} "
                f"{root_left_bottom_points_right_row[index]} {root_left_bottom_points_right_row[index]+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]}) "
                f"({partition_X} {number_of_bristles} {8*G_D}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
                f"\thex ({id} {id+1} {bottom_ids_left_corner_right_row[index+1]+1} {bottom_ids_left_corner_right_row[index+1]} "
                f"{root_left_bottom_points_right_row[index]} {root_left_bottom_points_right_row[index]+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]}) "
                f"({partition_X} {partition_Y_outside} {8*G_D}) simpleGrading (1 1 1)\n"
            )
        output_blocks.append(block_line)
    output_blocks.append("\n")
    #翅膀上下边界，翅膀根
    bottom_top_left_corner = sorted(list(set(bottom_ids)-set(bristle_left_ids_not_full[1:-2])-set(bottom_ids_left_corner_right_row)-set(bottom_ids_left_corner_left_row)))
    for index, id in enumerate(bottom_top_left_corner):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {id+5} {id+4} "
                f"{root_ids_left_corner[index]} {root_ids_left_corner[index]+1} {root_ids_left_corner[index]+5} {root_ids_left_corner[index]+4}) "
                f"({partition_Y_outside} {number_of_bristles} {8*G_D}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
            f"\thex ({id} {id+1} {id+5} {id+4} "
            f"{root_ids_left_corner[index]} {root_ids_left_corner[index]+1} {root_ids_left_corner[index]+5} {root_ids_left_corner[index]+4}) "
            f"({partition_Y_outside} {number_of_bristles} {8*G_D}) simpleGrading (1 1 1)\n"
        )
        output_blocks.append(block_line)
    output_blocks.append("\n")
    bottom_ids_left_corner = bottom_top_left_corner + bottom_ids_left_corner_left_row + bottom_ids_left_corner_right_row
    #入口边界，翅膀等高的部分
    for index, id in enumerate(root_left_bottom_points_left_row[0:-1]):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]} "
                f"{bristle_top_left_bottom_points_left_row[index]} {bristle_top_left_bottom_points_left_row[index]+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]}) "
                f"({partition_X} {number_of_bristles} {partition_Z}) simpleGrading (1 1 1)\n"
            )
        elif index == len(root_left_bottom_points_left_row[0:-1])-1:
            block_line = (
                f"\thex ({id} {id+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]} "
                f"{bristle_top_left_bottom_points_left_row[index]} {bristle_top_left_bottom_points_left_row[index]+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]}) "
                f"({partition_X} {number_of_bristles} {partition_Z}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
                f"\thex ({id} {id+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]} "
                f"{bristle_top_left_bottom_points_left_row[index]} {bristle_top_left_bottom_points_left_row[index]+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]}) "
                f"({partition_X} {partition_Y_outside} {partition_Z}) simpleGrading (1 1 1)\n"
            )
        output_blocks.append(block_line)
    output_blocks.append("\n")
    #出口边界，翅膀等高的部分
    for index, id in enumerate(root_left_bottom_points_right_row[0:-1]):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]} "
                f"{bristle_top_left_bottom_points_right_row[index]} {bristle_top_left_bottom_points_right_row[index]+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]}) "
                f"({partition_X} {number_of_bristles} {partition_Z}) simpleGrading (1 1 1)\n"
            )
        elif index == len(root_left_bottom_points_right_row[0:-1])-1:
            block_line = (
                f"\thex ({id} {id+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]} "
                f"{bristle_top_left_bottom_points_right_row[index]} {bristle_top_left_bottom_points_right_row[index]+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]}) "
                f"({partition_X} {number_of_bristles} {partition_Z}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
                f"\thex ({id} {id+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]} "
                f"{bristle_top_left_bottom_points_right_row[index]} {bristle_top_left_bottom_points_right_row[index]+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]}) "
                f"({partition_X} {partition_Y_outside} {partition_Z}) simpleGrading (1 1 1)\n"
            )
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    root_left_vertices_ids = set(root_ids) & set(bristle_left_ids)
    root_left_vertices_ids = root_left_vertices_ids - set(stream_right_wall) - set(stream_left_wall)
    root_left_vertices_ids_sorted = sort_ids_by_axis(vertices, root_left_vertices_ids, axis='y')
    #翅膀两侧边界，翅膀等高的部分
    bristle_ids, bristle_points_num = find_vertices(vertices, cubic_width/2-radius/(2**(0.5)), XYZ="X")
    bristle_ids_right, bristle_points_num = find_vertices(vertices, cubic_width/2+radius/(2**(0.5)), XYZ="X")
    root_ids_left_corner = sorted(list(set(root_ids)-set(bristle_left_ids_not_full_root[1:-2])-set(root_left_bottom_points_right_row)-set(root_left_bottom_points_left_row)-set(bristle_ids_right)-set(bristle_ids)))
    for index, id in enumerate(root_ids_left_corner):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {id+5} {id+4} "
                f"{top_ids_left_corner[index]} {top_ids_left_corner[index]+1} {top_ids_left_corner[index]+5} {top_ids_left_corner[index]+4}) "
                f"({partition_Y_outside} {number_of_bristles} {partition_Z}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
            f"\thex ({id} {id+1} {id+5} {id+4} "
            f"{top_ids_left_corner[index]} {top_ids_left_corner[index]+1} {top_ids_left_corner[index]+5} {top_ids_left_corner[index]+4}) "
            f"({partition_Y_outside} {number_of_bristles} {partition_Z}) simpleGrading (1 1 1)\n"
        )
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    root_bristle_vertices_ids = set(root_ids) & set(bristle_ids)
    root_bristle_vertices_ids_sorted = sort_ids_by_axis(vertices, root_bristle_vertices_ids, axis='y')
    root_bristle_vertices_ids_sorted = root_bristle_vertices_ids_sorted[::2]
    
    top_bristle_vertices_ids = set(bristle_top_ids) & set(bristle_ids)
    top_bristle_vertices_ids_sorted = sort_ids_by_axis(vertices, top_bristle_vertices_ids, axis='y')
    top_bristle_vertices_ids_sorted = top_bristle_vertices_ids_sorted[::2]
    
    bristle_top_left_vertices_ids = set(bristle_top_ids) & set(bristle_left_ids)
    bristle_top_left_vertices_ids = bristle_top_left_vertices_ids - set(stream_right_wall) - set(stream_left_wall)
    bristle_top_left_vertices_ids_sorted = sort_ids_by_axis(vertices, bristle_top_left_vertices_ids, axis='y')
    #翅膀周围那一圈的网格，翅膀根到翅膀顶
    root_patches = []
    for index, id in enumerate(root_bristle_vertices_ids_sorted):
        hex_line = (f"\thex ({root_left_vertices_ids_sorted[index]} {id} {id+2} {root_left_vertices_ids_sorted[index+1]} "
                    f"{bristle_top_left_vertices_ids_sorted[index]} {top_bristle_vertices_ids_sorted[index]} {top_bristle_vertices_ids_sorted[index]+6} {bristle_top_left_vertices_ids_sorted[index+1]}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z}) simpleGrading (0.5 1 1)\n"
                    
                    f"\thex ({root_left_vertices_ids_sorted[index]+1} {id+1} {id} {root_left_vertices_ids_sorted[index]} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+1} {top_bristle_vertices_ids_sorted[index]+1} {top_bristle_vertices_ids_sorted[index]} {bristle_top_left_vertices_ids_sorted[index]}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z}) simpleGrading (0.5 1 1)\n"
                    
                    f"\thex ({root_left_vertices_ids_sorted[index+1]+1} {id+3} {id+1} {root_left_vertices_ids_sorted[index]+1} "
                    f"{bristle_top_left_vertices_ids_sorted[index+1]+1} {top_bristle_vertices_ids_sorted[index]+7} {top_bristle_vertices_ids_sorted[index]+1} {bristle_top_left_vertices_ids_sorted[index]+1}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z}) simpleGrading (0.5 1 1)\n"
                    
                    f"\thex ({root_left_vertices_ids_sorted[index+1]} {id+2} {id+3} {root_left_vertices_ids_sorted[index+1]+1} "
                    f"{bristle_top_left_vertices_ids_sorted[index+1]} {top_bristle_vertices_ids_sorted[index]+6} {top_bristle_vertices_ids_sorted[index]+7} {bristle_top_left_vertices_ids_sorted[index+1]+1}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z}) simpleGrading (0.5 1 1)\n"
        )
        root_patch = [
            [root_left_vertices_ids_sorted[index], id, id+2, root_left_vertices_ids_sorted[index+1]],
            [root_left_vertices_ids_sorted[index]+1, id+1, id, root_left_vertices_ids_sorted[index]],
            [root_left_vertices_ids_sorted[index+1]+1, id+3, id+1, root_left_vertices_ids_sorted[index]+1],
            [root_left_vertices_ids_sorted[index+1], id+2, id+3, root_left_vertices_ids_sorted[index+1]+1]
        ]
        root_patches.extend(root_patch)
        output_blocks.append(hex_line)
    output_blocks.append("\n")
    
    top_patches = []
    #入口边界，翅膀顶
    for index, id in enumerate(bristle_top_left_bottom_points_left_row[0:-1]):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]} "
                f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num}) "
                f"({partition_X} {number_of_bristles} {partition_Z_top}) simpleGrading (1 1 1)\n"
            )
        elif index == len(bristle_top_left_bottom_points_left_row[0:-1])-1:
            block_line = (
                f"\thex ({id} {id+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]} "
                f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num}) "
                f"({partition_X} {number_of_bristles} {partition_Z_top}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
                f"\thex ({id} {id+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]} "
                f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num}) "
                f"({partition_X} {partition_Y_outside} {partition_Z_top}) simpleGrading (1 1 1)\n"
            )
        top_patches.append([id+bristle_top_points_num, id+1+bristle_top_points_num, bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num, bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num])
        output_blocks.append(block_line)
    output_blocks.append("\n")
    #出口边界，翅膀顶
    for index, id in enumerate(bristle_top_left_bottom_points_right_row[0:-1]):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]} "
                f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num}) "
                f"({partition_X} {number_of_bristles} {partition_Z_top}) simpleGrading (1 1 1)\n"
            )
        elif index == len(bristle_top_left_bottom_points_right_row[0:-1])-1:
            block_line = (
                f"\thex ({id} {id+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]} "
                f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num}) "
                f"({partition_X} {number_of_bristles} {partition_Z_top}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
                f"\thex ({id} {id+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]} "
                f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num}) "
                f"({partition_X} {partition_Y_outside} {partition_Z_top}) simpleGrading (1 1 1)\n"
            )
        top_patches.append([id+bristle_top_points_num, id+1+bristle_top_points_num, bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num, bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num])
        output_blocks.append(block_line)
    output_blocks.append("\n")
    #翅膀上下边界，翅膀顶
    for index, id in enumerate(top_ids_left_corner):
        if index == 0:
            block_line = (
                f"\thex ({id} {id+1} {id+5} {id+4} "
                f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {id+5+bristle_top_points_num} {id+4+bristle_top_points_num}) "
                f"({partition_Y_outside} {number_of_bristles} {partition_Z_top}) simpleGrading (1 1 1)\n"
            )
        else:
            block_line = (
                f"\thex ({id} {id+1} {id+5} {id+4} "
                f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {id+5+bristle_top_points_num} {id+4+bristle_top_points_num}) "
                f"({partition_Y_outside} {number_of_bristles} {partition_Z_top}) simpleGrading (1 1 1)\n"
            )
        top_patches.append([id+bristle_top_points_num, id+1+bristle_top_points_num, id+5+bristle_top_points_num, id+4+bristle_top_points_num])
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    bristle_top_vertices_ids = set(bristle_top_ids) & set(bristle_ids)
    bristle_top_vertices_ids_sorted = sort_ids_by_axis(vertices, bristle_top_vertices_ids, axis='y')
    bristle_top_vertices_ids_sorted = bristle_top_vertices_ids_sorted[::2]
    
    #翅膀顶周围那一圈的网格，翅膀顶
    for index, id in enumerate(bristle_top_vertices_ids_sorted):
        hex_line = (f"\thex ({bristle_top_left_vertices_ids_sorted[index]} {id} {id+6} {bristle_top_left_vertices_ids_sorted[index+1]} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num} {id+bristle_top_points_num} {id+6+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index+1]+bristle_top_points_num}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z_top}) simpleGrading (0.5 1 1)\n"
                    
                    f"\thex ({bristle_top_left_vertices_ids_sorted[index]+1} {id+1} {id} {bristle_top_left_vertices_ids_sorted[index]} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num} {id+1+bristle_top_points_num} {id+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z_top}) simpleGrading (0.5 1 1)\n"
                    
                    f"\thex ({bristle_top_left_vertices_ids_sorted[index+1]+1} {id+7} {id+1} {bristle_top_left_vertices_ids_sorted[index]+1} "
                    f"{bristle_top_left_vertices_ids_sorted[index+1]+1+bristle_top_points_num} {id+7+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z_top}) simpleGrading (0.5 1 1)\n"
                    
                    f"\thex ({bristle_top_left_vertices_ids_sorted[index+1]} {id+6} {id+7} {bristle_top_left_vertices_ids_sorted[index+1]+1} "
                    f"{bristle_top_left_vertices_ids_sorted[index+1]+bristle_top_points_num} {id+6+bristle_top_points_num} {id+7+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index+1]+1+bristle_top_points_num}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z_top}) simpleGrading (0.5 1 1)\n"
        )
        top_patch_outside_bristle = [
            [bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num, id+bristle_top_points_num, id+6+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index+1]+bristle_top_points_num],
            [bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num, id+1+bristle_top_points_num, id+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num],
            [bristle_top_left_vertices_ids_sorted[index+1]+1+bristle_top_points_num, id+7+bristle_top_points_num, id+1+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num],
            [bristle_top_left_vertices_ids_sorted[index+1]+bristle_top_points_num, id+6+bristle_top_points_num, id+7+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index+1]+1+bristle_top_points_num]
        ]
        top_patches.extend(top_patch_outside_bristle)
        output_blocks.append(hex_line)
    output_blocks.append("\n")
    
    inner_bristle_ids, bristle_points_num = find_vertices(vertices, cubic_width/2-radius/2/(2**(0.5)), XYZ="X")
    bristle_top_inner_left_vertices_ids = set(bristle_top_ids) & set(inner_bristle_ids)
    bristle_top_inner_vertices_ids_sorted = sort_ids_by_axis(vertices, bristle_top_inner_left_vertices_ids, axis='y')
    bristle_top_inner_vertices_ids_sorted = bristle_top_inner_vertices_ids_sorted[::2]
    
    #翅膀顶填补的网格
    top_inner_patches = []
    for index, id in enumerate(bristle_top_inner_vertices_ids_sorted):
        hex_line = (f"\thex ({id} {id+1} {id+3} {id+2} "
                    f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {id+3+bristle_top_points_num} {id+2+bristle_top_points_num}) "
                    f"({partition_Y_outside} {partition_Y_outside} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({bristle_top_vertices_ids_sorted[index]} {id} {id+2} {bristle_top_vertices_ids_sorted[index]+6} "
                    f"{bristle_top_vertices_ids_sorted[index]+bristle_top_points_num} {id+bristle_top_points_num} {id+2+bristle_top_points_num} {bristle_top_vertices_ids_sorted[index]+6+bristle_top_points_num}) "
                    f"(2 {partition_Y_outside} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({bristle_top_vertices_ids_sorted[index]+1} {id+1} {id} {bristle_top_vertices_ids_sorted[index]} "
                    f"{bristle_top_vertices_ids_sorted[index]+1+bristle_top_points_num} {id+1+bristle_top_points_num} {id+bristle_top_points_num} {bristle_top_vertices_ids_sorted[index]+bristle_top_points_num}) "
                    f"(2 {partition_Y_outside} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({bristle_top_vertices_ids_sorted[index]+7} {id+3} {id+1} {bristle_top_vertices_ids_sorted[index]+1} "
                    f"{bristle_top_vertices_ids_sorted[index]+7+bristle_top_points_num} {id+3+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_vertices_ids_sorted[index]+1+bristle_top_points_num}) "
                    f"(2 {partition_Y_outside} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({bristle_top_vertices_ids_sorted[index]+6} {id+2} {id+3} {bristle_top_vertices_ids_sorted[index]+7} "
                    f"{bristle_top_vertices_ids_sorted[index]+6+bristle_top_points_num} {id+2+bristle_top_points_num} {id+3+bristle_top_points_num} {bristle_top_vertices_ids_sorted[index]+7+bristle_top_points_num}) "
                    f"(2 {partition_Y_outside} {partition_Z_top}) simpleGrading (1 1 1)\n"
        )
        top_patch = [
            [id+bristle_top_points_num, id+1+bristle_top_points_num, id+3+bristle_top_points_num, id+2+bristle_top_points_num],
            [bristle_top_vertices_ids_sorted[index]+bristle_top_points_num, id+bristle_top_points_num, id+2+bristle_top_points_num, bristle_top_vertices_ids_sorted[index]+6+bristle_top_points_num],
            [bristle_top_vertices_ids_sorted[index]+1+bristle_top_points_num, id+1+bristle_top_points_num, id+bristle_top_points_num, bristle_top_vertices_ids_sorted[index]+bristle_top_points_num],
            [bristle_top_vertices_ids_sorted[index]+7+bristle_top_points_num, id+3+bristle_top_points_num, id+1+bristle_top_points_num, bristle_top_vertices_ids_sorted[index]+1+bristle_top_points_num],
            [bristle_top_vertices_ids_sorted[index]+6+bristle_top_points_num, id+2+bristle_top_points_num, id+3+bristle_top_points_num, bristle_top_vertices_ids_sorted[index]+7+bristle_top_points_num]
        ]
        top_patches.extend(top_patch)
        top_inner_patches.append([id, id+1, id+3, id+2])
        output_blocks.append(hex_line)
    
    # 展平二维list，提取所有唯一ID
    top_all_ids = set(pid for face in top_inner_patches for pid in face)
    bristle_root_surrounding_patch = [
                            [pid for pid in face]
                            for face in root_patches
                        ]
    # 展平二维list，提取所有唯一ID
    root_all_ids = set(pid for face in bristle_root_surrounding_patch for pid in face)

    # 获取所有对应坐标，形成列表 [(id, x, y)]
    id_xy_list = []
    for pid in top_all_ids:
        vertex = vertices.get_vertex(pid)
        if vertex:  # vertex 是 (x, y, z)
            id_xy_list.append((vertex[0], vertex[1]))
    for pid in root_all_ids:
        vertex = vertices.get_vertex(pid)
        if vertex:  # vertex 是 (x, y, z)
            id_xy_list.append((vertex[0], vertex[1]))

    # 按 Y 排序，再按 X 排序
    # solid_blocks_xy_vertices = sorted(id_xy_list, key=lambda item: (item[1], item[0]))  # item = (id, x, y)
    
    output_blocks.append("\n")
    output_blocks.append(");\n\n")
    return output_blocks, top_patches, root_patches, root_bristle_vertices_ids_sorted, id_xy_list, bristle_top_vertices_ids_sorted, bristle_top_points_num, bottom_ids_left_corner
    
def generate_solid_blocks(vertices, root_block_width, root_block_hight, bristle_length, radius, partition_X, partition_Y, partition_Z):
    output_blocks = ["blocks\n(\n"]

    bottom_ids, bottom_points_num = find_vertices(vertices, 0, XYZ="Z")
    root_ids, root_points_num = find_vertices(vertices, root_block_hight, XYZ="Z")
    top_ids, top_points_num = find_vertices(vertices, root_block_hight+bristle_length, XYZ="Z")
    #底层外框
    bristle_left_ids, bristle_left_points_num = find_vertices(vertices, cubic_width/2-root_block_width/2, XYZ="X")
    bottom_left_vertices_ids = set(bottom_ids) & set(bristle_left_ids)
    bottom_left_vertices_ids_sorted = sort_ids_by_axis(vertices, bottom_left_vertices_ids, axis='y')
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
    
    for index, id in enumerate(cylinder_inner_left_ids_sorted):
        #底层鬃毛内圈
        hex_line = f"\thex ({id} {id+1} {id+3} {id+2} {id+bottom_points_num} {id+1+bottom_points_num} {id+3+bottom_points_num} {id+2+bottom_points_num}) ({partition_Y} {partition_Y} {8*G_D}) simpleGrading (1 1 1)\n"
        output_blocks.append(hex_line)
        #底层鬃毛外圈
        cylinder_hex_line = (
            f"\thex ({cylinder_left_ids_sorted[index]} {id} {id+2} {cylinder_left_ids_sorted[index]+6} " 
            f"{cylinder_left_ids_sorted[index]+bottom_points_num} {id+bottom_points_num} {id+2+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num}) "
            f"({partition_Y} {partition_Y} {8*G_D}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+1} {id+1} {id} {cylinder_left_ids_sorted[index]} " 
            f"{cylinder_left_ids_sorted[index]+1+bottom_points_num} {id+1+bottom_points_num} {id+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num}) "
            f"({partition_Y} {partition_Y} {8*G_D}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+7} {id+3} {id+1} {cylinder_left_ids_sorted[index]+1} " 
            f"{cylinder_left_ids_sorted[index]+7+bottom_points_num} {id+3+bottom_points_num} {id+1+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num}) "
            f"({partition_Y} {partition_Y} {8*G_D}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+6} {id+2} {id+3} {cylinder_left_ids_sorted[index]+7} " 
            f"{cylinder_left_ids_sorted[index]+6+bottom_points_num} {id+2+bottom_points_num} {id+3+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num}) "
            f"({partition_Y} {partition_Y} {8*G_D}) simpleGrading (1 1 1)\n"
        )
        output_blocks.append(cylinder_hex_line)
        #底层基座
        cylinder_out_hex_line = (
            f"\thex ({bottom_left_vertices_ids_sorted[index]} {cylinder_left_ids_sorted[index]} {cylinder_left_ids_sorted[index]+6} {bottom_left_vertices_ids_sorted[index+1]} " 
            f"{bottom_left_vertices_ids_sorted[index]+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num} {bottom_left_vertices_ids_sorted[index+1]+bottom_points_num}) "
            f"({partition_Y*2} {partition_Y} {8*G_D}) simpleGrading (0.2 1 1)\n"
            
            f"\thex ({bottom_left_vertices_ids_sorted[index]+1} {cylinder_left_ids_sorted[index]+1} {cylinder_left_ids_sorted[index]} {bottom_left_vertices_ids_sorted[index]} " 
            f"{bottom_left_vertices_ids_sorted[index]+1+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num} {bottom_left_vertices_ids_sorted[index]+bottom_points_num}) "
            f"({partition_Y*2} {partition_Y} {8*G_D}) simpleGrading (0.2 1 1)\n"
            
            f"\thex ({bottom_left_vertices_ids_sorted[index+1]+1} {cylinder_left_ids_sorted[index]+7} {cylinder_left_ids_sorted[index]+1} {bottom_left_vertices_ids_sorted[index]+1} " 
            f"{bottom_left_vertices_ids_sorted[index+1]+1+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num} {bottom_left_vertices_ids_sorted[index]+1+bottom_points_num}) "
            f"({partition_Y*2} {partition_Y} {8*G_D}) simpleGrading (0.2 1 1)\n"
            
            f"\thex ({bottom_left_vertices_ids_sorted[index+1]} {cylinder_left_ids_sorted[index]+6} {cylinder_left_ids_sorted[index]+7} {bottom_left_vertices_ids_sorted[index+1]+1} " 
            f"{bottom_left_vertices_ids_sorted[index+1]+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num} {bottom_left_vertices_ids_sorted[index+1]+1+bottom_points_num}) "
            f"({partition_Y*2} {partition_Y} {8*G_D}) simpleGrading (0.2 1 1)\n"
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
    
def generate_edges(bristle_length, root_block_hight, cubic_width, cubic_length, root_bristle_vertices_ids_sorted, bristle_top_vertices_ids_sorted, bristle_top_points_num):
    
    def edge_generation(ids, index, z):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_width/2+radius*np.sin(alpha)} {cubic_length/2-root_block_length/2-radius*np.cos(beta)+(index+1/2)*root_block_length/num_bristles} {z})\n"
            alpha += np.pi/2
            beta += np.pi/2
            output_edges.append(edge_line)

    output_edges = ["edges\n(\n"]
    
    for index, id in enumerate(root_bristle_vertices_ids_sorted):
        root_out_circle_ids = [id, id+1, id+3, id+2]
        edge_generation(root_out_circle_ids, index, root_block_hight)
    output_edges.append("\n")
    for index, id in enumerate(bristle_top_vertices_ids_sorted):
        bristle_out_circle_ids = [id, id+1, id+7, id+6]
        edge_generation(bristle_out_circle_ids, index, root_block_hight+bristle_length)
    output_edges.append("\n")
    for index, id in enumerate(bristle_top_vertices_ids_sorted):
        top_out_circle_ids = [id+bristle_top_points_num, id+1+bristle_top_points_num, id+7+bristle_top_points_num, id+6+bristle_top_points_num]
        edge_generation(top_out_circle_ids, index, root_block_hight+bristle_length*1.5)

    output_edges.append(");\n\n")
    return output_edges

def generate_solid_edges(cubic_width, cubic_length, radius, bristle_length, root_block_hight, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num):
    output_edges = ["edges\n(\n"]
    
    def edge_generation(ids, index, z):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_width/2+radius*np.sin(alpha)} {cubic_length/2-root_block_length/2-radius*np.cos(beta)+(index+1/2)*root_block_length/num_bristles} {z})\n"
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

def generate_patches(vertices, root_block_hight, top_patches, root_patches, bristle_length, cubic_width, cubic_length, root_bristle_vertices_ids_sorted, root_block_width, radius):
    output_patches = ["patches\n(\n"]
    output_patches.append("\tpatch bottom\n")
    output_patches.append("\t(\n")
    bottom_ids, bottom_points_num = find_left_bottom_vertices_simple(vertices, 0, XYZ="Z")
    bristle_left_ids, bristle_left_points_num = find_vertices(vertices, cubic_width/2-root_block_width/2, XYZ="X")
    bristle_left_ids_not_full = [i for i in bristle_left_ids if i<= bottom_points_num]
    bristle_left_ids_not_full = bristle_left_ids_not_full[1:-2]
    bottom_ids_left_corner = set(bottom_ids)-set(bristle_left_ids_not_full)
    for id in bottom_ids_left_corner:
        output_patches.append(f"\t\t({id+4} {id+5} {id+1} {id})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch top\n")
    output_patches.append("\t(\n")
    root_ids, root_points_num = find_left_bottom_vertices_simple(vertices, root_block_hight, XYZ="Z")
    bristle_top_ids, bristle_top_points_num = find_left_bottom_vertices_simple(vertices, root_block_hight+bristle_length, XYZ="Z")
    for ids in top_patches:
        output_patches.append(f"\t\t({ids[0]} {ids[1]} {ids[2]} {ids[3]})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch inlet\n")
    output_patches.append("\t(\n")
    inlet_left_coner_ids, inlet_points_num = find_vertices(vertices, 0, "X")
    bottom_left_coner_ids, bottom_points_num = find_vertices(vertices, 0, "Z")
    root_left_coner_ids, root_points_num = find_vertices(vertices, root_block_hight, "Z")
    bristle_top_left_coner_ids, bristle_top_points_num = find_vertices(vertices, root_block_hight+bristle_length, "Z")
    roof_left_coner_ids, roof_points_num = find_vertices(vertices, root_block_hight+bristle_length*1.5, "Z")
    inlet_bottom_left_coner_ids = sorted(list(set(inlet_left_coner_ids) & set(bottom_left_coner_ids)))
    inlet_root_left_coner_ids = sorted(list(set(inlet_left_coner_ids) & set(root_left_coner_ids)))
    inlet_bristle_top_left_coner_ids = sorted(list(set(inlet_left_coner_ids) & set(bristle_top_left_coner_ids)))
    inlet_roof_left_coner_ids = sorted(list(set(inlet_left_coner_ids) & set(roof_left_coner_ids)))
    for index, id in enumerate(inlet_bottom_left_coner_ids[0:-1]):
        output_patches.append(f"\t\t({inlet_root_left_coner_ids[index]} {inlet_root_left_coner_ids[index+1]} {inlet_bottom_left_coner_ids[index+1]} {inlet_bottom_left_coner_ids[index]})\n")
        output_patches.append(f"\t\t({inlet_bristle_top_left_coner_ids[index]} {inlet_bristle_top_left_coner_ids[index+1]} {inlet_root_left_coner_ids[index+1]} {inlet_root_left_coner_ids[index]})\n")
        output_patches.append(f"\t\t({inlet_roof_left_coner_ids[index]} {inlet_roof_left_coner_ids[index+1]} {inlet_bristle_top_left_coner_ids[index+1]} {inlet_bristle_top_left_coner_ids[index]})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch outlet\n")
    output_patches.append("\t(\n")
    outlet_left_coner_ids, outlet_points_num = find_vertices(vertices, cubic_width, "X")
    outlet_bottom_left_coner_ids = sorted(list(set(outlet_left_coner_ids) & set(bottom_left_coner_ids)))
    outlet_root_left_coner_ids = sorted(list(set(outlet_left_coner_ids) & set(root_left_coner_ids)))
    outlet_roof_left_coner_ids = sorted(list(set(outlet_left_coner_ids) & set(roof_left_coner_ids)))
    outlet_bristle_top_left_coner_ids = sorted(list(set(outlet_left_coner_ids) & set(bristle_top_left_coner_ids)))
    for index, id in enumerate(outlet_bottom_left_coner_ids[0:-1]):
        output_patches.append(f"\t\t({outlet_bottom_left_coner_ids[index]} {outlet_bottom_left_coner_ids[index+1]} {outlet_root_left_coner_ids[index+1]} {outlet_root_left_coner_ids[index]})\n")
        output_patches.append(f"\t\t({outlet_root_left_coner_ids[index]} {outlet_root_left_coner_ids[index+1]} {outlet_bristle_top_left_coner_ids[index+1]} {outlet_bristle_top_left_coner_ids[index]})\n")
        output_patches.append(f"\t\t({outlet_bristle_top_left_coner_ids[index]} {outlet_bristle_top_left_coner_ids[index+1]} {outlet_roof_left_coner_ids[index+1]} {outlet_roof_left_coner_ids[index]})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch bristle\n")
    output_patches.append("\t(\n")
    root_block_corner = bristle_left_ids[1:-1]
    bottom_block_left_up_corner = sorted(list(set(bottom_ids) & set(root_block_corner)))
    bottom_block_left_up_corner_ids_sorted = sort_ids_by_axis(vertices, bottom_block_left_up_corner, axis='y')
    root_block_left_up_corner = sorted(list(set(root_left_coner_ids) & set(root_block_corner)))
    root_block_left_up_corner_ids_sorted = sort_ids_by_axis(vertices, root_block_left_up_corner, axis='y')
    root_block_left_up_corner_ids_sorted = root_block_left_up_corner_ids_sorted[1:-1]
    
    bristle_out_left_coner_ids, bristle_top_inner_points_num = find_vertices(vertices, cubic_width/2-radius/(2**(0.5)), "X")
    top_block_left_up_corner = sorted(list(set(bristle_top_left_coner_ids) & set(bristle_out_left_coner_ids)))
    top_block_left_up_corner_ids_sorted = sort_ids_by_axis(vertices, top_block_left_up_corner, axis='y')
    top_block_left_up_corner_ids_sorted = top_block_left_up_corner_ids_sorted[::2]
    for index, id in enumerate(bottom_block_left_up_corner_ids_sorted[0:-1]):
        root_block_patches = (f"\t\t({bottom_block_left_up_corner_ids_sorted[index+1]} {root_block_left_up_corner_ids_sorted[index+1]} {root_block_left_up_corner_ids_sorted[index]} {id})\n"
                              f"\t\t({bottom_block_left_up_corner_ids_sorted[index+1]+1} {root_block_left_up_corner_ids_sorted[index+1]+1} {root_block_left_up_corner_ids_sorted[index]+1} {id+1})\n"
                              )
        output_patches.append(root_block_patches)
    root_block_side_patches = (
        f"\t\t({bottom_block_left_up_corner_ids_sorted[0]} {root_block_left_up_corner_ids_sorted[0]} {root_block_left_up_corner_ids_sorted[0]+1} {bottom_block_left_up_corner_ids_sorted[0]+1})\n"
        f"\t\t({bottom_block_left_up_corner_ids_sorted[-1]+1} {root_block_left_up_corner_ids_sorted[-1]+1} {root_block_left_up_corner_ids_sorted[-1]} {bottom_block_left_up_corner_ids_sorted[-1]})\n"
    )
    output_patches.append(root_block_side_patches)
    for i in range(len(root_patches)):
        output_patches.append(f"\t\t({root_patches[i][3]} {root_patches[i][2]} {root_patches[i][1]} {root_patches[i][0]})\n")
    
    bristle_top_inner_left_coner_ids, bristle_top_inner_points_num = find_vertices(vertices, cubic_width/2-radius/2/(2**(0.5)), "X")
    bristle_top_inner_left_vertices_ids = set(bristle_top_left_coner_ids) & set(bristle_top_inner_left_coner_ids)
    bristle_top_inner_left_ids_sorted = sort_ids_by_axis(vertices, bristle_top_inner_left_vertices_ids, axis='y')
    bristle_top_inner_left_ids_sorted = bristle_top_inner_left_ids_sorted[::2]
    for index, id in enumerate(root_bristle_vertices_ids_sorted):
        bristle_top_patch = (
            f"\t\t({bristle_top_inner_left_ids_sorted[index]+2} {bristle_top_inner_left_ids_sorted[index]+3} {bristle_top_inner_left_ids_sorted[index]+1} {bristle_top_inner_left_ids_sorted[index]})\n"
            f"\t\t({top_block_left_up_corner_ids_sorted[index]+6} {bristle_top_inner_left_ids_sorted[index]+2} {bristle_top_inner_left_ids_sorted[index]} {top_block_left_up_corner_ids_sorted[index]})\n"
            f"\t\t({top_block_left_up_corner_ids_sorted[index]} {bristle_top_inner_left_ids_sorted[index]} {bristle_top_inner_left_ids_sorted[index]+1} {top_block_left_up_corner_ids_sorted[index]+1})\n"
            f"\t\t({top_block_left_up_corner_ids_sorted[index]+1} {bristle_top_inner_left_ids_sorted[index]+1} {bristle_top_inner_left_ids_sorted[index]+3} {top_block_left_up_corner_ids_sorted[index]+7})\n"
            f"\t\t({top_block_left_up_corner_ids_sorted[index]+7} {bristle_top_inner_left_ids_sorted[index]+3} {bristle_top_inner_left_ids_sorted[index]+2} {top_block_left_up_corner_ids_sorted[index]+6})\n"
        )
        output_patches.append(bristle_top_patch)
        bristle_side_patch = (
            f"\t\t({id+2} {top_block_left_up_corner_ids_sorted[index]+6} {top_block_left_up_corner_ids_sorted[index]} {id})\n"
            f"\t\t({id} {top_block_left_up_corner_ids_sorted[index]} {top_block_left_up_corner_ids_sorted[index]+1} {id+1})\n"
            f"\t\t({id+1} {top_block_left_up_corner_ids_sorted[index]+1} {top_block_left_up_corner_ids_sorted[index]+7} {id+3})\n"
            f"\t\t({id+3} {top_block_left_up_corner_ids_sorted[index]+7} {top_block_left_up_corner_ids_sorted[index]+6} {id+2})\n"
        )
        output_patches.append(bristle_side_patch)
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch frontAndBackPlanes\n")
    output_patches.append("\t(\n")
    stream_right_wall_left_coner_ids, stream_right_wall_points_num = find_vertices(vertices, 0, "Y")
    stream_left_wall_left_coner_ids, stream_left_wall_points_num = find_vertices(vertices, cubic_length, "Y")
    stream_right_wall_bottom_left_coner_ids = sorted(list(set(stream_right_wall_left_coner_ids) & set(bottom_left_coner_ids)))
    stream_right_wall_root_left_coner_ids = sorted(list(set(stream_right_wall_left_coner_ids) & set(root_left_coner_ids)))
    stream_right_wall_top_left_coner_ids = sorted(list(set(stream_right_wall_left_coner_ids) & set(bristle_top_left_coner_ids)))
    stream_right_wall_roof_left_coner_ids = sorted(list(set(stream_right_wall_left_coner_ids) & set(roof_left_coner_ids)))
    stream_left_wall_bottom_left_coner_ids = sorted(list(set(stream_left_wall_left_coner_ids) & set(bottom_left_coner_ids)))
    stream_left_wall_root_left_coner_ids = sorted(list(set(stream_left_wall_left_coner_ids) & set(root_left_coner_ids)))
    stream_left_wall_top_left_coner_ids = sorted(list(set(stream_left_wall_left_coner_ids) & set(bristle_top_left_coner_ids)))
    stream_left_wall_roof_left_coner_ids = sorted(list(set(stream_left_wall_left_coner_ids) & set(roof_left_coner_ids)))
    for index, id in enumerate(stream_right_wall_bottom_left_coner_ids[0:-1]):
        output_patches.append(f"\t\t({stream_right_wall_bottom_left_coner_ids[index]} {stream_right_wall_bottom_left_coner_ids[index+1]} {stream_right_wall_root_left_coner_ids[index+1]} {stream_right_wall_root_left_coner_ids[index]})\n")
        output_patches.append(f"\t\t({stream_right_wall_root_left_coner_ids[index]} {stream_right_wall_root_left_coner_ids[index+1]} {stream_right_wall_top_left_coner_ids[index+1]} {stream_right_wall_top_left_coner_ids[index]})\n")
        output_patches.append(f"\t\t({stream_right_wall_top_left_coner_ids[index]} {stream_right_wall_top_left_coner_ids[index+1]} {stream_right_wall_roof_left_coner_ids[index+1]} {stream_right_wall_roof_left_coner_ids[index]})\n")
        
        output_patches.append(f"\t\t({stream_left_wall_bottom_left_coner_ids[index]} {stream_left_wall_root_left_coner_ids[index]} {stream_left_wall_root_left_coner_ids[index+1]} {stream_left_wall_bottom_left_coner_ids[index+1]})\n")
        output_patches.append(f"\t\t({stream_left_wall_root_left_coner_ids[index]} {stream_left_wall_top_left_coner_ids[index]} {stream_left_wall_top_left_coner_ids[index+1]} {stream_left_wall_root_left_coner_ids[index+1]})\n")
        output_patches.append(f"\t\t({stream_left_wall_top_left_coner_ids[index]} {stream_left_wall_roof_left_coner_ids[index]} {stream_left_wall_roof_left_coner_ids[index+1]} {stream_left_wall_top_left_coner_ids[index+1]})\n")
    
    output_patches.append("\t)\n")
    output_patches.append(");\n\n")

    return output_patches

def generate_solid_patches(bottom_left_vertices_ids_sorted, bottom_points_num, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted):    
    output_patches = ["patches\n(\n"]
    output_patches.append("\tpatch bristle\n")
    output_patches.append("\t(\n")
    for index, id in enumerate(cylinder_left_ids_sorted):
        output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[index]} {bottom_left_vertices_ids_sorted[index]+bottom_points_num} {bottom_left_vertices_ids_sorted[index+1]+bottom_points_num} {bottom_left_vertices_ids_sorted[index+1]})\n")
        output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[index+1]+1} {bottom_left_vertices_ids_sorted[index+1]+bottom_points_num+1} {bottom_left_vertices_ids_sorted[index]+bottom_points_num+1} {bottom_left_vertices_ids_sorted[index]+1})\n")
        
        bristle_root_patch = (
            f"\t\t({bottom_left_vertices_ids_sorted[index]+bottom_points_num} {id+bottom_points_num} "
            f"{id+6+bottom_points_num} {bottom_left_vertices_ids_sorted[index+1]+bottom_points_num})\n"
            
            f"\t\t({bottom_left_vertices_ids_sorted[index]+1+bottom_points_num} {id+1+bottom_points_num} "
            f"{id+bottom_points_num} {bottom_left_vertices_ids_sorted[index]+bottom_points_num})\n"
            
            f"\t\t({bottom_left_vertices_ids_sorted[index+1]+1+bottom_points_num} {id+7+bottom_points_num} "
            f"{id+1+bottom_points_num} {bottom_left_vertices_ids_sorted[index]+1+bottom_points_num})\n"
            
            f"\t\t({bottom_left_vertices_ids_sorted[index+1]+bottom_points_num} {id+6+bottom_points_num} "
            f"{id+7+bottom_points_num} {bottom_left_vertices_ids_sorted[index+1]+1+bottom_points_num})\n"
        )
        output_patches.append(bristle_root_patch)
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
        
    output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[0]} {bottom_left_vertices_ids_sorted[0]+1} {bottom_left_vertices_ids_sorted[0]+1+bottom_points_num} {bottom_left_vertices_ids_sorted[0]+bottom_points_num})\n")
    output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[-1]+1} {bottom_left_vertices_ids_sorted[-1]} {bottom_left_vertices_ids_sorted[-1]+bottom_points_num} {bottom_left_vertices_ids_sorted[-1]+1+bottom_points_num})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch plateFix\n")
    output_patches.append("\t(\n")
    
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
        
        bottom_patch = (
            f"\t\t({bottom_left_vertices_ids_sorted[index+1]} {id+6} "
            f"{id} {bottom_left_vertices_ids_sorted[index]})\n"
            
            f"\t\t({bottom_left_vertices_ids_sorted[index]} {id} "
            f"{id+1} {bottom_left_vertices_ids_sorted[index]+1})\n"
            
            f"\t\t({bottom_left_vertices_ids_sorted[index]+1} {id+1} "
            f"{id+7} {bottom_left_vertices_ids_sorted[index+1]+1})\n"
            
            f"\t\t({bottom_left_vertices_ids_sorted[index+1]+1} {id+7} "
            f"{id+6} {bottom_left_vertices_ids_sorted[index+1]})\n"
        )
        output_patches.append(bottom_patch)
    
    output_patches.append("\t)\n\n")
    
    # output_patches.append("\tpatch bottomPlate\n")
    # output_patches.append("\t(\n")
    # for index, id in enumerate(cylinder_left_ids_sorted):
    #     bottom_patch = (
    #         f"\t\t({bottom_left_vertices_ids_sorted[index+1]} {id+6} "
    #         f"{id} {bottom_left_vertices_ids_sorted[index]})\n"
            
    #         f"\t\t({bottom_left_vertices_ids_sorted[index]} {id} "
    #         f"{id+1} {bottom_left_vertices_ids_sorted[index]+1})\n"
            
    #         f"\t\t({bottom_left_vertices_ids_sorted[index]+1} {id+1} "
    #         f"{id+7} {bottom_left_vertices_ids_sorted[index+1]+1})\n"
            
    #         f"\t\t({bottom_left_vertices_ids_sorted[index+1]+1} {id+7} "
    #         f"{id+6} {bottom_left_vertices_ids_sorted[index+1]})\n"
    #     )
    #     output_patches.append(bottom_patch)
    # output_patches.append("\t)\n\n")
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
fluid_mesh = "fluid/constant/polyMesh/blockMeshDict"
head = generate_FOAM_head()

G_D = 3
bristle_length = 80
radius = 0.5 # 140是翅尖部分的bristle长度，1.4是实际bristle的直径
num_bristles = 7
bristle_gap = radius * 2 * 5 * G_D # 这个数字是 gap/diameter

# mesh_size = radius / (partition_XY * 3 / 2)
if G_D == 1:
    outside_bristle_partition_half = 5
    partition_XY = 18
    outside_partition_Y = 45
elif G_D == 2:
    outside_bristle_partition_half = 8
    partition_XY = 16
    outside_partition_Y = 50
elif G_D == 3:
    outside_bristle_partition_half = 11
    partition_XY = 14
    outside_partition_Y = 56
partition_Z_top = 50
partition_Z = 120# int(bristle_length / mesh_size)

root_block_hight = G_D * 5 + radius * 2
root_block_length = (radius * 2 + bristle_gap) * num_bristles
root_block_width = radius * 2 + bristle_gap
cubic_width = 50
cubic_length = 100 + G_D * 50


vertices, solid_blocks_xy_vertices = generate_vertices(cubic_width, cubic_length, radius, bristle_length, num_bristles, bristle_gap, root_block_hight, root_block_length, root_block_width)
blocks, top_patches, root_patches, root_bristle_vertices_ids_sorted, id_xy_list, bristle_top_vertices_ids_sorted, bristle_top_points_num, bottom_ids_left_corner = generate_blocks(vertices, bristle_length, partition_XY, outside_bristle_partition_half, partition_Z, root_block_hight, root_block_width, cubic_length, radius, outside_partition_Y, partition_Z_top)
edges = generate_edges(bristle_length, root_block_hight, cubic_width, cubic_length, root_bristle_vertices_ids_sorted, bristle_top_vertices_ids_sorted, bristle_top_points_num)
patches = generate_patches(vertices, root_block_hight, top_patches, root_patches, bristle_length, cubic_width, cubic_length, root_bristle_vertices_ids_sorted, root_block_width, radius)
end = generate_ends()
# **修正写入文件的方式**
with open(fluid_mesh, 'w') as file:
    file.write(head)
    file.write(vertices.get_output())  # **修正点**
    file.write("".join(blocks))
    file.write("".join(edges))
    file.write("".join(patches))
    file.write("".join(end))

solid_partition_XY = 4
solid_partition_Z = 100
solid_mesh = "solid/constant/polyMesh/blockMeshDict"#"blockMeshDict.solid"
solid_vertices = generate_solid_vertices(solid_blocks_xy_vertices, root_block_hight, bristle_length, root_block_width)
solid_blocks, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num, bottom_left_vertices_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted = generate_solid_blocks(solid_vertices, root_block_width, root_block_hight, bristle_length, radius, solid_partition_XY, solid_partition_XY, solid_partition_Z)
solid_edges = generate_solid_edges(cubic_width, cubic_length, radius, bristle_length, root_block_hight, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num)
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