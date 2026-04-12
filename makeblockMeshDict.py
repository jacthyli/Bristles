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

convertToMeters 1;
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
        sorted_points = sorted(self.id_to_vertex.values(), key=lambda p: (round(p[2], 4), round(p[1], 4), round(p[0], 4)))

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

def get_centers(num_bristles, cubic_width, cubic_length, bristle_gap, radius_base):
    centers = [[cubic_width / 2, cubic_length / 2]]
    count = (num_bristles - 1) / 2
    for i in range(1, num_bristles):
        offset = count * (bristle_gap + radius_base * 2)
        direction = np.cos(i * np.pi)
        centers.append([cubic_width / 2, cubic_length / 2 + direction * offset])
        if i % 2 == 0 and i != 0:
            count -= 1
    return centers

def scale_xy(x, y, centers, radius_base, radius_top):
    for cx, cy in centers:
        dist = np.hypot(x - cx, y - cy)
        # Check if it matches radius_base, radius_base*2, or radius_base*0.7
        if np.isclose(dist, radius_base) or np.isclose(dist, radius_base * 2) or np.isclose(dist, radius_base * 0.7):
            scale = radius_top / radius_base
            return cx + (x - cx) * scale, cy + (y - cy) * scale
    return x, y

def generate_vertices(cubic_width, cubic_length, radius_base, radius_top, bristle_length, num_bristles, bristle_gap, root_block_hight, root_block_length, root_block_width, G_D):
    vertices_manager = VertexManager()
    solid_blocks_xy_vertices = []
    bottom_vertices = [
        [0, 0, 0],
        [cubic_width/2-root_block_width/2, 0, 0],
        [cubic_width/2+root_block_width/2, 0, 0],
        [cubic_width+50, 0, 0],
        [0, cubic_length/2-root_block_length/2, 0],
        [cubic_width/2-root_block_width/2, cubic_length/2-root_block_length/2, 0],
        [cubic_width/2+root_block_width/2, cubic_length/2-root_block_length/2, 0],
        [cubic_width+50, cubic_length/2-root_block_length/2, 0]
    ]
    for i in range(num_bristles-1):
        solid_blocks_xy = [
            [cubic_width/2-root_block_width/2, cubic_length/2-root_block_length/2 + (i+1)*(bristle_gap+radius_base*2), 0],
            [cubic_width/2+root_block_width/2, cubic_length/2-root_block_length/2 + (i+1)*(bristle_gap+radius_base*2), 0]
        ]
        bottom_middle_sector = [
            [0, cubic_length/2-root_block_length/2 + (i+1)*(bristle_gap+radius_base*2), 0],
            [cubic_width/2-root_block_width/2, cubic_length/2-root_block_length/2 + (i+1)*(bristle_gap+radius_base*2), 0],
            [cubic_width/2+root_block_width/2, cubic_length/2-root_block_length/2 + (i+1)*(bristle_gap+radius_base*2), 0],
            [cubic_width+50, cubic_length/2-root_block_length/2 + (i+1)*(bristle_gap+radius_base*2), 0]
        ]
        solid_blocks_xy_vertices.extend(solid_blocks_xy)
        bottom_vertices.extend(bottom_middle_sector)
    for i in range(num_bristles):
        bottom_middle_sector = [
            [0, cubic_length/2-root_block_length/2 + (i+1/2)*(bristle_gap+radius_base*2) - root_block_width/2 , 0],
            [cubic_width/2-root_block_width/2, cubic_length/2-root_block_length/2 + (i+1/2)*(bristle_gap+radius_base*2) - root_block_width/2, 0],
            [cubic_width/2+root_block_width/2, cubic_length/2-root_block_length/2 + (i+1/2)*(bristle_gap+radius_base*2) - root_block_width/2, 0],
            [cubic_width+50, cubic_length/2-root_block_length/2 + (i+1/2)*(bristle_gap+radius_base*2) - root_block_width/2, 0],
            [0, cubic_length/2-root_block_length/2 + (i+1/2)*(bristle_gap+radius_base*2) + root_block_width/2 , 0],
            [cubic_width/2-root_block_width/2, cubic_length/2-root_block_length/2 + (i+1/2)*(bristle_gap+radius_base*2) + root_block_width/2, 0],
            [cubic_width/2+root_block_width/2, cubic_length/2-root_block_length/2 + (i+1/2)*(bristle_gap+radius_base*2) + root_block_width/2, 0],
            [cubic_width+50, cubic_length/2-root_block_length/2 + (i+1/2)*(bristle_gap+radius_base*2) + root_block_width/2, 0]
        ]
        bottom_vertices.extend(bottom_middle_sector)
    bottom_top_sector = [
        [0, cubic_length/2+root_block_length/2, 0],
        [cubic_width/2-root_block_width/2, cubic_length/2+root_block_length/2, 0],
        [cubic_width/2+root_block_width/2, cubic_length/2+root_block_length/2, 0],
        [cubic_width+50, cubic_length/2+root_block_length/2, 0],
        [0, cubic_length, 0],
        [cubic_width/2-root_block_width/2, cubic_length, 0],
        [cubic_width/2+root_block_width/2, cubic_length, 0],
        [cubic_width+50, cubic_length, 0]
    ]
    solid_blocks_xy = [
        [cubic_width/2-root_block_width/2, cubic_length/2+root_block_length/2, 0],
        [cubic_width/2+root_block_width/2, cubic_length/2+root_block_length/2, 0],
        [cubic_width/2-root_block_width/2, cubic_length/2-root_block_length/2, 0],
        [cubic_width/2+root_block_width/2, cubic_length/2-root_block_length/2, 0]
    ]
    solid_blocks_xy_vertices.extend(solid_blocks_xy)
    bottom_vertices.extend(bottom_top_sector)
    vertices_manager.add_vertices(bottom_vertices)
    
    # === 1. 生成 root_vertices（Z = 0，不包含 inner_circle_points） ===
    root_vertices = [[x, y, root_block_hight] for x, y, _ in bottom_vertices]
    centers = get_centers(num_bristles, cubic_width, cubic_length, bristle_gap, radius_base)

    bristle_bit_vertices = [] 

    for idx, (cx, cy) in enumerate(centers):
        out_circle_points = bristle_points(cx, cy, radius_base)
        add_rest_vertices = [
            [out_circle_points[0][0], out_circle_points[0][1], root_block_hight],
            [out_circle_points[1][0], out_circle_points[1][1], root_block_hight],
            [out_circle_points[3][0], out_circle_points[3][1], root_block_hight],
            [out_circle_points[2][0], out_circle_points[2][1], root_block_hight]
        ]
        bristle_bit_vertices.extend(add_rest_vertices)
        root_vertices.extend(add_rest_vertices)
        
        middle_circle_points_i = bristle_points(cx, cy, radius_base * 2)
        middle_layer_vertices_i = [
            [middle_circle_points_i[0][0], middle_circle_points_i[0][1], root_block_hight],
            [middle_circle_points_i[1][0], middle_circle_points_i[1][1], root_block_hight],
            [middle_circle_points_i[3][0], middle_circle_points_i[3][1], root_block_hight],
            [middle_circle_points_i[2][0], middle_circle_points_i[2][1], root_block_hight]
        ]
        root_vertices.extend(middle_layer_vertices_i)
        
        middle_layer_4_solid_blocks_xy_i = [
            [cubic_width/2-root_block_width/2, cy-root_block_width/2, 0],
            [cubic_width/2+root_block_width/2, cy-root_block_width/2, 0],
            [cubic_width/2-root_block_width/2, cy+root_block_width/2, 0],
            [cubic_width/2+root_block_width/2, cy+root_block_width/2, 0]
        ]
        solid_blocks_xy_vertices.extend(middle_layer_4_solid_blocks_xy_i)

        middle_layer_4_solid_blocks_xy_middle_i = [
            [middle_circle_points_i[0][0], middle_circle_points_i[0][1], 0],
            [middle_circle_points_i[1][0], middle_circle_points_i[1][1], 0],
            [middle_circle_points_i[3][0], middle_circle_points_i[3][1], 0],
            [middle_circle_points_i[2][0], middle_circle_points_i[2][1], 0]
        ]
        solid_blocks_xy_vertices.extend(middle_layer_4_solid_blocks_xy_middle_i)

    solid_blocks_xy_vertices.extend(bristle_bit_vertices)
    vertices_manager.add_vertices(root_vertices)

    # === 2. 生成 bristle_end_vertices ===
    bristle_end_vertices = []
    for x, y, _ in root_vertices:
        nx, ny = scale_xy(x, y, centers, radius_base, radius_top)
        bristle_end_vertices.append([nx, ny, bristle_length + root_block_hight])
        
    bristle_inner_vertices = []
    for cx, cy in centers:
        inner_circle_points_top = bristle_points(cx, cy, radius_top * 0.7)
        bristle_end_vertices += [[x, y, bristle_length + root_block_hight] for x, y in inner_circle_points_top]
        
        # for solid_blocks_xy_vertices base projection
        inner_circle_points_base = bristle_points(cx, cy, radius_base * 0.7)
        bristle_inner_vertices += [[x, y, bristle_length] for x, y in inner_circle_points_base]

    solid_blocks_xy_vertices.extend(bristle_inner_vertices)
    vertices_manager.add_vertices(bristle_end_vertices)

    # === 3. 生成 roof_vertices ===
    roof_vertices = [[x, y, bristle_length*1.5 + root_block_hight] for x, y, _ in bristle_end_vertices]
    vertices_manager.add_vertices(roof_vertices)
    vertices_manager.sort_vertices_by_zyx()

    return vertices_manager, solid_blocks_xy_vertices

def generate_solid_vertices(solid_blocks_xy_vertices, root_block_hight, bristle_length, root_block_width, centers, radius_base, radius_top):
    vertices_manager = VertexManager()
    
    # === 1. 生成 root_vertices ===
    root_vertices = [[x, y, 0] for x, y, _ in solid_blocks_xy_vertices]
    vertices_manager.add_vertices(root_vertices)

    root_vertices = [[x, y, root_block_hight] for x, y, _ in solid_blocks_xy_vertices]
    vertices_manager.add_vertices(root_vertices)
    
    # === 2. 生成 bristle_end_vertices ===
    bristle_end_vertices = []
    for x, y, _ in solid_blocks_xy_vertices:
        if not np.isclose(x, root_block_width) and not np.isclose(x, root_block_width * 2):
            nx, ny = scale_xy(x, y, centers, radius_base, radius_top)
            bristle_end_vertices.append([nx, ny, root_block_hight+bristle_length])
            
    vertices_manager.add_vertices(bristle_end_vertices)
    vertices_manager.sort_vertices_by_zyx()
    
    return vertices_manager

def sort_ids_by_axis(vertex_manager, id_list, axis='z'):
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    if axis not in axis_map:
        raise ValueError("axis 参数必须为 'x'、'y' 或 'z'")
    axis_idx = axis_map[axis]
    return sorted(id_list, key=lambda vid: vertex_manager.get_vertex(vid)[axis_idx])

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
    max_x = max(filtered_points, key=lambda item: item[0])[0]
    max_y = max(filtered_points, key=lambda item: item[1])[1]
    result_ids = []
    for x,y,point_id in filtered_points:
        if np.isclose(x, max_x) or  np.isclose(y, max_y):
            continue
        result_ids.append(point_id)
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
        return [], 0
    num_of_points = len(filtered_points)
    result_ids = []
    for x,y,point_id in filtered_points:
        result_ids.append(point_id)
    return result_ids, num_of_points

def generate_blocks(vertices, bristle_length, partition_X, partition_Y_bristle, partition_Z, root_block_hight, root_block_width, cubic_length, radius_base, radius_top, partition_Y_up_bottom, partition_Z_top, partition_X_out, partition_X_middle, G_D, partition_Y_gap):
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
    
    bristle_ids_base, bristle_points_num = find_vertices(vertices, cubic_width/2-radius_base/(2**(0.5)), XYZ="X")
    bristle_ids_right_base, bristle_points_num = find_vertices(vertices, cubic_width/2+radius_base/(2**(0.5)), XYZ="X")
    bristle_ids_top, _ = find_vertices(vertices, cubic_width/2-radius_top/(2**(0.5)), XYZ="X")
    bristle_ids_right_top, _ = find_vertices(vertices, cubic_width/2+radius_top/(2**(0.5)), XYZ="X")
    
    root_ids_left_corner = sorted(list(set(root_ids)-set(bristle_left_ids_not_full_root[1:-2])-set(root_left_bottom_points_right_row)
                                       -set(root_left_bottom_points_left_row)-set(bristle_ids_right_base)-set(bristle_ids_base)))

    inner_bristle_ids_left_base, bristle_points_num = find_vertices(vertices, cubic_width/2-radius_base*0.7/(2**(0.5)), XYZ="X")
    inner_bristle_ids_right_base, bristle_points_num = find_vertices(vertices, cubic_width/2+radius_base*0.7/(2**(0.5)), XYZ="X")
    inner_bristle_ids_left_top, _ = find_vertices(vertices, cubic_width/2-radius_top*0.7/(2**(0.5)), XYZ="X")
    inner_bristle_ids_right_top, _ = find_vertices(vertices, cubic_width/2+radius_top*0.7/(2**(0.5)), XYZ="X")

    #入口处网格，翅膀根
    bristle_left_corner_id = bottom_ids_left_corner_left_row[2::3]
    for index, id in enumerate(bottom_ids_left_corner_left_row[0:-1]):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {bottom_ids_left_corner_left_row[index+1]+1} {bottom_ids_left_corner_left_row[index+1]} "
                          f"{root_left_bottom_points_left_row[index]} {root_left_bottom_points_left_row[index]+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]}) "
                          f"({partition_X} {partition_Y_up_bottom} 5) simpleGrading (0.5 0.2 1)\n")
        elif index == len(bottom_ids_left_corner_left_row[0:-1])-1:
            block_line = (f"\thex ({id} {id+1} {bottom_ids_left_corner_left_row[index+1]+1} {bottom_ids_left_corner_left_row[index+1]} "
                          f"{root_left_bottom_points_left_row[index]} {root_left_bottom_points_left_row[index]+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]}) "
                          f"({partition_X} {partition_Y_up_bottom} 5) simpleGrading (0.5 5 1)\n")
        elif id in bristle_left_corner_id[0:-1]:
            block_line = (f"\thex ({id} {id+1} {bottom_ids_left_corner_left_row[index+1]+1} {bottom_ids_left_corner_left_row[index+1]} "
                          f"{root_left_bottom_points_left_row[index]} {root_left_bottom_points_left_row[index]+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]}) "
                          f"({partition_X} {partition_Y_bristle} 5) simpleGrading (0.5 1 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {bottom_ids_left_corner_left_row[index+1]+1} {bottom_ids_left_corner_left_row[index+1]} "
                          f"{root_left_bottom_points_left_row[index]} {root_left_bottom_points_left_row[index]+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]}) "
                          f"({partition_X} {partition_Y_gap} 5) simpleGrading (0.5 1 1)\n")
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    #出口网格，翅膀根
    bristle_right_corner_id = bottom_ids_left_corner_right_row[2::3]
    for index, id in enumerate(bottom_ids_left_corner_right_row[0:-1]):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {bottom_ids_left_corner_right_row[index+1]+1} {bottom_ids_left_corner_right_row[index+1]} "
                          f"{root_left_bottom_points_right_row[index]} {root_left_bottom_points_right_row[index]+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]}) "
                          f"({partition_X*2} {partition_Y_up_bottom} 5) simpleGrading (2 0.2 1)\n")
        elif index == len(bottom_ids_left_corner_right_row[0:-1])-1:
            block_line = (f"\thex ({id} {id+1} {bottom_ids_left_corner_right_row[index+1]+1} {bottom_ids_left_corner_right_row[index+1]} "
                          f"{root_left_bottom_points_right_row[index]} {root_left_bottom_points_right_row[index]+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]}) "
                          f"({partition_X*2} {partition_Y_up_bottom} 5) simpleGrading (2 5 1)\n")
        elif id in bristle_right_corner_id[0:-1]:
            block_line = (f"\thex ({id} {id+1} {bottom_ids_left_corner_right_row[index+1]+1} {bottom_ids_left_corner_right_row[index+1]} "
                          f"{root_left_bottom_points_right_row[index]} {root_left_bottom_points_right_row[index]+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]}) "
                          f"({partition_X*2} {partition_Y_bristle} 5) simpleGrading (2 1 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {bottom_ids_left_corner_right_row[index+1]+1} {bottom_ids_left_corner_right_row[index+1]} "
                          f"{root_left_bottom_points_right_row[index]} {root_left_bottom_points_right_row[index]+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]}) "
                          f"({partition_X*2} {partition_Y_gap} 5) simpleGrading (2 1 1)\n")
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    #翅膀上下边界，翅膀根
    bottom_top_left_corner = sorted(list(set(bottom_ids)-set(bristle_left_ids_not_full[1:-2])-set(bottom_ids_left_corner_right_row)-set(bottom_ids_left_corner_left_row)))
    for index, id in enumerate(bottom_top_left_corner):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {id+5} {id+4} {root_ids_left_corner[index]} {root_ids_left_corner[index]+1} {root_ids_left_corner[index]+5} {root_ids_left_corner[index]+4}) "
                          f"({partition_Y_bristle} {partition_Y_up_bottom} 5) simpleGrading (1 0.2 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {id+5} {id+4} {root_ids_left_corner[-1]} {root_ids_left_corner[-1]+1} {root_ids_left_corner[-1]+5} {root_ids_left_corner[-1]+4}) "
                          f"({partition_Y_bristle} {partition_Y_up_bottom} 5) simpleGrading (1 5 1)\n")
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    bottom_ids_left_corner = bottom_top_left_corner + bottom_ids_left_corner_left_row + bottom_ids_left_corner_right_row
    
    #入口边界，翅膀等高的部分
    bristle_root_right_corner_id = root_left_bottom_points_left_row[2::3]
    for index, id in enumerate(root_left_bottom_points_left_row[0:-1]):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]} "
                          f"{bristle_top_left_bottom_points_left_row[index]} {bristle_top_left_bottom_points_left_row[index]+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]}) "
                          f"({partition_X} {partition_Y_up_bottom} {partition_Z}) simpleGrading (0.5 0.2 1)\n")
        elif index == len(root_left_bottom_points_left_row[0:-1])-1:
            block_line = (f"\thex ({id} {id+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]} "
                          f"{bristle_top_left_bottom_points_left_row[index]} {bristle_top_left_bottom_points_left_row[index]+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]}) "
                          f"({partition_X} {partition_Y_up_bottom} {partition_Z}) simpleGrading (0.5 5 1)\n")
        elif id in bristle_root_right_corner_id[0:-1]:
            block_line = (f"\thex ({id} {id+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]} "
                          f"{bristle_top_left_bottom_points_left_row[index]} {bristle_top_left_bottom_points_left_row[index]+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]}) "
                          f"({partition_X} {partition_Y_bristle} {partition_Z}) simpleGrading (0.5 1 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {root_left_bottom_points_left_row[index+1]+1} {root_left_bottom_points_left_row[index+1]} "
                          f"{bristle_top_left_bottom_points_left_row[index]} {bristle_top_left_bottom_points_left_row[index]+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]}) "
                          f"({partition_X} {partition_Y_gap} {partition_Z}) simpleGrading (0.5 1 1)\n")
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    #出口边界，翅膀等高的部分
    bristle_root_right_corner_id = root_left_bottom_points_right_row[2::3]
    for index, id in enumerate(root_left_bottom_points_right_row[0:-1]):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]} "
                          f"{bristle_top_left_bottom_points_right_row[index]} {bristle_top_left_bottom_points_right_row[index]+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]}) "
                          f"({partition_X*2} {partition_Y_up_bottom} {partition_Z}) simpleGrading (2 0.2 1)\n")
        elif index == len(root_left_bottom_points_right_row[0:-1])-1:
            block_line = (f"\thex ({id} {id+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]} "
                          f"{bristle_top_left_bottom_points_right_row[index]} {bristle_top_left_bottom_points_right_row[index]+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]}) "
                          f"({partition_X*2} {partition_Y_up_bottom} {partition_Z}) simpleGrading (2 5 1)\n")
        elif id in bristle_root_right_corner_id[0:-1]:
            block_line = (f"\thex ({id} {id+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]} "
                          f"{bristle_top_left_bottom_points_right_row[index]} {bristle_top_left_bottom_points_right_row[index]+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]}) "
                          f"({partition_X*2} {partition_Y_bristle} {partition_Z}) simpleGrading (2 1 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {root_left_bottom_points_right_row[index+1]+1} {root_left_bottom_points_right_row[index+1]} "
                          f"{bristle_top_left_bottom_points_right_row[index]} {bristle_top_left_bottom_points_right_row[index]+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]}) "
                          f"({partition_X*2} {partition_Y_gap} {partition_Z}) simpleGrading (2 1 1)\n")
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    root_left_vertices_ids = set(root_ids) & set(bristle_left_ids)
    root_left_vertices_ids = root_left_vertices_ids - set(stream_right_wall) - set(stream_left_wall)
    root_left_vertices_ids_sorted = sort_ids_by_axis(vertices, root_left_vertices_ids, axis='y')
    root_left_bottom_vertices_ids_sorted = root_left_vertices_ids_sorted[0::3]
    root_left_top_vertices_ids_sorted = root_left_vertices_ids_sorted[3::3]
    root_left_vertices_ids_sorted = root_left_vertices_ids_sorted[1::3]
    
    #翅膀两侧边界，翅膀等高的部分
    bristle_middle_ids_base, _ = find_vertices(vertices, cubic_width/2-radius_base*2/(2**(0.5)), XYZ="X")
    bristle_middle_right_ids_base, _ = find_vertices(vertices, cubic_width/2+radius_base*2/(2**(0.5)), XYZ="X")
    bristle_middle_ids_top, _ = find_vertices(vertices, cubic_width/2-radius_top*2/(2**(0.5)), XYZ="X")
    bristle_middle_right_ids_top, _ = find_vertices(vertices, cubic_width/2+radius_top*2/(2**(0.5)), XYZ="X")
    
    root_ids_left_corner = sorted(list(set(root_ids)-set(bristle_left_ids_not_full_root[1:-2])-set(root_left_bottom_points_right_row)
                                       -set(root_left_bottom_points_left_row)-set(bristle_ids_right_base)-set(bristle_ids_base)
                                       -set(bristle_middle_right_ids_base)-set(bristle_middle_ids_base)))
                                       
    top_ids_left_corner = sorted(list(set(bristle_top_ids)-set(bristle_left_ids_not_full_top[1:-2])
                                       -set(bristle_top_left_bottom_points_right_row)-set(bristle_top_left_bottom_points_left_row)
                                       -set(bristle_ids_right_top)-set(bristle_ids_top)-set(inner_bristle_ids_left_top)-set(inner_bristle_ids_right_top)
                                       -set(bristle_middle_right_ids_top)-set(bristle_middle_ids_top)))
                                       
    for index, id in enumerate(root_ids_left_corner):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {id+5} {id+4} {top_ids_left_corner[index]} {top_ids_left_corner[index]+1} {top_ids_left_corner[index]+5} {top_ids_left_corner[index]+4}) "
                          f"({partition_Y_bristle} {partition_Y_up_bottom} {partition_Z}) simpleGrading (1 0.2 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {id+5} {id+4} {top_ids_left_corner[index]} {top_ids_left_corner[index]+1} {top_ids_left_corner[index]+5} {top_ids_left_corner[index]+4}) "
                          f"({partition_Y_bristle} {partition_Y_up_bottom} {partition_Z}) simpleGrading (1 5 1)\n")
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    root_bristle_vertices_ids = set(root_ids) & set(bristle_ids_base)
    root_bristle_vertices_ids_sorted = sort_ids_by_axis(vertices, root_bristle_vertices_ids, axis='y')
    root_bristle_vertices_ids_sorted = root_bristle_vertices_ids_sorted[::2]
    
    top_bristle_vertices_ids = set(bristle_top_ids) & set(bristle_ids_top)
    top_bristle_vertices_ids_sorted = sort_ids_by_axis(vertices, top_bristle_vertices_ids, axis='y')
    top_bristle_vertices_ids_sorted = top_bristle_vertices_ids_sorted[::2]
    
    root_middle_bristle_vertices_ids = set(root_ids) & set(bristle_middle_ids_base)
    root_middle_bristle_vertices_ids_sorted = sort_ids_by_axis(vertices, root_middle_bristle_vertices_ids, axis='y')
    root_middle_bristle_vertices_ids_sorted = root_middle_bristle_vertices_ids_sorted[::2]
    
    top_middle_bristle_vertices_ids = set(bristle_top_ids) & set(bristle_middle_ids_top)
    top_middle_bristle_vertices_ids_sorted = sort_ids_by_axis(vertices, top_middle_bristle_vertices_ids, axis='y')
    top_middle_bristle_vertices_ids_sorted = top_middle_bristle_vertices_ids_sorted[::2]
    
    bristle_top_left_vertices_ids = set(bristle_top_ids) & set(bristle_left_ids)
    bristle_top_left_vertices_ids = bristle_top_left_vertices_ids - set(stream_right_wall) - set(stream_left_wall)
    bristle_top_left_vertices_ids_sorted = sort_ids_by_axis(vertices, bristle_top_left_vertices_ids, axis='y')
    bristle_top_left_bottom_vertices_ids_sorted = bristle_top_left_vertices_ids_sorted[0::3]
    bristle_top_left_top_vertices_ids_sorted = bristle_top_left_vertices_ids_sorted[3::3]
    bristle_top_left_vertices_ids_sorted = bristle_top_left_vertices_ids_sorted[1::3]
    
    #翅膀周围那一圈的网格，翅膀根到翅膀顶
    root_patches = []
    for index, id in enumerate(root_bristle_vertices_ids_sorted):
        hex_line = (f"\thex ({root_left_vertices_ids_sorted[index]} {root_middle_bristle_vertices_ids_sorted[index]} {root_middle_bristle_vertices_ids_sorted[index]+6} {root_left_vertices_ids_sorted[index]+12} "
                    f"{bristle_top_left_vertices_ids_sorted[index]} {top_middle_bristle_vertices_ids_sorted[index]} {top_middle_bristle_vertices_ids_sorted[index]+10} {bristle_top_left_vertices_ids_sorted[index]+16}) "
                    f"({partition_X_out} {partition_Y_bristle} {partition_Z}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({root_left_vertices_ids_sorted[index]+1} {root_middle_bristle_vertices_ids_sorted[index]+1} {root_middle_bristle_vertices_ids_sorted[index]} {root_left_vertices_ids_sorted[index]} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+1} {top_middle_bristle_vertices_ids_sorted[index]+1} {top_middle_bristle_vertices_ids_sorted[index]} {bristle_top_left_vertices_ids_sorted[index]}) "
                    f"({partition_X_out} {partition_Y_bristle} {partition_Z}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({root_left_vertices_ids_sorted[index]+13} {root_middle_bristle_vertices_ids_sorted[index]+7} {root_middle_bristle_vertices_ids_sorted[index]+1} {root_left_vertices_ids_sorted[index]+1} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+17} {top_middle_bristle_vertices_ids_sorted[index]+11} {top_middle_bristle_vertices_ids_sorted[index]+1} {bristle_top_left_vertices_ids_sorted[index]+1}) "
                    f"({partition_X_out} {partition_Y_bristle} {partition_Z}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({root_left_vertices_ids_sorted[index]+12} {root_middle_bristle_vertices_ids_sorted[index]+6} {root_middle_bristle_vertices_ids_sorted[index]+7} {root_left_vertices_ids_sorted[index]+13} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+16} {top_middle_bristle_vertices_ids_sorted[index]+10} {top_middle_bristle_vertices_ids_sorted[index]+11} {bristle_top_left_vertices_ids_sorted[index]+17}) "
                    f"({partition_X_out} {partition_Y_bristle} {partition_Z}) simpleGrading (1 1 1)\n"
                    #内圈
                    f"\thex ({root_middle_bristle_vertices_ids_sorted[index]} {root_bristle_vertices_ids_sorted[index]} {root_bristle_vertices_ids_sorted[index]+2} {root_middle_bristle_vertices_ids_sorted[index]+6} "
                    f"{top_middle_bristle_vertices_ids_sorted[index]} {top_bristle_vertices_ids_sorted[index]} {top_bristle_vertices_ids_sorted[index]+6} {top_middle_bristle_vertices_ids_sorted[index]+10}) "
                    f"({partition_X_middle} {partition_Y_bristle} {partition_Z}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({root_middle_bristle_vertices_ids_sorted[index]+1} {root_bristle_vertices_ids_sorted[index]+1} {root_bristle_vertices_ids_sorted[index]} {root_middle_bristle_vertices_ids_sorted[index]} "
                    f"{top_middle_bristle_vertices_ids_sorted[index]+1} {top_bristle_vertices_ids_sorted[index]+1} {top_bristle_vertices_ids_sorted[index]} {top_middle_bristle_vertices_ids_sorted[index]}) "
                    f"({partition_X_middle} {partition_Y_bristle} {partition_Z}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({root_middle_bristle_vertices_ids_sorted[index]+7} {root_bristle_vertices_ids_sorted[index]+3} {root_bristle_vertices_ids_sorted[index]+1} {root_middle_bristle_vertices_ids_sorted[index]+1} "
                    f"{top_middle_bristle_vertices_ids_sorted[index]+11} {top_bristle_vertices_ids_sorted[index]+7} {top_bristle_vertices_ids_sorted[index]+1} {top_middle_bristle_vertices_ids_sorted[index]+1}) "
                    f"({partition_X_middle} {partition_Y_bristle} {partition_Z}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({root_middle_bristle_vertices_ids_sorted[index]+6} {root_bristle_vertices_ids_sorted[index]+2} {root_bristle_vertices_ids_sorted[index]+3} {root_middle_bristle_vertices_ids_sorted[index]+7} "
                    f"{top_middle_bristle_vertices_ids_sorted[index]+10} {top_bristle_vertices_ids_sorted[index]+6} {top_bristle_vertices_ids_sorted[index]+7} {top_middle_bristle_vertices_ids_sorted[index]+11}) "
                    f"({partition_X_middle} {partition_Y_bristle} {partition_Z}) simpleGrading (1 1 1)\n"
                    #鬃毛的上下部分
                    f"\thex ({root_left_bottom_vertices_ids_sorted[index]} {root_left_bottom_vertices_ids_sorted[index]+1} {root_left_vertices_ids_sorted[index]+1} {root_left_vertices_ids_sorted[index]} "
                    f"{bristle_top_left_bottom_vertices_ids_sorted[index]} {bristle_top_left_bottom_vertices_ids_sorted[index]+1} {bristle_top_left_vertices_ids_sorted[index]+1} {bristle_top_left_vertices_ids_sorted[index]}) "
                    f"({partition_Y_bristle} {partition_Y_gap} {partition_Z}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({root_left_vertices_ids_sorted[index]+12} {root_left_vertices_ids_sorted[index]+13} {root_left_top_vertices_ids_sorted[index]+1} {root_left_top_vertices_ids_sorted[index]} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+16} {bristle_top_left_vertices_ids_sorted[index]+17} {bristle_top_left_top_vertices_ids_sorted[index]+1} {bristle_top_left_top_vertices_ids_sorted[index]}) "
                    f"({partition_Y_bristle} {partition_Y_gap} {partition_Z}) simpleGrading (1 1 1)\n"
        )
        root_patch = [
            [root_left_vertices_ids_sorted[index], root_middle_bristle_vertices_ids_sorted[index], root_middle_bristle_vertices_ids_sorted[index]+6, root_left_vertices_ids_sorted[index]+12],
            [root_left_vertices_ids_sorted[index]+1, root_middle_bristle_vertices_ids_sorted[index]+1, root_middle_bristle_vertices_ids_sorted[index], root_left_vertices_ids_sorted[index]],
            [root_left_vertices_ids_sorted[index]+13, root_middle_bristle_vertices_ids_sorted[index]+7, root_middle_bristle_vertices_ids_sorted[index]+1, root_left_vertices_ids_sorted[index]+1],
            [root_left_vertices_ids_sorted[index]+12, root_middle_bristle_vertices_ids_sorted[index]+6, root_middle_bristle_vertices_ids_sorted[index]+7, root_left_vertices_ids_sorted[index]+13],
            
            [root_middle_bristle_vertices_ids_sorted[index], root_bristle_vertices_ids_sorted[index], root_bristle_vertices_ids_sorted[index]+2, root_middle_bristle_vertices_ids_sorted[index]+6],
            [root_middle_bristle_vertices_ids_sorted[index]+1, root_bristle_vertices_ids_sorted[index]+1, root_bristle_vertices_ids_sorted[index], root_middle_bristle_vertices_ids_sorted[index]],
            [root_middle_bristle_vertices_ids_sorted[index]+7, root_bristle_vertices_ids_sorted[index]+3, root_bristle_vertices_ids_sorted[index]+1, root_middle_bristle_vertices_ids_sorted[index]+1],
            [root_middle_bristle_vertices_ids_sorted[index]+6, root_bristle_vertices_ids_sorted[index]+2, root_bristle_vertices_ids_sorted[index]+3, root_middle_bristle_vertices_ids_sorted[index]+7],
            
            [root_left_bottom_vertices_ids_sorted[index], root_left_bottom_vertices_ids_sorted[index]+1, root_left_vertices_ids_sorted[index]+1, root_left_vertices_ids_sorted[index]],
            [root_left_vertices_ids_sorted[index]+12, root_left_vertices_ids_sorted[index]+13, root_left_top_vertices_ids_sorted[index]+1, root_left_top_vertices_ids_sorted[index]]
        ]
        root_patches.extend(root_patch)
        output_blocks.append(hex_line)
    output_blocks.append("\n")
    
    top_patches = []
    #入口边界，翅膀顶
    bristle_top_right_corner_id = bristle_top_left_bottom_points_left_row[2::3]
    for index, id in enumerate(bristle_top_left_bottom_points_left_row[0:-1]):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]} "
                          f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num}) "
                          f"({partition_X} {partition_Y_up_bottom} {partition_Z_top}) simpleGrading (0.5 0.2 1)\n")
        elif index == len(bristle_top_left_bottom_points_left_row[0:-1])-1:
            block_line = (f"\thex ({id} {id+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]} "
                          f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num}) "
                          f"({partition_X} {partition_Y_up_bottom} {partition_Z_top}) simpleGrading (0.5 5 1)\n")
        elif id in bristle_top_right_corner_id[0:-1]:
            block_line = (f"\thex ({id} {id+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]} "
                          f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num}) "
                          f"({partition_X} {partition_Y_bristle} {partition_Z_top}) simpleGrading (0.5 1 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {bristle_top_left_bottom_points_left_row[index+1]+1} {bristle_top_left_bottom_points_left_row[index+1]} "
                          f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num}) "
                          f"({partition_X} {partition_Y_gap} {partition_Z_top}) simpleGrading (0.5 1 1)\n")
        top_patches.append([id+bristle_top_points_num, id+1+bristle_top_points_num, bristle_top_left_bottom_points_left_row[index+1]+1+bristle_top_points_num, bristle_top_left_bottom_points_left_row[index+1]+bristle_top_points_num])
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    #出口边界，翅膀顶
    bristle_top_right_corner_id = bristle_top_left_bottom_points_right_row[2::3]
    for index, id in enumerate(bristle_top_left_bottom_points_right_row[0:-1]):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]} "
                          f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num}) "
                          f"({partition_X*2} {partition_Y_up_bottom} {partition_Z_top}) simpleGrading (2 0.2 1)\n")
        elif index == len(bristle_top_left_bottom_points_right_row[0:-1])-1:
            block_line = (f"\thex ({id} {id+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]} "
                          f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num}) "
                          f"({partition_X*2} {partition_Y_up_bottom} {partition_Z_top}) simpleGrading (2 5 1)\n")
        elif id in bristle_top_right_corner_id[0:-1]:
            block_line = (f"\thex ({id} {id+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]} "
                          f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num}) "
                          f"({partition_X*2} {partition_Y_bristle} {partition_Z_top}) simpleGrading (2 1 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {bristle_top_left_bottom_points_right_row[index+1]+1} {bristle_top_left_bottom_points_right_row[index+1]} "
                          f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num} {bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num}) "
                          f"({partition_X*2} {partition_Y_gap} {partition_Z_top}) simpleGrading (2 1 1)\n")
        top_patches.append([id+bristle_top_points_num, id+1+bristle_top_points_num, bristle_top_left_bottom_points_right_row[index+1]+1+bristle_top_points_num, bristle_top_left_bottom_points_right_row[index+1]+bristle_top_points_num])
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    #翅膀上下边界，翅膀顶
    for index, id in enumerate(top_ids_left_corner):
        if index == 0:
            block_line = (f"\thex ({id} {id+1} {id+5} {id+4} {id+bristle_top_points_num} {id+1+bristle_top_points_num} {id+5+bristle_top_points_num} {id+4+bristle_top_points_num}) "
                          f"({partition_Y_bristle} {partition_Y_up_bottom} {partition_Z_top}) simpleGrading (1 0.2 1)\n")
        else:
            block_line = (f"\thex ({id} {id+1} {id+5} {id+4} {id+bristle_top_points_num} {id+1+bristle_top_points_num} {id+5+bristle_top_points_num} {id+4+bristle_top_points_num}) "
                          f"({partition_Y_bristle} {partition_Y_up_bottom} {partition_Z_top}) simpleGrading (1 5 1)\n")
        top_patches.append([id+bristle_top_points_num, id+1+bristle_top_points_num, id+5+bristle_top_points_num, id+4+bristle_top_points_num])
        output_blocks.append(block_line)
    output_blocks.append("\n")
    
    #翅膀顶周围那一圈的网格，翅膀顶
    for index, id in enumerate(top_bristle_vertices_ids_sorted):
        hex_line = (f"\thex ({bristle_top_left_vertices_ids_sorted[index]} {top_middle_bristle_vertices_ids_sorted[index]} {top_middle_bristle_vertices_ids_sorted[index]+10} {bristle_top_left_vertices_ids_sorted[index]+16} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+10+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+16+bristle_top_points_num}) "
                    f"({partition_X_out} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({bristle_top_left_vertices_ids_sorted[index]+1} {top_middle_bristle_vertices_ids_sorted[index]+1} {top_middle_bristle_vertices_ids_sorted[index]} {bristle_top_left_vertices_ids_sorted[index]} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num}) "
                    f"({partition_X_out} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({bristle_top_left_vertices_ids_sorted[index]+17} {top_middle_bristle_vertices_ids_sorted[index]+11} {top_middle_bristle_vertices_ids_sorted[index]+1} {bristle_top_left_vertices_ids_sorted[index]+1} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+17+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+11+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num}) "
                    f"({partition_X_out} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({bristle_top_left_vertices_ids_sorted[index]+16} {top_middle_bristle_vertices_ids_sorted[index]+10} {top_middle_bristle_vertices_ids_sorted[index]+11} {bristle_top_left_vertices_ids_sorted[index]+17} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+16+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+10+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+11+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+17+bristle_top_points_num}) "
                    f"({partition_X_out} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    #内圈
                    f"\thex ({top_middle_bristle_vertices_ids_sorted[index]} {id} {id+6} {top_middle_bristle_vertices_ids_sorted[index]+10} "
                    f"{top_middle_bristle_vertices_ids_sorted[index]+bristle_top_points_num} {id+bristle_top_points_num} {id+6+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+10+bristle_top_points_num}) "
                    f"({partition_X_middle} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({top_middle_bristle_vertices_ids_sorted[index]+1} {id+1} {id} {top_middle_bristle_vertices_ids_sorted[index]} "
                    f"{top_middle_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num} {id+1+bristle_top_points_num} {id+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+bristle_top_points_num}) "
                    f"({partition_X_middle} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({top_middle_bristle_vertices_ids_sorted[index]+11} {id+7} {id+1} {top_middle_bristle_vertices_ids_sorted[index]+1} "
                    f"{top_middle_bristle_vertices_ids_sorted[index]+11+bristle_top_points_num} {id+7+bristle_top_points_num} {id+1+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num}) "
                    f"({partition_X_middle} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({top_middle_bristle_vertices_ids_sorted[index]+10} {id+6} {id+7} {top_middle_bristle_vertices_ids_sorted[index]+11} "
                    f"{top_middle_bristle_vertices_ids_sorted[index]+10+bristle_top_points_num} {id+6+bristle_top_points_num} {id+7+bristle_top_points_num} {top_middle_bristle_vertices_ids_sorted[index]+11+bristle_top_points_num}) "
                    f"({partition_X_middle} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    #鬃毛的上下部分
                    f"\thex ({bristle_top_left_bottom_vertices_ids_sorted[index]} {bristle_top_left_bottom_vertices_ids_sorted[index]+1} {bristle_top_left_vertices_ids_sorted[index]+1} {bristle_top_left_vertices_ids_sorted[index]} "
                    f"{bristle_top_left_bottom_vertices_ids_sorted[index]+bristle_top_points_num} {bristle_top_left_bottom_vertices_ids_sorted[index]+1+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num}) "
                    f"({partition_Y_bristle} {partition_Y_gap} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({bristle_top_left_vertices_ids_sorted[index]+16} {bristle_top_left_vertices_ids_sorted[index]+17} {bristle_top_left_top_vertices_ids_sorted[index]+1} {bristle_top_left_top_vertices_ids_sorted[index]} "
                    f"{bristle_top_left_vertices_ids_sorted[index]+16+bristle_top_points_num} {bristle_top_left_vertices_ids_sorted[index]+17+bristle_top_points_num} {bristle_top_left_top_vertices_ids_sorted[index]+1+bristle_top_points_num} {bristle_top_left_top_vertices_ids_sorted[index]+bristle_top_points_num}) "
                    f"({partition_Y_bristle} {partition_Y_gap} {partition_Z_top}) simpleGrading (1 1 1)\n"
        )
        top_patch_outside_bristle = [
            [bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+10+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+16+bristle_top_points_num],
            [bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num],
            [bristle_top_left_vertices_ids_sorted[index]+17+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+11+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num],
            [bristle_top_left_vertices_ids_sorted[index]+16+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+10+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+11+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+17+bristle_top_points_num], 
            
            [top_middle_bristle_vertices_ids_sorted[index]+bristle_top_points_num, id+bristle_top_points_num, id+6+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+10+bristle_top_points_num],
            [top_middle_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num, id+1+bristle_top_points_num, id+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+bristle_top_points_num],
            [top_middle_bristle_vertices_ids_sorted[index]+11+bristle_top_points_num, id+7+bristle_top_points_num, id+1+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num],
            [top_middle_bristle_vertices_ids_sorted[index]+10+bristle_top_points_num, id+6+bristle_top_points_num, id+7+bristle_top_points_num, top_middle_bristle_vertices_ids_sorted[index]+11+bristle_top_points_num],
            
            [bristle_top_left_bottom_vertices_ids_sorted[index]+bristle_top_points_num, bristle_top_left_bottom_vertices_ids_sorted[index]+1+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+1+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+bristle_top_points_num],
            [bristle_top_left_vertices_ids_sorted[index]+16+bristle_top_points_num, bristle_top_left_vertices_ids_sorted[index]+17+bristle_top_points_num, bristle_top_left_top_vertices_ids_sorted[index]+1+bristle_top_points_num, bristle_top_left_top_vertices_ids_sorted[index]+bristle_top_points_num]
        ]
        top_patches.extend(top_patch_outside_bristle)
        output_blocks.append(hex_line)
    output_blocks.append("\n")
    
    inner_bristle_ids_top, _ = find_vertices(vertices, cubic_width/2-radius_top*0.7/(2**(0.5)), XYZ="X")
    bristle_top_inner_left_vertices_ids = set(bristle_top_ids) & set(inner_bristle_ids_top)
    bristle_top_inner_vertices_ids_sorted = sort_ids_by_axis(vertices, bristle_top_inner_left_vertices_ids, axis='y')
    bristle_top_inner_vertices_ids_sorted = bristle_top_inner_vertices_ids_sorted[::2]
    
    #翅膀顶填补的网格
    top_inner_patches = []
    for index, id in enumerate(bristle_top_inner_vertices_ids_sorted):
        hex_line = (f"\thex ({id} {id+1} {id+3} {id+2} "
                    f"{id+bristle_top_points_num} {id+1+bristle_top_points_num} {id+3+bristle_top_points_num} {id+2+bristle_top_points_num}) "
                    f"({partition_Y_bristle} {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({top_bristle_vertices_ids_sorted[index]} {id} {id+2} {top_bristle_vertices_ids_sorted[index]+6} "
                    f"{top_bristle_vertices_ids_sorted[index]+bristle_top_points_num} {id+bristle_top_points_num} {id+2+bristle_top_points_num} {top_bristle_vertices_ids_sorted[index]+6+bristle_top_points_num}) "
                    f"(2 {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({top_bristle_vertices_ids_sorted[index]+1} {id+1} {id} {top_bristle_vertices_ids_sorted[index]} "
                    f"{top_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num} {id+1+bristle_top_points_num} {id+bristle_top_points_num} {top_bristle_vertices_ids_sorted[index]+bristle_top_points_num}) "
                    f"(2 {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({top_bristle_vertices_ids_sorted[index]+7} {id+3} {id+1} {top_bristle_vertices_ids_sorted[index]+1} "
                    f"{top_bristle_vertices_ids_sorted[index]+7+bristle_top_points_num} {id+3+bristle_top_points_num} {id+1+bristle_top_points_num} {top_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num}) "
                    f"(2 {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
                    
                    f"\thex ({top_bristle_vertices_ids_sorted[index]+6} {id+2} {id+3} {top_bristle_vertices_ids_sorted[index]+7} "
                    f"{top_bristle_vertices_ids_sorted[index]+6+bristle_top_points_num} {id+2+bristle_top_points_num} {id+3+bristle_top_points_num} {top_bristle_vertices_ids_sorted[index]+7+bristle_top_points_num}) "
                    f"(2 {partition_Y_bristle} {partition_Z_top}) simpleGrading (1 1 1)\n"
        )
        top_patch = [
            [id+bristle_top_points_num, id+1+bristle_top_points_num, id+3+bristle_top_points_num, id+2+bristle_top_points_num],
            [top_bristle_vertices_ids_sorted[index]+bristle_top_points_num, id+bristle_top_points_num, id+2+bristle_top_points_num, top_bristle_vertices_ids_sorted[index]+6+bristle_top_points_num],
            [top_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num, id+1+bristle_top_points_num, id+bristle_top_points_num, top_bristle_vertices_ids_sorted[index]+bristle_top_points_num],
            [top_bristle_vertices_ids_sorted[index]+7+bristle_top_points_num, id+3+bristle_top_points_num, id+1+bristle_top_points_num, top_bristle_vertices_ids_sorted[index]+1+bristle_top_points_num],
            [top_bristle_vertices_ids_sorted[index]+6+bristle_top_points_num, id+2+bristle_top_points_num, id+3+bristle_top_points_num, top_bristle_vertices_ids_sorted[index]+7+bristle_top_points_num]
        ]
        top_patches.extend(top_patch)
        top_inner_patches.append([id, id+1, id+3, id+2])
        output_blocks.append(hex_line)
    
    top_all_ids = set(pid for face in top_inner_patches for pid in face)
    bristle_root_surrounding_patch = [
                            [pid for pid in face]
                            for face in root_patches
                        ]
    root_all_ids = set(pid for face in bristle_root_surrounding_patch for pid in face)

    id_xy_list = []
    for pid in top_all_ids:
        vertex = vertices.get_vertex(pid)
        if vertex:
            id_xy_list.append((vertex[0], vertex[1]))
    for pid in root_all_ids:
        vertex = vertices.get_vertex(pid)
        if vertex:
            id_xy_list.append((vertex[0], vertex[1]))

    output_blocks.append("\n")
    output_blocks.append(");\n\n")
    return output_blocks, top_patches, root_patches, root_bristle_vertices_ids_sorted, id_xy_list, top_bristle_vertices_ids_sorted, bristle_top_points_num, bottom_ids_left_corner, root_middle_bristle_vertices_ids_sorted, top_middle_bristle_vertices_ids_sorted

def generate_solid_blocks(vertices, root_block_width, root_block_hight, bristle_length, radius_base, radius_top, partition_X_in, partition_X_out, partition_Z_base, partition_Y, partition_Z, partition_X_gap, partition_Y_out):
    output_blocks = ["blocks\n(\n"]

    bottom_ids, bottom_points_num = find_vertices(vertices, 0, XYZ="Z")
    root_ids, root_points_num = find_vertices(vertices, root_block_hight, XYZ="Z")
    top_ids, top_points_num = find_vertices(vertices, root_block_hight+bristle_length, XYZ="Z")
    
    #底层外框
    bristle_left_ids, bristle_left_points_num = find_vertices(vertices, cubic_width/2-root_block_width/2, XYZ="X")
    bottom_left_vertices_ids = set(bottom_ids) & set(bristle_left_ids)
    bottom_left_vertices_ids_sorted = sort_ids_by_axis(vertices, bottom_left_vertices_ids, axis='y')
    bristle_left_ids_only_4_bristles = bottom_left_vertices_ids_sorted[1::3]
    gap_left_ids = bottom_left_vertices_ids_sorted[0::3]
    
    # 底层中间层
    middle_left_ids, middle_left_points_num = find_vertices(vertices, cubic_width/2-radius_base*2/(2**(0.5)), XYZ="X")
    middle_left_vertices_ids = set(bottom_ids) & set(middle_left_ids)
    middle_left_ids_sorted = sort_ids_by_axis(vertices, middle_left_vertices_ids, axis='y')
    middle_left_ids_sorted = middle_left_ids_sorted[::2]
    
    # 底层毛
    cylinder_left_ids, cylinder_left_points_num = find_vertices(vertices, cubic_width/2-radius_base/(2**(0.5)), XYZ="X")
    cylinder_left_vertices_ids = set(bottom_ids) & set(cylinder_left_ids)
    cylinder_left_ids_sorted = sort_ids_by_axis(vertices, cylinder_left_vertices_ids, axis='y')
    cylinder_left_ids_sorted = cylinder_left_ids_sorted[::2]
    
    # 顶层毛 (使用 radius_top)
    cylinder_top_left_ids, _ = find_vertices(vertices, cubic_width/2-radius_top/(2**(0.5)), XYZ="X")
    cylinder_top_left_vertices_ids = set(top_ids) & set(cylinder_top_left_ids)
    cylinder_top_left_ids_sorted = sort_ids_by_axis(vertices, cylinder_top_left_vertices_ids, axis='y')
    cylinder_top_left_ids_sorted = cylinder_top_left_ids_sorted[::2]
    
    # 底层毛内框
    cylinder_inner_left_ids, cylinder_inner_left_points_num = find_vertices(vertices, cubic_width/2-radius_base*0.7/(2**(0.5)), XYZ="X")
    cylinder_inner_left_vertices_ids = set(bottom_ids) & set(cylinder_inner_left_ids)
    cylinder_inner_left_ids_sorted = sort_ids_by_axis(vertices, cylinder_inner_left_vertices_ids, axis='y')
    cylinder_inner_left_ids_sorted = cylinder_inner_left_ids_sorted[::2]
    
    # 顶层毛内框 (使用 radius_top)
    cylinder_top_inner_left_ids, _ = find_vertices(vertices, cubic_width/2-radius_top*0.7/(2**(0.5)), XYZ="X")
    cylinder_top_inner_left_vertices_ids = set(top_ids) & set(cylinder_top_inner_left_ids)
    cylinder_top_inner_left_ids_sorted = sort_ids_by_axis(vertices, cylinder_top_inner_left_vertices_ids, axis='y')
    cylinder_top_inner_left_ids_sorted = cylinder_top_inner_left_ids_sorted[::2]
    
    for index, id in enumerate(cylinder_inner_left_ids_sorted):
        #底层鬃毛内圈
        hex_line = f"\thex ({id} {id+1} {id+3} {id+2} {id+bottom_points_num} {id+1+bottom_points_num} {id+3+bottom_points_num} {id+2+bottom_points_num}) ({partition_Y} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
        output_blocks.append(hex_line)
        #底层鬃毛外圈
        cylinder_hex_line = (
            f"\thex ({cylinder_left_ids_sorted[index]} {id} {id+2} {cylinder_left_ids_sorted[index]+6} " 
            f"{cylinder_left_ids_sorted[index]+bottom_points_num} {id+bottom_points_num} {id+2+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num}) "
            f"({partition_Y_out} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+1} {id+1} {id} {cylinder_left_ids_sorted[index]} " 
            f"{cylinder_left_ids_sorted[index]+1+bottom_points_num} {id+1+bottom_points_num} {id+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num}) "
            f"({partition_Y_out} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+7} {id+3} {id+1} {cylinder_left_ids_sorted[index]+1} " 
            f"{cylinder_left_ids_sorted[index]+7+bottom_points_num} {id+3+bottom_points_num} {id+1+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num}) "
            f"({partition_Y_out} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+6} {id+2} {id+3} {cylinder_left_ids_sorted[index]+7} " 
            f"{cylinder_left_ids_sorted[index]+6+bottom_points_num} {id+2+bottom_points_num} {id+3+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num}) "
            f"({partition_Y_out} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
        )
        output_blocks.append(cylinder_hex_line)
        
        gap_section_hex_line = (
            f"\thex ({gap_left_ids[index]} {gap_left_ids[index]+1} {gap_left_ids[index]+3} {gap_left_ids[index]+2} "
            f"{gap_left_ids[index]+bottom_points_num} {gap_left_ids[index]+1+bottom_points_num} {gap_left_ids[index]+3+bottom_points_num} {gap_left_ids[index]+2+bottom_points_num}) "
            f"({partition_Y} {partition_X_gap} {partition_Z_base}) simpleGrading (1 1 1)\n"
            
            f"\thex ({gap_left_ids[index]+16} {gap_left_ids[index]+17} {gap_left_ids[index]+19} {gap_left_ids[index]+18} "
            f"{gap_left_ids[index]+16+bottom_points_num} {gap_left_ids[index]+17+bottom_points_num} {gap_left_ids[index]+19+bottom_points_num} {gap_left_ids[index]+18+bottom_points_num}) "
            f"({partition_Y} {partition_X_gap} {partition_Z_base}) simpleGrading (1 1 1)\n"
        )
        output_blocks.append(gap_section_hex_line)
        
        #底层基座外圈
        cylinder_out_hex_line = (
            f"\thex ({bristle_left_ids_only_4_bristles[index]} {middle_left_ids_sorted[index]} {middle_left_ids_sorted[index]+10} {bristle_left_ids_only_4_bristles[index]+14} " 
            f"{bristle_left_ids_only_4_bristles[index]+bottom_points_num} {middle_left_ids_sorted[index]+bottom_points_num} {middle_left_ids_sorted[index]+10+bottom_points_num} {bristle_left_ids_only_4_bristles[index]+14+bottom_points_num}) "
            f"({partition_X_out} {partition_Y} {partition_Z_base}) simpleGrading (0.5 1 1)\n"
            
            f"\thex ({bristle_left_ids_only_4_bristles[index]+1} {middle_left_ids_sorted[index]+1} {middle_left_ids_sorted[index]} {bristle_left_ids_only_4_bristles[index]} " 
            f"{bristle_left_ids_only_4_bristles[index]+1+bottom_points_num} {middle_left_ids_sorted[index]+1+bottom_points_num} {middle_left_ids_sorted[index]+bottom_points_num} {bristle_left_ids_only_4_bristles[index]+bottom_points_num}) "
            f"({partition_X_out} {partition_Y} {partition_Z_base}) simpleGrading (0.5 1 1)\n"
            
            f"\thex ({bristle_left_ids_only_4_bristles[index]+15} {middle_left_ids_sorted[index]+11} {middle_left_ids_sorted[index]+1} {bristle_left_ids_only_4_bristles[index]+1} " 
            f"{bristle_left_ids_only_4_bristles[index]+15+bottom_points_num} {middle_left_ids_sorted[index]+11+bottom_points_num} {middle_left_ids_sorted[index]+1+bottom_points_num} {bristle_left_ids_only_4_bristles[index]+1+bottom_points_num}) "
            f"({partition_X_out} {partition_Y} {partition_Z_base}) simpleGrading (0.5 1 1)\n"
            
            f"\thex ({bristle_left_ids_only_4_bristles[index]+14} {middle_left_ids_sorted[index]+10} {middle_left_ids_sorted[index]+11} {bristle_left_ids_only_4_bristles[index]+15} " 
            f"{bristle_left_ids_only_4_bristles[index]+14+bottom_points_num} {middle_left_ids_sorted[index]+10+bottom_points_num} {middle_left_ids_sorted[index]+11+bottom_points_num} {bristle_left_ids_only_4_bristles[index]+15+bottom_points_num}) "
            f"({partition_X_out} {partition_Y} {partition_Z_base}) simpleGrading (0.5 1 1)\n"
        )
        output_blocks.append(cylinder_out_hex_line)

        #底层基座内圈
        cylinder_out_hex_line = (
            f"\thex ({middle_left_ids_sorted[index]} {cylinder_left_ids_sorted[index]} {cylinder_left_ids_sorted[index]+6} {middle_left_ids_sorted[index]+10} " 
            f"{middle_left_ids_sorted[index]+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num} {middle_left_ids_sorted[index]+10+bottom_points_num}) "
            f"({partition_X_in} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
            
            f"\thex ({middle_left_ids_sorted[index]+1} {cylinder_left_ids_sorted[index]+1} {cylinder_left_ids_sorted[index]} {middle_left_ids_sorted[index]} " 
            f"{middle_left_ids_sorted[index]+1+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num} {middle_left_ids_sorted[index]+bottom_points_num}) "
            f"({partition_X_in} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
            
            f"\thex ({middle_left_ids_sorted[index]+11} {cylinder_left_ids_sorted[index]+7} {cylinder_left_ids_sorted[index]+1} {middle_left_ids_sorted[index]+1} " 
            f"{middle_left_ids_sorted[index]+11+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num} {middle_left_ids_sorted[index]+1+bottom_points_num}) "
            f"({partition_X_in} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
            
            f"\thex ({middle_left_ids_sorted[index]+10} {cylinder_left_ids_sorted[index]+6} {cylinder_left_ids_sorted[index]+7} {middle_left_ids_sorted[index]+11} " 
            f"{middle_left_ids_sorted[index]+10+bottom_points_num} {cylinder_left_ids_sorted[index]+6+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num} {middle_left_ids_sorted[index]+11+bottom_points_num}) "
            f"({partition_X_in} {partition_Y} {partition_Z_base}) simpleGrading (1 1 1)\n"
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
            f"({partition_Y_out} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+1+bottom_points_num} {id+1+bottom_points_num} {id+bottom_points_num} {cylinder_left_ids_sorted[index]+bottom_points_num} " 
            f"{cylinder_top_left_ids_sorted[index]+1} {cylinder_top_inner_left_ids_sorted[index]+1} {cylinder_top_inner_left_ids_sorted[index]} {cylinder_top_left_ids_sorted[index]}) "
            f"({partition_Y_out} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+7+bottom_points_num} {id+3+bottom_points_num} {id+1+bottom_points_num} {cylinder_left_ids_sorted[index]+1+bottom_points_num} " 
            f"{cylinder_top_left_ids_sorted[index]+7} {cylinder_top_inner_left_ids_sorted[index]+3} {cylinder_top_inner_left_ids_sorted[index]+1} {cylinder_top_left_ids_sorted[index]+1}) "
            f"({partition_Y_out} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
            
            f"\thex ({cylinder_left_ids_sorted[index]+6+bottom_points_num} {id+2+bottom_points_num} {id+3+bottom_points_num} {cylinder_left_ids_sorted[index]+7+bottom_points_num} " 
            f"{cylinder_top_left_ids_sorted[index]+6} {cylinder_top_inner_left_ids_sorted[index]+2} {cylinder_top_inner_left_ids_sorted[index]+3} {cylinder_top_left_ids_sorted[index]+7}) "
            f"({partition_Y_out} {partition_Y} {partition_Z}) simpleGrading (1 1 1)\n"
        )
        output_blocks.append(bristle_hex_line)
    
    output_blocks.append("\n")
    output_blocks.append(");\n\n")
    return output_blocks, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num, bottom_left_vertices_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted, middle_left_ids_sorted, bristle_left_ids_only_4_bristles, gap_left_ids
    
def generate_edges(bristle_length, root_block_hight, cubic_width, cubic_length, root_bristle_vertices_ids_sorted, bristle_top_vertices_ids_sorted, bristle_top_points_num, root_middle_bristle_vertices_ids_sorted, top_middle_bristle_vertices_ids_sorted, radius_base, radius_top):
    
    def edge_generation(ids, index, z, current_radius):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_width/2+current_radius*np.sin(alpha)} {cubic_length/2-root_block_length/2-current_radius*np.cos(beta)+(index+1/2)*root_block_length/num_bristles} {z})\n"
            alpha += np.pi/2
            beta += np.pi/2
            output_edges.append(edge_line)
            
    def edge_generation_middle_layer(ids, index, z, current_radius):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_width/2+current_radius*np.sin(alpha)} {cubic_length/2-root_block_length/2-current_radius*np.cos(beta)+(index+1/2)*root_block_length/num_bristles} {z})\n"
            alpha += np.pi/2
            beta += np.pi/2
            output_edges.append(edge_line)

    output_edges = ["edges\n(\n"]
    
    for index, id in enumerate(root_middle_bristle_vertices_ids_sorted):
        root_out_circle_ids = [id, id+1, id+7, id+6]
        edge_generation_middle_layer(root_out_circle_ids, index, root_block_hight, radius_base * 2)
    output_edges.append("\n")
    for index, id in enumerate(root_bristle_vertices_ids_sorted):
        root_out_circle_ids = [id, id+1, id+3, id+2]
        edge_generation(root_out_circle_ids, index, root_block_hight, radius_base)
    output_edges.append("\n")
    for index, id in enumerate(top_middle_bristle_vertices_ids_sorted):
        bristle_out_circle_ids = [id, id+1, id+11, id+10]
        edge_generation_middle_layer(bristle_out_circle_ids, index, root_block_hight+bristle_length, radius_top * 2)
    output_edges.append("\n")
    for index, id in enumerate(bristle_top_vertices_ids_sorted):
        bristle_out_circle_ids = [id, id+1, id+7, id+6]
        edge_generation(bristle_out_circle_ids, index, root_block_hight+bristle_length, radius_top)
    output_edges.append("\n")
    for index, id in enumerate(top_middle_bristle_vertices_ids_sorted):
        top_out_circle_ids = [id+bristle_top_points_num, id+1+bristle_top_points_num, id+11+bristle_top_points_num, id+10+bristle_top_points_num]
        edge_generation_middle_layer(top_out_circle_ids, index, root_block_hight+bristle_length*1.5, radius_top * 2)
    output_edges.append("\n")
    for index, id in enumerate(bristle_top_vertices_ids_sorted):
        top_out_circle_ids = [id+bristle_top_points_num, id+1+bristle_top_points_num, id+7+bristle_top_points_num, id+6+bristle_top_points_num]
        edge_generation(top_out_circle_ids, index, root_block_hight+bristle_length*1.5, radius_top)

    output_edges.append(");\n\n")
    return output_edges

def generate_solid_edges(cubic_width, cubic_length, radius_base, radius_top, bristle_length, root_block_hight, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num, middle_left_ids_sorted):
    output_edges = ["edges\n(\n"]
    
    def edge_generation(ids, index, z, current_radius):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_width/2+current_radius*np.sin(alpha)} {cubic_length/2-root_block_length/2-current_radius*np.cos(beta)+(index+1/2)*root_block_length/num_bristles} {z})\n"
            alpha += np.pi/2
            beta += np.pi/2
            output_edges.append(edge_line)

    def edge_generation_middle_layer(ids, index, z, current_radius):
        alpha = 0
        beta = 0
        num_points = len(ids)
        for i in range(num_points):
            start_id = ids[i]
            end_id = ids[(i + 1) % num_points]
            edge_line = f"\tarc {start_id} {end_id} ({cubic_width/2+current_radius*np.sin(alpha)} {cubic_length/2-root_block_length/2-current_radius*np.cos(beta)+(index+1/2)*root_block_length/num_bristles} {z})\n"
            alpha += np.pi/2
            beta += np.pi/2
            output_edges.append(edge_line)

    for index, id in enumerate(cylinder_left_ids_sorted):
        bottom_out_circle_ids = [id, id+1, id+7, id+6]
        root_out_circle_ids = [i+bottom_points_num for i in bottom_out_circle_ids]
        edge_generation(bottom_out_circle_ids, index, 0, radius_base)
        edge_generation(root_out_circle_ids, index, root_block_hight, radius_base)
    for index, id in enumerate(cylinder_top_left_ids_sorted):
        root_out_circle_ids = [id, id+1, id+7, id+6]
        edge_generation(root_out_circle_ids, index, root_block_hight+bristle_length, radius_top)

    for index, id in enumerate(middle_left_ids_sorted):
        bottom_out_circle_ids = [id, id+1, id+11, id+10]
        root_out_circle_ids = [i+bottom_points_num for i in bottom_out_circle_ids]
        edge_generation_middle_layer(bottom_out_circle_ids, index, 0, radius_base * 2)
        edge_generation_middle_layer(root_out_circle_ids, index, root_block_hight, radius_base * 2)

    output_edges.append(");\n\n")
    return output_edges

def generate_patches(vertices, root_block_hight, top_patches, root_patches, bristle_length, cubic_width, cubic_length, root_bristle_vertices_ids_sorted, root_block_width, radius_base, radius_top):
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
    for ids in top_patches:
        output_patches.append(f"\t\t({ids[0]} {ids[1]} {ids[2]} {ids[3]})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch inlet\n")
    output_patches.append("\t(\n")
    inlet_left_coner_ids, _ = find_vertices(vertices, 0, "X")
    bottom_left_coner_ids, _ = find_vertices(vertices, 0, "Z")
    root_left_coner_ids, _ = find_vertices(vertices, root_block_hight, "Z")
    bristle_top_left_coner_ids, _ = find_vertices(vertices, root_block_hight+bristle_length, "Z")
    roof_left_coner_ids, _ = find_vertices(vertices, root_block_hight+bristle_length*1.5, "Z")
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
    outlet_left_coner_ids, _ = find_vertices(vertices, cubic_width+50, "X")
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
    
    bristle_out_left_coner_ids_top, _ = find_vertices(vertices, cubic_width/2-radius_top/(2**(0.5)), "X")
    top_block_left_up_corner = sorted(list(set(bristle_top_left_coner_ids) & set(bristle_out_left_coner_ids_top)))
    top_block_left_up_corner_ids_sorted = sort_ids_by_axis(vertices, top_block_left_up_corner, axis='y')
    top_block_left_up_corner_ids_sorted = top_block_left_up_corner_ids_sorted[::2]
    for index, id in enumerate(bottom_block_left_up_corner_ids_sorted[0:-1]):
        root_block_patches = (f"\t\t({bottom_block_left_up_corner_ids_sorted[index+1]} {root_block_left_up_corner_ids_sorted[index+1]} {root_block_left_up_corner_ids_sorted[index]} {id})\n"
                              f"\t\t({id+1} {root_block_left_up_corner_ids_sorted[index]+1} {root_block_left_up_corner_ids_sorted[index+1]+1} {bottom_block_left_up_corner_ids_sorted[index+1]+1})\n")
        output_patches.append(root_block_patches)
    root_block_side_patches = (
        f"\t\t({bottom_block_left_up_corner_ids_sorted[0]} {root_block_left_up_corner_ids_sorted[0]} {root_block_left_up_corner_ids_sorted[0]+1} {bottom_block_left_up_corner_ids_sorted[0]+1})\n"
        f"\t\t({bottom_block_left_up_corner_ids_sorted[-1]+1} {root_block_left_up_corner_ids_sorted[-1]+1} {root_block_left_up_corner_ids_sorted[-1]} {bottom_block_left_up_corner_ids_sorted[-1]})\n"
    )
    output_patches.append(root_block_side_patches)
    for i in range(len(root_patches)):
        output_patches.append(f"\t\t({root_patches[i][3]} {root_patches[i][2]} {root_patches[i][1]} {root_patches[i][0]})\n")
    
    bristle_top_inner_left_coner_ids, _ = find_vertices(vertices, cubic_width/2-radius_top*0.7/(2**(0.5)), "X")
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
    stream_right_wall_left_coner_ids, _ = find_vertices(vertices, 0, "Y")
    stream_left_wall_left_coner_ids, _ = find_vertices(vertices, cubic_length, "Y")
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

def generate_solid_patches(bottom_left_vertices_ids_sorted, bottom_points_num, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted, middle_left_ids_sorted, bristle_left_ids_only_4_bristles, gap_left_ids):    
    output_patches = ["patches\n(\n"]
    output_patches.append("\tpatch bristle\n")
    output_patches.append("\t(\n")
    for index, id in enumerate(bottom_left_vertices_ids_sorted):
        if index < len(bottom_left_vertices_ids_sorted)-1:
            output_patches.append(f"\t\t({id} {id+bottom_points_num} {bottom_left_vertices_ids_sorted[index+1]+bottom_points_num} {bottom_left_vertices_ids_sorted[index+1]})\n")
            output_patches.append(f"\t\t({bottom_left_vertices_ids_sorted[index+1]+1} {bottom_left_vertices_ids_sorted[index+1]+1+bottom_points_num} {id+1+bottom_points_num} {id+1} )\n")

    for index, id in enumerate(cylinder_left_ids_sorted): 
        bristle_root_patch = (
            f"\t\t({bristle_left_ids_only_4_bristles[index]+bottom_points_num} {middle_left_ids_sorted[index]+bottom_points_num} "
            f"{middle_left_ids_sorted[index]+10+bottom_points_num} {bristle_left_ids_only_4_bristles[index]+14+bottom_points_num})\n"
            
            f"\t\t({bristle_left_ids_only_4_bristles[index]+1+bottom_points_num} {middle_left_ids_sorted[index]+1+bottom_points_num} "
            f"{middle_left_ids_sorted[index]+bottom_points_num} {bristle_left_ids_only_4_bristles[index]+bottom_points_num})\n"
            
            f"\t\t({bristle_left_ids_only_4_bristles[index]+15+bottom_points_num} {middle_left_ids_sorted[index]+11+bottom_points_num} "
            f"{middle_left_ids_sorted[index]+1+bottom_points_num} {bristle_left_ids_only_4_bristles[index]+1+bottom_points_num})\n"
            
            f"\t\t({bristle_left_ids_only_4_bristles[index]+14+bottom_points_num} {middle_left_ids_sorted[index]+10+bottom_points_num} "
            f"{middle_left_ids_sorted[index]+11+bottom_points_num} {bristle_left_ids_only_4_bristles[index]+15+bottom_points_num})\n"
            
            f"\t\t({gap_left_ids[index]+bottom_points_num} {gap_left_ids[index]+1+bottom_points_num} "
            f"{gap_left_ids[index]+3+bottom_points_num} {gap_left_ids[index]+2+bottom_points_num})\n"
            
            f"\t\t({gap_left_ids[index]+16+bottom_points_num} {gap_left_ids[index]+17+bottom_points_num} "
            f"{gap_left_ids[index]+19+bottom_points_num} {gap_left_ids[index]+18+bottom_points_num})\n"

            f"\t\t({middle_left_ids_sorted[index]+bottom_points_num} {id+bottom_points_num} "
            f"{id+6+bottom_points_num} {middle_left_ids_sorted[index]+10+bottom_points_num})\n"
            
            f"\t\t({middle_left_ids_sorted[index]+1+bottom_points_num} {id+1+bottom_points_num} "
            f"{id+bottom_points_num} {middle_left_ids_sorted[index]+bottom_points_num})\n"
            
            f"\t\t({middle_left_ids_sorted[index]+11+bottom_points_num} {id+7+bottom_points_num} "
            f"{id+1+bottom_points_num} {middle_left_ids_sorted[index]+1+bottom_points_num})\n"
            
            f"\t\t({middle_left_ids_sorted[index]+10+bottom_points_num} {id+6+bottom_points_num} "
            f"{id+7+bottom_points_num} {middle_left_ids_sorted[index]+11+bottom_points_num})\n"
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
            f"\t\t({bristle_left_ids_only_4_bristles[index]+14} {middle_left_ids_sorted[index]+10} "
            f"{middle_left_ids_sorted[index]} {bristle_left_ids_only_4_bristles[index]})\n"
            
            f"\t\t({bristle_left_ids_only_4_bristles[index]} {middle_left_ids_sorted[index]} "
            f"{middle_left_ids_sorted[index]+1} {bristle_left_ids_only_4_bristles[index]+1})\n"
            
            f"\t\t({bristle_left_ids_only_4_bristles[index]+1} {middle_left_ids_sorted[index]+1} "
            f"{middle_left_ids_sorted[index]+11} {bristle_left_ids_only_4_bristles[index]+15})\n"
            
            f"\t\t({bristle_left_ids_only_4_bristles[index]+15} {middle_left_ids_sorted[index]+11} "
            f"{middle_left_ids_sorted[index]+10} {bristle_left_ids_only_4_bristles[index]+14})\n"
            
            f"\t\t({gap_left_ids[index]} {gap_left_ids[index]+1} "
            f"{gap_left_ids[index]+3} {gap_left_ids[index]+2})\n"
            
            f"\t\t({gap_left_ids[index]+16} {gap_left_ids[index]+17} "
            f"{gap_left_ids[index]+19} {gap_left_ids[index]+18})\n"
            
            f"\t\t({middle_left_ids_sorted[index]+10} {id+6} "
            f"{id} {middle_left_ids_sorted[index]})\n"
            
            f"\t\t({middle_left_ids_sorted[index]} {id} "
            f"{id+1} {middle_left_ids_sorted[index]+1})\n"
            
            f"\t\t({middle_left_ids_sorted[index]+1} {id+1} "
            f"{id+7} {middle_left_ids_sorted[index]+11})\n"
            
            f"\t\t({middle_left_ids_sorted[index]+11} {id+7} "
            f"{id+6} {middle_left_ids_sorted[index]+10})\n"
        )
        output_patches.append(bottom_patch)
    
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

# ================= 全局参数区 ================= #

fluid_mesh = "fluid/constant/polyMesh/blockMeshDict"
head = generate_FOAM_head()

G_D = 1
bristle_length = 140

# --- 核心修改：将原来的统一 radius 拆分为底部与顶部半径 ---
radius_base = 1.0     # 根部的半径
radius_top  = 0.4     # 顶部的半径，设置成比 1.0 小即可成为圆台/圆锥。不可设置为 0 以免导致零体积网格。
# --------------------------------------------------------

num_bristles = 7
bristle_gap = radius_base * 2 * 5 * G_D # 这个数字是 gap/diameter

outside_bristle_partition_half = 4
partition_X_out = 2 # 如果是cone shape 这里需要是 4， reverse cone 则保持是2就好了
partition_X_middle = 3
partition_XY = 20
partition_Z_top = 80
partition_Z = 160

if G_D == 1:
    cubic_length = 300
    outside_partition_Y = 30 # 如果是reverse cone，这里要稍微降低一些，因为 root_block_width变宽了。
    partition_Y_gap = 2
    partition_X_gap = 2 
elif G_D == 2:
    cubic_length = 460
    outside_partition_Y = 40
    partition_Y_gap = 5
    partition_X_gap = 5 
elif G_D == 3:
    cubic_length = 300
    partition_Y_gap = 8
    partition_X_gap = 8 

root_block_hight = 4
root_block_length = (radius_base * 2 + bristle_gap) * num_bristles
root_block_width = 6 # 如果是reverse cone，这里需要增加宽度到8
cubic_width = 106

# ================= 函数调用区 ================= #

vertices, solid_blocks_xy_vertices = generate_vertices(cubic_width, cubic_length, radius_base, radius_top, bristle_length, num_bristles, bristle_gap, root_block_hight, root_block_length, root_block_width, G_D)
blocks, top_patches, root_patches, root_bristle_vertices_ids_sorted, id_xy_list, bristle_top_vertices_ids_sorted, bristle_top_points_num, bottom_ids_left_corner, root_middle_bristle_vertices_ids_sorted, top_middle_bristle_vertices_ids_sorted = generate_blocks(vertices, bristle_length, partition_XY, outside_bristle_partition_half, partition_Z, root_block_hight, root_block_width, cubic_length, radius_base, radius_top, outside_partition_Y, partition_Z_top, partition_X_out, partition_X_middle, G_D, partition_Y_gap)
edges = generate_edges(bristle_length, root_block_hight, cubic_width, cubic_length, root_bristle_vertices_ids_sorted, bristle_top_vertices_ids_sorted, bristle_top_points_num, root_middle_bristle_vertices_ids_sorted, top_middle_bristle_vertices_ids_sorted, radius_base, radius_top)
patches = generate_patches(vertices, root_block_hight, top_patches, root_patches, bristle_length, cubic_width, cubic_length, root_bristle_vertices_ids_sorted, root_block_width, radius_base, radius_top)
end = generate_ends()

with open(fluid_mesh, 'w') as file:
    file.write(head)
    file.write(vertices.get_output())
    file.write("".join(blocks))
    file.write("".join(edges))
    file.write("".join(patches))
    file.write("".join(end))

solid_partition_XY = 4
partition_Y_out = 2
solid_partition_Z = 150
partition_X_in = 2
solid_partition_X_out = 2 # 如果是cone shape 需要和 partition_X_out 保持一致
partition_Z_base = 3

solid_mesh = "solid/constant/polyMesh/blockMeshDict"

# 计算圆心供缩放使用
centers = get_centers(num_bristles, cubic_width, cubic_length, bristle_gap, radius_base)

solid_vertices = generate_solid_vertices(solid_blocks_xy_vertices, root_block_hight, bristle_length, root_block_width, centers, radius_base, radius_top)
solid_blocks, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num, bottom_left_vertices_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted ,middle_left_ids_sorted, bristle_left_ids_only_4_bristles, gap_left_ids= generate_solid_blocks(solid_vertices, root_block_width, root_block_hight, bristle_length, radius_base, radius_top, partition_X_in, solid_partition_X_out, partition_Z_base, solid_partition_XY, solid_partition_Z, partition_X_gap, partition_Y_out)
solid_edges = generate_solid_edges(cubic_width, cubic_length, radius_base, radius_top, bristle_length, root_block_hight, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, bottom_points_num, middle_left_ids_sorted)
solid_patches = generate_solid_patches(bottom_left_vertices_ids_sorted, bottom_points_num, cylinder_left_ids_sorted, cylinder_top_left_ids_sorted, cylinder_inner_left_ids_sorted, cylinder_top_inner_left_ids_sorted, middle_left_ids_sorted, bristle_left_ids_only_4_bristles, gap_left_ids)

with open(solid_mesh, 'w') as file:
    file.write(head)
    file.write(solid_vertices.get_output())
    file.write("".join(solid_blocks))
    file.write("".join(solid_edges))
    file.write("".join(solid_patches))
    file.write("".join(end))
