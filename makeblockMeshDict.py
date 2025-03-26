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

convertToMeters 0.000001;
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
        [cubic_size, cubic_size, 0],
    ]
    
    length_each_layer = len(root_vertices)
    vertices_manager.add_vertices(root_vertices)

    # === 2. 生成 bristle_end_vertices（Z = bristle_length，包含 inner_circle_points） ===
    bristle_end_vertices = [[x, y, bristle_length] for x, y, _ in root_vertices]
    bristle_end_vertices += [[x, y, bristle_length] for x, y in inner_circle_points]  # 这里添加 inner_circle_points

    # 按 X 方向排序
    # bristle_end_vertices.sort(key=lambda v: (v[0], v[1]))

    vertices_manager.add_vertices(bristle_end_vertices)

    # === 3. 生成 roof_vertices（Z = cubic_size，包含 inner_circle_points） ===
    roof_vertices = [[x, y, cubic_size] for x, y, _ in root_vertices]
    roof_vertices += [[x, y, cubic_size] for x, y in inner_circle_points]  # 这里添加 inner_circle_points

    # 按 Y 方向排序
    # roof_vertices.sort(key=lambda v: (v[0], v[1]))

    vertices_manager.add_vertices(roof_vertices)

    return vertices_manager, length_each_layer, inner_circle_points, out_circle_points

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
        output_blocks.append(surrounding_blocks(id, bristle_top_points_num, partition_XY, partition_Z))
        
    quad_groups = [
    [out_circle_points[0], inner_circle_points[0], inner_circle_points[3], out_circle_points[3]],
    [out_circle_points[0], out_circle_points[1], inner_circle_points[1], inner_circle_points[0]],
    [inner_circle_points[1], out_circle_points[1], out_circle_points[2], inner_circle_points[2]],
    [inner_circle_points[3], inner_circle_points[2], out_circle_points[2], out_circle_points[3]]
    ]
    
    output_blocks.append("\n")
    output_blocks.append(");\n\n")
    return output_blocks
    
def generate_edges(cubic_size, radius, length_each_layer, bristle_length):
    output_edges = ["edges\n(\n"]
    z = 0
    for k in range(3):
        out_down_arc = f"\t//arc {5+k*length_each_layer} {6+k*length_each_layer} ({cubic_size/2} {cubic_size/2-radius*3} {z})\n"
        out_right_arc = f"\t//arc {6+k*length_each_layer} {14+k*length_each_layer} ({cubic_size/2+radius*3} {cubic_size/2} {z})\n"
        out_top_arc = f"\t//arc {14+k*length_each_layer} {13+k*length_each_layer} ({cubic_size/2} {cubic_size/2+radius*3} {z})\n"
        out_left_arc = f"\t//arc {13+k*length_each_layer} {5+k*length_each_layer} ({cubic_size/2-radius*3} {cubic_size/2} {z})\n"
        inner_down_arc = f"\tarc {8+k*length_each_layer} {9+k*length_each_layer} ({cubic_size/2} {cubic_size/2-radius} {z})\n"
        inner_right_arc = f"\tarc {9+k*length_each_layer} {11+k*length_each_layer} ({cubic_size/2+radius} {cubic_size/2} {z})\n"
        inner_top_arc = f"\tarc {11+k*length_each_layer} {10+k*length_each_layer} ({cubic_size/2} {cubic_size/2+radius} {z})\n"
        inner_left_arc = f"\tarc {10+k*length_each_layer} {8+k*length_each_layer} ({cubic_size/2-radius} {cubic_size/2} {z})\n"
        if k == 0:
            z = bristle_length
        elif k == 1:
            z = cubic_size
        output_edges.append(out_down_arc)
        output_edges.append(out_right_arc)
        output_edges.append(out_top_arc)
        output_edges.append(out_left_arc)
        output_edges.append(inner_down_arc)
        output_edges.append(inner_right_arc)
        output_edges.append(inner_top_arc)
        output_edges.append(inner_left_arc)
        
        output_edges.append("\n")
    output_edges.append(");\n\n")
    return output_edges

def generate_patches(length_each_layer):
    output_patches = ["patches\n(\n"]
    output_patches.append("\tpatch bottom\n")
    output_patches.append("\t(\n")
    bottom_id = [0, 1, 2, 4, 5, 9, 6, 10, 12, 13, 14]
    for i in bottom_id:
        if i == 0 or i == 1 or i == 2 or i == 12 or i == 13 or i == 14:
            output_patches.append(f"\t\t({i} {i+1} {i+5} {i+4})\n")
        elif i == 4 or i == 6:
            output_patches.append(f"\t\t({i} {i+1} {i+9} {i+8})\n")
        elif i == 5:
            output_patches.append(f"\t\t({i} {i+3} {i+5} {i+8})\n")
            output_patches.append(f"\t\t({i} {i+1} {i+4} {i+3})\n")
        elif i == 9:
            output_patches.append(f"\t\t({i} {i-3} {i+5} {i+2})\n")
        elif i == 10:
            output_patches.append(f"\t\t({i} {i+1} {i+4} {i+3})\n")
    output_patches.append("\t)\n\n")
    
    top_id = [0, 1, 2, 4, 5, 9, 6, 10, 12, 13, 14]
    output_patches.append("\tpatch top\n")
    output_patches.append("\t(\n")
    for i in top_id:
        if i == 0 or i == 1 or i == 2 or i == 12 or i == 13 or i == 14:
            output_patches.append(f"\t\t({i+length_each_layer*2} {i+1+length_each_layer*2} {i+5+length_each_layer*2} {i+4+length_each_layer*2})\n")
        elif i == 4 or i == 6:
            output_patches.append(f"\t\t({i+length_each_layer*2} {i+1+length_each_layer*2} {i+9+length_each_layer*2} {i+8+length_each_layer*2})\n")
        elif i == 5:
            output_patches.append(f"\t\t({i+length_each_layer*2} {i+3+length_each_layer*2} {i+5+length_each_layer*2} {i+8+length_each_layer*2})\n")
            output_patches.append(f"\t\t({i+length_each_layer*2} {i+1+length_each_layer*2} {i+4+length_each_layer*2} {i+3+length_each_layer*2})\n")
        elif i == 9:
            output_patches.append(f"\t\t({i+length_each_layer*2} {i-3+length_each_layer*2} {i+5+length_each_layer*2} {i+2+length_each_layer*2})\n")
        elif i == 10:
            output_patches.append(f"\t\t({i+length_each_layer*2} {i-2+length_each_layer*2} 61 61)\n")
            output_patches.append(f"\t\t({i-2+length_each_layer*2} {i-1+length_each_layer*2} 61 61)\n")
            output_patches.append(f"\t\t({i-1+length_each_layer*2} {i+1+length_each_layer*2} 61 61)\n")
            output_patches.append(f"\t\t({i+1+length_each_layer*2} {i+length_each_layer*2} 61 61)\n")
            output_patches.append(f"\t\t({i+length_each_layer*2} {i+1+length_each_layer*2} {i+4+length_each_layer*2} {i+3+length_each_layer*2})\n")
    output_patches.append("\t)\n\n")
    
    inlet_id = [40, 20, 44, 24, 52, 32]
    output_patches.append("\tpatch inlet\n")
    output_patches.append("\t(\n")
    for i in inlet_id:
        if i == 40 or i == 20 or i == 52 or i == 32:
            output_patches.append(f"\t\t({i} {i-length_each_layer} {i-length_each_layer+4} {i+4})\n")
        elif i == 44 or i == 24:
            output_patches.append(f"\t\t({i} {i-length_each_layer} {i-length_each_layer+8} {i+8})\n")
    output_patches.append("\t)\n\n")
    
    outlet_id = [40, 20, 44, 24, 52, 32]
    output_patches.append("\tpatch outlet\n")
    output_patches.append("\t(\n")
    for i in outlet_id:
        if i == 40 or i == 20 or i == 52 or i == 32:
            output_patches.append(f"\t\t({i+3} {i-length_each_layer+3} {i-length_each_layer+7} {i+7})\n")
        elif i == 44 or i == 24:
            output_patches.append(f"\t\t({i+3} {i-length_each_layer+3} {i-length_each_layer+11} {i+11})\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tpatch bristle\n")
    output_patches.append("\t(\n")
    def patch_line_1(i):
        return f"\t\t({i} {i + 1} {i+length_each_layer+1} {i+length_each_layer})\n"
    def patch_line_2(i):
        return f"\t\t({i} {i + 2} {i+length_each_layer+2} {i+length_each_layer})\n"
    output_patches.append(patch_line_1(8))
    output_patches.append(patch_line_2(9))
    output_patches.append(patch_line_1(10))
    output_patches.append(patch_line_2(8))
    output_patches.append(f"\t\t(30 28 60 60)\n")
    output_patches.append(f"\t\t(28 29 60 60)\n")
    output_patches.append(f"\t\t(29 31 60 60)\n")
    output_patches.append(f"\t\t(31 30 60 60)\n")
    output_patches.append("\t)\n\n")
    
    output_patches.append("\tempty frontAndBackPlanes\n")
    output_patches.append("\t(\n")
    def empty_line(i):
        return f"\t\t({i} {i-length_each_layer} {i-length_each_layer+1} {i+1})\n"
    empty_id = [40, 20, 41, 21, 42, 22, 56, 36, 57, 37, 58, 38]
    for i in empty_id:
        output_patches.append(empty_line(i))
    output_patches.append("\t)\n")
    
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
output_file = "blockMeshDict"
head = generate_FOAM_head()
cubic_size = 2000
bristle_length = 1500
radius = 250
partition_XY = 5
partition_Z = 10

vertices, length_each_layer, inner_circle_points, out_circle_points = generate_vertices(cubic_size, radius, bristle_length)
test = get_specific_circle_points(vertices, bristle_length, inner_circle_points, out_circle_points)
print(test)
out_circle_ids, inner_circle_ids = get_all_circle_points_separately(vertices, bristle_length, inner_circle_points, out_circle_points)
blocks = generate_blocks(vertices, bristle_length, inner_circle_points, out_circle_points, partition_XY, partition_Z)
# edges = generate_edges(cubic_size, radius, length_each_layer, bristle_length)
# patches = generate_patches(length_each_layer)
end = generate_ends()
# **修正写入文件的方式**
# with open(output_file, 'w') as file:
#     file.write(head)
#     file.write(vertices.get_output())  # **修正点**
#     # file.write("".join(blocks))
#     # file.write("".join(edges))
#     # file.write("".join(patches))
#     file.write("".join(end))

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