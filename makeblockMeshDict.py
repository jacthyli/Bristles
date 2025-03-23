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
    points[0] 为右上， 1 为右下， 2 为左上， 3 为左下
    """
    angles = np.radians([45, -45, 135, -135])
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
    
    root_vertices = [
        [0, 0, 0],
        [out_circle_points[3][0], 0, 0],
        [out_circle_points[1][0], 0, 0],
        [cubic_size, 0, 0],
        [0, out_circle_points[3][1], 0],
        [out_circle_points[3][0], out_circle_points[3][1], 0],
        [out_circle_points[1][0], out_circle_points[1][1], 0],
        [cubic_size, out_circle_points[3][1], 0],
        [inner_circle_points[3][0], inner_circle_points[3][1], 0],
        [inner_circle_points[1][0], inner_circle_points[1][1], 0],
        [inner_circle_points[2][0], inner_circle_points[2][1], 0],
        [inner_circle_points[0][0], inner_circle_points[0][1], 0],
        [0, out_circle_points[2][1], 0],
        [out_circle_points[2][0], out_circle_points[2][1], 0],
        [out_circle_points[0][0], out_circle_points[0][1], 0],
        [cubic_size, out_circle_points[0][1], 0],
        [0, cubic_size, 0],
        [out_circle_points[2][0], cubic_size, 0],
        [out_circle_points[0][0], cubic_size, 0],
        [cubic_size, cubic_size, 0],
    ]
    length_each_layer = len(root_vertices)
    vertices_manager.add_vertices(root_vertices)
    
    bristle_end_vertices = [
        [x, y, z + bristle_length] for x, y, z in root_vertices
        ]
    vertices_manager.add_vertices(bristle_end_vertices)
    
    roof_vertices = [
        [x, y, z + cubic_size] for x, y, z in root_vertices
        ]
    vertices_manager.add_vertices(roof_vertices)
    
    # bristle_center_vertices = [
    #     [cubic_size / 2, cubic_size / 2, bristle_length],
    #     [cubic_size / 2, cubic_size / 2, cubic_size]
    # ]
    # vertices_manager.add_vertices(bristle_center_vertices)
    
    return vertices_manager, length_each_layer

def corner_blocks(start_id, length_each_layer, partition_num, Z):
    block_line = f"\thex ({start_id} {start_id+1} {start_id+5} {start_id+4} {start_id+length_each_layer} {start_id+length_each_layer+1} {start_id+length_each_layer+5} {start_id+length_each_layer+4}) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    return block_line
def up_down_blocks(start_id, length_each_layer, partition_num, Z):
    block_line = f"\thex ({start_id} {start_id+1} {start_id+5} {start_id+4} {start_id+length_each_layer} {start_id+length_each_layer+1} {start_id+length_each_layer+5} {start_id+length_each_layer+4}) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    return block_line
def left_right_blocks(start_id, length_each_layer, partition_num, Z):
    block_line = f"\thex ({start_id} {start_id+1} {start_id+9} {start_id+8} {start_id+length_each_layer} {start_id+length_each_layer+1} {start_id+length_each_layer+9} {start_id+length_each_layer+8}) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    return block_line
def circle_blocks(start_id, length_each_layer, partition_num, Z):
    left_block_line = f"\thex ({start_id} {start_id+3} {start_id+5} {start_id+8} {start_id+length_each_layer} {start_id+3+length_each_layer} {start_id+5+length_each_layer} {start_id+8+length_each_layer}) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    down_block_line = f"\thex ({start_id} {start_id+1} {start_id+4} {start_id+3} {start_id+length_each_layer} {start_id+1+length_each_layer} {start_id+4+length_each_layer} {start_id+3+length_each_layer}) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    right_block_line = f"\thex ({start_id} {start_id-3} {start_id+5} {start_id+2} {start_id+length_each_layer} {start_id-3+length_each_layer} {start_id+5+length_each_layer} {start_id+2+length_each_layer}) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    up_block_line = f"\thex ({start_id} {start_id+1} {start_id+4} {start_id+3} {start_id+length_each_layer} {start_id+1+length_each_layer} {start_id+4+length_each_layer} {start_id+3+length_each_layer}) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    return left_block_line, down_block_line, right_block_line, up_block_line

def bristle_top_blocks(start_id, length_each_layer, partition_num, Z):
    left_block_line = f"\thex ({start_id+2} {start_id} 60 60 {start_id+length_each_layer+2} {start_id+length_each_layer} 61 61) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    down_block_line = f"\thex ({start_id} {start_id+1} 60 60 {start_id+length_each_layer} {start_id+length_each_layer+1} 61 61) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    right_block_line = f"\thex ({start_id+1} {start_id+3} 60 60 {start_id+length_each_layer+1} {start_id+length_each_layer+3} 61 61) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    up_block_line = f"\thex ({start_id+3} {start_id+2} 60 60 {start_id+length_each_layer+3} {start_id+length_each_layer+2} 61 61) ({partition_num} {partition_num} {int(Z)}) simpleGrading (1 1 1)\n"
    return left_block_line, down_block_line, right_block_line, up_block_line

def generate_blocks(length_each_layer):
    output_blocks = ["blocks\n(\n"]
    partition_num = 20
    for k in range(2):
        points_id = [0, 1, 2, 4, 5, 9, 6, 10, 12, 13, 14, 28]
        if k == 0:
            Z = 40
        elif k == 1:
            Z = 13
        for i in points_id:
            if i == 0 or i == 2 or i == 12 or i == 14:
                output_blocks.append(corner_blocks(i + k * length_each_layer, length_each_layer, partition_num, Z))
            elif i == 4 or i == 6:
                output_blocks.append(left_right_blocks(i + k * length_each_layer, length_each_layer, partition_num, Z))
            elif i == 1 or i == 13:
                output_blocks.append(up_down_blocks(i + k * length_each_layer, length_each_layer, partition_num, Z))
            elif i == 5:
                left_block_line, down_block_line, right_block_line, up_block_line = circle_blocks(i + k * length_each_layer, length_each_layer, partition_num, Z)
                output_blocks.append(left_block_line)
                output_blocks.append(down_block_line)
            elif i == 9:
                left_block_line, down_block_line, right_block_line, up_block_line = circle_blocks(i + k * length_each_layer, length_each_layer, partition_num, Z)
                output_blocks.append(right_block_line)
            elif i == 10:
                left_block_line, down_block_line, right_block_line, up_block_line = circle_blocks(i + k * length_each_layer, length_each_layer, partition_num, Z)
                output_blocks.append(up_block_line)
            elif i == 28 and k == 1:
                output_blocks.append("\n")
                left_block_line, down_block_line, right_block_line, up_block_line = bristle_top_blocks(i, length_each_layer, partition_num, Z)
                output_blocks.append(left_block_line)
                output_blocks.append(down_block_line)
                output_blocks.append(right_block_line)
                output_blocks.append(up_block_line)
            # else:
            #     return ValueError
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

vertices, length_each_layer = generate_vertices(cubic_size, radius, bristle_length)
# blocks = generate_blocks(length_each_layer)
# edges = generate_edges(cubic_size, radius, length_each_layer, bristle_length)
# patches = generate_patches(length_each_layer)
end = generate_ends()
# **修正写入文件的方式**
with open(output_file, 'w') as file:
    file.write(head)
    file.write(vertices.get_output())  # **修正点**
    # file.write("".join(blocks))
    # file.write("".join(edges))
    # file.write("".join(patches))
    file.write("".join(end))

# 提取顶点数据
vertices = extract_vertices(vertices)

# 绘制3D散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')

ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o', s=5)

# 设置坐标轴标签
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("3D Scatter Plot of Extracted Vertices")

plt.show()