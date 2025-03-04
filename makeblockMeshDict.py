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
    
def generate_vertices(cubic_size, blade_length, blade_hight, blade_thickness, bristles_num, radius, num_points, bristle_length):
    global global_points_id
    global_points_id = 0
    ground_level_points = [
        [0, 0, 0],
        [cubic_size / 2 - blade_thickness / 2, 0, 0],
        [cubic_size / 2 + blade_thickness / 2, 0, 0],
        [cubic_size, 0, 0],
        [0, cubic_size / 2 - blade_length / 2, 0],
        [cubic_size / 2 - blade_thickness / 2, cubic_size / 2 - blade_length / 2, 0],
        [cubic_size / 2 + blade_thickness / 2, cubic_size / 2 - blade_length / 2, 0],
        [cubic_size, cubic_size / 2 - blade_length / 2, 0],
        [0, cubic_size / 2 + blade_length / 2, 0],
        [cubic_size / 2 - blade_thickness / 2, cubic_size / 2 + blade_length / 2, 0],
        [cubic_size / 2 + blade_thickness / 2, cubic_size / 2 + blade_length / 2, 0],
        [cubic_size, cubic_size / 2 + blade_length / 2, 0],
        [0, cubic_size, 0],
        [cubic_size / 2 - blade_thickness / 2, cubic_size, 0],
        [cubic_size / 2 + blade_thickness / 2, cubic_size, 0],
        [cubic_size, cubic_size, 0]
    ]

    top_level_points = [
        [0, 0, cubic_size],
        [cubic_size / 2 - blade_thickness / 2, 0, cubic_size],
        [cubic_size / 2 + blade_thickness / 2, 0, cubic_size],
        [cubic_size, 0, cubic_size],
        [0, cubic_size / 2 - blade_length / 2, cubic_size],
        [cubic_size / 2 - blade_thickness / 2, cubic_size / 2 - blade_length / 2, cubic_size],
        [cubic_size / 2 + blade_thickness / 2, cubic_size / 2 - blade_length / 2, cubic_size],
        [cubic_size, cubic_size / 2 - blade_length / 2, cubic_size],
        [0, cubic_size / 2 + blade_length / 2, cubic_size],
        [cubic_size / 2 - blade_thickness / 2, cubic_size / 2 + blade_length / 2, cubic_size],
        [cubic_size / 2 + blade_thickness / 2, cubic_size / 2 + blade_length / 2, cubic_size],
        [cubic_size, cubic_size / 2 + blade_length / 2, cubic_size],
        [0, cubic_size, cubic_size],
        [cubic_size / 2 - blade_thickness / 2, cubic_size, cubic_size],
        [cubic_size / 2 + blade_thickness / 2, cubic_size, cubic_size],
        [cubic_size, cubic_size, cubic_size]
    ]
    
    
    center_blade = np.array([cubic_size/2, cubic_size/2, 0])
    
    # 生成数据
    all_points, edge_points = generate_circle_points(radius, num_points)
    zigzag_points = generate_zigzag_points(edge_points)

    merged_sorted_points = merge_and_sort_points(zigzag_points, edge_points)
    
    # plot_sorted_points_and_circle(merged_sorted_points, radius)

    blade_hight_array = np.full((merged_sorted_points.shape[0], 1), blade_hight)
    bristle_lower_surface = np.hstack((merged_sorted_points, blade_hight_array)) + center_blade
    
    bristle_top_surface_array = np.full((merged_sorted_points.shape[0], 1), blade_hight + bristle_length)
    bristle_top_surface = np.hstack((merged_sorted_points, bristle_top_surface_array)) + center_blade
    
    
    bristle_center_y = []
    bristle_gap = int(blade_length / bristles_num)
    for i in range(bristles_num):
        bristle_center_y.append(bristle_gap / 2 + bristle_gap * i)
    
    output_vertices = ["vertices\n(\n"]
    for i,point in enumerate(ground_level_points):
        formatted_point = " ".join(format_number(p) for p in point)
        output_vertices.append(f"    ({formatted_point})      //{i}\n")
    global_points_id += len(ground_level_points) 
    
    unique_x_values = set()
    unique_y_values = set()
    existing_points = set()
    for i,point in enumerate(bristle_lower_surface):
        formatted_point = " ".join(format_number(p) for p in point)
        output_vertices.append(f"    ({formatted_point})      //{i+global_points_id}\n")
        x, y = point[:2]
        unique_x_values.add(x)
        unique_y_values.add(y)
        existing_points.add((x, y))
    
    global_points_id += len(bristle_lower_surface)
    sorted_x_values = sorted(unique_x_values)
    sorted_y_values = sorted(unique_y_values)
    grid_x, grid_y = np.meshgrid(sorted_x_values, sorted_y_values)
    grid_points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
    filtered_grid_points = [tuple(p) for p in grid_points 
                            if tuple(p) not in existing_points and np.sqrt((p[0] - cubic_size/2)**2 + (p[1] - cubic_size/2)**2) >= radius]
    filtered_grid_points = sorted(filtered_grid_points, key=lambda p: (p[0], p[1]))
    filtered_grid_points = np.array(filtered_grid_points)
    
    x_min, y_min = filtered_grid_points.min(axis=0)
    x_max, y_max = filtered_grid_points.max(axis=0)

    blade_hight_grid_array = np.full((filtered_grid_points.shape[0], 1), blade_hight)
    bristle_lower_surface_grid = np.hstack((filtered_grid_points, blade_hight_grid_array))
    
    bristle_x = [point[0] for point in bristle_lower_surface]
    bristle_y = [point[1] for point in bristle_lower_surface]
    # 提取 filtered_grid_points 的 X 和 Y 坐标
    grid_x = filtered_grid_points[:, 0]
    grid_y = filtered_grid_points[:, 1]

    # 绘制
    plt.figure(figsize=(8, 8))

    # 绘制 bristle_lower_surface（红色圆点）
    plt.scatter(bristle_x, bristle_y, color='red', marker='o', label='bristle_lower_surface')

    # 绘制 filtered_grid_points（蓝色叉号）
    plt.scatter(grid_x, grid_y, color='blue', marker='x', label='filtered_grid_points')

    # 添加标题和图例
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Bristle Lower Surface and Filtered Grid Points')
    plt.legend()
    plt.grid(True)

    # 保存图片到文件
    plt.savefig("bristle_grid.png", dpi=300, bbox_inches='tight')

    # 关闭图像，避免显示
    plt.close()
    
    blade_top_level_points = [
        [0, 0, blade_hight],
        [cubic_size / 2 - blade_thickness / 2, 0, blade_hight],
        [cubic_size / 2 + blade_thickness / 2, 0, blade_hight],
        [cubic_size, 0, blade_hight],
        [0, cubic_size / 2 - blade_length / 2, blade_hight],
        [cubic_size / 2 - blade_thickness / 2, cubic_size / 2 - blade_length / 2, blade_hight],
        [x_min, cubic_size / 2 - blade_length / 2, blade_hight],
        [x_max, cubic_size / 2 - blade_length / 2, blade_hight],
        [cubic_size / 2 + blade_thickness / 2, cubic_size / 2 - blade_length / 2, blade_hight],
        [cubic_size, cubic_size / 2 - blade_length / 2, blade_hight],
        [cubic_size / 2 - blade_thickness / 2, y_min, blade_hight],
        [cubic_size / 2 + blade_thickness / 2, y_min, blade_hight],
        [cubic_size / 2 - blade_thickness / 2, y_max, blade_hight],
        [cubic_size / 2 + blade_thickness / 2, y_max, blade_hight],
        [0, cubic_size / 2 + blade_length / 2, blade_hight],
        [cubic_size / 2 - blade_thickness / 2, cubic_size / 2 + blade_length / 2, blade_hight],
        [x_min, cubic_size / 2 + blade_length / 2, blade_hight],
        [x_max, cubic_size / 2 + blade_length / 2, blade_hight],
        [cubic_size / 2 + blade_thickness / 2, cubic_size / 2 + blade_length / 2, blade_hight],
        [cubic_size, cubic_size / 2 + blade_length / 2, blade_hight],
        [0, cubic_size, blade_hight],
        [cubic_size / 2 - blade_thickness / 2, cubic_size, blade_hight],
        [cubic_size / 2 + blade_thickness / 2, cubic_size, blade_hight],
        [cubic_size, cubic_size, blade_hight]
    ]
        
    for i,point in enumerate(blade_top_level_points):
        formatted_point = " ".join(format_number(p) for p in point)
        output_vertices.append(f"    ({formatted_point})      //{i+global_points_id}\n")
    global_points_id += len(blade_top_level_points)
    
    for i,point in enumerate(bristle_lower_surface_grid):
        formatted_point = " ".join(format_number(p) for p in point)
        output_vertices.append(f"    ({formatted_point})      //{i+global_points_id}\n")
    global_points_id += len(bristle_lower_surface_grid)
    
    bristle_top_level_points = [
        [0, 0, blade_hight + bristle_length],
        [cubic_size / 2 - blade_thickness / 2, 0, blade_hight + bristle_length],
        [cubic_size / 2 + blade_thickness / 2, 0, blade_hight + bristle_length],
        [cubic_size, 0, blade_hight + bristle_length],
        [0, cubic_size / 2 - blade_length / 2, blade_hight + bristle_length],
        [cubic_size / 2 - blade_thickness / 2, cubic_size / 2 - blade_length / 2, blade_hight + bristle_length],
        [x_min, cubic_size / 2 - blade_length / 2, blade_hight + bristle_length],
        [x_max, cubic_size / 2 - blade_length / 2, blade_hight + bristle_length],
        [cubic_size / 2 + blade_thickness / 2, cubic_size / 2 - blade_length / 2, blade_hight + bristle_length],
        [cubic_size, cubic_size / 2 - blade_length / 2, blade_hight + bristle_length],
        [cubic_size / 2 - blade_thickness / 2, y_min, blade_hight + bristle_length],
        [cubic_size / 2 + blade_thickness / 2, y_min, blade_hight + bristle_length],
        [cubic_size / 2 - blade_thickness / 2, y_max, blade_hight + bristle_length],
        [cubic_size / 2 + blade_thickness / 2, y_max, blade_hight + bristle_length],
        [0, cubic_size / 2 + blade_length / 2, blade_hight + bristle_length],
        [cubic_size / 2 - blade_thickness / 2, cubic_size / 2 + blade_length / 2, blade_hight + bristle_length],
        [x_min, cubic_size / 2 + blade_length / 2, blade_hight + bristle_length],
        [x_max, cubic_size / 2 + blade_length / 2, blade_hight + bristle_length],
        [cubic_size / 2 + blade_thickness / 2, cubic_size / 2 + blade_length / 2, blade_hight + bristle_length],
        [cubic_size, cubic_size / 2 + blade_length / 2, blade_hight + bristle_length],
        [0, cubic_size, blade_hight + bristle_length],
        [cubic_size / 2 - blade_thickness / 2, cubic_size, blade_hight + bristle_length],
        [cubic_size / 2 + blade_thickness / 2, cubic_size, blade_hight + bristle_length],
        [cubic_size, cubic_size, blade_hight + bristle_length]
    ]
    
    
    for i,point in enumerate(bristle_top_level_points):
        formatted_point = " ".join(format_number(p) for p in point)
        output_vertices.append(f"    ({formatted_point})      //{i+global_points_id}\n")
    global_points_id += len(bristle_top_level_points)
    bristle_top_surface_grid_array = np.full((filtered_grid_points.shape[0], 1), blade_hight + bristle_length)
    bristle_top_surface_grid = np.hstack((filtered_grid_points, bristle_top_surface_grid_array))
    
    for i,point in enumerate(bristle_top_surface_grid):
        formatted_point = " ".join(format_number(p) for p in point)
        output_vertices.append(f"    ({formatted_point})      //{i+global_points_id}\n")
    global_points_id += len(bristle_top_surface_grid)
    
    for i,point in enumerate(bristle_top_surface):
        formatted_point = " ".join(format_number(p) for p in point)
        output_vertices.append(f"    ({formatted_point})      //{i+global_points_id}\n")
    global_points_id += len(bristle_top_surface)
    
    for i,point in enumerate(top_level_points):
        formatted_point = " ".join(format_number(p) for p in point)
        output_vertices.append(f"    ({formatted_point})      //{i+global_points_id}\n")
    output_vertices.append(");\n\n")
    
    return output_vertices
    

def format_number(num):
    """ 格式化数字，保留4位小数，但如果是整数则不保留小数 """
    if isinstance(num, float):
        return "{:.4f}".format(num).rstrip('0').rstrip('.')
    return str(num)

def generate_circle_points(radius, num_points):
    """ 生成网格内的点，并找到最靠近圆弧的点 """
    # 生成网格
    x = np.linspace(-radius, radius, num_points)
    y = np.linspace(-radius, radius, num_points)
    grid_x, grid_y = np.meshgrid(x, y)
    
    # 筛选出在圆内的点
    distances = np.sqrt(grid_x**2 + grid_y**2)
    inside_circle = distances <= radius
    points = np.column_stack((grid_x[inside_circle], grid_y[inside_circle]))

    # 记录靠近圆弧的点
    edge_points = []
    for x_val in np.unique(points[:, 0]):
        # 获取当前 x 的所有点
        col_points = points[points[:, 0] == x_val]
        
        if len(col_points) > 0:
            # 按离圆弧的距离排序
            col_points = col_points[np.argsort(np.abs(np.sqrt(col_points[:, 0]**2 + col_points[:, 1]**2) - radius))]
            # 取离圆弧最近的两个点
            edge_points.append(col_points[0])
            edge_points.append(col_points[1])
    
    # 按 y 先排序，再按 x 排序
    edge_points = sorted(edge_points, key=lambda p: (p[1], p[0]))

    return points, edge_points

def generate_zigzag_points(edge_points):
    """ 生成额外的锯齿点 """
    zigzag_points = []

    # 处理成字典，key为x坐标，value为对应y的两个点
    grouped_points = {}
    for x, y in edge_points:
        if x not in grouped_points:
            grouped_points[x] = []
        grouped_points[x].append(y)

    # 确保每列只有两个点
    for x in grouped_points:
        grouped_points[x] = sorted(grouped_points[x])[:2]

    # 遍历字典，构建锯齿点
    x_values = sorted(grouped_points.keys())  # 按x排序
    for i in range(len(x_values) - 1):
        if x_values[i] < 0:
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = grouped_points[x1]  # 当前列的两个y值
            zigzag_points.append([x2, y1])  # 右侧列的 x2，y1
            zigzag_points.append([x2, y2])  # 右侧列的 x2，y2
        elif x_values[i] > 0:
            x1, x2 = x_values[i], x_values[i + 1]
            y1, y2 = grouped_points[x2]  # 前一列的两个y值
            zigzag_points.append([x1, y1])  # 左侧列的 x1，y1
            zigzag_points.append([x1, y2])  # 左侧列的 x1，y2
    
    return zigzag_points

def merge_and_sort_points(zigzag_points, edge_points):
    # 合并点集并去重
    unique_points = set(tuple(point) for point in zigzag_points + edge_points)
    
    # angles = np.arctan2(unique_points[:, 1], unique_points[:, 0])
    # sorted_indices = np.argsort(-angles)
    # sorted_points = unique_points[sorted_indices]
    
    # 按 X 递增排序，若 X 相同则按 Y 递增排序
    sorted_points = sorted(unique_points, key=lambda p: (p[0], p[1]))
    
    return np.array(sorted_points)

def plot_sorted_points_and_circle(sorted_points, radius):
    """ 绘制 sorted_points 和 圆形轮廓 """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 绘制 sorted_points
    ax.scatter(sorted_points[:, 0], sorted_points[:, 1], c='b', marker='o', s=10, label="Sorted Points")
    
    # 绘制圆的轮廓
    circle = plt.Circle((0, 0), radius, color='r', fill=False, linestyle='dashed', label="Circle Boundary")
    ax.add_patch(circle)
    
    # 设置坐标轴范围
    ax.set_xlim(-radius - 10, radius + 10)
    ax.set_ylim(-radius - 10, radius + 10)
    
    # 使坐标轴比例相等
    ax.set_aspect('equal', adjustable='datalim')
    
    # 添加网格和图例
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    
    # 设置标题
    ax.set_title("Sorted Points and Circle Outline")
    
    plt.show()

output_file = "blockMeshDict" 
head = generate_FOAM_head()
# 参数
cubic_size = 30000
blade_length = 10000
blade_hight = 2500
blade_thickness = 1000
bristles_num = 1
radius = 250
num_points = 50
bristle_length = 15000 
output_vertices = generate_vertices(cubic_size, blade_length, blade_hight, blade_thickness, bristles_num, radius, num_points, bristle_length)
with open(output_file, 'w') as file:
        file.write(head)
        file.write("".join(output_vertices))

def extract_vertices(output_vertices):
    vertices = []
    pattern = r"\(\s*([\d\.\-]+)\s+([\d\.\-]+)\s+([\d\.\-]+)\s*\)"  # 正则匹配 ( x y z )
    
    for line in output_vertices:
        match = re.search(pattern, line)
        if match:
            x, y, z = map(float, match.groups())
            vertices.append((x, y, z))
    
    return np.array(vertices)

# 假设 output_vertices 已经从用户提供的代码生成
vertices = extract_vertices(output_vertices)

# 绘制3D散点图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(projection='3d')  # 直接使用 add_subplot(projection='3d')

ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='b', marker='o', s=5)

# 设置坐标轴标签
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("3D Scatter Plot of Extracted Vertices")

plt.show()