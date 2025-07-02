
from shapely.geometry import JOIN_STYLE, MultiPolygon, Polygon
import geopandas as gpd
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString
from scipy.spatial import Delaunay

def create_split_gdf(buildings: gpd.GeoDataFrame, hull: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    对给定范围（hull）内的建筑进行缓冲和合并，生成街道分割用的多边形图层。

    参数：
        buildings: 输入建筑 GeoDataFrame
        hull: 用于裁剪的区域 Polygon 或 MultiPolygon

    返回：
        split_gdf: 合并后生成的 GeoDataFrame（多面域）
    """
    # 只保留在 hull 范围内的建筑
    buildings_in_hull = buildings[buildings.within(hull)].copy()
    buildings_in_hull.plot()
    # 第一步：初始缓冲（5 米）
    buildings_buffered_5m = buildings_in_hull.copy()
    buildings_buffered_5m["geometry"] = buildings_buffered_5m.geometry.buffer(
        5, join_style=JOIN_STYLE.mitre
    )

    # 第二步：二次缓冲（2 米）
    buildings_buffered_7m = buildings_buffered_5m.copy()
    buildings_buffered_7m["geometry"] = buildings_buffered_7m.geometry.buffer(
        2, join_style=JOIN_STYLE.mitre
    )

    # 第三步：合并所有图形
    merged_geometry = buildings_buffered_7m.geometry.unary_union

    if isinstance(merged_geometry, MultiPolygon):
        polygons = list(merged_geometry.geoms)
    elif isinstance(merged_geometry, Polygon):
        polygons = [merged_geometry]
    else:
        raise ValueError("Unexpected geometry type")

    # 第四步：创建 GeoDataFrame
    split_gdf = gpd.GeoDataFrame(geometry=polygons, crs=buildings.crs)

    return split_gdf

def compute_circumcenter(p1, p2, p3):
    A = np.array(p1)
    B = np.array(p2)
    C = np.array(p3)

    mid_ab = (A + B) / 2
    dir_ab = B - A
    perp_ab = np.array([-dir_ab[1], dir_ab[0]])

    mid_bc = (B + C) / 2
    dir_bc = C - B
    perp_bc = np.array([-dir_bc[1], dir_bc[0]])

    try:
        t = np.linalg.solve(np.column_stack((perp_ab, -perp_bc)), mid_bc - mid_ab)
        center = mid_ab + t[0] * perp_ab
        return center
    except np.linalg.LinAlgError:
        return None

# 主函数：输入 enclosure 和 buildings，输出骨架线
def extract_skeleton_from_enclosure(buildings_in, enclosure_geom, spacing):
    # 合并建筑为一个几何体
    buildings_union = buildings_in.unary_union
    blank_area = enclosure_geom.difference(buildings_union)
    # print("buildings:",buildings.type)
    # print("buildings_union:",buildings_union.type)
    # print("enclosure_geom:",enclosure_geom.type)
    # print("blank_area:",blank_area.type)
    # ax =blank_area.plot(figsize=(8, 8))
    # ax.set_axis_off()
    # 统一为列表格式
    if isinstance(blank_area, Polygon):
        blank_area = [blank_area]
    elif isinstance(blank_area, MultiPolygon):
        # blank_area = list(blank_area)
        blank_area = list(blank_area.geoms)

    # Step 1：采样空白区域边界上的点（包括 exterior 和 holes）
    coords = []

    for poly in blank_area:
        if isinstance(poly, MultiPolygon):
            polys = list(poly.geoms)  # 拆成单个 Polygon
        else:
            polys = [poly]

        for single_poly in polys:
            # 外边界
            ext_length =single_poly.exterior.length
            ext_n = int(ext_length // spacing)
            for i in range(ext_n):
                pt = single_poly.exterior.interpolate(i * spacing)
                coords.append([pt.x, pt.y])
            # 内部孔洞
            for interior in single_poly.interiors:
                int_length = interior.length
                int_n = int(int_length // spacing)
                for i in range(int_n):
                    pt = interior.interpolate(i * spacing)
                    coords.append([pt.x, pt.y])

    if len(coords) < 3:
        return gpd.GeoDataFrame(geometry=[], crs=buildings_in.crs)

    # Step 2：构建 Delaunay 三角网
    points = np.array(coords)
    tri = Delaunay(points)

    # Step 3：计算每个三角形的外接圆心
    centers = []
    for simplex in tri.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        center = compute_circumcenter(p1, p2, p3)
        centers.append(center if center is not None else None)

    # Step 4：连接相邻三角形的圆心 → 构建骨架线
    lines = []
    for i, neighbors in enumerate(tri.neighbors):
        c1 = centers[i]
        if c1 is None:
            continue
        for j, neighbor in enumerate(neighbors):
            if neighbor != -1 and i < neighbor:  # 防止重复
                c2 = centers[neighbor]
                if c2 is not None:
                    line = LineString([c1, c2])
                    # 检查线段是否在空白区域内部
                    mid = line.interpolate(0.5, normalized=True)
                    if any(poly.contains(Point(mid)) for poly in blank_area):
                        lines.append(line)

    # Step 5：转为 GeoDataFrame 输出
    return gpd.GeoDataFrame(geometry=lines, crs=buildings_in.crs)

import networkx as nx
from shapely.geometry import LineString, Point
def clean_skeleton_network(skeleton_gdf, min_length):
    """
    输入为初步骨架线 GeoDataFrame，输出为简化的、干净的“道路骨架网络”
    """
    import networkx as nx
    from shapely.geometry import LineString, MultiLineString

    G = nx.Graph()

    for geom in skeleton_gdf.geometry:
        # 兼容 MultiLineString
        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = list(geom.geoms)
        else:
            continue

        for line in lines:
            if line.length < min_length:
                continue
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                p1 = tuple(np.round(coords[i], 3))
                p2 = tuple(np.round(coords[i+1], 3))
                G.add_edge(p1, p2, geometry=LineString([p1, p2]))

    # 保留所有有效子图（至少有2条边）
    cleaned_lines = []
    for component in nx.connected_components(G):
        subG = G.subgraph(component).copy()
        if subG.number_of_edges() < 2:
            continue  # 忽略孤立路径

        # 清理挂链节点
        dangling = [n for n in subG.nodes if subG.degree[n] == 1]
        subG.remove_nodes_from(dangling)

        for u, v, data in subG.edges(data=True):
            cleaned_lines.append(data["geometry"])

    return gpd.GeoDataFrame(geometry=cleaned_lines, crs=skeleton_gdf.crs)


def angle_between(p1, p2, p3):
    """计算 ∠p2 的夹角（度数）"""
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    v1 = a - b
    v2 = c - b
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))

def merge_zigzag_segments(skeleton_gdf, angle_threshold):
    """
    将由短线段组成的 zigzag 骨架线合并成方向稳定的长直线
    """
    edge_geom_map = {}  # 记录每条边对应的原始线段
    G = nx.Graph()
    for line in skeleton_gdf.geometry:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            p1 = tuple(np.round(coords[i], 3))
            p2 = tuple(np.round(coords[i+1], 3))
            G.add_edge(p1, p2)
            edge_geom_map[frozenset([p1, p2])] = LineString([p1, p2])

    merged_lines = []
    used_edges = set()

    for component in nx.connected_components(G):
        subG = G.subgraph(component)
        deg = dict(subG.degree())
        endpoints = [n for n in subG.nodes if deg[n] == 1]

        if len(endpoints) < 2:
            continue

        try:
            path = nx.shortest_path(subG, source=endpoints[0], target=endpoints[1])
        except:
            continue

        # 分段拟合：当角度变化过大就断开
        segment = [path[0], path[1]]
        for i in range(2, len(path)):
            angle = angle_between(path[i - 2], path[i - 1], path[i])
            if angle < (180 - angle_threshold):
                if len(segment) >= 2:
                    merged_lines.append(LineString([segment[0], segment[-1]]))
                    for j in range(len(segment) - 1):
                        used_edges.add(frozenset([segment[j], segment[j + 1]]))
                segment = [path[i - 1], path[i]]
            else:
                segment.append(path[i])

        if len(segment) >= 2:
            merged_lines.append(LineString([segment[0], segment[-1]]))
            for j in range(len(segment) - 1):
                used_edges.add(frozenset([segment[j], segment[j + 1]]))

    # 补回未被处理的原始线段
    remaining_edges = [
        geom for edge, geom in edge_geom_map.items()
        if edge not in used_edges
    ]
    merged_lines.extend(remaining_edges)

    return gpd.GeoDataFrame(geometry=merged_lines, crs=skeleton_gdf.crs)


import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt


def generate_skeletons_from_buildings(buildings, enclosures, plot=True):
    """
    对每个enclosure提取建筑骨架，并返回三种版本的GeoDataFrame：raw, clean, total。

    参数:
        buildings: GeoDataFrame，建筑数据
        enclosures: GeoDataFrame，街区边界数据
        plot: 是否绘图，默认True

    返回:
        skeleton_raw, skeleton_clean, skeleton_total：三种阶段的骨架线GeoDataFrame
    """
    all_skeletons_raw = []
    all_skeletons_clean = []
    all_skeletons = []

    # 遍历每个街区
    for idx, row in enclosures.iterrows():
        enclosure_geom = row["geometry"]

        # 获取该enclosure内的建筑（用建筑中心点判断是否落在enclosure中）
        inside = buildings[buildings.geometry.centroid.within(enclosure_geom)]

        if len(inside) < 2:
            continue

        # 提取骨架
        skel_raw = extract_skeleton_from_enclosure(inside, enclosure_geom, 50)
        skel_clean = clean_skeleton_network(skel_raw, 5)
        skel_straight = merge_zigzag_segments(skel_clean, 30)

        # 添加到列表
        all_skeletons_raw.append(skel_raw)
        all_skeletons_clean.append(skel_clean)
        all_skeletons.append(skel_straight)

    # 合并为GeoDataFrame
    skeleton_raw = gpd.GeoDataFrame(pd.concat(all_skeletons_raw, ignore_index=True), crs=buildings.crs)
    skeleton_clean = gpd.GeoDataFrame(pd.concat(all_skeletons_clean, ignore_index=True), crs=buildings.crs)
    skeleton_total = gpd.GeoDataFrame(pd.concat(all_skeletons, ignore_index=True), crs=buildings.crs)

    # 可选：绘图
    if plot:
        ax = enclosures.plot(edgecolor="black", facecolor="none", figsize=(15, 15))
        buildings.plot(ax=ax, color="black")
        skeleton_total.plot(ax=ax, color="blue", linewidth=0.8)
        plt.title("Skeleton Extraction from Enclosures")
        plt.show()

    return skeleton_raw, skeleton_clean, skeleton_total
