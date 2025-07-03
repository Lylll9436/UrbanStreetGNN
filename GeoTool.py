
from shapely.geometry import JOIN_STYLE, MultiPolygon, Polygon
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, Point, LineString, MultiLineString
from shapely.ops import linemerge
from scipy.spatial import Delaunay
import networkx as nx
from shapely.geometry import LineString, Point
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import momepy
from shapely.ops import snap


def create_split_gdf(buildings: gpd.GeoDataFrame, hull: gpd.GeoDataFrame,tolerance =10) -> gpd.GeoDataFrame:
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
    # buildings_in_hull.plot()
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
        polygons = [geom.simplify(tolerance=tolerance, preserve_topology=True) for geom in merged_geometry.geoms]
    elif isinstance(merged_geometry, Polygon):
        polygons = [merged_geometry.simplify(tolerance=tolerance, preserve_topology=True)]
    else:
        raise ValueError("Unexpected geometry type")

    # 第四步：创建 GeoDataFrame
    split_gdf = gpd.GeoDataFrame(geometry=polygons, crs=buildings.crs)

    return split_gdf, buildings_in_hull

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

def merge_zigzag_segments(skeleton_gdf, epsilon=100):
    """
    替代 merge_zigzag_segments：先合并骨架为整体线形，再简化曲折线段。

    参数:
        skeleton_gdf: GeoDataFrame，输入骨架线段
        epsilon: 简化公差（越大线越直）

    返回:
        simplified_gdf: GeoDataFrame，合并并简化后的骨架线段
    """
    # 合并线段为一条或多条连续线
    merged_line = linemerge(MultiLineString(skeleton_gdf.geometry.values))

    # 构造 GeoSeries
    merged_series = gpd.GeoSeries([merged_line], crs=skeleton_gdf.crs)

    # 简化几何
    simplified_geom = merged_series.apply(lambda geom: geom.simplify(epsilon, preserve_topology=False))

    # 转为 GeoDataFrame 返回
    simplified_gdf = gpd.GeoDataFrame(geometry=simplified_geom, crs=skeleton_gdf.crs)

    return simplified_gdf


def generate_skeletons_from_buildings(buildings, enclosures,spacing,min_length ,epsilon,plot=True):
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
        skel_raw = extract_skeleton_from_enclosure(inside, enclosure_geom,spacing)
        skel_clean = clean_skeleton_network(skel_raw,min_length)
        skel_straight = merge_zigzag_segments(skel_clean,epsilon)

        # 添加到列表
        all_skeletons_raw.append(skel_raw)
        all_skeletons_clean.append(skel_clean)
        all_skeletons.append(skel_straight)

    # 合并为GeoDataFrame
    skeleton_raw = gpd.GeoDataFrame(pd.concat(all_skeletons_raw, ignore_index=True), crs=buildings.crs)
    skeleton_clean = gpd.GeoDataFrame(pd.concat(all_skeletons_clean, ignore_index=True), crs=buildings.crs)
    skeleton_total = gpd.GeoDataFrame(pd.concat(all_skeletons, ignore_index=True), crs=buildings.crs)

    # 可选：绘图
    # if plot:
    #     ax = enclosures.plot(edgecolor="black", facecolor="none", figsize=(15, 15))
    #     buildings.plot(ax=ax, color="black")
    #
    #     skeleton_raw.plot(ax=ax, color="grey", linewidth=0.8)
    #     skeleton_clean.plot(ax=ax, color="blue", linewidth=0.8)
    #     skeleton_total.plot(ax=ax, color="red", linewidth=0.8)
    #     plt.title("Skeleton Extraction from Enclosures")
    #     plt.show()

    return skeleton_raw, skeleton_clean, skeleton_total


def plot_extended_road_closure(buildings, roads, skeleton_total, enclosures, convex_hull, crs_epsg=32650):
    """
    绘制道路延伸合并图，包含原始道路、骨架线、闭合区域和凸包边界。

    参数：
        buildings: GeoDataFrame, 建筑数据
        roads: GeoDataFrame, 原始道路网络
        skeleton_total: GeoDataFrame, 所有街区合并后的骨架线
        enclosures: GeoDataFrame, 包围区域（用于坐标系统一）
        convex_hull: shapely Polygon or MultiPolygon, 用于裁剪范围
        crs_epsg: int, 投影坐标系 EPSG（默认32650）
    """

    # 凸包转 Polygon，再转为 GeoDataFrame
    hull_polygon = Polygon(convex_hull.boundary)
    hull = gpd.GeoDataFrame(geometry=[hull_polygon], crs=f"EPSG:{crs_epsg}")
    hull = hull.to_crs(epsg=crs_epsg)

    # 投影转换
    roads_proj = roads.to_crs(epsg=crs_epsg)
    enclosures_proj = enclosures.to_crs(epsg=crs_epsg)
    skeleton_proj = skeleton_total.to_crs(epsg=crs_epsg)
    buildings_proj = buildings.to_crs(epsg=crs_epsg)

    # 合并道路线与骨架线
    all_roads = gpd.GeoDataFrame(
        pd.concat([roads_proj, skeleton_proj,hull.geometry], ignore_index=True),
        crs=f"EPSG:{crs_epsg}"
    )

    # 使用 momepy 进行道路延伸
    closed = momepy.extend_lines(all_roads, tolerance=800)

    # # 可视化
    # fig, ax = plt.subplots(figsize=(15, 15))
    # closed.plot(ax=ax, color="red", linewidth=1)
    # hull.boundary.plot(ax=ax, color="black", linewidth=1)
    # buildings_proj.plot(ax=ax, color="black")
    # roads_proj.plot(ax=ax, color="grey", linewidth=0.8)
    # plt.title("Extended Road + Buildings + Convex Hull Boundary")
    # plt.axis("equal")
    # plt.show()
    return closed


def safe_linemerge(lines):
    merged = linemerge(MultiLineString(lines))
    if isinstance(merged, LineString):
        return [merged]
    elif isinstance(merged, MultiLineString):
        return list(merged.geoms)
    else:
        return []  # fallback

def merge_lines(*gdfs):
    """
    将多个 GeoDataFrame 中的线要素合并并转换为 NetworkX 无向图。

    参数：
        *gdfs: 多个 GeoDataFrame，要求其 geometry 为 LineString 或 MultiLineString

    返回：
        G: NetworkX Graph 对象
    """
    tolerance = 5
    # 合并所有线段
    combined = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)
    lines = [geom for geom in combined.geometry if geom is not None]

    # 使用 linemerge 前，先 snap 所有线段首尾点，增强连接性
    snapped_lines = []
    for i in range(len(lines)):
        base = lines[i]
        for j in range(i + 1, len(lines)):
            base = snap(base, lines[j], tolerance)
        snapped_lines.append(base)

    merged = safe_linemerge(snapped_lines)


    return merged