#!/usr/bin/env python
# coding: utf-8

# In[2]:
def process_area(area_name: str):

    import geopandas as gpd
    import momepy
    from shapely.geometry import JOIN_STYLE,MultiPolygon, Polygon
    import matplotlib.pyplot as plt
    import libpysal
    from shapely.ops import voronoi_diagram
    from shapely.ops import unary_union
    from shapely.geometry import JOIN_STYLE,MultiPolygon, Polygon
    import matplotlib.pyplot as plt
    import GeoTool
    import os
    import geopandas as gpd


    # 构造输入输出路径
    input_dir = f"data/road_network_cache/{area_name}/"
    output_dir = f"data/road_network_gen/{area_name}/"
    img_dir = f"data/road_network_img"
    os.makedirs(output_dir, exist_ok=True)
    # os.makedirs(img_dir, exist_ok=True)
    # # 数据获取

    # In[3]:


    # 读取边（街道轴线）
    edges = gpd.read_file(os.path.join(input_dir, f"{area_name}_edges.shp")).to_crs("EPSG:32650")
    # 读取节点（交叉点）
    nodes = gpd.read_file(os.path.join(input_dir, f"{area_name}_nodes.shp")).to_crs("EPSG:32650")
    # 读取POI（百度地图点）
    pois = gpd.read_file(os.path.join(input_dir, f"{area_name}_poi_baidu.shp")).to_crs("EPSG:32650")
    # 读取Building（建筑地图）
    buildings = gpd.read_file(os.path.join(input_dir, f"{area_name}_buildings.shp")).to_crs("EPSG:32650")


    # # 计算地块获取

    # In[4]:


    roads = edges.copy()
    convex_hull = roads.union_all().convex_hull
    enclosures = momepy.enclosures(roads, limit=convex_hull)
    enclosures = enclosures[enclosures.area >10000]
    hull =gpd.GeoDataFrame(geometry=[convex_hull.boundary],crs="EPSG:32650")
    # ax = roads.plot(figsize=(10, 10))
    # hull.plot(ax=ax, color="r")
    # ax.set_axis_off()
    # enclosures.plot(figsize=(10, 10), edgecolor="w").set_axis_off()
    # print(enclosures.area)


    # # 建筑数据预处理

    # In[5]:

    split_gdf,buildings_in_hull= GeoTool.create_split_gdf(buildings,convex_hull)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # split_gdf.plot(ax=ax, color="black",edgecolor="red")
    # roads.plot(ax=ax, color="grey",)
    # hull.plot(ax=ax,color="grey")


    # In[6]:


    import GeoTool

    skeleton_raw, skeleton_clean, skeleton_total = GeoTool.generate_skeletons_from_buildings(buildings=split_gdf, enclosures=enclosures,spacing=50,min_length=2,epsilon=20)


    # In[7]:


    closed = GeoTool.plot_extended_road_closure(buildings_in_hull, roads, skeleton_total, enclosures, convex_hull)


    # In[8]:


    # print(closed)


    # # 合并三类 street

    # In[32]:


    # 粗暴打标
    roads["label"] = 1
    hull["label"] = 1
    closed["label"] = 0


    # In[33]:


    from shapely.ops import snap
    import pandas as pd

    tolerance = 5
    # 合并所有线段
    combined = gpd.GeoDataFrame(pd.concat([roads, closed,hull]), crs=roads.crs)
    lines = [
    part
    for geom in combined.geometry if geom is not None
    for part in (geom.geoms if geom.geom_type == 'MultiLineString' else [geom])
    ]
    # print(combined)
    # 使用 linemerge 前，先 snap 所有线段首尾点，增强连接性
    snapped_lines = []
    for i in range(len(lines)):
        base = lines[i]
        for j in range(i + 1, len(lines)):
            base = snap(base, lines[j], tolerance)
        snapped_lines.append(base)

    merged = GeoTool.safe_linemerge(snapped_lines)


    # In[34]:



    merged_gdf=gpd.GeoDataFrame(geometry=merged,crs=roads.crs)
    # 使用空间最近邻匹配，给 merged_gdf 添加 label
    merged_gdf = gpd.sjoin_nearest(merged_gdf, combined[["geometry", "label"]], how="left", distance_col="dist")
    # print(merged_gdf)
    # fig, ax = plt.subplots(figsize=(15, 15))
    # merged_gdf.plot(ax=ax, color="black",edgecolor="red")
    # roads.plot(ax=ax, color="black")
    # nodes.plot(ax=ax, color="red")


    # In[38]:


    import momepy
    import geopandas as gpd

    # 1. 创建 networkx 图（这一步会自动打断线段）
    G = momepy.gdf_to_nx(merged_gdf, approach='primal', multigraph=False)
    # print(G)
    # 2. 提取 edges 和 nodes 为 GeoDataFrame
    # edges_gdf = momepy.nx_to_gdf(G, points=False)  # edges
    nodes_gdf,edges_gdf= momepy.nx_to_gdf(G, points=True)   # nodes
    fig, ax = plt.subplots(figsize=(15, 15))
    edges_gdf.plot(ax=ax, color="red")
    nodes_gdf .plot(ax=ax, color="blue")
    # closed.plot(ax=ax, color="orange")
    roads.plot(ax=ax, color="black")
    buildings_in_hull.plot(ax=ax,color="black")
    buildings.plot(ax=ax,color="grey",alpha=0.3)
    ax.set_axis_off()
    # 保存图像
    plt.savefig(os.path.join(img_dir, f"{area_name}.png"), dpi=300, bbox_inches='tight')

    # In[41]:


    edges_gdf.to_file(os.path.join(output_dir, f"{area_name}_edges.shp"))
    nodes_gdf.to_file(os.path.join(output_dir, f"{area_name}_nodes.shp"))
    pois.to_file(os.path.join(output_dir, f"{area_name}_poi_baidu.shp"))
    buildings.to_file(os.path.join(output_dir, f"{area_name}_buildings.shp"))

