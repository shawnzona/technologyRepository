# # -*- coding: utf-8 -*-
# # ********************************************************************
# # Author: zhangxu
# # Create Time: 2019/10/31 15:33
# # ********************************************************************
import os
import pandas as pd
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt, degrees, atan2
from sklearn.cluster import DBSCAN  # 基于密度的空间聚类
from sklearn import metrics  # 模型评估
from myScript.getAddress import getAddress


def centerGeolocation(geolocations):
    """已知多个点的经纬度，计算这些点的中心点坐标"""
    x = 0
    y = 0
    z = 0
    length = len(geolocations)
    for lon, lat in geolocations:
        lon = radians(float(lon))
        lat = radians(float(lat))
        x += cos(lat) * cos(lon)
        y += cos(lat) * sin(lon)
        z += sin(lat)
    x = float(x / length)
    y = float(y / length)
    z = float(z / length)
    return (degrees(atan2(y, x)), degrees(atan2(z, sqrt(x * x + y * y))))


def get_distance(array_1, array_2):
    """已知两点经纬度计算两点间距离"""
    EARTH_RADIUS = 6371.393  # 地球半径
    lon_1 = radians(array_1[0])
    lat_1 = radians(array_1[1])
    lon_2 = radians(array_2[0])
    lat_2 = radians(array_2[1])
    dlon = abs(lon_1 - lon_2)
    dlat = abs(lat_1 - lat_2)
    h = sin(dlat / 2) * sin(dlat / 2) + cos(lat_1) * cos(lat_2) * sin(dlon / 2) * sin(dlon / 2)
    distance = 2 * EARTH_RADIUS * asin(sqrt(h))
    return distance


def dbscanMain(factory_name):
    """聚类，可视化"""
    df = pd.read_excel("{}{}".format(os.getcwd(), "/data/客户数据.xlsx"))
    data = df[df["factory_name"] == factory_name]
    lon_lat_data = data[["longitude", "latitude"]]
    dbscan = DBSCAN(eps=5.5, min_samples=5, algorithm='ball_tree', metric=get_distance)
    labels = dbscan.fit(lon_lat_data).labels_
    s = pd.Series(labels)  # 标签转为series类型
    op = lon_lat_data.iloc[s[s.values == -1].index.tolist()]  # 离群点datafram类型
    outlier_points = [tuple(op.iloc[i].tolist()) for i in range(op.shape[0])]  # 离群点列表类型
    print('分为%d类' % (len(set(labels)) - 1))
    print("轮廓系数: %0.3f" % metrics.silhouette_score(lon_lat_data, labels))
    print('离群点个数: %d' % labels.tolist().count(-1))
    print("{}{}{}".format("离群点：", outlier_points, "（方圆8.5公里内公司数小于3家）"))
    data = lon_lat_data.drop(op.index.tolist())  # 剔除离群点
    center = centerGeolocation([tuple(data.iloc[i].tolist()) for i in range(data.shape[0])])
    print("中心区域点：" + str(center) + ", " + str(getAddress(center)) + "附近")
    plt.scatter(lon_lat_data.iloc[:, 0].tolist(), lon_lat_data.iloc[:, 1].tolist(), c=labels, alpha=1)  # 描绘各个点
    plt.scatter(center[0], center[1], color="red", alpha=1, label="中心区域")  # 描绘中心区域
    plt.scatter([i[0] for i in outlier_points], [j[1] for j in outlier_points], color="blue", alpha=1, label="离群点")  # 描绘离群点
    plt.xlabel("经度", fontproperties="KaiTi")  # 设置x轴描述，楷体
    plt.ylabel("纬度", fontproperties="KaiTi")  # 设置y轴描述
    plt.title(factory_name + "(DBSCAN)", fontproperties="KaiTi")  # 设置标题
    font = {'family': "KaiTi", 'weight': 'normal', 'size': 10}
    plt.legend(prop=font, loc="upper left")  # 多个图形时设置图例用以区分，前面不同的y轴要加label
    plt.show()


if __name__ == "__main__":
    dbscanMain("曙邦")