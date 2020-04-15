# -*- coding: utf-8 -*-
# ********************************************************************
# Author: zhangxu
# Create Time: 2019/11/1 11:33
# ********************************************************************
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn import metrics  # 模型评估
from math import cos, sin, atan2, sqrt, radians, degrees, asin
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


def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """已知两点的经纬度，计算两点间距离"""
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371.393  # 地球平均半径，单位为KM
    return c * r


def outlierPiontAround(points):
    """依据点周围点的个数，判断离群点"""
    outlierPionts = []
    for i in points:
        num = 0
        for lon2, lat2 in points:
            r = haversine(i[0], i[1], lon2, lat2)
            if r <= 8.5:  # 半径为三公里
                num += 1
            if num >= 3:  # 周围有三个则判定为非离群点
                break
        if num < 3:
            outlierPionts.append(i)
    return outlierPionts


def outlierPiontDistance(points):
    """依据点到其它所有点距离总和，判断离群点"""
    r_dict = {}
    for i in points:
        r = 0
        for lon2, lat2 in points:
            r += haversine(i[0], i[1], lon2, lat2)
        r_dict[r] = i
    return [r_dict[j] for j in sorted(list(r_dict.keys()), reverse=True)[:int(round(len(points) * 0.02, 0))]]  # 按最大百分比判定离群点


def getData(KMM):
    """提取要分析的数据"""
    def innerLayer(name):
        df = pd.read_excel("{}{}".format(os.getcwd(), "/data/客户数据.xlsx"))
        data = df[df["factory_name"] == name]
        KMM(data[["longitude", "latitude"]])
    return innerLayer


@getData
def kmeansModel(X):
    """"聚类，可视化"""
    n = 3
    km = KMeans(n_clusters=n).fit(X)
    rs_labels = km.labels_  # 标签结果
    print("簇数：" + str(n))
    print("{}{}".format("轮廓系数: ", round(metrics.silhouette_score(X, rs_labels), 3)))
    # center = centerGeolocation(km.cluster_centers_)  # 基于质点的中心区域计算
    lon_lat_data = [tuple(X.iloc[i].tolist()) for i in range(X.shape[0])]  # 所有点经纬度
    outlier_points = outlierPiontAround(lon_lat_data)
    print("离群点个数：" + str(len(outlier_points)) + "（方圆8.5公里内公司数小于3家）")
    print("{}{}".format("离群点：", outlier_points))
    for j in outlier_points:  # 剔除离群点
        lon_lat_data.remove(j)
    center = centerGeolocation(lon_lat_data)  # 剔除离群点后根据所有点计算中心区域
    print("中心区域点：" + str(center) + ", " + str(getAddress(center)) + "附近")
    plt.scatter(X.iloc[:, 0].tolist(), X.iloc[:, 1].tolist(), c=rs_labels, alpha=1)  # 描绘各个点
    plt.scatter(center[0], center[1], color="red", alpha=1, label="中心区域")  # 描绘中心区域
    plt.scatter([i[0] for i in outlier_points], [j[1] for j in outlier_points], color="blue", alpha=1, label="离群点")  # 描绘离群点


if __name__ == "__main__":
    factory_name = "曙邦"
    kmeansModel(factory_name)
    plt.xlabel("经度", fontproperties="KaiTi")  # 设置x轴描述，楷体
    plt.ylabel("纬度", fontproperties="KaiTi")  # 设置y轴描述
    plt.title(factory_name, fontproperties="KaiTi")  # 设置标题
    plt.title(factory_name + "(KMeans)", fontproperties="KaiTi")  # 设置标题
    font = {'family': "KaiTi", 'weight': 'normal', 'size': 10}
    plt.legend(prop=font, loc="upper right")  # 多个图形时设置图例用以区分，前面不同的y轴要加label
    plt.show()
