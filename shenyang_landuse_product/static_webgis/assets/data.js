window.DASHBOARD_DATA = {
  "meta": {
    "title": "沈阳市土地利用变化预测展示系统",
    "subtitle": "CLCD 2015-2025 回测验证与 2030 情景预测",
    "generated_at": "2026-06-23",
    "study_area": "沈阳市",
    "data_source": "CLCD",
    "pixel_area_km2": 0.0009
  },
  "classes": [
    {
      "code": 1,
      "name": "Cropland",
      "cn": "耕地",
      "color": "#f4e04d"
    },
    {
      "code": 2,
      "name": "Forest",
      "cn": "林地",
      "color": "#267300"
    },
    {
      "code": 3,
      "name": "Shrub",
      "cn": "灌木",
      "color": "#6dbb75"
    },
    {
      "code": 4,
      "name": "Grassland",
      "cn": "草地",
      "color": "#9cc36b"
    },
    {
      "code": 5,
      "name": "Water",
      "cn": "水体",
      "color": "#4b9cd3"
    },
    {
      "code": 6,
      "name": "Snow/Ice",
      "cn": "冰雪",
      "color": "#f7fbff"
    },
    {
      "code": 7,
      "name": "Barren",
      "cn": "裸地",
      "color": "#bdbdbd"
    },
    {
      "code": 8,
      "name": "Impervious",
      "cn": "不透水面",
      "color": "#d7191c"
    },
    {
      "code": 9,
      "name": "Wetland",
      "cn": "湿地",
      "color": "#41b6c4"
    }
  ],
  "layers": [
    {
      "id": "actual_2015",
      "title": "CLCD 2015 真实土地利用",
      "group": "真实 CLCD",
      "year": 2015,
      "model": "Observed",
      "mode": "historical",
      "description": "研究区 2015 年 CLCD 原始 9 类土地利用。",
      "src": "assets/maps/actual_2015.png",
      "source": "E:\\model\\RF-CA\\data\\processed\\landuse\\shenyang\\shenyang_clcd_v01_2015_original.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1087868,
        "2": 35593,
        "4": 3977,
        "5": 26427,
        "7": 327,
        "8": 229332
      }
    },
    {
      "id": "actual_2020",
      "title": "CLCD 2020 真实土地利用",
      "group": "真实 CLCD",
      "year": 2020,
      "model": "Observed",
      "mode": "historical",
      "description": "作为 2025 回测和 2030 预测的重要基准年份。",
      "src": "assets/maps/actual_2020.png",
      "source": "E:\\model\\RF-CA\\data\\processed\\landuse\\shenyang\\shenyang_clcd_v01_2020_original.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1078213,
        "2": 36900,
        "4": 7223,
        "5": 22185,
        "7": 326,
        "8": 238677
      }
    },
    {
      "id": "actual_2025",
      "title": "CLCD 2025 真实土地利用",
      "group": "真实 CLCD",
      "year": 2025,
      "model": "Observed",
      "mode": "validation",
      "description": "用于验证 2015-2020 训练、2020-2025 预测结果的真实参照。",
      "src": "assets/maps/actual_2025.png",
      "source": "E:\\model\\RF-CA\\data\\processed\\landuse\\shenyang\\shenyang_clcd_v01_2025_original.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1042527,
        "2": 42997,
        "3": 6,
        "4": 10105,
        "5": 24748,
        "7": 608,
        "8": 262533
      }
    },
    {
      "id": "markov_2025",
      "title": "Markov baseline 2025 预测",
      "group": "2025 回测预测",
      "year": 2025,
      "model": "Markov baseline",
      "mode": "validation",
      "description": "非空间 Markov 基准模型，作为传统基线。",
      "src": "assets/maps/markov_2025.png",
      "source": "E:\\model\\CA-Markov\\output\\markov_baseline\\shenyang\\shenyang_markov_fit_2015_2020_predict_2025.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1068624,
        "2": 38229,
        "4": 9608,
        "5": 18628,
        "7": 325,
        "8": 248110
      }
    },
    {
      "id": "ca_markov_2025",
      "title": "CA-Markov 2025 预测",
      "group": "2025 回测预测",
      "year": 2025,
      "model": "CA-Markov",
      "mode": "validation",
      "description": "无外部驱动因子的 CA-Markov，突出邻域约束作用。",
      "src": "assets/maps/ca_markov_2025.png",
      "source": "E:\\model\\CA-Markov\\output\\ca_markov\\shenyang\\shenyang_ca_markov_fit_2015_2020_predict_2025_n5_i5.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1068691,
        "2": 38173,
        "4": 9566,
        "5": 18829,
        "7": 324,
        "8": 247941
      }
    },
    {
      "id": "ca_markov_suit_2025",
      "title": "适宜性因子 CA-Markov 2025 预测",
      "group": "2025 回测预测",
      "year": 2025,
      "model": "CA-Markov suitability",
      "mode": "validation",
      "description": "最优权重组：neighbor_weight=0.90, suitability_weight=0.10。",
      "src": "assets/maps/ca_markov_suit_2025.png",
      "source": "E:\\model\\CA-Markov\\output\\ca_markov_suitability\\shenyang\\shenyang_ca_markov_suitability_fit_2015_2020_predict_2025_n5_i5_nw090_sw010.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1068738,
        "2": 38160,
        "4": 9536,
        "5": 18828,
        "7": 323,
        "8": 247939
      }
    },
    {
      "id": "logistic_ca_2025",
      "title": "Logistic-CA 2025 预测",
      "group": "2025 回测预测",
      "year": 2025,
      "model": "Logistic-CA",
      "mode": "validation",
      "description": "Logistic Regression 学习转移概率，再结合 CA 空间分配。",
      "src": "assets/maps/logistic_ca_2025.png",
      "source": "E:\\model\\Logistic-CA\\output\\logistic_ca\\shenyang\\shenyang_logistic_ca_fit_2015_2020_predict_2025_n5_i5.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1068718,
        "2": 38140,
        "4": 9546,
        "5": 18877,
        "7": 316,
        "8": 247927
      }
    },
    {
      "id": "rf_ca_2025",
      "title": "RF-CA 2025 预测",
      "group": "2025 回测预测",
      "year": 2025,
      "model": "RF-CA",
      "mode": "validation",
      "description": "随机森林转移概率 + CA 空间分配，本研究主模型之一。",
      "src": "assets/maps/rf_ca_2025.png",
      "source": "E:\\model\\RF-CA\\output\\rf_ca\\shenyang\\shenyang_rf_ca_fit_2015_2020_predict_2025_driver2020_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1068787,
        "2": 38161,
        "4": 9550,
        "5": 18818,
        "7": 321,
        "8": 247887
      }
    },
    {
      "id": "ca_markov_2030",
      "title": "CA-Markov 2030 预测",
      "group": "2030 未来预测",
      "year": 2030,
      "model": "CA-Markov",
      "mode": "projection",
      "description": "基于 2020-2025 训练、以 2025 为基准的 2030 预测。",
      "src": "assets/maps/ca_markov_2030.png",
      "source": "E:\\model\\CA-Markov\\output\\ca_markov\\shenyang\\shenyang_ca_markov_fit_2020_2025_predict_2030_n5_i5.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1009168,
        "2": 48788,
        "3": 14,
        "4": 11598,
        "5": 27180,
        "7": 880,
        "8": 285896
      }
    },
    {
      "id": "ca_markov_suit_2030",
      "title": "适宜性因子 CA-Markov 2030 预测",
      "group": "2030 未来预测",
      "year": 2030,
      "model": "CA-Markov suitability",
      "mode": "projection",
      "description": "考虑外部适宜性因子的 CA-Markov 2030 情景预测。",
      "src": "assets/maps/ca_markov_suit_2030.png",
      "source": "E:\\model\\CA-Markov\\output\\ca_markov_suitability\\shenyang\\shenyang_ca_markov_suitability_fit_2020_2025_predict_2030_n5_i5.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1009118,
        "2": 48729,
        "3": 14,
        "4": 11606,
        "5": 27127,
        "7": 873,
        "8": 286057
      }
    },
    {
      "id": "logistic_ca_2030",
      "title": "Logistic-CA 2030 预测",
      "group": "2030 未来预测",
      "year": 2030,
      "model": "Logistic-CA",
      "mode": "projection",
      "description": "Logistic-CA 未来预测结果。",
      "src": "assets/maps/logistic_ca_2030.png",
      "source": "E:\\model\\Logistic-CA\\output\\logistic_ca\\shenyang\\shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1009252,
        "2": 48767,
        "3": 15,
        "4": 11592,
        "5": 27176,
        "7": 874,
        "8": 285848
      }
    },
    {
      "id": "rf_ca_2030",
      "title": "RF-CA 2030 预测",
      "group": "2030 未来预测",
      "year": 2030,
      "model": "RF-CA",
      "mode": "projection",
      "description": "RF-CA 未来预测结果，可作为报告主展示图层。",
      "src": "assets/maps/rf_ca_2030.png",
      "source": "E:\\model\\RF-CA\\output\\rf_ca\\shenyang\\shenyang_rf_ca_fit_2020_2025_predict_2030_driver2025_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42.tif",
      "width": 1250,
      "height": 2246,
      "class_counts_preview": {
        "1": 1009001,
        "2": 48787,
        "3": 14,
        "4": 11627,
        "5": 27153,
        "7": 879,
        "8": 286063
      }
    }
  ],
  "drivers": [
    {
      "id": "road_closeness_2025",
      "title": "道路邻近性 2025",
      "description": "数值越高表示越接近道路，对建设用地扩张具有解释意义。",
      "src": "assets/drivers/driver_road_closeness_2025.png",
      "source": "E:\\model\\RF-CA\\data\\processed\\drivers\\shenyang\\shenyang_road_closeness_2025.tif",
      "width": 1250,
      "height": 2246,
      "display_min": 0.0,
      "display_max": 0.9801986813545227
    },
    {
      "id": "water_closeness_2025",
      "title": "水系邻近性 2025",
      "description": "数值越高表示越接近水系。",
      "src": "assets/drivers/driver_water_closeness_2025.png",
      "source": "E:\\model\\RF-CA\\data\\processed\\drivers\\shenyang\\shenyang_water_closeness_2025.tif",
      "width": 1250,
      "height": 2246,
      "display_min": 0.0,
      "display_max": 0.9186474680900574
    },
    {
      "id": "nightlight_2025",
      "title": "夜光强度 2025",
      "description": "反映城市活动强度，是 RF 模型中排名靠前的驱动因子。",
      "src": "assets/drivers/driver_nightlight_2025.png",
      "source": "E:\\model\\RF-CA\\data\\processed\\drivers\\shenyang\\shenyang_nightlight_2025.tif",
      "width": 1250,
      "height": 2246,
      "display_min": 0.0,
      "display_max": 0.6256677961349488
    },
    {
      "id": "elevation_norm_2025",
      "title": "高程归一化 2025",
      "description": "归一化高程因子。",
      "src": "assets/drivers/driver_elevation_norm_2025.png",
      "source": "E:\\model\\RF-CA\\data\\processed\\drivers\\shenyang\\shenyang_elevation_norm_2025.tif",
      "width": 1250,
      "height": 2246,
      "display_min": 0.0,
      "display_max": 0.8496732115745544
    },
    {
      "id": "low_slope_2025",
      "title": "低坡度因子 2025",
      "description": "低坡度区域更适宜城镇建设和农业利用。",
      "src": "assets/drivers/driver_low_slope_2025.png",
      "source": "E:\\model\\RF-CA\\data\\processed\\drivers\\shenyang\\shenyang_low_slope_2025.tif",
      "width": 1250,
      "height": 2246,
      "display_min": 0.0,
      "display_max": 1.0
    }
  ],
  "modelComparison": [
    {
      "model": "Markov baseline",
      "fit_from": 2015,
      "base_year": 2020,
      "target_year": 2025,
      "key_parameters": "non-spatial Markov allocation",
      "overall_accuracy": 0.949188,
      "kappa": 0.867277,
      "delta_oa_vs_markov_baseline": 0.0,
      "delta_kappa_vs_markov_baseline": 0.0,
      "notes": "Reference baseline from CA-Markov project"
    },
    {
      "model": "CA-Markov",
      "fit_from": 2015,
      "base_year": 2020,
      "target_year": 2025,
      "key_parameters": "n5_i5_random_weight0.03",
      "overall_accuracy": 0.959903,
      "kappa": 0.895266,
      "delta_oa_vs_markov_baseline": 0.010715,
      "delta_kappa_vs_markov_baseline": 0.027989,
      "notes": "No external suitability factors"
    },
    {
      "model": "CA-Markov suitability",
      "fit_from": 2015,
      "base_year": 2020,
      "target_year": 2025,
      "key_parameters": "n5_i5_nw090_sw010",
      "overall_accuracy": 0.959885,
      "kappa": 0.89522,
      "delta_oa_vs_markov_baseline": 0.010697,
      "delta_kappa_vs_markov_baseline": 0.027943,
      "notes": "Best suitability-factor run copied from CA-Markov project"
    },
    {
      "model": "Logistic-CA",
      "fit_from": 2015,
      "base_year": 2020,
      "target_year": 2025,
      "key_parameters": "n5_i5_lbfgs_samples40000_changed100000",
      "overall_accuracy": 0.959266,
      "kappa": 0.893603,
      "delta_oa_vs_markov_baseline": 0.010078,
      "delta_kappa_vs_markov_baseline": 0.026326,
      "notes": "Logistic transition probability plus CA allocation"
    },
    {
      "model": "RF-CA",
      "fit_from": 2015,
      "base_year": 2020,
      "target_year": 2025,
      "key_parameters": "n5_i5_rf300_rw070_nw030_seed42",
      "overall_accuracy": 0.959994,
      "kappa": 0.895504,
      "delta_oa_vs_markov_baseline": 0.010806,
      "delta_kappa_vs_markov_baseline": 0.028227,
      "notes": "Random forest transition probability plus CA allocation"
    }
  ],
  "areaProjection": [
    {
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "base_2025_area_km2": 9696.2013,
      "projected_2030_area_km2": 9385.9389,
      "change_area_km2": -310.26240000000143
    },
    {
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "base_2025_area_km2": 400.3101,
      "projected_2030_area_km2": 454.1616,
      "change_area_km2": 53.851500000000044
    },
    {
      "class_code": 3,
      "class_name": "Shrub",
      "class_cn": "灌木",
      "base_2025_area_km2": 0.0747,
      "projected_2030_area_km2": 0.1521,
      "change_area_km2": 0.07740000000000001
    },
    {
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "base_2025_area_km2": 94.5324,
      "projected_2030_area_km2": 108.4923,
      "change_area_km2": 13.959900000000005
    },
    {
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "base_2025_area_km2": 230.706,
      "projected_2030_area_km2": 253.4274,
      "change_area_km2": 22.721400000000017
    },
    {
      "class_code": 6,
      "class_name": "Snow/Ice",
      "class_cn": "冰雪",
      "base_2025_area_km2": 0.0,
      "projected_2030_area_km2": 0.0,
      "change_area_km2": 0.0
    },
    {
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "base_2025_area_km2": 5.7708,
      "projected_2030_area_km2": 8.4384,
      "change_area_km2": 2.6675999999999993
    },
    {
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "base_2025_area_km2": 2442.1608,
      "projected_2030_area_km2": 2659.1454,
      "change_area_km2": 216.98459999999977
    },
    {
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "base_2025_area_km2": 0.0,
      "projected_2030_area_km2": 0.0,
      "change_area_km2": 0.0
    }
  ],
  "historicalArea": [
    {
      "year": 2000,
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "area_km2": 10668.4794
    },
    {
      "year": 2000,
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "area_km2": 372.3219
    },
    {
      "year": 2000,
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "area_km2": 78.6924
    },
    {
      "year": 2000,
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "area_km2": 180.6246
    },
    {
      "year": 2000,
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "area_km2": 8.2431
    },
    {
      "year": 2000,
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "area_km2": 1561.3857
    },
    {
      "year": 2000,
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "area_km2": 0.009
    },
    {
      "year": 2005,
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "area_km2": 10578.9654
    },
    {
      "year": 2005,
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "area_km2": 347.7276
    },
    {
      "year": 2005,
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "area_km2": 67.9554
    },
    {
      "year": 2005,
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "area_km2": 178.7337
    },
    {
      "year": 2005,
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "area_km2": 6.057
    },
    {
      "year": 2005,
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "area_km2": 1690.3143
    },
    {
      "year": 2005,
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "area_km2": 0.0027
    },
    {
      "year": 2010,
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "area_km2": 10422.1872
    },
    {
      "year": 2010,
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "area_km2": 306.2169
    },
    {
      "year": 2010,
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "area_km2": 50.2119
    },
    {
      "year": 2010,
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "area_km2": 210.0555
    },
    {
      "year": 2010,
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "area_km2": 3.8115
    },
    {
      "year": 2010,
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "area_km2": 1877.2731
    },
    {
      "year": 2015,
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "area_km2": 10119.2139
    },
    {
      "year": 2015,
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "area_km2": 330.8391
    },
    {
      "year": 2015,
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "area_km2": 37.5183
    },
    {
      "year": 2015,
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "area_km2": 245.8008
    },
    {
      "year": 2015,
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "area_km2": 3.1014
    },
    {
      "year": 2015,
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "area_km2": 2133.2826
    },
    {
      "year": 2020,
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "area_km2": 10029.4974
    },
    {
      "year": 2020,
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "area_km2": 343.008
    },
    {
      "year": 2020,
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "area_km2": 67.7196
    },
    {
      "year": 2020,
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "area_km2": 206.4429
    },
    {
      "year": 2020,
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "area_km2": 2.9511
    },
    {
      "year": 2020,
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "area_km2": 2220.1371
    },
    {
      "year": 2025,
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "area_km2": 9696.2013
    },
    {
      "year": 2025,
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "area_km2": 400.3101
    },
    {
      "year": 2025,
      "class_code": 3,
      "class_name": "Shrub",
      "class_cn": "灌木",
      "area_km2": 0.0747
    },
    {
      "year": 2025,
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "area_km2": 94.5324
    },
    {
      "year": 2025,
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "area_km2": 230.706
    },
    {
      "year": 2025,
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "area_km2": 5.7708
    },
    {
      "year": 2025,
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "area_km2": 2442.1608
    },
    {
      "year": 2030,
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "area_km2": 9385.9389,
      "source": "RF-CA projection"
    },
    {
      "year": 2030,
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "area_km2": 454.1616,
      "source": "RF-CA projection"
    },
    {
      "year": 2030,
      "class_code": 3,
      "class_name": "Shrub",
      "class_cn": "灌木",
      "area_km2": 0.1521,
      "source": "RF-CA projection"
    },
    {
      "year": 2030,
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "area_km2": 108.4923,
      "source": "RF-CA projection"
    },
    {
      "year": 2030,
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "area_km2": 253.4274,
      "source": "RF-CA projection"
    },
    {
      "year": 2030,
      "class_code": 6,
      "class_name": "Snow/Ice",
      "class_cn": "冰雪",
      "area_km2": 0.0,
      "source": "RF-CA projection"
    },
    {
      "year": 2030,
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "area_km2": 8.4384,
      "source": "RF-CA projection"
    },
    {
      "year": 2030,
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "area_km2": 2659.1454,
      "source": "RF-CA projection"
    },
    {
      "year": 2030,
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "area_km2": 0.0,
      "source": "RF-CA projection"
    }
  ],
  "classAccuracy": [
    {
      "model": "Markov baseline",
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "precision": 0.957013,
      "recall": 0.98124,
      "f1_score": 0.968975
    },
    {
      "model": "Markov baseline",
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "precision": 0.891334,
      "recall": 0.790413,
      "f1_score": 0.837845
    },
    {
      "model": "Markov baseline",
      "class_code": 3,
      "class_name": "Shrub",
      "class_cn": "灌木",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "Markov baseline",
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "precision": 0.324442,
      "recall": 0.305724,
      "f1_score": 0.314805
    },
    {
      "model": "Markov baseline",
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "precision": 0.855168,
      "recall": 0.648178,
      "f1_score": 0.737423
    },
    {
      "model": "Markov baseline",
      "class_code": 6,
      "class_name": "Snow/Ice",
      "class_cn": "冰雪",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "Markov baseline",
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "precision": 0.41328,
      "recall": 0.207735,
      "f1_score": 0.276492
    },
    {
      "model": "Markov baseline",
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "precision": 0.956293,
      "recall": 0.90308,
      "f1_score": 0.928925
    },
    {
      "model": "Markov baseline",
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "CA-Markov",
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "precision": 0.964671,
      "recall": 0.989092,
      "f1_score": 0.976729
    },
    {
      "model": "CA-Markov",
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "precision": 0.941853,
      "recall": 0.835212,
      "f1_score": 0.885332
    },
    {
      "model": "CA-Markov",
      "class_code": 3,
      "class_name": "Shrub",
      "class_cn": "灌木",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "CA-Markov",
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "precision": 0.491937,
      "recall": 0.463555,
      "f1_score": 0.477325
    },
    {
      "model": "CA-Markov",
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "precision": 0.919977,
      "recall": 0.6973,
      "f1_score": 0.793309
    },
    {
      "model": "CA-Markov",
      "class_code": 6,
      "class_name": "Snow/Ice",
      "class_cn": "冰雪",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "CA-Markov",
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "precision": 0.580515,
      "recall": 0.291797,
      "f1_score": 0.388376
    },
    {
      "model": "CA-Markov",
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "precision": 0.963707,
      "recall": 0.910081,
      "f1_score": 0.936127
    },
    {
      "model": "CA-Markov",
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "precision": 0.96468,
      "recall": 0.989101,
      "f1_score": 0.976738
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "precision": 0.942152,
      "recall": 0.835477,
      "f1_score": 0.885614
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 3,
      "class_name": "Shrub",
      "class_cn": "灌木",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "precision": 0.491008,
      "recall": 0.462679,
      "f1_score": 0.476423
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "precision": 0.918603,
      "recall": 0.696259,
      "f1_score": 0.792124
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 6,
      "class_name": "Snow/Ice",
      "class_cn": "冰雪",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "precision": 0.581136,
      "recall": 0.292109,
      "f1_score": 0.388791
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "precision": 0.963663,
      "recall": 0.910039,
      "f1_score": 0.936084
    },
    {
      "model": "CA-Markov suitability",
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "Logistic-CA",
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "precision": 0.964143,
      "recall": 0.988551,
      "f1_score": 0.976195
    },
    {
      "model": "Logistic-CA",
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "precision": 0.937631,
      "recall": 0.831468,
      "f1_score": 0.881364
    },
    {
      "model": "Logistic-CA",
      "class_code": 3,
      "class_name": "Shrub",
      "class_cn": "灌木",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "Logistic-CA",
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "precision": 0.455929,
      "recall": 0.429624,
      "f1_score": 0.442386
    },
    {
      "model": "Logistic-CA",
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "precision": 0.922535,
      "recall": 0.699239,
      "f1_score": 0.795515
    },
    {
      "model": "Logistic-CA",
      "class_code": 6,
      "class_name": "Snow/Ice",
      "class_cn": "冰雪",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "Logistic-CA",
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "precision": 0.527769,
      "recall": 0.265284,
      "f1_score": 0.353088
    },
    {
      "model": "Logistic-CA",
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "precision": 0.964341,
      "recall": 0.91068,
      "f1_score": 0.936742
    },
    {
      "model": "Logistic-CA",
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "RF-CA",
      "class_code": 1,
      "class_name": "Cropland",
      "class_cn": "耕地",
      "precision": 0.964628,
      "recall": 0.989048,
      "f1_score": 0.976685
    },
    {
      "model": "RF-CA",
      "class_code": 2,
      "class_name": "Forest",
      "class_cn": "林地",
      "precision": 0.940732,
      "recall": 0.834218,
      "f1_score": 0.884279
    },
    {
      "model": "RF-CA",
      "class_code": 3,
      "class_name": "Shrub",
      "class_cn": "灌木",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "RF-CA",
      "class_code": 4,
      "class_name": "Grassland",
      "class_cn": "草地",
      "precision": 0.49219,
      "recall": 0.463793,
      "f1_score": 0.47757
    },
    {
      "model": "RF-CA",
      "class_code": 5,
      "class_name": "Water",
      "class_cn": "水体",
      "precision": 0.924686,
      "recall": 0.70087,
      "f1_score": 0.79737
    },
    {
      "model": "RF-CA",
      "class_code": 6,
      "class_name": "Snow/Ice",
      "class_cn": "冰雪",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    },
    {
      "model": "RF-CA",
      "class_code": 7,
      "class_name": "Barren",
      "class_cn": "裸地",
      "precision": 0.578343,
      "recall": 0.290705,
      "f1_score": 0.386923
    },
    {
      "model": "RF-CA",
      "class_code": 8,
      "class_name": "Impervious",
      "class_cn": "不透水面",
      "precision": 0.964208,
      "recall": 0.910555,
      "f1_score": 0.936614
    },
    {
      "model": "RF-CA",
      "class_code": 9,
      "class_name": "Wetland",
      "class_cn": "湿地",
      "precision": 0.0,
      "recall": 0.0,
      "f1_score": 0.0
    }
  ],
  "featureImportance": [
    {
      "rank": 1,
      "feature": "current_is_1_Cropland",
      "importance": 0.24258429
    },
    {
      "rank": 2,
      "feature": "current_is_8_Impervious",
      "importance": 0.20567031
    },
    {
      "rank": 3,
      "feature": "neighbor_frac_1_Cropland",
      "importance": 0.1427878
    },
    {
      "rank": 4,
      "feature": "neighbor_frac_8_Impervious",
      "importance": 0.10800124
    },
    {
      "rank": 5,
      "feature": "driver_nightlight_2020",
      "importance": 0.04637581
    },
    {
      "rank": 6,
      "feature": "driver_water_closeness_2020",
      "importance": 0.04299689
    },
    {
      "rank": 7,
      "feature": "driver_road_closeness_2020",
      "importance": 0.03910336
    },
    {
      "rank": 8,
      "feature": "driver_elevation_norm_2020",
      "importance": 0.03820159
    },
    {
      "rank": 9,
      "feature": "current_is_2_Forest",
      "importance": 0.03481657
    },
    {
      "rank": 10,
      "feature": "neighbor_frac_2_Forest",
      "importance": 0.03282045
    },
    {
      "rank": 11,
      "feature": "current_is_5_Water",
      "importance": 0.02086746
    },
    {
      "rank": 12,
      "feature": "neighbor_frac_5_Water",
      "importance": 0.0185012
    },
    {
      "rank": 13,
      "feature": "driver_low_slope_2020",
      "importance": 0.01478142
    },
    {
      "rank": 14,
      "feature": "neighbor_frac_4_Grassland",
      "importance": 0.00710401
    },
    {
      "rank": 15,
      "feature": "current_is_4_Grassland",
      "importance": 0.00449524
    },
    {
      "rank": 16,
      "feature": "neighbor_frac_7_Barren",
      "importance": 0.00071502
    },
    {
      "rank": 17,
      "feature": "current_is_7_Barren",
      "importance": 0.00017734
    },
    {
      "rank": 18,
      "feature": "current_is_3_Shrub",
      "importance": 0.0
    },
    {
      "rank": 19,
      "feature": "current_is_6_Snow/Ice",
      "importance": 0.0
    },
    {
      "rank": 20,
      "feature": "current_is_9_Wetland",
      "importance": 0.0
    },
    {
      "rank": 21,
      "feature": "neighbor_frac_3_Shrub",
      "importance": 0.0
    },
    {
      "rank": 22,
      "feature": "neighbor_frac_6_Snow/Ice",
      "importance": 0.0
    },
    {
      "rank": 23,
      "feature": "neighbor_frac_9_Wetland",
      "importance": 0.0
    }
  ],
  "figures": [
    {
      "title": "2025 模型精度对比",
      "src": "assets/figures/fig01_2025_model_accuracy_comparison.png",
      "source": "E:\\model\\RF-CA\\figures\\paper\\shenyang\\fig01_2025_model_accuracy_comparison.png"
    },
    {
      "title": "2025 各类别 F1 对比",
      "src": "assets/figures/fig02_2025_per_class_f1_comparison.png",
      "source": "E:\\model\\RF-CA\\figures\\paper\\shenyang\\fig02_2025_per_class_f1_comparison.png"
    },
    {
      "title": "2025 真实-预测图对比",
      "src": "assets/figures/fig03_2025_prediction_map_comparison.png",
      "source": "E:\\model\\RF-CA\\figures\\paper\\shenyang\\fig03_2025_prediction_map_comparison.png"
    },
    {
      "title": "2030 预测图对比",
      "src": "assets/figures/fig04_2030_projection_map_comparison.png",
      "source": "E:\\model\\RF-CA\\figures\\paper\\shenyang\\fig04_2030_projection_map_comparison.png"
    },
    {
      "title": "RF-CA 2030 面积变化",
      "src": "assets/figures/fig05_rf_ca_2030_area_change.png",
      "source": "E:\\model\\RF-CA\\figures\\paper\\shenyang\\fig05_rf_ca_2030_area_change.png"
    },
    {
      "title": "随机森林特征重要性",
      "src": "assets/figures/fig06_rf_feature_importance.png",
      "source": "E:\\model\\RF-CA\\figures\\paper\\shenyang\\fig06_rf_feature_importance.png"
    },
    {
      "title": "RF-CA 2025 混淆矩阵",
      "src": "assets/figures/fig07_rf_ca_2025_confusion_matrix.png",
      "source": "E:\\model\\RF-CA\\figures\\paper\\shenyang\\fig07_rf_ca_2025_confusion_matrix.png"
    }
  ],
  "paths": {
    "model_comparison": "E:\\model\\RF-CA\\tables\\model_comparison\\shenyang_2025_validation_model_comparison.csv",
    "historical_area": "E:\\model\\CA-Markov\\tables\\shenyang_clcd_original_class_summary.csv",
    "rf_feature_importance": "E:\\model\\RF-CA\\tables\\random_forest\\shenyang\\shenyang_rf_fit_2020_2025_driver2020_n5_sampseed42_rf300_depthnone_leaf1_wtransition_trainseed42_feature_importance.csv",
    "rf_ca_area_2030": "E:\\model\\RF-CA\\tables\\rf_ca\\shenyang\\shenyang_rf_ca_fit_2020_2025_predict_2030_driver2025_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42_area_projection.csv",
    "logistic_area_2030": "E:\\model\\Logistic-CA\\tables\\shenyang_logistic_ca_fit_2020_2025_predict_2030_n5_i5_area_projection.csv",
    "ca_markov_area_2030": "E:\\model\\CA-Markov\\tables\\shenyang_ca_markov_fit_2020_2025_predict_2030_n5_i5_area_projection.csv",
    "rf_ca_f1": "E:\\model\\RF-CA\\tables\\rf_ca\\shenyang\\shenyang_rf_ca_fit_2015_2020_predict_2025_driver2020_n5_i5_rf300_depthnone_leaf1_rw070_nw030_seed42_per_class_accuracy.csv",
    "logistic_f1": "E:\\model\\Logistic-CA\\tables\\shenyang_logistic_ca_fit_2015_2020_predict_2025_n5_i5_per_class_accuracy.csv",
    "ca_markov_f1": "E:\\model\\CA-Markov\\tables\\shenyang_ca_markov_fit_2015_2020_predict_2025_n5_i5_per_class_accuracy.csv",
    "ca_markov_suit_f1": "E:\\model\\CA-Markov\\tables\\shenyang_ca_markov_suitability_fit_2015_2020_predict_2025_n5_i5_nw090_sw010_per_class_accuracy.csv",
    "markov_f1": "E:\\model\\CA-Markov\\tables\\shenyang_markov_fit_2015_2020_predict_2025_per_class_accuracy.csv"
  }
};
