# 沈阳市土地利用变化预测展示系统

这是一个免安装的静态轻量 WebGIS 展示原型，用于展示 RF-CA、Logistic-CA、CA-Markov 与 Markov baseline 的 2025 回测和 2030 预测结果。

## 打开方式

直接双击打开 `index.html`，或在浏览器中打开该文件。

## 当前包含内容

- 2015、2020、2025 年真实 CLCD 土地利用图
- Markov baseline、CA-Markov、适宜性因子 CA-Markov、Logistic-CA、RF-CA 的 2025 回测预测图
- CA-Markov、适宜性因子 CA-Markov、Logistic-CA、RF-CA 的 2030 未来预测图
- 2025 年模型精度对比：OA、Kappa、相对 Markov baseline 提升
- RF-CA 2030 面积变化表和条形图
- 各模型分类别 Precision、Recall、F1
- 道路邻近性、水系邻近性、夜光强度、高程、低坡度等驱动因子预览
- RF-CA 项目已有论文图件汇总

## 数据刷新

如果 E 盘三个模型项目的结果更新，可以回到本工作区运行：

```powershell
D:\anaconda\envs\ca\python.exe work\build_webgis_assets.py
```

脚本会重新读取 E 盘项目结果，并刷新 `outputs/shenyang_landuse_webgis/assets` 中的展示图层和数据文件。

## 报告中可写

为增强成果表达和应用展示，本研究构建了一个沈阳市土地利用变化预测结果可视化展示原型。系统集成 CLCD 真实土地利用图、多模型 2025 回测结果、2030 预测结果、模型精度评价、面积变化统计和驱动因子图层，可交互查看不同模型、年份和指标结果，形成面向规划分析与结果汇报的轻量化 WebGIS 产品化输出。
