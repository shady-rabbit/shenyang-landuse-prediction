# 沈阳市土地利用变化预测产品化展示系统

本目录包含两个版本的成果展示：

- `app.py`：Streamlit + Plotly 交互式展示系统
- `static_webgis/index.html`：免安装静态 WebGIS 展示页

## 推荐目录

建议后续统一放在：

```text
E:\model\shenyang_landuse_product
```

同时建议三个模型项目保持如下结构：

```text
E:\model\RF-CA
E:\model\Logistic-CA
E:\model\CA-Markov
E:\model\shenyang_landuse_product
```

## 运行 Streamlit 版本

```powershell
D:\anaconda\envs\ca\python.exe -m streamlit run E:\model\shenyang_landuse_product\app.py
```

如果还没有复制到 E 盘，也可以在当前工作区运行：

```powershell
D:\anaconda\envs\ca\python.exe -m streamlit run outputs\shenyang_landuse_product\app.py
```

## 刷新展示资产

当三个模型项目的输出结果更新后，运行：

```powershell
D:\anaconda\envs\ca\python.exe E:\model\shenyang_landuse_product\build_assets.py
```

刷新脚本会读取 `E:\model` 下的三个项目。如果某个项目缺少关键文件，脚本会直接报错，便于及时检查复制是否完整。

也可以在 Streamlit 页面左侧点击“刷新展示资产”。

## 产品化输出说明

该系统集成了沈阳市 CLCD 土地利用数据、多模型 2025 回测结果、2030 预测结果、模型精度评价、面积变化统计、类别误差分析和驱动因子图层。相比静态论文图件，系统支持模型、年份、图层和指标的交互式切换，可作为土地利用变化预测研究的轻量化 WebGIS/可视化产品原型。
