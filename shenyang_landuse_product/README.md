# 沈阳市土地利用变化预测产品化展示系统

本目录包含两种展示方式：

- `app.py`：Streamlit + Plotly 交互式展示系统。
- `static_webgis/index.html`：免安装的静态 WebGIS 展示页。

## 新电脑快速查看

如果只是查看成果，不需要安装 Python。直接用浏览器打开：

```text
shenyang_landuse_product/static_webgis/index.html
```

仓库中已经包含 `static_webgis/assets` 展示资产，因此静态页面可以独立查看。

## 新电脑运行 Streamlit

建议安装 Anaconda 或 Miniconda，然后打开 Anaconda Prompt 或 PowerShell：

```powershell
conda create -n ca python=3.11 -y
conda activate ca
python -m pip install streamlit plotly pandas
```

进入仓库根目录后运行：

```powershell
python -m streamlit run shenyang_landuse_product\app.py
```

如果已经进入 `shenyang_landuse_product` 目录，也可以运行：

```powershell
python -m streamlit run app.py
```

启动成功后浏览器通常会自动打开；若没有自动打开，访问：

```text
http://localhost:8501
```

## 重新生成展示资产

一般展示时不需要重新生成资产。只有在 RF-CA、Logistic-CA、CA-Markov 的结果更新后，才需要运行 `build_assets.py`。

默认情况下，脚本会从 `E:\model` 读取三个完整项目：

```text
E:\model\RF-CA
E:\model\Logistic-CA
E:\model\CA-Markov
```

需要额外依赖：

```powershell
conda activate ca
python -m pip install numpy rasterio pillow
```

运行：

```powershell
cd E:\model\shenyang_landuse_product
python build_assets.py
```

如果三个完整模型项目放在其他位置，可以通过 `MODEL_ROOT` 指定其上一级目录：

```powershell
$env:MODEL_ROOT = "D:\your_model_folder"
python build_assets.py
```

## 说明

本产品系统集成了沈阳市 CLCD 土地利用数据、多模型 2025 回测结果、2030 预测结果、模型精度评价、面积变化统计、类别误差分析和论文图件。相较于静态报告图件，Streamlit 版本支持模型、年份、图层和指标的交互式切换，可作为土地利用变化预测研究的轻量 WebGIS/可视化产品原型。
