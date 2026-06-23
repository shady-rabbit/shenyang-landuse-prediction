# 沈阳市土地利用变化预测与产品化展示

本仓库为《地理空间信息工程与技术》期末研究报告配套项目，围绕沈阳市土地利用变化预测问题，基于 CLCD 多时相土地利用数据，构建 RF-CA、Logistic-CA、CA-Markov 与 Markov baseline 多模型对比框架，并完成可视化产品化输出。

## 项目亮点

- 真实空间问题：沈阳市土地利用变化预测。
- 空间数据基础：CLCD 2015、2020、2025 年土地利用数据，保留原始 9 类。
- 结果验证：2015-2020 训练，基于 2020 预测 2025，并用真实 2025 CLCD 验证。
- 方法对比：Markov baseline、CA-Markov、适宜性因子 CA-Markov、Logistic-CA、RF-CA。
- 产品化输出：Streamlit + Plotly 交互系统与静态 WebGIS 展示页。

## 核心精度

| 模型 | OA | Kappa |
|---|---:|---:|
| RF-CA | 0.959994 | 0.895504 |
| CA-Markov | 0.959903 | 0.895266 |
| CA-Markov suitability | 0.959885 | 0.895220 |
| Logistic-CA | 0.959266 | 0.893603 |
| Markov baseline | 0.949188 | 0.867277 |

## 目录说明

```text
reports/                         # 研究报告、产品说明文档、GitHub 上传说明
shenyang_landuse_product/         # Streamlit/Plotly 与静态 WebGIS 展示系统
RF-CA/scripts, tables, figures    # RF-CA 代码、关键表格和图件
Logistic-CA/scripts, tables, figures
CA-Markov/scripts, tables, figures
```

## 运行环境与软件包

建议使用 Windows + Anaconda/Miniconda 运行本项目。当前已测试环境如下：

| 项目 | 建议/当前环境 |
|---|---|
| Conda 环境名 | ca |
| Python 路径 | `D:\anaconda\envs\ca\python.exe` |
| Streamlit | 1.58.0 |
| Plotly | 6.8.0 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| rasterio | 1.4.4 |
| Pillow | 12.2.0 |

如果在新电脑或新环境中运行，可先进入项目目录，再安装核心展示依赖：

```powershell
D:\anaconda\envs\ca\python.exe -m pip install streamlit plotly
```

如需重新生成展示系统资产，建议在确认 `RF-CA`、`Logistic-CA`、`CA-Markov` 三个项目目录已放在 `E:\model` 下后运行：

```powershell
Set-Location E:\model\shenyang_landuse_product
D:\anaconda\envs\ca\python.exe build_assets.py
```

说明：`build_assets.py` 会尽量自动配置 GDAL/PROJ 相关路径；如果出现 `GDAL_DATA is not defined` 一类警告，但系统仍能正常启动和显示图表，通常不影响本展示系统使用。

## 运行产品系统

如果在本机保持原始目录结构：

```powershell
D:\anaconda\envs\ca\python.exe -m streamlit run E:\model\shenyang_landuse_product\app.py
```

若从本仓库目录运行，可进入 `shenyang_landuse_product` 后运行：

```powershell
D:\anaconda\envs\ca\python.exe -m streamlit run app.py
```

静态展示页：

```text
shenyang_landuse_product/static_webgis/index.html
```

## 说明

为避免 GitHub 仓库过大，本上传包不包含原始 CLCD 数据、大体量 GeoTIFF、缓存文件、模型权重和压缩包。完整数据与模型运行环境保存在本地 `E:\model` 目录下。

## AI 使用说明

项目报告和说明文档在结构整理、语言润色、排版和上传说明编写环节使用AI辅助；模型结果、精度表、预测图件和产品系统均来自本地项目输出。
