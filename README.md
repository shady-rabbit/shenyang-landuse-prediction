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

本仓库已经包含展示系统所需的轻量资产，因此在新电脑上“查看展示结果”和“重新生成展示资产”是两种不同情况：

| 使用场景 | 是否需要完整模型数据 | 需要的软件 |
|---|---:|---|
| 打开静态 WebGIS 页面 | 不需要 | 浏览器 |
| 运行 Streamlit 交互系统 | 不需要 | Python、Streamlit、Plotly、pandas |
| 重新生成地图 PNG 和数据资产 | 需要 | 上述环境 + numpy、rasterio、Pillow，并准备完整模型输出目录 |

本机测试环境如下，仅作为参考，不要求新电脑路径完全一致：

| 项目 | 测试环境 |
|---|---|
| Conda 环境名 | ca |
| Python 路径 | `D:\anaconda\envs\ca\python.exe` |
| Streamlit | 1.58.0 |
| Plotly | 6.8.0 |
| pandas | 3.0.3 |
| numpy | 2.4.6 |
| rasterio | 1.4.4 |
| Pillow | 12.2.0 |

## 新电脑运行产品系统

### 1. 获取项目文件

在 GitHub 仓库页面点击 `Code`，可以任选一种方式获取项目：

- 使用 GitHub 客户端：在网页中选择用本地客户端打开，选择本地保存位置并克隆。
- 下载压缩包：点击 `Download ZIP`，下载后解压。

下面假设解压或克隆后的仓库目录为：

```text
D:\shenyang-landuse-prediction
```

实际路径可以不同，后续命令中的路径按自己的电脑修改即可。

### 2. 最简单查看方式：打开静态 WebGIS

如果只是查看地图展示、模型对比图表和成果页面，不需要安装 Python。直接用浏览器打开：

```text
D:\shenyang-landuse-prediction\shenyang_landuse_product\static_webgis\index.html
```

这种方式适合老师快速查看成果，缺点是交互能力比 Streamlit 版本弱一些。

### 3. 交互展示方式：运行 Streamlit

如果新电脑还没有 Python 环境，建议先安装 Anaconda 或 Miniconda，然后打开 Anaconda Prompt 或 PowerShell，执行：

```powershell
conda create -n ca python=3.11 -y
conda activate ca
python -m pip install streamlit plotly pandas
```

进入仓库目录并启动系统：

```powershell
cd D:\shenyang-landuse-prediction
python -m streamlit run shenyang_landuse_product\app.py
```

启动后浏览器通常会自动打开。如果没有自动打开，可在浏览器地址栏输入：

```text
http://localhost:8501
```

### 4. 可选：重新生成展示资产

一般查看 GitHub 仓库成果时不需要执行本步骤，因为仓库中已经包含 `static_webgis/assets` 展示资产。

只有当三个模型的输出结果更新后，才需要重新生成资产。此时需要把完整项目放成如下结构：

```text
E:\model\RF-CA
E:\model\Logistic-CA
E:\model\CA-Markov
E:\model\shenyang_landuse_product
```

然后安装额外依赖并运行：

```powershell
conda activate ca
python -m pip install numpy rasterio pillow
cd E:\model\shenyang_landuse_product
python build_assets.py
```

如果完整模型项目不放在 `E:\model`，可以先设置 `MODEL_ROOT` 为三个模型项目所在的上一级目录：

```powershell
$env:MODEL_ROOT = "D:\your_model_folder"
python build_assets.py
```

## 说明

为避免 GitHub 仓库过大，本上传包不包含原始 CLCD 数据、大体量 GeoTIFF、缓存文件、模型权重和压缩包。完整数据与模型运行环境保存在本地 `E:\model` 目录下。

## AI 使用说明

项目报告和说明文档在结构整理、语言润色、排版和上传说明编写环节使用AI辅助；模型结果、精度表、预测图件和产品系统均来自本地项目输出。
