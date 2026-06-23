# RF-CA 土地利用变化预测项目

本项目用于在沈阳市 CLCD 数据上实现 RF-CA 模型，并与已有 Markov baseline、无驱动 CA-Markov 结果进行对比。

## 当前项目路径

- 当前 RF-CA 项目：`E:\RF-CA`
- 旧 CA-Markov 参考项目：`E:\CA-Markov`
- Python 环境：`D:\anaconda\envs\ca\python.exe`

后续代码、输出和实验记录应以 `E:\RF-CA` 为准；`E:\CA-Markov` 只作为数据、脚本和结果参考来源。

## 已复制数据

### 土地利用栅格

目录：`data/processed/landuse/shenyang`

- `shenyang_clcd_v01_2015_original.tif`
- `shenyang_clcd_v01_2020_original.tif`
- `shenyang_clcd_v01_2025_original.tif`

这些栅格来自 CLCD，保留原始 9 类：

1. Cropland
2. Forest
3. Shrub
4. Grassland
5. Water
6. Snow/Ice
7. Barren
8. Impervious
9. Wetland

### 驱动因子栅格

目录：`data/processed/drivers/shenyang`

已复制 2020 和 2025 两期处理后驱动因子：

- `shenyang_road_closeness_*.tif`
- `shenyang_water_closeness_*.tif`
- `shenyang_nightlight_*.tif`
- `shenyang_elevation_norm_*.tif`
- `shenyang_low_slope_*.tif`

这些因子已与沈阳 CLCD 栅格保持相同空间网格，可用于随机森林特征构建。

### CA-Markov 对比表

目录：`references/ca_markov_tables`

已复制 Markov baseline、无驱动 CA-Markov、适宜性因子 CA-Markov 的关键指标表和转移矩阵表，供 RF-CA 结果对比使用。

## RF-CA 建模目标

1. 使用 2015 和 2020 CLCD 构建训练样本，学习 2015 -> 2020 土地利用转移规则。
2. 使用随机森林输出各像元转为不同土地利用类型的概率或适宜性。
3. 结合 CA 空间分配和 Markov/历史转移需求，基于 2020 预测 2025。
4. 使用 2025 CLCD 验证预测结果，输出 OA、Kappa、per-class F1 和混淆矩阵。
5. 使用 2020 -> 2025 训练，基于 2025 预测 2030。

## 输出命名约定

实验输出文件名应包含模型、研究区、训练年份、预测年份和关键参数，例如：

- `shenyang_rf_ca_fit_2015_2020_predict_2025_n5_rf300_seed42.tif`
- `shenyang_rf_ca_fit_2015_2020_predict_2025_n5_rf300_seed42_summary.csv`
- `shenyang_rf_ca_fit_2020_2025_predict_2030_n5_rf300_seed42.tif`

这样可以避免不同实验结果互相覆盖。

## 样本构建

脚本：`scripts/build_rf_samples.py`

该脚本按土地利用转移类型进行分层抽样，把两期 CLCD 转换成随机森林可用的 `.npz` 样本包。样本特征包括：

- 当前土地利用类别 one-hot；
- 邻域 9 类土地利用比例；
- 道路邻近性、水系邻近性、夜光、高程归一化、低坡度因子；
- 可选的当前类别编码和行列号归一化特征。

推荐先做 dry-run 检查输入路径和栅格对齐：

```powershell
D:\anaconda\envs\ca\python.exe scripts\build_rf_samples.py --dry-run
```

构建 2015 -> 2020 训练样本：

```powershell
D:\anaconda\envs\ca\python.exe scripts\build_rf_samples.py --from-year 2015 --to-year 2020
```

构建 2020 -> 2025 训练样本：

```powershell
D:\anaconda\envs\ca\python.exe scripts\build_rf_samples.py --from-year 2020 --to-year 2025
```

输出目录：`data/samples/shenyang`

每次运行会生成：

- `.npz`：压缩样本包，包含 `X`、`y`、`rows`、`cols`、`from_class`、`to_class` 和 `feature_names`；
- `_transition_summary.csv`：每一种 `from_class -> to_class` 的可用像元数和实际抽样数；
- `_metadata.json`：样本构建参数、输入路径、特征名和警告信息。

当前项目尚未包含 2015 年处理后驱动因子。因此构建 2015 -> 2020 样本时，脚本默认使用 2020 年驱动因子，并在输出文件名和元数据中记录为 `driver2020`。

## 随机森林训练

脚本：`scripts/train_rf.py`

该脚本读取样本 `.npz`，训练随机森林转移规则模型。默认会自动查找对应年份的样本文件，并按 `_transition_summary.csv` 计算转移样本权重：

```text
weight = available_pixels / sampled_pixels
```

这样既保留分层抽样对稀有转移的覆盖，又能在训练和验证指标中反映真实像元数量。若要做均衡样本对比实验，可加 `--sample-weight-mode none`。

推荐先 dry-run：

```powershell
D:\anaconda\envs\ca\python.exe scripts\train_rf.py --from-year 2015 --to-year 2020 --dry-run
```

自动查找样本时默认匹配样本构建的 `--seed 42`。如果同一年份有多个样本版本，可用 `--sample-seed` 或 `--sample-file` 指定，例如：

```powershell
D:\anaconda\envs\ca\python.exe scripts\train_rf.py --from-year 2015 --to-year 2020 --sample-seed 43 --dry-run
```

训练 2015 -> 2020 RF 模型，用于后续 2020 -> 2025 回测：

```powershell
D:\anaconda\envs\ca\python.exe scripts\train_rf.py --from-year 2015 --to-year 2020 --n-estimators 300
```

训练 2020 -> 2025 RF 模型，用于后续基于 2025 预测 2030：

```powershell
D:\anaconda\envs\ca\python.exe scripts\train_rf.py --from-year 2020 --to-year 2025 --n-estimators 300
```

输出位置：

- 模型：`models/random_forest/shenyang`
- 指标表：`tables/random_forest/shenyang`

每次训练会输出：

- `.joblib`：包含随机森林模型、特征名、类别编码和训练参数；
- `_summary.csv`：训练集/验证集 OA、Kappa、F1；
- `_confusion_matrix.csv`：验证集未加权混淆矩阵；
- `_weighted_confusion_matrix.csv`：按真实转移像元数加权的验证集混淆矩阵；
- `_per_class_accuracy.csv`：各类别 precision、recall、F1；
- `_feature_importance.csv`：随机森林特征重要性。

## RF-CA 概率预测与 CA 分配

脚本：`scripts/rf_ca_predict.py`

该脚本把训练好的 RF 模型用于全图预测：

1. 按模型特征顺序为基期土地利用图分块构建特征；
2. 输出各像元转为 1-9 类的 RF 概率；
3. 用 Markov 转移矩阵控制目标期各类别需求量；
4. 用 `RF 概率 + CA 邻域吸引力` 为空间分配排序；
5. 如果目标年份真实 CLCD 存在，则输出全图 OA、Kappa、per-class F1 和混淆矩阵。

推荐先 dry-run 检查输入和输出路径：

```powershell
D:\anaconda\envs\ca\python.exe scripts\rf_ca_predict.py --fit-from 2015 --base-year 2020 --target-year 2025 --dry-run
```

使用 2015 -> 2020 RF 模型，基于 2020 预测 2025 并验证：

```powershell
D:\anaconda\envs\ca\python.exe scripts\rf_ca_predict.py --fit-from 2015 --base-year 2020 --target-year 2025
```

使用 2020 -> 2025 RF 模型，基于 2025 预测 2030：

```powershell
D:\anaconda\envs\ca\python.exe scripts\rf_ca_predict.py --fit-from 2020 --base-year 2025 --target-year 2030
```

默认空间分配参数：

- `--neighborhood-size 5`
- `--iterations 5`
- `--rf-weight 0.7`
- `--neighbor-weight 0.3`
- `--random-weight 0.02`

输出位置：

- 预测 GeoTIFF：`output/rf_ca/shenyang`
- 指标表：`tables/rf_ca/shenyang`

默认使用临时 memmap 缓存 RF 概率，运行结束会删除。若要保留调试缓存，可加 `--keep-probability-cache`；若要额外输出 9 波段概率 GeoTIFF，可加 `--write-probability-raster`。

## 论文图件绘制

脚本：`scripts/plot_paper_figures.py`

该脚本汇总 RF-CA、Logistic-CA、CA-Markov 和 Markov baseline 的结果，生成可放入论文的图件。当前绘图脚本使用 Pillow 直接出图，不依赖 matplotlib。

先检查输入路径：

```powershell
D:\anaconda\envs\ca\python.exe scripts\plot_paper_figures.py --dry-run
```

生成 PNG 图件：

```powershell
D:\anaconda\envs\ca\python.exe scripts\plot_paper_figures.py --formats png
```

输出目录：`figures/paper/shenyang`

默认生成：

- `fig01_2025_model_accuracy_comparison.png`
- `fig02_2025_per_class_f1_comparison.png`
- `fig03_2025_prediction_map_comparison.png`
- `fig04_2030_projection_map_comparison.png`
- `fig05_rf_ca_2030_area_change.png`
- `fig06_rf_feature_importance.png`
- `fig07_rf_ca_2025_confusion_matrix.png`
