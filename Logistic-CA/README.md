# Logistic-CA

Logistic-CA project for land-use type prediction in Shenyang.

## Data Kept

- `data/processed/landuse/shenyang/`
  - `shenyang_clcd_v01_2015_original.tif`
  - `shenyang_clcd_v01_2020_original.tif`
  - `shenyang_clcd_v01_2025_original.tif`
- `data/processed/suitability/shenyang/`
  - `elevation_norm`
  - `low_slope`
  - `nightlight`
  - `road_closeness`
  - `water_closeness`
  - each factor has 2020 and 2025 rasters.

## Planned Workflow

1. Train Logistic-CA transition rules with `2015 -> 2020`.
2. Predict `2025` from the `2020` land-use map.
3. Validate against the real `2025` CLCD map.
4. Train with `2020 -> 2025` and predict `2030`.

## Outputs

- Scripts: `scripts/`
- Tables: `tables/`
- Predicted rasters: `output/logistic_ca/shenyang/`
- Figures: `figures/logistic_ca/shenyang/`

Recommended conda environment: `ca`.

## Run Order

Activate the environment first:

```powershell
conda activate ca
```

Step 1: validation experiment, train with `2015 -> 2020` and predict `2025`.

```powershell
python E:\Logistic-CA\scripts\logistic_ca.py
```

Step 2: draw validation figures after Step 1 finishes.

```powershell
python E:\Logistic-CA\scripts\plot_logistic_ca_results.py
```

Step 3: future projection, train with `2020 -> 2025` and predict `2030`.
Use the dedicated 2030 script. It uses an SGD-based logistic model that is
more stable for this large raster sample.

```powershell
python E:\Logistic-CA\scripts\predict_2030_logistic_ca.py
```

Step 4: draw future projection figures after Step 3 finishes.

```powershell
python E:\Logistic-CA\scripts\plot_2030_logistic_ca.py
```

For a faster trial run, reduce samples and iterations:

```powershell
python E:\Logistic-CA\scripts\logistic_ca.py --samples-per-class 5000 --max-changed-samples 20000 --iterations 1
```
