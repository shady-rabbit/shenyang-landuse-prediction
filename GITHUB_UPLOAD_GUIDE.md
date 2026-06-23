# GitHub 上传说明

## 重要提醒

不要直接把整个 `E:\model` 原样上传到 GitHub。当前 `CA-Markov` 项目约 33GB，包含大量原始数据、GeoTIFF、缓存和模型输出，容易超过 GitHub 单文件与仓库限制。

推荐公开上传：

- 三个项目的 `scripts/`
- `README.md`
- 关键 `tables/*.csv`
- 论文图件 `figures/**/*.png`
- 产品化展示系统 `shenyang_landuse_product`
- 期末研究报告成果 `期末研究报告成果`

推荐排除：

- `data/raw/`
- `data/processed/`
- `output/**/*.tif`
- `models/**/*.joblib`
- `*.npz`
- `*.zip`
- `_cache/`

## 第一次上传步骤

1. 安装 Git 和 GitHub CLI（如果还没有）。
2. 登录 GitHub：

```powershell
gh auth login
```

3. 进入项目目录：

```powershell
cd /d E:\model
```

4. 初始化 Git：

```powershell
git init
git branch -M main
```

5. 检查将要上传的文件：

```powershell
git status --short
```

6. 添加文件并提交：

```powershell
git add README.md .gitignore GITHUB_UPLOAD_GUIDE.md shenyang_landuse_product 期末研究报告成果 RF-CA\scripts RF-CA\tables RF-CA\figures Logistic-CA\scripts Logistic-CA\tables Logistic-CA\figures CA-Markov\scripts CA-Markov\tables CA-Markov\figures
git commit -m "Add Shenyang land-use prediction project"
```

7. 创建公开仓库并推送：

```powershell
gh repo create shenyang-landuse-prediction --public --source . --remote origin --push
```

8. 浏览器打开仓库页面，确认文件、README 和报告可以查看。

## 如果要我帮你实际上传

请先告诉我：

- GitHub 仓库名
- 是否确定公开
- 是否只上传轻量发布版（推荐）

并确保本机已经完成 `gh auth login`。
