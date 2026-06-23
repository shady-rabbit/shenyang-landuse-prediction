from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static_webgis"
ASSETS_DIR = STATIC_DIR / "assets"
DATA_JSON = ASSETS_DIR / "data.json"
DATA_JS = ASSETS_DIR / "data.js"
MODEL_ROOT = Path(os.environ.get("MODEL_ROOT", r"E:\model"))

PROJECT_REQUIREMENTS = {
    "RF-CA": r"tables\model_comparison\shenyang_2025_validation_model_comparison.csv",
    "Logistic-CA": r"output\logistic_ca\shenyang\shenyang_logistic_ca_fit_2015_2020_predict_2025_n5_i5.tif",
    "CA-Markov": r"tables\shenyang_clcd_original_class_summary.csv",
}

MODEL_LABELS = {
    "Markov baseline": "Markov baseline",
    "CA-Markov": "CA-Markov",
    "CA-Markov suitability": "适宜性因子 CA-Markov",
    "Logistic-CA": "Logistic-CA",
    "RF-CA": "RF-CA",
}


st.set_page_config(
    page_title="沈阳市土地利用变化预测展示系统",
    page_icon="map",
    layout="wide",
)


def inject_style() -> None:
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
        [data-testid="stMetric"] {
            background: #f8fbfd;
            border: 1px solid #dce7ef;
            border-radius: 8px;
            padding: 14px 16px;
        }
        div[data-testid="stExpander"] {
            border: 1px solid #dce7ef;
            border-radius: 8px;
        }
        .small-note {
            color: #5f6f7c;
            font-size: 0.92rem;
        }
        .layer-caption {
            color: #425563;
            margin-top: -0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_data() -> dict:
    if DATA_JSON.exists():
        return json.loads(DATA_JSON.read_text(encoding="utf-8"))
    if DATA_JS.exists():
        text = DATA_JS.read_text(encoding="utf-8")
        prefix = "window.DASHBOARD_DATA = "
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
        if text.endswith(";"):
            text = text[:-1]
        return json.loads(text)
    st.error("未找到展示数据，请先运行 build_assets.py。")
    st.stop()


def dataframe(records: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(records)


def asset_path(relative: str) -> Path:
    return STATIC_DIR / relative


def project_status() -> pd.DataFrame:
    rows = []
    for name, required in PROJECT_REQUIREMENTS.items():
        root = MODEL_ROOT / name
        required_path = root / required
        rows.append(
            {
                "项目": name,
                "路径": str(root),
                "状态": "可用" if required_path.exists() else "未检测到完整项目",
                "关键文件": required,
            }
        )
    return pd.DataFrame(rows)


def refresh_assets() -> tuple[bool, str]:
    script = BASE_DIR / "build_assets.py"
    if not script.exists():
        return False, "未找到 build_assets.py。"
    env = os.environ.copy()
    env["MODEL_ROOT"] = str(MODEL_ROOT)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(BASE_DIR),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    message = (result.stdout + "\n" + result.stderr).strip()
    return result.returncode == 0, message


def model_name(value: str) -> str:
    return MODEL_LABELS.get(value, value)


def render_sidebar(data: dict) -> dict:
    st.sidebar.title("展示控制")
    st.sidebar.caption("当前系统统一读取 E:\\model 下的三个模型项目。")

    if st.sidebar.button("刷新展示资产", width="stretch"):
        with st.spinner("正在重新生成地图 PNG 和数据文件..."):
            ok, message = refresh_assets()
        if ok:
            st.sidebar.success("刷新完成，页面会使用最新资产。")
            st.cache_data.clear()
        else:
            st.sidebar.error("刷新失败。")
        with st.sidebar.expander("刷新日志", expanded=not ok):
            st.code(message or "无输出")

    layer_mode = st.sidebar.radio(
        "图层类型",
        ["土地利用与预测结果", "驱动因子"],
        horizontal=False,
    )
    layers = data["drivers"] if layer_mode == "驱动因子" else data["layers"]
    if layer_mode == "驱动因子":
        groups = ["驱动因子"]
    else:
        groups = list(dict.fromkeys(layer["group"] for layer in layers))
    default_group = "2030 未来预测" if "2030 未来预测" in groups else groups[0]
    group = st.sidebar.selectbox("图层分组", groups, index=groups.index(default_group))
    group_layers = layers if layer_mode == "驱动因子" else [item for item in layers if item["group"] == group]
    default_layer_index = 0
    if layer_mode != "驱动因子":
        for index, item in enumerate(group_layers):
            if item["id"] == "rf_ca_2030":
                default_layer_index = index
                break
    layer = st.sidebar.selectbox(
        "当前图层",
        group_layers,
        index=default_layer_index,
        format_func=lambda item: item["title"],
    )

    st.sidebar.divider()
    st.sidebar.subheader("项目状态")
    status = project_status()
    for row in status.to_dict("records"):
        prefix = "[OK]" if row["状态"] == "可用" else "[WAIT]"
        st.sidebar.caption(f"{prefix} {row['项目']}：{row['状态']}")

    return {"layer_mode": layer_mode, "group": group, "layer": layer}


def render_header(data: dict) -> None:
    meta = data["meta"]
    st.caption("CLCD 2015-2025 回测验证 · 2030 情景预测")
    st.title(meta["title"])
    st.markdown(
        f"<div class='small-note'>{meta['subtitle']}｜研究区：{meta['study_area']}｜数据源：{meta['data_source']}</div>",
        unsafe_allow_html=True,
    )


def render_layer_view(data: dict, controls: dict) -> None:
    layer = controls["layer"]
    left, right = st.columns([0.72, 0.28], gap="large")
    with left:
        st.subheader(layer["title"])
        image_path = asset_path(layer["src"])
        if image_path.exists():
            st.image(str(image_path), width="stretch")
        else:
            st.warning(f"未找到图层图片：{image_path}")
    with right:
        st.subheader("图层信息")
        st.markdown(f"**模型/类型：** {layer.get('model', 'Driver')}")
        if layer.get("year"):
            st.markdown(f"**年份：** {layer['year']}")
        st.markdown(f"<p class='layer-caption'>{layer.get('description', '')}</p>", unsafe_allow_html=True)
        st.markdown("**土地利用图例**" if controls["layer_mode"] != "驱动因子" else "**驱动因子图例**")
        if controls["layer_mode"] == "驱动因子":
            st.markdown(
                """
                <div style="height:14px;border-radius:8px;background:linear-gradient(90deg,#f7fbff,#08519c);"></div>
                <div class="small-note">颜色越深，数值越高。</div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for cls in data["classes"]:
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin:6px 0;'>"
                    f"<span style='width:16px;height:16px;border-radius:4px;border:1px solid #999;background:{cls['color']};display:inline-block;'></span>"
                    f"<span>{cls['code']} {cls['name']} / {cls['cn']}</span></div>",
                    unsafe_allow_html=True,
                )
        with st.expander("源文件路径"):
            st.code(layer.get("source", ""))


def render_metrics(data: dict) -> None:
    comparison = dataframe(data["modelComparison"])
    rf = comparison[comparison["model"] == "RF-CA"].iloc[0]
    ca = comparison[comparison["model"] == "CA-Markov"].iloc[0]
    baseline = comparison[comparison["model"] == "Markov baseline"].iloc[0]
    best = comparison.sort_values("overall_accuracy", ascending=False).iloc[0]

    cols = st.columns(4)
    cols[0].metric("最佳模型", best["model"], f"OA {best['overall_accuracy']:.6f}")
    cols[1].metric("RF-CA OA", f"{rf['overall_accuracy']:.6f}", f"Kappa {rf['kappa']:.6f}")
    cols[2].metric("较 Markov baseline", f"+{(rf['overall_accuracy'] - baseline['overall_accuracy']) * 100:.3f} p.p.")
    cols[3].metric("较 CA-Markov", f"+{(rf['overall_accuracy'] - ca['overall_accuracy']) * 100:.4f} p.p.")


def render_overview(data: dict) -> None:
    render_metrics(data)
    col1, col2 = st.columns([0.52, 0.48], gap="large")
    with col1:
        st.subheader("核心结论")
        st.markdown(
            """
            - RF-CA 在 2025 回测中取得最高精度，可作为主模型展示。
            - RF-CA 相比无驱动 CA-Markov 提升很小，说明 CA 邻域约束是空间格局模拟的主要贡献。
            - 小面积类别如 Shrub、Barren、Wetland 识别能力较弱，适合作为误差分析重点。
            - 2030 面积需求在多个 CA 模型中较接近，模型差异主要体现在空间分配位置。
            """
        )
    with col2:
        st.subheader("RF 特征重要性 Top 12")
        features = dataframe(data["featureImportance"]).head(12).sort_values("importance")
        fig = px.bar(
            features,
            x="importance",
            y="feature",
            orientation="h",
            text="importance",
            color_discrete_sequence=["#0f6fae"],
        )
        fig.update_traces(texttemplate="%{text:.4f}", textposition="outside")
        fig.update_layout(height=430, margin=dict(l=10, r=40, t=10, b=10), xaxis_title="Importance", yaxis_title="")
        st.plotly_chart(fig, width="stretch")


def render_accuracy(data: dict) -> None:
    st.subheader("2025 回测精度对比")
    comparison = dataframe(data["modelComparison"]).copy()
    comparison["model_label"] = comparison["model"].map(model_name)
    comparison = comparison.sort_values("overall_accuracy", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=comparison["model_label"],
            y=comparison["overall_accuracy"],
            name="OA",
            marker_color="#0f6fae",
            text=comparison["overall_accuracy"].map(lambda x: f"{x:.6f}"),
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            x=comparison["model_label"],
            y=comparison["kappa"],
            name="Kappa",
            marker_color="#1f8a5b",
            text=comparison["kappa"].map(lambda x: f"{x:.6f}"),
            textposition="outside",
        )
    )
    fig.update_layout(
        barmode="group",
        height=430,
        margin=dict(l=10, r=10, t=20, b=20),
        yaxis=dict(range=[0.84, 0.975]),
        xaxis_title="",
        yaxis_title="Score",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        comparison[
            [
                "model",
                "fit_from",
                "base_year",
                "target_year",
                "key_parameters",
                "overall_accuracy",
                "kappa",
                "delta_oa_vs_markov_baseline",
                "notes",
            ]
        ],
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "下载模型精度对比 CSV",
        comparison.to_csv(index=False).encode("utf-8-sig"),
        "shenyang_2025_model_comparison.csv",
        "text/csv",
    )


def render_area(data: dict) -> None:
    st.subheader("RF-CA 2025-2030 面积变化")
    area = dataframe(data["areaProjection"]).copy()
    area["label"] = area["class_name"] + " / " + area["class_cn"]
    area = area.sort_values("change_area_km2")
    colors = ["#cf3b35" if value < 0 else "#1f8a5b" for value in area["change_area_km2"]]
    fig = go.Figure(
        go.Bar(
            x=area["change_area_km2"],
            y=area["label"],
            orientation="h",
            marker_color=colors,
            text=area["change_area_km2"].map(lambda value: f"{value:+.3f}"),
            textposition="outside",
        )
    )
    fig.update_layout(height=460, margin=dict(l=10, r=60, t=20, b=20), xaxis_title="面积变化 km²", yaxis_title="")
    fig.add_vline(x=0, line_color="#6b7884", line_width=1)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(area.drop(columns=["label"]), width="stretch", hide_index=True)


def render_class_accuracy(data: dict) -> None:
    st.subheader("类别精度与误差分析")
    class_acc = dataframe(data["classAccuracy"]).copy()
    models = class_acc["model"].drop_duplicates().tolist()
    selected_models = st.multiselect("选择模型", models, default=["RF-CA", "CA-Markov", "Logistic-CA"])
    filtered = class_acc[class_acc["model"].isin(selected_models)].copy()
    filtered["label"] = filtered["class_name"] + " / " + filtered["class_cn"]
    fig = px.bar(
        filtered,
        x="label",
        y="f1_score",
        color="model",
        barmode="group",
        text="f1_score",
        color_discrete_sequence=["#8c6bb1", "#0f6fae", "#1f8a5b", "#d95f02", "#cf3b35"],
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig.update_layout(height=470, margin=dict(l=10, r=10, t=20, b=20), xaxis_title="", yaxis_title="F1 score", yaxis_range=[0, 1.08])
    st.plotly_chart(fig, width="stretch")
    st.dataframe(filtered.drop(columns=["label"]), width="stretch", hide_index=True)


def render_figures(data: dict) -> None:
    st.subheader("论文图件汇总")
    figure_cols = st.columns(2)
    for index, figure in enumerate(data["figures"]):
        with figure_cols[index % 2]:
            st.markdown(f"**{figure['title']}**")
            path = asset_path(figure["src"])
            if path.exists():
                st.image(str(path), width="stretch")
            else:
                st.warning(f"未找到图片：{path}")


def render_data_sources(data: dict) -> None:
    st.subheader("数据源与项目路径")
    st.dataframe(project_status(), width="stretch", hide_index=True)
    st.markdown("**展示数据文件**")
    st.code(str(DATA_JSON if DATA_JSON.exists() else DATA_JS))
    st.markdown("**CSV 源路径**")
    path_rows = [{"名称": key, "路径": value} for key, value in data.get("paths", {}).items()]
    st.dataframe(pd.DataFrame(path_rows), width="stretch", hide_index=True)


def main() -> None:
    inject_style()
    data = load_data()
    controls = render_sidebar(data)
    render_header(data)
    st.divider()
    render_layer_view(data, controls)
    st.divider()

    tabs = st.tabs(["综合概览", "精度对比", "面积变化", "类别误差", "论文图件", "数据来源"])
    with tabs[0]:
        render_overview(data)
    with tabs[1]:
        render_accuracy(data)
    with tabs[2]:
        render_area(data)
    with tabs[3]:
        render_class_accuracy(data)
    with tabs[4]:
        render_figures(data)
    with tabs[5]:
        render_data_sources(data)


if __name__ == "__main__":
    main()
