# -*- coding: utf-8 -*-
"""
app.py — 世界杯比赛结果智能预测与分析系统(Streamlit 主程序)
================================================================
绿茵草地主题 · 侧边栏导航 · 六大功能页面:
    📊 数据概览      数据量、字段、类别分布
    📈 可视化分析    9 张核心图表 + 文字解释
    🤖 模型对比      指标表、混淆矩阵、最佳模型
    ⚽ 单场预测      两队下拉选择,卡片式结果 + 概率仪表盘
    📁 批量预测      上传 CSV → 批量预测 → 下载结果
    📝 分析报告      一键生成自然语言分析报告 + 下载

运行方式:
    streamlit run code/app.py
依赖:
    streamlit, pandas, numpy, joblib, plotly, streamlit-option-menu
"""

import os
import json
import base64
import html
from io import BytesIO

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_option_menu import option_menu

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

CLEAN_CSV = os.path.join(DATA_DIR, "world_cup_clean.csv")
FEATURE_CSV = os.path.join(DATA_DIR, "world_cup_features.csv")
TEAM_STATE_JSON = os.path.join(DATA_DIR, "team_state.json")
EVAL_JSON = os.path.join(OUTPUT_DIR, "eval_results.json")

# 三分类标签
LABELS = ["home_win", "draw", "away_win"]
LABEL_NAMES = {"home_win": "队伍A胜", "draw": "平局", "away_win": "队伍B胜"}

FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff", "elo_abs_diff",
    "home_recent_winrate", "away_recent_winrate",
    "home_recent_goal_diff", "away_recent_goal_diff",
    "home_avg_goals", "away_avg_goals",
    "h2h_diff", "home_wc_exp", "away_wc_exp",
    "neutral", "is_world_cup_final", "match_year",
]

# ------------------------------------------------------------------
# 页面配置
# ------------------------------------------------------------------
st.set_page_config(
    page_title="世界杯比赛结果智能预测系统",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# 自定义 CSS —— 深色体育数据产品主题
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
    --bg: #07110f;
    --sidebar: #091411;
    --surface: #0d1917;
    --surface-2: #12201d;
    --surface-3: #182824;
    --border: #24332f;
    --border-strong: #344842;
    --text: #f4f7f5;
    --muted: #91a39d;
    --accent: #b7f34a;
    --accent-soft: rgba(183, 243, 74, 0.10);
    --cyan: #4fc7c9;
    --amber: #f7b84b;
    --danger: #ff7b73;
}

html, body, [class*="css"] {
    font-family: "Inter", "Noto Sans SC", "Microsoft YaHei", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 76% -20%, rgba(74, 132, 106, 0.12), transparent 34rem),
        var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: transparent;
    height: 0;
}

[data-testid="stToolbar"] {
    right: 1rem;
    top: .65rem;
}

#MainMenu, footer { visibility: hidden; }

.block-container {
    max-width: 1480px;
    padding: 2.4rem 3rem 4rem;
}

.hero {
    padding: 0 0 1.65rem;
    margin-bottom: 1.1rem;
    border-bottom: 1px solid var(--border);
}
.hero-title {
    max-width: 980px;
    margin: 0;
    color: var(--text);
    font-size: clamp(2rem, 3.2vw, 3.15rem);
    font-weight: 780;
    line-height: 1.14;
    letter-spacing: -0.035em;
}
.hero-sub {
    max-width: 900px;
    margin-top: .7rem;
    color: var(--muted);
    font-size: .98rem;
    line-height: 1.75;
}

.glass {
    margin-bottom: 1rem;
    padding: 1.2rem 1.3rem;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    box-shadow: 0 16px 38px rgba(0, 0, 0, .13);
}
.glass h3, .glass h4 {
    margin: 0 0 .9rem;
    color: var(--text);
    font-size: 1rem;
    font-weight: 680;
    letter-spacing: -.01em;
}
.section-heading {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 1.5rem 0 .8rem;
}
.section-heading h3 {
    margin: 0;
    color: var(--text);
    font-size: 1.05rem;
    font-weight: 680;
}
.section-heading span {
    color: var(--muted);
    font-size: .78rem;
}

.metric-card {
    min-height: 132px;
    padding: 1.15rem 1.2rem;
    background: transparent;
    border-top: 1px solid var(--border-strong);
    border-bottom: 1px solid var(--border);
}
.metric-index {
    color: var(--accent);
    font-size: .72rem;
    font-weight: 750;
    letter-spacing: .12em;
    text-transform: uppercase;
}
.metric-value {
    margin-top: .62rem;
    color: var(--text);
    font-size: clamp(1.65rem, 2.35vw, 2.35rem);
    font-weight: 780;
    letter-spacing: -.035em;
}
.metric-label {
    margin-top: .22rem;
    color: var(--muted);
    font-size: .82rem;
}

.match-table {
    width: 100%;
    border-collapse: collapse;
    font-size: .8rem;
}
.match-table th {
    padding: .7rem .65rem;
    color: var(--muted);
    font-weight: 560;
    text-align: left;
    border-bottom: 1px solid var(--border-strong);
}
.match-table td {
    padding: .72rem .65rem;
    color: #dce5e1;
    border-bottom: 1px solid rgba(52, 72, 66, .55);
}
.match-table tr:last-child td { border-bottom: 0; }
.match-table .score {
    color: var(--text);
    font-weight: 700;
    white-space: nowrap;
}
.match-table .result {
    color: var(--accent);
    font-weight: 650;
    white-space: nowrap;
}
.field-name {
    color: var(--accent);
    font-family: "Cascadia Code", "SFMono-Regular", monospace;
    font-size: .76rem;
}

.vs-wrap { display: flex; align-items: stretch; gap: 14px; margin: 10px 0; }
.team-card {
    flex: 1;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem 1.2rem;
    text-align: center;
}
.team-card.winner {
    border-color: rgba(183, 243, 74, .7);
    box-shadow: inset 0 0 0 1px rgba(183, 243, 74, .16);
    background: var(--accent-soft);
}
.team-flag { font-size: 52px; }
.team-name { font-size: 22px; font-weight: 700; color: var(--text); margin-top: 6px; }
.vs-badge {
    display: flex; align-items: center; justify-content: center;
    font-size: 20px; font-weight: 800; color: var(--accent);
    padding: 0 8px;
}

.result-banner {
    margin: 1rem 0;
    padding: 1rem 1.2rem;
    color: var(--accent);
    background: var(--accent-soft);
    border: 1px solid rgba(183, 243, 74, .42);
    border-radius: 12px;
    text-align: center;
    font-size: 1.05rem;
    font-weight: 720;
}

[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid var(--border);
    border-radius: 10px;
}
[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--surface);
    border-color: var(--border) !important;
    border-radius: 14px;
    box-shadow: 0 16px 38px rgba(0, 0, 0, .13);
}
[data-testid="stDataFrame"] * { font-size: .8rem; }

.stPlotlyChart {
    overflow: hidden;
    border-radius: 10px;
}
[data-testid="stImage"] img {
    background: #f7faf8;
    border: 1px solid var(--border);
    border-radius: 10px;
}
[data-testid="stImageCaption"] {
    color: var(--muted);
    font-size: .75rem;
}

[data-baseweb="tab-list"] {
    gap: .4rem;
    border-bottom: 1px solid var(--border);
}
[data-baseweb="tab"] {
    padding: .7rem 1rem;
    color: var(--muted);
    font-size: .84rem;
}
[aria-selected="true"][data-baseweb="tab"] {
    color: var(--text);
    border-bottom-color: var(--accent);
}

.stSelectbox label, .stRadio label, .stFileUploader label {
    color: var(--muted) !important;
    font-size: .8rem !important;
    font-weight: 560 !important;
}

[data-baseweb="select"] > div,
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface-2);
    border-color: var(--border-strong);
    color: var(--text);
}

.stButton>button {
    min-height: 2.75rem;
    background: var(--accent);
    color: #13200d;
    border: 1px solid var(--accent);
    border-radius: 9px;
    font-weight: 750;
    transition: transform .15s ease, box-shadow .15s ease;
}
.stButton>button:hover {
    color: #13200d;
    border-color: #cbff68;
    transform: translateY(-1px);
    box-shadow: 0 8px 20px rgba(183, 243, 74, .15);
}

.stDownloadButton>button {
    background: transparent;
    color: var(--accent);
    border: 1px solid rgba(183, 243, 74, .45);
    border-radius: 9px;
}

[data-testid="stAlert"] {
    background: var(--surface-2);
    border: 1px solid var(--border);
    color: var(--text);
}

[data-testid="stSidebar"] {
    min-width: 250px;
    background: var(--sidebar);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1rem;
}
.sidebar-brand {
    padding: .7rem .65rem 1.45rem;
    border-bottom: 1px solid var(--border);
    margin-bottom: .8rem;
}
.brand-mark {
    display: flex;
    align-items: center;
    gap: .7rem;
}
.brand-ball {
    width: 34px;
    height: 34px;
    display: grid;
    place-items: center;
    border: 1px solid var(--accent);
    border-radius: 50%;
    color: var(--accent);
    font-size: 16px;
}
.brand-name {
    color: var(--text);
    font-size: .98rem;
    font-weight: 720;
}
.brand-sub {
    margin: .25rem 0 0 2.8rem;
    color: var(--muted);
    font-size: .67rem;
}

hr { border-color: var(--border) !important; }

@media (max-width: 900px) {
    .block-container { padding: 1.4rem 1rem 3rem; }
    .hero-title { font-size: 1.8rem; }
    .metric-card { min-height: 108px; padding: .9rem .75rem; }
    .metric-value { font-size: 1.45rem; }
    .vs-wrap { flex-direction: column; }
    .vs-badge { min-height: 36px; }
    .match-table { min-width: 680px; }
    .table-scroll { overflow-x: auto; }
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Plotly 统一草地配色主题
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#b9c6c1", family="Inter, Microsoft YaHei, sans-serif", size=12),
    margin=dict(l=28, r=20, t=24, b=36),
    xaxis=dict(gridcolor="rgba(145,163,157,0.12)", zerolinecolor="rgba(145,163,157,0.16)"),
    yaxis=dict(gridcolor="rgba(145,163,157,0.12)", zerolinecolor="rgba(145,163,157,0.16)"),
    hoverlabel=dict(bgcolor="#182824", bordercolor="#344842", font_color="#f4f7f5"),
)
RESULT_COLORS = {"队伍A胜": "#b7f34a", "平局": "#4fc7c9", "队伍B胜": "#f7b84b"}


# ------------------------------------------------------------------
# 数据加载(缓存)
# ------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_clean():
    return pd.read_csv(CLEAN_CSV, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_features():
    return pd.read_csv(FEATURE_CSV, parse_dates=["date"])


@st.cache_data(show_spinner=False)
def load_team_state():
    with open(TEAM_STATE_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_eval():
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_resource(show_spinner=False)
def load_models():
    best = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    all_models = joblib.load(os.path.join(MODEL_DIR, "all_models.pkl"))
    feat_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))
    return best, scaler, all_models, feat_cols


def models_ready():
    """检查模型与数据产物是否已生成。"""
    need = [CLEAN_CSV, FEATURE_CSV, TEAM_STATE_JSON, EVAL_JSON,
            os.path.join(MODEL_DIR, "best_model.pkl")]
    return all(os.path.exists(p) for p in need)


# 国旗 emoji(常见球队,缺省用足球)
FLAG = {
    "Brazil": "🇧🇷", "Argentina": "🇦🇷", "France": "🇫🇷", "Germany": "🇩🇪",
    "Spain": "🇪🇸", "Italy": "🇮🇹", "England": "🏴", "Portugal": "🇵🇹",
    "Netherlands": "🇳🇱", "Belgium": "🇧🇪", "Croatia": "🇭🇷", "Uruguay": "🇺🇾",
    "Mexico": "🇲🇽", "United States": "🇺🇸", "Japan": "🇯🇵", "South Korea": "🇰🇷",
    "Morocco": "🇲🇦", "Poland": "🇵🇱", "Switzerland": "🇨🇭", "Denmark": "🇩🇰",
    "Sweden": "🇸🇪", "Russia": "🇷🇺", "Serbia": "🇷🇸", "Nigeria": "🇳🇬",
    "Ghana": "🇬🇭", "Senegal": "🇸🇳", "Cameroon": "🇨🇲", "Australia": "🇦🇺",
    "Colombia": "🇨🇴", "Chile": "🇨🇱", "Peru": "🇵🇪", "Ecuador": "🇪🇨",
    "Turkey": "🇹🇷", "Greece": "🇬🇷", "Austria": "🇦🇹", "Czech Republic": "🇨🇿",
    "Saudi Arabia": "🇸🇦", "Iran": "🇮🇷", "Egypt": "🇪🇬", "Canada": "🇨🇦",
}


def flag_of(team):
    return FLAG.get(team, "⚽")


def render_page_header(title, subtitle):
    st.markdown(
        f"<div class='hero'><div class='hero-title'>{html.escape(title)}</div>"
        f"<div class='hero-sub'>{html.escape(subtitle)}</div></div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------------------------
# 预测核心:为任意两队构造特征向量
# ------------------------------------------------------------------
def build_feature_vector(team_state, h2h, end_year, team_a, team_b,
                         neutral=1, is_final=1):
    """
    根据各队最新状态,为 team_a(队伍A) vs team_b(队伍B) 构造特征向量。
    用于单场/批量预测。所有数据来自训练数据末尾的累计状态。
    """
    default = {"elo": 1500.0, "recent_winrate": 0.5, "recent_goal_diff": 0.0,
               "avg_goals": 1.0, "wc_exp": 0}
    sa = team_state.get(team_a, default)
    sb = team_state.get(team_b, default)

    a_h2h = h2h.get(team_a, {}).get(team_b, 0)
    b_h2h = h2h.get(team_b, {}).get(team_a, 0)

    vec = {
        "home_elo": sa["elo"],
        "away_elo": sb["elo"],
        "elo_diff": sa["elo"] - sb["elo"],
        "elo_abs_diff": sa["elo"] + sb["elo"],
        "home_recent_winrate": sa["recent_winrate"],
        "away_recent_winrate": sb["recent_winrate"],
        "home_recent_goal_diff": sa["recent_goal_diff"],
        "away_recent_goal_diff": sb["recent_goal_diff"],
        "home_avg_goals": sa["avg_goals"],
        "away_avg_goals": sb["avg_goals"],
        "h2h_diff": a_h2h - b_h2h,
        "home_wc_exp": sa["wc_exp"],
        "away_wc_exp": sb["wc_exp"],
        "neutral": int(neutral),
        "is_world_cup_final": int(is_final),
        "match_year": int(end_year),
    }
    return vec


def predict_match(model, scaler, feat_cols, vec):
    """返回 (预测标签中文, 三类概率 dict)。"""
    X = np.array([[vec[c] for c in feat_cols]], dtype=float)
    Xs = scaler.transform(X)
    pred = model.predict(Xs)[0]

    proba = {}
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(Xs)[0]
        classes = list(model.classes_)
        for lab in LABELS:
            if lab in classes:
                proba[LABEL_NAMES[lab]] = float(p[classes.index(lab)])
            else:
                proba[LABEL_NAMES[lab]] = 0.0
    else:
        for lab in LABELS:
            proba[LABEL_NAMES[lab]] = 1.0 if lab == pred else 0.0

    return LABEL_NAMES.get(pred, str(pred)), proba


# ==================================================================
# 侧边栏导航
# ==================================================================
with st.sidebar:
    st.markdown(
        "<div class='sidebar-brand'>"
        "<div class='brand-mark'>"
        "<div class='brand-ball'>◉</div>"
        "<div class='brand-name'>世界杯预测系统</div>"
        "</div>"
        "<div class='brand-sub'>WORLD CUP INTELLIGENCE</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    page = option_menu(
        menu_title=None,
        options=["数据概览", "可视化分析", "模型对比",
                 "单场预测", "批量预测", "分析报告"],
        icons=["clipboard-data", "bar-chart-line", "cpu",
               "trophy", "upload", "file-earmark-text"],
        default_index=0,
        styles={
            "container": {"background-color": "rgba(0,0,0,0)", "padding": "4px"},
            "icon": {"color": "#91a39d", "font-size": "16px"},
            "nav-link": {
                "color": "#aab8b3", "font-size": "14px",
                "text-align": "left", "margin": "4px 0",
                "border-radius": "8px",
                "padding": "11px 12px",
                "--hover-color": "#12201d",
            },
            "nav-link-selected": {
                "background-color": "#12201d",
                "color": "#f4f7f5", "font-weight": "700",
                "border-left": "3px solid #b7f34a",
            },
        },
    )

# 测试钩子:允许通过环境变量强制指定页面(仅用于自动化测试,不影响正常使用)
_test_page = os.environ.get("WC_TEST_PAGE")
if _test_page:
    page = _test_page

# 检查产物是否就绪
if not models_ready():
    st.markdown(
        "<div class='hero'><div class='hero-title'>⚠️ 数据与模型尚未生成</div>"
        "<div class='hero-sub'>请先在 code 目录依次运行以下脚本生成数据与模型:</div>"
        "</div>", unsafe_allow_html=True)
    st.code(
        "python data_prep.py\n"
        "python feature_engineering.py\n"
        "python train_models.py\n"
        "python evaluate.py",
        language="bash")
    st.stop()


# ==================================================================
# 页面 1:数据概览
# ==================================================================
def page_overview():
    df = load_clean()
    st.markdown(
        "<div class='hero'>"
        "<div class='hero-title'>世界杯比赛结果智能预测与分析系统</div>"
        "<div class='hero-sub'>基于历史比赛数据与 ELO 评分的三分类预测。"
        "系统覆盖数据探索、模型比较与赛前结果推断，所有动态特征均遵循赛前可得原则。</div>"
        "</div>", unsafe_allow_html=True)

    # 指标卡
    n = len(df)
    n_final = int(df["is_world_cup_final"].sum())
    n_qual = n - n_final
    yr_min, yr_max = df["date"].dt.year.min(), df["date"].dt.year.max()
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        (c1, "01", f"{n:,}", "世界杯相关比赛"),
        (c2, "02", f"{n_final:,}", "正赛场次"),
        (c3, "03", f"{n_qual:,}", "预选赛场次"),
        (c4, "04", f"{yr_min}–{yr_max}", "时间跨度"),
    ]
    for col, idx, val, lab in cards:
        col.markdown(
            f"<div class='metric-card'><div class='metric-index'>{idx}</div>"
            f"<div class='metric-value'>{val}</div>"
            f"<div class='metric-label'>{lab}</div></div>",
            unsafe_allow_html=True)

    st.markdown(
        "<div class='section-heading'><h3>数据概览</h3>"
        "<span>最近更新：训练数据末次比赛</span></div>",
        unsafe_allow_html=True)
    left, right = st.columns([1.48, 1])

    with left:
        with st.container(border=True):
            st.markdown("#### 最近比赛")
            show = df.sort_values("date", ascending=False).head(8)[
                ["date", "home_team", "away_team", "home_score", "away_score",
                 "tournament", "result"]].copy()
            rows = []
            for _, row in show.iterrows():
                result_name = LABEL_NAMES.get(row["result"], row["result"])
                rows.append(
                    "<tr>"
                    f"<td>{row['date'].strftime('%Y-%m-%d')}</td>"
                    f"<td>{html.escape(str(row['home_team']))}</td>"
                    f"<td class='score'>{int(row['home_score'])} - {int(row['away_score'])}</td>"
                    f"<td>{html.escape(str(row['away_team']))}</td>"
                    f"<td class='result'>{html.escape(result_name)}</td>"
                    "</tr>"
                )
            table_html = (
                "<div class='table-scroll'><table class='match-table'>"
                "<thead><tr><th>日期</th><th>队伍 A</th><th>比分</th>"
                "<th>队伍 B</th><th>结果</th></tr></thead>"
                f"<tbody>{''.join(rows)}</tbody></table></div>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

    with right:
        with st.container(border=True):
            st.markdown("#### 结果分布")
            cnt = df["result"].map(LABEL_NAMES).value_counts()
            fig = go.Figure(data=[go.Pie(
                labels=cnt.index.tolist(), values=cnt.values.tolist(),
                hole=0.64,
                marker=dict(colors=[RESULT_COLORS[k] for k in cnt.index]),
                sort=False,
                textinfo="percent", textfont=dict(size=13, color="#07110f"),
                hovertemplate="%{label}<br>%{value:,} 场 · %{percent}<extra></extra>")])
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=290,
                showlegend=True,
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.03,
                    xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#91a39d", size=11),
                ),
                annotations=[dict(
                    text=f"<b>{n:,}</b><br><span style='font-size:11px'>总场次</span>",
                    x=.5, y=.5, showarrow=False, font=dict(color="#f4f7f5", size=18),
                )],
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                "<p style='font-size:12px;color:#91a39d;line-height:1.6'>"
                "平局样本占比最低，因此模型评价同时关注 Macro-F1，"
                "避免准确率被多数类别主导。</p>",
                unsafe_allow_html=True)

    # 字段说明
    with st.container(border=True):
        st.markdown("#### 字段说明")
        fields = pd.DataFrame({
            "字段": ["date", "home_team / away_team", "home_score / away_score",
                    "tournament", "neutral", "is_world_cup_final", "result"],
            "含义": ["比赛日期", "对阵双方(队伍A / 队伍B,中立场不代表真实主客)",
                    "双方进球数", "赛事类型(正赛 / 预选赛)",
                    "是否中立场地", "是否世界杯正赛(1正赛/0预选赛)",
                    "三分类标签:队伍A胜 / 平局 / 队伍B胜"],
            "类型": ["日期", "类别", "数值", "类别", "布尔", "类别", "类别(目标)"],
        })
        st.dataframe(
            fields,
            use_container_width=True,
            hide_index=True,
            column_config={
                "字段": st.column_config.TextColumn("字段", width="medium"),
                "含义": st.column_config.TextColumn("含义", width="large"),
                "类型": st.column_config.TextColumn("类型", width="small"),
            },
        )


# ==================================================================
# 页面 2:可视化分析(展示 9 张核心静态图 + 文字解释)
# ==================================================================
def page_visual():
    render_page_header(
        "数据可视化分析",
        "九张核心图表从数据结构、足球规律和建模结果三个层面解释预测依据。",
    )

    groups = [
        ("一、数据理解类", [
            ("fig01_match_goal_trend.png", "世界杯比赛数量与进球趋势",
             "各届世界杯的比赛场次与总进球随时间变化,反映赛事规模扩张与进攻趋势。"),
            ("fig02_result_distribution.png", "比赛结果类别分布",
             "三类结果中主胜占比最高、平局最少,印证类别不平衡问题。"),
            ("fig03_goal_distribution.png", "进球数分布",
             "多数比赛进球集中在 0–3 球,符合足球低比分特征。"),
        ]),
        ("二、足球规律类", [
            ("fig04_strong_team_winrate.png", "传统强队历史胜率对比",
             "巴西、德国、阿根廷等传统强队胜率显著更高,说明实力差异真实存在。"),
            ("fig05_home_vs_neutral.png", "主场 vs 中立场结果差异",
             "非中立场时队伍A(主场)胜率更高,说明主场优势是有效信号。"),
            ("fig06_elo_distribution.png", "ELO 评分分布",
             "ELO 近似正态分布,强队集中于高分区,验证 ELO 作为实力刻画的合理性。"),
        ]),
        ("三、建模分析类", [
            ("fig07_corr_heatmap.png", "特征相关性热力图",
             "ELO 相关特征与结果相关性最强,特征间无严重冗余。"),
            ("fig08_model_compare.png", "多模型准确率 / Macro-F1 对比",
             "所有模型均显著超过多数类基准;准确率高的模型 Macro-F1 未必高。"),
            ("fig09_feature_importance.png", "最佳模型(随机森林)特征重要性",
             "ELO 差值与绝对实力是最重要特征,近期状态与历史交锋次之。"),
        ]),
    ]

    tabs = st.tabs(["数据理解", "足球规律", "建模分析"])
    for tab, (title, figs) in zip(tabs, groups):
        with tab:
            st.markdown(
                f"<div class='section-heading'><h3>{title}</h3>"
                "<span>点击图片可查看完整尺寸</span></div>",
                unsafe_allow_html=True,
            )
            cols = st.columns(len(figs))
            for col, (fname, cap, desc) in zip(cols, figs):
                path = os.path.join(FIG_DIR, fname)
                with col:
                    if os.path.exists(path):
                        st.image(path, use_container_width=True, caption=cap)
                    else:
                        st.warning(f"缺少图片:{fname}")
                    st.markdown(
                        f"<p style='font-size:12px;color:#91a39d;line-height:1.65'>{desc}</p>",
                        unsafe_allow_html=True)


# ==================================================================
# 页面 3:模型对比
# ==================================================================
def page_models():
    ev = load_eval()
    metrics = ev["metrics"]
    best_name = ev["best_model_name"]
    best_f1_name = ev["best_macro_f1_name"]

    render_page_header(
        "模型训练与对比",
        "两个基准模型、六个机器学习模型与 KNN 参数搜索，以准确率和 Macro-F1 综合评价。",
    )

    # 最佳模型横幅
    st.markdown(
        f"<div class='result-banner'>系统默认模型：{best_name} "
        f"(综合分 {metrics[best_name].get('composite','-')} · "
        f"准确率 {metrics[best_name]['accuracy']} · "
        f"Macro-F1 {metrics[best_name]['macro_f1']})</div>",
        unsafe_allow_html=True)
    st.markdown(
        f"<p style='text-align:center;color:#91a39d;font-size:13px'>Macro-F1 单项最高为 "
        f"<b>{best_f1_name}</b>({metrics[best_f1_name]['macro_f1']}),"
        f"但综合准确率与可解释性后,系统默认选用 {best_name}(可输出特征重要性,便于分析)。</p>",
        unsafe_allow_html=True)

    # 指标表
    st.markdown("<div class='glass'><h3>模型指标汇总</h3>",
                unsafe_allow_html=True)
    rows = []
    for name, m in metrics.items():
        rows.append({
            "模型": name,
            "准确率": m["accuracy"],
            "Macro-F1": m["macro_f1"],
            "综合分": m.get("composite", float("nan")),
            "类型": "基准" if name in ("多数类基准", "ELO规则基准") else "机器学习",
        })
    mdf = pd.DataFrame(rows)
    st.dataframe(mdf, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # 对比柱状图(Plotly 交互)
    st.markdown("<div class='glass'><h3>准确率与 Macro-F1 对比</h3>",
                unsafe_allow_html=True)
    ml = mdf.copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ml["模型"], y=ml["准确率"], name="准确率",
                         marker_color="#b7f34a",
                         text=ml["准确率"], textposition="outside"))
    fig.add_trace(go.Bar(x=ml["模型"], y=ml["Macro-F1"], name="Macro-F1",
                         marker_color="#4fc7c9",
                         text=ml["Macro-F1"], textposition="outside"))
    fig.add_hline(y=1/3, line_dash="dot", line_color="#ff7675",
                  annotation_text="随机基线 0.33", annotation_font_color="#ffb3b0")
    fig.update_layout(**PLOTLY_LAYOUT, height=440, barmode="group",
                      yaxis_title="指标值", xaxis_title="模型")
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # KNN 调参曲线
    if "knn_tuning" in ev:
        st.markdown("<div class='glass'><h3>KNN 参数调优曲线</h3>",
                    unsafe_allow_html=True)
        kt = ev["knn_tuning"]
        # knn_tuning 结构为 {k: {"accuracy":..,"macro_f1":..}},按 K 升序排列
        k_sorted = sorted(kt.keys(), key=lambda x: int(x))
        ks = [str(k) for k in k_sorted]
        accs = [kt[k]["accuracy"] for k in k_sorted]
        f1s = [kt[k]["macro_f1"] for k in k_sorted]
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=ks, y=accs, mode="lines+markers",
                                  name="准确率", line=dict(color="#b7f34a", width=3)))
        fig2.add_trace(go.Scatter(x=ks, y=f1s, mode="lines+markers",
                                  name="Macro-F1", line=dict(color="#4fc7c9", width=3)))
        fig2.update_layout(**PLOTLY_LAYOUT, height=360,
                           xaxis_title="K 值", yaxis_title="指标值")
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})
        st.markdown(
            f"<p style='color:#91a39d;font-size:13px'>最佳 K = <b>{ev.get('best_k','-')}</b>。"
            f"K 太小易受噪声影响,K 太大则趋于多数类,需折中选择。</p>"
            "</div>", unsafe_allow_html=True)

    # 混淆矩阵(下拉选择查看)
    st.markdown("<div class='glass'><h3>混淆矩阵</h3>", unsafe_allow_html=True)
    cm_files = {
        "随机森林": "fig_cm_随机森林.png", "逻辑回归": "fig_cm_逻辑回归.png",
        "KNN": "fig_cm_KNN.png", "决策树": "fig_cm_决策树.png",
        "朴素贝叶斯": "fig_cm_朴素贝叶斯.png", "SVM": "fig_cm_SVM.png",
        "ELO规则基准": "fig_cm_ELO规则基准.png", "多数类基准": "fig_cm_多数类基准.png",
    }
    sel = st.selectbox("选择要查看混淆矩阵的模型", list(cm_files.keys()))
    cm_path = os.path.join(FIG_DIR, cm_files[sel])
    if os.path.exists(cm_path):
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.image(cm_path, use_container_width=True,
                     caption=f"{sel} 在测试集上的混淆矩阵")
    else:
        st.warning(f"缺少混淆矩阵图:{cm_files[sel]}")
    st.markdown("</div>", unsafe_allow_html=True)


# ==================================================================
# 页面 4:单场预测
# ==================================================================
def page_predict():
    render_page_header(
        "单场比赛预测",
        "选择对阵双方、比赛环境和模型，查看胜平负概率及模型置信度。",
    )

    best, scaler, all_models, feat_cols = load_models()
    state_data = load_team_state()
    team_state = state_data["team_state"]
    h2h = state_data["h2h"]
    end_year = state_data["data_end_year"]
    teams = sorted(team_state.keys())

    # 默认给两支强队
    def_a = teams.index("Brazil") if "Brazil" in teams else 0
    def_b = teams.index("Argentina") if "Argentina" in teams else 1

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 2])
    with c1:
        team_a = st.selectbox("队伍 A", teams, index=def_a)
    with c2:
        st.markdown(
            "<div style='text-align:center;font-size:30px;margin-top:26px;"
            "color:#b7f34a;font-weight:900'>VS</div>", unsafe_allow_html=True)
    with c3:
        team_b = st.selectbox("队伍 B", teams, index=def_b)

    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        neutral = st.radio("比赛场地", ["中立场", "队伍A主场"],
                           horizontal=True) == "中立场"
    with cc2:
        is_final = st.radio("比赛阶段", ["世界杯正赛", "预选赛"],
                            horizontal=True) == "世界杯正赛"
    with cc3:
        model_name = st.selectbox(
            "预测模型", ["最佳模型(默认)"] + list(all_models.keys()))

    go_pred = st.button("开始预测", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not go_pred:
        return

    if team_a == team_b:
        st.error("请选择两支不同的球队。")
        return

    model = best if model_name.startswith("最佳") else all_models[model_name]
    vec = build_feature_vector(team_state, h2h, end_year, team_a, team_b,
                               neutral=int(neutral), is_final=int(is_final))
    pred_label, proba = predict_match(model, scaler, feat_cols, vec)

    # 对战卡
    win_a = pred_label == "队伍A胜"
    win_b = pred_label == "队伍B胜"
    st.markdown(
        f"""<div class='vs-wrap'>
        <div class='team-card {'winner' if win_a else ''}'>
            <div class='team-flag'>{flag_of(team_a)}</div>
            <div class='team-name'>{team_a}</div>
            <div style='color:#91a39d'>ELO {team_state.get(team_a,{}).get('elo','-')}</div>
        </div>
        <div class='vs-badge'>VS</div>
        <div class='team-card {'winner' if win_b else ''}'>
            <div class='team-flag'>{flag_of(team_b)}</div>
            <div class='team-name'>{team_b}</div>
            <div style='color:#91a39d'>ELO {team_state.get(team_b,{}).get('elo','-')}</div>
        </div></div>""",
        unsafe_allow_html=True)

    st.markdown(
        f"<div class='result-banner'>预测结果:{pred_label}</div>",
        unsafe_allow_html=True)

    # 概率仪表盘(三类条形 + 环形)
    colL, colR = st.columns([1.2, 1])
    with colL:
        st.markdown("<div class='glass'><h4>三种结果概率</h4>",
                    unsafe_allow_html=True)
        pdf = pd.DataFrame({
            "结果": list(proba.keys()),
            "概率": [round(v * 100, 1) for v in proba.values()],
        })
        fig = go.Figure(go.Bar(
            x=pdf["概率"], y=pdf["结果"], orientation="h",
            marker_color=[RESULT_COLORS[k] for k in pdf["结果"]],
            text=[f"{v}%" for v in pdf["概率"]], textposition="outside"))
        fig.update_layout(**PLOTLY_LAYOUT, height=260,
                          xaxis_title="概率 (%)", xaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with colR:
        st.markdown("<div class='glass'><h4>最可能结果置信度</h4>",
                    unsafe_allow_html=True)
        top_p = max(proba.values()) * 100
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=top_p,
            number={"suffix": "%", "font": {"color": "#b7f34a"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#f0f5f0"},
                "bar": {"color": "#b7f34a"},
                "bgcolor": "#12201d",
                "steps": [
                    {"range": [0, 40], "color": "rgba(231,76,60,0.3)"},
                    {"range": [40, 70], "color": "rgba(241,196,15,0.3)"},
                    {"range": [70, 100], "color": "rgba(46,204,113,0.3)"},
                ],
            }))
        gauge.update_layout(**PLOTLY_LAYOUT, height=260)
        st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.info("⚠️ 足球比赛存在较强偶然性,预测结果仅作辅助参考,不代表确定性判断。")


# ==================================================================
# 页面 5:批量预测
# ==================================================================
def page_batch():
    render_page_header(
        "批量预测",
        "使用标准 CSV 模板一次预测多场比赛，并导出包含概率与备注的结果文件。",
    )

    best, scaler, all_models, feat_cols = load_models()
    state_data = load_team_state()
    team_state = state_data["team_state"]
    h2h = state_data["h2h"]
    end_year = state_data["data_end_year"]

    # 模板下载
    template = pd.DataFrame({
        "home_team": ["Brazil", "France", "Argentina"],
        "away_team": ["Argentina", "Germany", "Spain"],
        "neutral": [1, 1, 0],
        "is_world_cup_final": [1, 1, 1],
    })
    st.markdown("<div class='glass'><h3>01 下载模板</h3>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#91a39d;font-size:13px'>CSV 需包含 <b>home_team</b>(队伍A)、"
        "<b>away_team</b>(队伍B)列;可选 <b>neutral</b>(1中立/0主场)、"
        "<b>is_world_cup_final</b>(1正赛/0预选赛)。</p>", unsafe_allow_html=True)
    csv_t = template.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("下载预测模板 CSV", csv_t, "predict_template.csv",
                       "text/csv")
    st.dataframe(template, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='glass'><h3>02 上传文件并预测</h3>",
                unsafe_allow_html=True)
    up = st.file_uploader("上传 CSV 文件", type=["csv"])
    if up is None:
        st.markdown("</div>", unsafe_allow_html=True)
        return

    try:
        bdf = pd.read_csv(up)
    except Exception as e:
        st.error(f"读取失败:{e}")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "home_team" not in bdf.columns or "away_team" not in bdf.columns:
        st.error("CSV 必须包含 home_team 和 away_team 两列。")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    if "neutral" not in bdf.columns:
        bdf["neutral"] = 1
    if "is_world_cup_final" not in bdf.columns:
        bdf["is_world_cup_final"] = 1

    results, probs_a, probs_d, probs_b, warns = [], [], [], [], []
    for _, r in bdf.iterrows():
        ta, tb = str(r["home_team"]), str(r["away_team"])
        miss = [t for t in (ta, tb) if t not in team_state]
        vec = build_feature_vector(team_state, h2h, end_year, ta, tb,
                                   neutral=int(r["neutral"]),
                                   is_final=int(r["is_world_cup_final"]))
        lab, proba = predict_match(best, scaler, feat_cols, vec)
        results.append(lab)
        probs_a.append(round(proba["队伍A胜"] * 100, 1))
        probs_d.append(round(proba["平局"] * 100, 1))
        probs_b.append(round(proba["队伍B胜"] * 100, 1))
        warns.append("未知球队按默认值处理" if miss else "")

    bdf_out = bdf.copy()
    bdf_out["预测结果"] = results
    bdf_out["队伍A胜%"] = probs_a
    bdf_out["平局%"] = probs_d
    bdf_out["队伍B胜%"] = probs_b
    bdf_out["备注"] = warns

    st.success(f"已完成 {len(bdf_out)} 场比赛预测")
    st.dataframe(bdf_out, use_container_width=True, hide_index=True)

    out_csv = bdf_out.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button("下载预测结果 CSV", out_csv,
                       "batch_predictions.csv", "text/csv")
    st.markdown("</div>", unsafe_allow_html=True)


# ==================================================================
# 页面 6:分析报告
# ==================================================================
def page_report():
    render_page_header(
        "自动分析报告",
        "依据真实数据统计和模型评价结果生成 Markdown 报告，默认无需网络即可运行。",
    )

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#91a39d;font-size:13px'>点击下方按钮,系统将基于真实的数据统计与"
        "模型评价结果,自动生成一份 Markdown 分析报告。</p>", unsafe_allow_html=True)
    gen = st.button("生成分析报告", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if not gen:
        return

    # 调用报告生成器
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from report_generator import generate_report
        md = generate_report()
    except Exception as e:
        st.error(f"报告生成失败:{e}")
        return

    st.markdown("<div class='glass'>", unsafe_allow_html=True)
    st.markdown(md)
    st.markdown("</div>", unsafe_allow_html=True)

    st.download_button("下载报告 (Markdown)", md.encode("utf-8"),
                       "世界杯预测分析报告.md", "text/markdown")


# ==================================================================
# 路由
# ==================================================================
PAGES = {
    "数据概览": page_overview,
    "可视化分析": page_visual,
    "模型对比": page_models,
    "单场预测": page_predict,
    "批量预测": page_batch,
    "分析报告": page_report,
}
PAGES[page]()

# 页脚
st.markdown(
    "<hr>"
    "<p style='text-align:center;color:#6f827b;font-size:11px'>"
    "世界杯比赛结果智能预测与分析系统 · 人工智能课程大作业 · "
    "数据来源 martj42/international_results (CC0)</p>",
    unsafe_allow_html=True)
