# -*- coding: utf-8 -*-
"""
evaluate.py — 模型评价与数据可视化模块
================================================
功能:
    1. 读取清洗数据、特征数据、训练阶段保存的评价结果(eval_results.json)
    2. 生成 9 张核心可视化图(分三组),全部保存到 outputs/figures/
        【数据理解类】
            fig01_match_goal_trend     世界杯比赛数量与进球趋势
            fig02_result_distribution  比赛结果类别分布
            fig03_goal_distribution    进球数分布(箱线图)
        【足球规律类】
            fig04_strong_team_winrate  强队历史胜率对比
            fig05_home_vs_neutral      主场 vs 中立场结果差异
            fig06_elo_distribution     ELO 评分分布
        【建模分析类】
            fig07_corr_heatmap         特征相关性热力图
            fig08_model_compare        多模型 准确率/Macro-F1 对比
            fig09_feature_importance   最佳模型特征重要性
    3. 额外生成各模型混淆矩阵(放模型对比页用),保存 fig_cm_*.png
    4. 输出一份指标汇总表 outputs/metrics_summary.csv

说明:
    所有图表均带标题、坐标轴说明,配色采用克制、色盲友好的科学出版风格。
    静态图用于项目报告存档;Streamlit 界面另用 Plotly 交互版本展示。
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.metrics import confusion_matrix

# ------------------------------------------------------------------
# 中文字体与全局样式
# ------------------------------------------------------------------
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["figure.dpi"] = 110

# 科学出版风格配色。以高辨识度、色盲友好和印刷可读性为优先。
SCI_BLUE = "#3C5488"
SCI_CYAN = "#4DBBD5"
SCI_TEAL = "#00A087"
SCI_RED = "#E64B35"
SCI_ORANGE = "#F39B7F"
SCI_PURPLE = "#8491B4"
SCI_GOLD = "#D6A53A"
PAPER = "#FFFFFF"
GRID = "#D9DEE7"
DARK_TEXT = "#263238"
RESULT_COLORS = {
    "home_win": SCI_TEAL,
    "draw": SCI_GOLD,
    "away_win": SCI_RED,
}

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CLEAN_CSV = os.path.join(DATA_DIR, "world_cup_clean.csv")
FEATURE_CSV = os.path.join(DATA_DIR, "world_cup_features.csv")
EVAL_JSON = os.path.join(OUTPUT_DIR, "eval_results.json")

LABELS = ["home_win", "draw", "away_win"]
LABEL_NAMES = {"home_win": "队伍A胜", "draw": "平局", "away_win": "队伍B胜"}


def _style_ax(ax, title, xlabel="", ylabel=""):
    """统一图表样式。"""
    ax.set_title(title, fontsize=14, fontweight="bold", color=DARK_TEXT, pad=12)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=DARK_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=DARK_TEXT)
    ax.set_facecolor(PAPER)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.65, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(colors="#4F5B62", labelsize=9)
    ax.spines["left"].set_color("#9AA5AD")
    ax.spines["bottom"].set_color("#9AA5AD")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _save(fig, name):
    """保存图片到 figures 目录。"""
    path = os.path.join(FIG_DIR, name)
    fig.patch.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"    [图] 已保存 {name}")


# ==================================================================
# 数据理解类
# ==================================================================
def fig01_match_goal_trend(clean):
    """世界杯比赛数量与进球趋势(按年份)。"""
    clean = clean.copy()
    clean["year"] = pd.to_datetime(clean["date"]).dt.year
    clean["total_goals"] = clean["home_score"] + clean["away_score"]
    grp = clean.groupby("year").agg(
        matches=("result", "count"),
        avg_goals=("total_goals", "mean"),
    ).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(grp["year"], grp["matches"], color=SCI_CYAN, alpha=0.72,
            label="比赛场数", width=0.8, edgecolor="white", linewidth=0.35)
    _style_ax(ax1, "世界杯比赛数量与场均进球趋势(1930–2026)",
              "年份", "比赛场数")

    ax2 = ax1.twinx()
    ax2.plot(grp["year"], grp["avg_goals"], color=SCI_RED,
             marker="o", markersize=3, linewidth=2, label="场均进球")
    ax2.set_ylabel("场均进球数", fontsize=11, color=SCI_RED)
    ax2.tick_params(axis="y", labelcolor=SCI_RED)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_color("#9AA5AD")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    _save(fig, "fig01_match_goal_trend.png")


def fig02_result_distribution(clean):
    """比赛结果类别分布。"""
    counts = clean["result"].value_counts().reindex(LABELS)
    names = [LABEL_NAMES[k] for k in LABELS]
    colors = [RESULT_COLORS[k] for k in LABELS]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, counts.values, color=colors, edgecolor="white")
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 30,
                f"{v}\n({v/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    _style_ax(ax, "世界杯比赛结果类别分布", "比赛结果", "场数")
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, "fig02_result_distribution.png")


def fig03_goal_distribution(clean):
    """进球数分布(箱线图):主队/客队/总进球。"""
    clean = clean.copy()
    clean["total_goals"] = clean["home_score"] + clean["away_score"]
    data = [clean["home_score"], clean["away_score"], clean["total_goals"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(data, tick_labels=["队伍A进球", "队伍B进球", "总进球"],
                    patch_artist=True, showmeans=True)
    for patch, c in zip(bp["boxes"], [SCI_TEAL, SCI_RED, SCI_BLUE]):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    for median in bp["medians"]:
        median.set_color(DARK_TEXT)
        median.set_linewidth(2)
    _style_ax(ax, "世界杯比赛进球数分布(箱线图)", "", "进球数")
    _save(fig, "fig03_goal_distribution.png")


# ==================================================================
# 足球规律类
# ==================================================================
def fig04_strong_team_winrate(clean):
    """出场最多的强队历史胜率对比。"""
    # 统计每支队的出场数与胜场(分主客)
    records = {}
    for _, r in clean.iterrows():
        h, a, res = r["home_team"], r["away_team"], r["result"]
        records.setdefault(h, [0, 0])
        records.setdefault(a, [0, 0])
        records[h][0] += 1
        records[a][0] += 1
        if res == "home_win":
            records[h][1] += 1
        elif res == "away_win":
            records[a][1] += 1
    stat = pd.DataFrame(
        [(t, n, w, w / n) for t, (n, w) in records.items()],
        columns=["team", "matches", "wins", "winrate"],
    )
    # 取出场数最多的前 12 支队
    top = stat.sort_values("matches", ascending=False).head(12)
    top = top.sort_values("winrate")

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.Blues(np.linspace(0.38, 0.88, len(top)))
    bars = ax.barh(top["team"], top["winrate"] * 100,
                   color=colors, edgecolor="white")
    bars[-1].set_color(SCI_RED)
    for b, v in zip(bars, top["winrate"] * 100):
        ax.text(v + 0.5, b.get_y() + b.get_height() / 2,
                f"{v:.1f}%", va="center", fontsize=10)
    _style_ax(ax, "出场最多的 12 支球队历史胜率对比", "胜率 (%)", "")
    ax.set_xlim(0, top["winrate"].max() * 100 * 1.15)
    _save(fig, "fig04_strong_team_winrate.png")


def fig05_home_vs_neutral(clean):
    """主场 vs 中立场结果差异。"""
    out = []
    for neutral_flag, label in [(False, "非中立场(有主场)"), (True, "中立场")]:
        sub = clean[clean["neutral"] == neutral_flag]
        if len(sub) == 0:
            continue
        dist = sub["result"].value_counts(normalize=True).reindex(LABELS).fillna(0)
        out.append((label, dist))

    x = np.arange(len(LABELS))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, (label, dist) in enumerate(out):
        ax.bar(x + (i - 0.5) * width, dist.values * 100, width,
               label=label,
               color=[SCI_BLUE, SCI_ORANGE][i], edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL_NAMES[k] for k in LABELS])
    _style_ax(ax, "主场 vs 中立场:比赛结果差异", "比赛结果", "占比 (%)")
    ax.legend()
    _save(fig, "fig05_home_vs_neutral.png")


def fig06_elo_distribution(feat):
    """ELO 评分分布(赛前)。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(feat["home_elo"], bins=40, color=SCI_CYAN, alpha=0.58,
            label="队伍A 赛前 ELO", edgecolor="white")
    ax.hist(feat["away_elo"], bins=40, color=SCI_RED, alpha=0.46,
            label="队伍B 赛前 ELO", edgecolor="white")
    ax.axvline(1500, color=DARK_TEXT, linestyle="--", linewidth=1.5,
               label="初始 ELO=1500")
    _style_ax(ax, "赛前 ELO 实力评分分布", "ELO 评分", "频数")
    ax.legend()
    _save(fig, "fig06_elo_distribution.png")


# ==================================================================
# 建模分析类
# ==================================================================
def fig07_corr_heatmap(feat, feature_cols):
    """特征相关性热力图。"""
    num = feat[feature_cols].select_dtypes(include=[np.number])
    corr = num.corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.columns, fontsize=9)
    # 标注相关系数
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="black" if abs(v) < 0.6 else "white", fontsize=7)
    ax.set_title("特征相关性热力图", fontsize=14, fontweight="bold",
                 color=DARK_TEXT, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="相关系数")
    _save(fig, "fig07_corr_heatmap.png")


def fig08_model_compare(metrics):
    """多模型 准确率 / Macro-F1 对比(含基准)。"""
    names = list(metrics.keys())
    acc = [metrics[n]["accuracy"] for n in names]
    f1 = [metrics[n]["macro_f1"] for n in names]

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(x - width / 2, acc, width, label="准确率 Accuracy",
                color=SCI_BLUE, edgecolor="white")
    b2 = ax.bar(x + width / 2, f1, width, label="Macro-F1",
                color=SCI_ORANGE, edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.3f}", ha="center", va="bottom",
                    fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    _style_ax(ax, "各模型性能对比:准确率 vs Macro-F1", "模型", "指标值")
    ax.set_ylim(0, max(max(acc), max(f1)) * 1.18)
    ax.legend()
    ax.axhline(1/3, color=SCI_RED, linestyle=":", linewidth=1,
               label="随机猜测基线")
    _save(fig, "fig08_model_compare.png")


def fig09_feature_importance(feature_cols):
    """最佳模型(随机森林)特征重要性。若最佳模型无该属性,则用随机森林补出。"""
    import joblib
    model_dir = os.path.join(BASE_DIR, "models")
    best = joblib.load(os.path.join(model_dir, "best_model.pkl"))

    if hasattr(best, "feature_importances_"):
        importances = best.feature_importances_
        title = "最佳模型(随机森林)特征重要性"
    else:
        # 兜底:从 all_models 里取随机森林
        all_models = joblib.load(os.path.join(model_dir, "all_models.pkl"))
        rf = all_models.get("随机森林")
        importances = rf.feature_importances_
        title = "随机森林特征重要性(用于特征解释)"

    order = np.argsort(importances)
    cols = np.array(feature_cols)[order]
    vals = importances[order]

    fig, ax = plt.subplots(figsize=(9, 7))
    colors = plt.cm.Blues(np.linspace(0.38, 0.86, len(vals)))
    bars = ax.barh(cols, vals, color=colors, edgecolor="white")
    bars[-1].set_color(SCI_RED)
    for b, v in zip(bars, vals):
        ax.text(v + 0.002, b.get_y() + b.get_height() / 2,
                f"{v:.3f}", va="center", fontsize=9)
    _style_ax(ax, title, "重要性", "")
    ax.set_xlim(0, vals.max() * 1.15)
    _save(fig, "fig09_feature_importance.png")


def fig_confusion_matrices(eval_data):
    """各模型混淆矩阵(供模型对比页)。"""
    y_test = eval_data["y_test"]
    all_preds = eval_data["all_preds"]
    label_names = [LABEL_NAMES[k] for k in LABELS]

    for name, preds in all_preds.items():
        cm = confusion_matrix(y_test, preds, labels=LABELS)
        fig, ax = plt.subplots(figsize=(5.5, 4.8))
        im = ax.imshow(cm, cmap="Greens")
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(label_names)
        ax.set_yticklabels(label_names)
        ax.set_xlabel("预测类别")
        ax.set_ylabel("真实类别")
        thresh = cm.max() / 2
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black",
                        fontsize=12, fontweight="bold")
        ax.set_title(f"混淆矩阵 — {name}", fontsize=13, fontweight="bold",
                     color=DARK_TEXT, pad=10)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        safe = name.replace("(", "_").replace(")", "").replace("=", "")
        _save(fig, f"fig_cm_{safe}.png")


def export_metrics_table(metrics):
    """导出指标汇总表。"""
    rows = []
    for name, m in metrics.items():
        rows.append({
            "模型": name,
            "准确率": m["accuracy"],
            "宏精确率": m.get("macro_precision", ""),
            "宏召回率": m.get("macro_recall", ""),
            "Macro-F1": m["macro_f1"],
            "综合分": m.get("composite", ""),
        })
    df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, "metrics_summary.csv")
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"    [表] 指标汇总已保存 metrics_summary.csv")
    return df


def run_all():
    print("=" * 60)
    print("评价与可视化流程开始")
    print("=" * 60)

    clean = pd.read_csv(CLEAN_CSV)
    feat = pd.read_csv(FEATURE_CSV)
    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        eval_data = json.load(f)
    metrics = eval_data["metrics"]
    feature_cols = eval_data["feature_cols"]

    print("\n[数据理解类]")
    fig01_match_goal_trend(clean)
    fig02_result_distribution(clean)
    fig03_goal_distribution(clean)

    print("\n[足球规律类]")
    fig04_strong_team_winrate(clean)
    fig05_home_vs_neutral(clean)
    fig06_elo_distribution(feat)

    print("\n[建模分析类]")
    fig07_corr_heatmap(feat, feature_cols)
    fig08_model_compare(metrics)
    fig09_feature_importance(feature_cols)

    print("\n[混淆矩阵]")
    fig_confusion_matrices(eval_data)

    print("\n[指标汇总表]")
    export_metrics_table(metrics)

    print("=" * 60)
    print(f"全部图表已生成 -> {FIG_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    run_all()
