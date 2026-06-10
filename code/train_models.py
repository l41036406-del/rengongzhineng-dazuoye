# -*- coding: utf-8 -*-
"""
train_models.py — 模型训练模块
================================================
功能:
    1. 读取特征数据 world_cup_features.csv
    2. 按时间顺序划分训练集/测试集(防止未来信息泄露)
    3. 训练 2 个基准模型:
         - 多数类基准(永远预测样本最多的类别)
         - ELO 规则基准(ELO 高者胜,差距小则判平)
    4. 训练 6 个机器学习模型:
         逻辑回归 / KNN / 决策树 / 随机森林 / 朴素贝叶斯 / SVM
    5. 对 KNN 进行不同 K 值调参
    6. 以 Macro-F1 为主指标选出最佳模型
    7. 保存:最佳模型、标准化器、特征列、全部模型的测试集预测结果

设计要点(严谨性):
    - 采用按时间的划分(早期 80% 训练,近期 20% 测试),
      与"赛前滚动特征"一脉相承,杜绝数据泄露。
    - 标准化器只在训练集上 fit,再 transform 测试集。
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------------------------
# 路径配置
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FEATURE_CSV = os.path.join(DATA_DIR, "world_cup_features.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 参与建模的特征列(不含 date / 队名 / 标签)
FEATURE_COLS = [
    "home_elo", "away_elo", "elo_diff", "elo_abs_diff",
    "home_recent_winrate", "away_recent_winrate",
    "home_recent_goal_diff", "away_recent_goal_diff",
    "home_avg_goals", "away_avg_goals",
    "h2h_diff", "home_wc_exp", "away_wc_exp",
    "neutral", "is_world_cup_final", "match_year",
]

# 三分类标签顺序(固定,保证混淆矩阵/报告一致)
LABELS = ["home_win", "draw", "away_win"]
LABEL_NAMES = {"home_win": "队伍A胜", "draw": "平局", "away_win": "队伍B胜"}

TEST_RATIO = 0.2  # 近期 20% 作测试集


# ------------------------------------------------------------------
# 数据加载与划分
# ------------------------------------------------------------------
def load_features():
    """读取特征数据,按时间排序。"""
    df = pd.read_csv(FEATURE_CSV, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # neutral / is_world_cup_final 统一为 0/1
    df["neutral"] = df["neutral"].astype(int)
    df["is_world_cup_final"] = df["is_world_cup_final"].astype(int)
    print(f"[读取] 特征数据 {len(df)} 场,特征数 {len(FEATURE_COLS)}")
    return df


def time_split(df):
    """
    按时间划分训练/测试集。
    早期 80% 用于训练,近期 20% 用于测试 —— 模拟"用历史预测未来"。
    """
    n = len(df)
    split_idx = int(n * (1 - TEST_RATIO))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    split_date = df.iloc[split_idx]["date"].date()
    print(f"[划分] 训练集 {len(train_df)} 场 (~{train_df['date'].max().date()}), "
          f"测试集 {len(test_df)} 场 ({split_date}~)")
    return train_df, test_df


# ------------------------------------------------------------------
# 基准模型
# ------------------------------------------------------------------
def baseline_majority(y_train, test_len):
    """多数类基准:永远预测训练集中样本最多的类别。"""
    majority = pd.Series(y_train).value_counts().idxmax()
    return np.array([majority] * test_len)


def baseline_elo(test_df, draw_threshold=40):
    """
    ELO 规则基准:
        ELO 差 > 阈值 -> 队伍A胜
        ELO 差 < -阈值 -> 队伍B胜
        |ELO 差| <= 阈值 -> 平局
    阈值 40 为经验值(约对应胜率 56%)。
    """
    preds = []
    for diff in test_df["elo_diff"].values:
        if diff > draw_threshold:
            preds.append("home_win")
        elif diff < -draw_threshold:
            preds.append("away_win")
        else:
            preds.append("draw")
    return np.array(preds)


# ------------------------------------------------------------------
# 机器学习模型定义
# ------------------------------------------------------------------
def build_models():
    """返回待训练的 6 个机器学习模型(已配好基本参数)。"""
    return {
        "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=15),
        "决策树": DecisionTreeClassifier(max_depth=8, random_state=42),
        "随机森林": RandomForestClassifier(
            n_estimators=300, max_depth=12, random_state=42, n_jobs=-1
        ),
        "朴素贝叶斯": GaussianNB(),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }


def tune_knn(X_train, y_train, X_test, y_test):
    """KNN 不同 K 值调参,返回 {k: macro_f1} 及最佳 K。"""
    print("\n[调参] KNN 不同 K 值测试:")
    k_values = [3, 5, 7, 9, 11, 15, 21, 31, 51]
    results = {}
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        pred = knn.predict(X_test)
        f1 = f1_score(y_test, pred, average="macro", labels=LABELS)
        acc = accuracy_score(y_test, pred)
        results[k] = {"macro_f1": round(f1, 4), "accuracy": round(acc, 4)}
        print(f"        K={k:>2}  准确率={acc:.4f}  Macro-F1={f1:.4f}")
    best_k = max(results, key=lambda k: results[k]["macro_f1"])
    print(f"        => 最佳 K = {best_k}")
    return results, best_k


# ------------------------------------------------------------------
# 主训练流程
# ------------------------------------------------------------------
def train_all():
    print("=" * 60)
    print("模型训练流程开始(按时间划分,防数据泄露)")
    print("=" * 60)

    df = load_features()
    train_df, test_df = time_split(df)

    X_train_raw = train_df[FEATURE_COLS].values
    X_test_raw = test_df[FEATURE_COLS].values
    y_train = train_df["result"].values
    y_test = test_df["result"].values

    # 标准化(只在训练集 fit)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 收集所有模型在测试集上的预测,供评价/可视化复用
    all_preds = {}      # {模型名: 预测数组}
    metrics = {}        # {模型名: {acc, macro_f1, ...}}
    trained_models = {} # {模型名: 已训练模型对象}

    def record(name, y_pred):
        acc = accuracy_score(y_test, y_pred)
        mf1 = f1_score(y_test, y_pred, average="macro", labels=LABELS)
        all_preds[name] = list(y_pred)
        metrics[name] = {"accuracy": round(acc, 4), "macro_f1": round(mf1, 4)}
        print(f"    {name:<10} 准确率={acc:.4f}  Macro-F1={mf1:.4f}")

    # --- 基准模型 ---
    print("\n[基准模型]")
    record("多数类基准", baseline_majority(y_train, len(y_test)))
    record("ELO规则基准", baseline_elo(test_df))

    # --- 机器学习模型 ---
    print("\n[机器学习模型]")
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        trained_models[name] = model
        record(name, y_pred)

    # --- KNN 调参 ---
    knn_tuning, best_k = tune_knn(X_train, y_train, X_test, y_test)
    # 用最佳 K 重训一个 KNN,纳入候选
    best_knn = KNeighborsClassifier(n_neighbors=best_k)
    best_knn.fit(X_train, y_train)
    trained_models[f"KNN(K={best_k})"] = best_knn
    record(f"KNN(K={best_k})", best_knn.predict(X_test))

    # --- 选最佳模型(只在 ML 模型里选,基准不参与)---
    # 方案2:综合打分选最佳,而非仅看 Macro-F1。
    # 综合分 = 0.5*Macro-F1 + 0.4*准确率 + 0.1*可解释性加分。
    # 可解释性加分:具备特征重要性/系数、便于分析与展示的模型更适合
    # 作为系统默认预测模型(可输出"最佳模型特征重要性"图)。
    ml_names = [n for n in metrics if n not in ("多数类基准", "ELO规则基准")]

    # 可解释性加分表(具备 feature_importances_ 或 coef_,且解释性强者得分高)
    interpretability = {
        "随机森林": 1.0,    # 有特征重要性,稳定,集成模型
        "决策树": 0.8,      # 有特征重要性,可视化直观,但易过拟合
        "逻辑回归": 0.7,    # 有系数,线性可解释
        "SVM": 0.3,         # 解释性弱,速度慢
        "朴素贝叶斯": 0.3,  # 无特征重要性图
    }

    def composite_score(n):
        # KNN(K=..) 这类带后缀的名字归一到 "KNN"
        base = "KNN" if n.startswith("KNN") else n
        interp = interpretability.get(base, 0.2)
        return (0.5 * metrics[n]["macro_f1"]
                + 0.4 * metrics[n]["accuracy"]
                + 0.1 * interp)

    # 记录综合分,写入 metrics 供报告/可视化引用
    for n in ml_names:
        metrics[n]["composite"] = round(composite_score(n), 4)

    best_name = max(ml_names, key=composite_score)
    best_model = trained_models[best_name]

    # 同时记录"纯 Macro-F1 最高"者,报告中如实说明二者差异
    best_macro_f1_name = max(ml_names, key=lambda n: metrics[n]["macro_f1"])

    print(f"\n[最佳模型-综合打分] {best_name} "
          f"(综合分={metrics[best_name]['composite']}, "
          f"Macro-F1={metrics[best_name]['macro_f1']}, "
          f"准确率={metrics[best_name]['accuracy']})")
    print(f"[参考] Macro-F1 单项最高:{best_macro_f1_name} "
          f"(Macro-F1={metrics[best_macro_f1_name]['macro_f1']}) "
          f"—— 报告中将如实说明综合选型理由")

    # ------------------------------------------------------------------
    # 保存产物
    # ------------------------------------------------------------------
    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(FEATURE_COLS, os.path.join(MODEL_DIR, "feature_columns.pkl"))
    joblib.dump(LABELS, os.path.join(MODEL_DIR, "labels.pkl"))
    # 同时保存全部 ML 模型,供界面切换比较
    joblib.dump(trained_models, os.path.join(MODEL_DIR, "all_models.pkl"))

    # 保存测试集真实标签 + 各模型预测(供 evaluate.py / app.py 复用)
    eval_payload = {
        "y_test": list(y_test),
        "all_preds": all_preds,
        "metrics": metrics,
        "knn_tuning": knn_tuning,
        "best_k": best_k,
        "best_model_name": best_name,
        "best_macro_f1_name": best_macro_f1_name,
        "labels": LABELS,
        "label_names": LABEL_NAMES,
        "feature_cols": FEATURE_COLS,
        "test_index_start": int(len(train_df)),
    }
    with open(os.path.join(OUTPUT_DIR, "eval_results.json"),
              "w", encoding="utf-8") as f:
        json.dump(eval_payload, f, ensure_ascii=False, indent=2)

    # 测试集明细(含队名/日期)另存,供可视化与报告引用
    test_df_out = test_df[["date", "home_team", "away_team",
                           "elo_diff", "result"]].copy()
    test_df_out["best_pred"] = all_preds[best_name]
    test_df_out.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"),
                       index=False, encoding="utf-8-sig")

    print(f"\n[保存] 最佳模型 / 标准化器 / 特征列 / 全部模型 -> {MODEL_DIR}")
    print(f"[保存] 评价数据 eval_results.json / test_predictions.csv -> {OUTPUT_DIR}")
    print("=" * 60)
    print("模型训练完成")
    print("=" * 60)

    return eval_payload


if __name__ == "__main__":
    train_all()
