# -*- coding: utf-8 -*-
"""
data_prep.py — 数据清洗模块
================================================
功能:
    1. 读取国际足球历史比赛原始数据 results.csv
    2. 精确筛选出 FIFA 世界杯相关比赛(正赛 + 预选赛)
    3. 基本清洗:去重、缺失值检查、类型转换、按时间排序
    4. 生成比赛结果三分类标签:队伍A胜 / 平局 / 队伍B胜
    5. 标注每场比赛是"正赛"还是"预选赛"
    6. 输出清洗后的数据 data/world_cup_clean.csv

说明:
    数据来源 martj42/international_results (CC0 开源许可)。
    home_team / away_team 仅表示数据记录中的双方位置,
    在中立场(neutral=True)比赛中并不代表真实主客场。
"""

import os
import pandas as pd

# ------------------------------------------------------------------
# 路径配置(相对本文件定位,保证在任意目录运行都能找到数据)
# ------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_CSV = os.path.join(DATA_DIR, "results.csv")
CLEAN_CSV = os.path.join(DATA_DIR, "world_cup_clean.csv")


def load_raw_data(path=RAW_CSV):
    """读取原始比赛数据。"""
    df = pd.read_csv(path)
    print(f"[读取] 原始数据共 {len(df)} 场比赛,字段:{list(df.columns)}")
    return df


def filter_world_cup(df):
    """
    精确筛选 FIFA 世界杯比赛。
    注意:数据中还存在 CONIFA World Cup、Viva World Cup 等非 FIFA 赛事,
    必须用精确匹配排除,只保留 'FIFA World Cup' 和 'FIFA World Cup qualification'。
    """
    mask = df["tournament"].isin(["FIFA World Cup", "FIFA World Cup qualification"])
    wc = df[mask].copy()
    print(f"[筛选] FIFA 世界杯相关比赛共 {len(wc)} 场")
    print(wc["tournament"].value_counts().to_string())
    return wc


def basic_clean(df):
    """基本清洗:类型转换、缺失值处理、去重、排序。"""
    # 日期转 datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 比分转数值
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    # 缺失值检查
    n_before = len(df)
    miss = df[["date", "home_score", "away_score"]].isna().sum()
    print(f"[缺失值] 关键字段缺失情况:\n{miss.to_string()}")

    # 丢弃关键字段缺失的行(比分缺失说明比赛未进行/未记录,无法用于训练)
    df = df.dropna(subset=["date", "home_score", "away_score"]).copy()

    # neutral 字段统一为布尔值
    if df["neutral"].dtype == object:
        df["neutral"] = df["neutral"].astype(str).str.upper().map(
            {"TRUE": True, "FALSE": False}
        ).fillna(False)
    df["neutral"] = df["neutral"].astype(bool)

    # 去重(完全相同的记录)
    df = df.drop_duplicates()

    # 比分转整数
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # 按时间排序——这是后续"赛前滚动特征"防数据泄露的基础
    df = df.sort_values("date").reset_index(drop=True)

    print(f"[清洗] 清洗后剩余 {len(df)} 场(删除 {n_before - len(df)} 场无效记录)")
    return df


def add_result_label(df):
    """
    生成三分类标签 result:
        'home_win'  -> 队伍A(home)胜
        'draw'      -> 平局
        'away_win'  -> 队伍B(away)胜
    """
    def label(row):
        if row["home_score"] > row["away_score"]:
            return "home_win"
        elif row["home_score"] < row["away_score"]:
            return "away_win"
        else:
            return "draw"

    df["result"] = df.apply(label, axis=1)
    print("[标签] 三分类结果分布:")
    print(df["result"].value_counts().to_string())
    print("[标签] 各类别占比:")
    print((df["result"].value_counts(normalize=True) * 100).round(2).to_string())
    return df


def add_stage_flag(df):
    """
    标注比赛阶段:
        is_world_cup_final = 1  -> 世界杯正赛
        is_world_cup_final = 0  -> 世界杯预选赛
    用于区分两类比赛分布差异,并作为模型特征之一。
    """
    df["is_world_cup_final"] = (df["tournament"] == "FIFA World Cup").astype(int)
    n_final = int(df["is_world_cup_final"].sum())
    print(f"[阶段] 正赛 {n_final} 场 / 预选赛 {len(df) - n_final} 场")
    return df


def prepare_data(save=True):
    """主流程:读取 -> 筛选 -> 清洗 -> 加标签 -> 加阶段标志 -> 保存。"""
    print("=" * 60)
    print("数据清洗流程开始")
    print("=" * 60)

    df = load_raw_data()
    df = filter_world_cup(df)
    df = basic_clean(df)
    df = add_result_label(df)
    df = add_stage_flag(df)

    # 只保留后续需要的列
    keep_cols = [
        "date", "home_team", "away_team", "home_score", "away_score",
        "tournament", "city", "country", "neutral",
        "result", "is_world_cup_final",
    ]
    df = df[keep_cols]

    if save:
        df.to_csv(CLEAN_CSV, index=False, encoding="utf-8-sig")
        print(f"[保存] 清洗后数据已写入:{CLEAN_CSV}")

    print("=" * 60)
    print(f"数据清洗完成,最终 {len(df)} 场比赛,{df.shape[1]} 个字段")
    print(f"时间跨度:{df['date'].min().date()} ~ {df['date'].max().date()}")
    print("=" * 60)
    return df


if __name__ == "__main__":
    prepare_data()
