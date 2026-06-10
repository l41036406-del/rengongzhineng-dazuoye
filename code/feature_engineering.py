# -*- coding: utf-8 -*-
"""
feature_engineering.py — 赛前滚动特征工程模块(项目核心)
================================================================
本模块是整个项目最关键的部分,核心原则只有一条:

    ★ 严格的"赛前可获得信息"原则,杜绝数据泄露 ★

即:每一场比赛的所有动态特征(ELO、近期胜率、近期净胜球、历史交锋、
场均进球、世界杯经验),都只允许使用【该场比赛日期之前】已经发生的
历史数据计算得到,绝不能使用这场比赛本身或之后的任何信息。

实现方式:
    将全部比赛按时间从早到晚排序,逐场遍历。处理第 i 场比赛时:
      第一步:用"截至目前已积累的历史状态"计算出第 i 场的特征向量;
      第二步:等特征写好之后,再用第 i 场的真实比分去更新两队的状态
              (ELO、进球记录、出场次数等),供后面的比赛使用。
    "先取特征、后更新状态"的顺序,从机制上保证了不会用到未来信息。

构造的特征(共 14 个):
    elo_diff            赛前 ELO 评分差 (A - B)
    elo_abs_diff        双方 ELO 绝对实力之和(反映整体水平高低)
    home_elo            队伍A 赛前 ELO
    away_elo            队伍B 赛前 ELO
    home_recent_winrate 队伍A 近 N 场胜率
    away_recent_winrate 队伍B 近 N 场胜率
    home_recent_goal_diff 队伍A 近 N 场场均净胜球
    away_recent_goal_diff 队伍B 近 N 场场均净胜球
    home_avg_goals      队伍A 历史场均进球
    away_avg_goals      队伍B 历史场均进球
    h2h_diff            两队历史交锋净胜场 (A胜场 - B胜场)
    home_wc_exp         队伍A 此前世界杯相关出场次数
    away_wc_exp         队伍B 此前世界杯相关出场次数
    neutral             是否中立场(0/1)
    is_world_cup_final  是否世界杯正赛(0/1)
    match_year          比赛年份(反映足球整体环境变化)
"""

import os
import json
from collections import defaultdict, deque

import pandas as pd

from data_prep import prepare_data, CLEAN_CSV, DATA_DIR

# ------------------------------------------------------------------
# 配置参数
# ------------------------------------------------------------------
FEATURE_CSV = os.path.join(DATA_DIR, "world_cup_features.csv")
# 各队"最新状态"导出文件:供 Streamlit 预测页为任意两队即时构造特征向量
TEAM_STATE_JSON = os.path.join(DATA_DIR, "team_state.json")

RECENT_N = 5          # "近期"窗口:最近 5 场
ELO_INIT = 1500.0     # ELO 初始分
ELO_K = 30.0          # ELO K 因子(每场最大调整幅度)


# ------------------------------------------------------------------
# ELO 评分:足球实力量化的经典方法
# ------------------------------------------------------------------
def expected_score(rating_a, rating_b):
    """根据双方 ELO 计算 A 的期望胜率(0~1)。"""
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def update_elo(rating_a, rating_b, score_a):
    """
    根据一场比赛的真实结果更新双方 ELO。
    score_a: A 的实际得分,胜=1,平=0.5,负=0。
    返回更新后的 (rating_a, rating_b)。
    """
    exp_a = expected_score(rating_a, rating_b)
    exp_b = 1.0 - exp_a
    score_b = 1.0 - score_a
    new_a = rating_a + ELO_K * (score_a - exp_a)
    new_b = rating_b + ELO_K * (score_b - exp_b)
    return new_a, new_b


# ------------------------------------------------------------------
# 主流程:逐场滚动构造特征
# ------------------------------------------------------------------
def build_features(df=None, save=True):
    """
    输入清洗后的比赛数据(按时间排序),输出带特征的数据集。
    若不传入 df,则自动调用 data_prep 生成。
    """
    if df is None:
        if os.path.exists(CLEAN_CSV):
            df = pd.read_csv(CLEAN_CSV, parse_dates=["date"])
        else:
            df = prepare_data(save=True)

    # 双保险:确保严格按时间升序(防泄露的前提)
    df = df.sort_values("date").reset_index(drop=True)

    print("=" * 60)
    print("赛前滚动特征工程开始(严格遵循赛前可获得原则)")
    print("=" * 60)

    # ---- 各队的"当前状态",随时间滚动更新 ----
    elo = defaultdict(lambda: ELO_INIT)              # 每队当前 ELO
    recent_results = defaultdict(lambda: deque(maxlen=RECENT_N))  # 近 N 场:1胜0.5平0负
    recent_gd = defaultdict(lambda: deque(maxlen=RECENT_N))       # 近 N 场净胜球
    goals_for = defaultdict(float)                   # 历史累计进球
    games_played = defaultdict(int)                  # 历史累计场次
    wc_exp = defaultdict(int)                         # 世界杯相关出场次数
    h2h = defaultdict(lambda: defaultdict(int))      # 历史交锋: h2h[A][B] = A 对 B 的胜场数

    rows = []  # 收集每场比赛的特征

    for _, m in df.iterrows():
        a = m["home_team"]    # 队伍A(数据中的 home)
        b = m["away_team"]    # 队伍B(数据中的 away)

        # ========== 第一步:用"赛前"已有状态计算特征 ==========
        a_elo = elo[a]
        b_elo = elo[b]

        # 近期胜率(无历史则给中性值 0.5)
        a_wr = sum(recent_results[a]) / len(recent_results[a]) if recent_results[a] else 0.5
        b_wr = sum(recent_results[b]) / len(recent_results[b]) if recent_results[b] else 0.5

        # 近期场均净胜球(无历史则 0)
        a_gd = sum(recent_gd[a]) / len(recent_gd[a]) if recent_gd[a] else 0.0
        b_gd = sum(recent_gd[b]) / len(recent_gd[b]) if recent_gd[b] else 0.0

        # 历史场均进球(无历史则给一个温和默认值 1.0)
        a_avg = goals_for[a] / games_played[a] if games_played[a] > 0 else 1.0
        b_avg = goals_for[b] / games_played[b] if games_played[b] > 0 else 1.0

        # 历史交锋净胜场
        h2h_diff = h2h[a][b] - h2h[b][a]

        feat = {
            "date": m["date"],
            "home_team": a,
            "away_team": b,
            # --- 实力 ---
            "home_elo": round(a_elo, 2),
            "away_elo": round(b_elo, 2),
            "elo_diff": round(a_elo - b_elo, 2),
            "elo_abs_diff": round(a_elo + b_elo, 2),
            # --- 近期状态 ---
            "home_recent_winrate": round(a_wr, 4),
            "away_recent_winrate": round(b_wr, 4),
            "home_recent_goal_diff": round(a_gd, 4),
            "away_recent_goal_diff": round(b_gd, 4),
            # --- 攻击力 ---
            "home_avg_goals": round(a_avg, 4),
            "away_avg_goals": round(b_avg, 4),
            # --- 历史交锋 ---
            "h2h_diff": h2h_diff,
            # --- 经验 ---
            "home_wc_exp": wc_exp[a],
            "away_wc_exp": wc_exp[b],
            # --- 环境 ---
            "neutral": int(bool(m["neutral"])),
            "is_world_cup_final": int(m["is_world_cup_final"]),
            "match_year": m["date"].year,
            # --- 标签 ---
            "result": m["result"],
        }
        rows.append(feat)

        # ========== 第二步:用这场的真实结果更新状态(供后续比赛使用)==========
        hs, as_ = int(m["home_score"]), int(m["away_score"])

        # A 的实际得分(胜1/平0.5/负0)
        if hs > as_:
            score_a = 1.0
            h2h[a][b] += 1
        elif hs < as_:
            score_a = 0.0
            h2h[b][a] += 1
        else:
            score_a = 0.5

        # 更新 ELO
        elo[a], elo[b] = update_elo(a_elo, b_elo, score_a)

        # 更新近期战绩队列
        recent_results[a].append(score_a)
        recent_results[b].append(1.0 - score_a)
        recent_gd[a].append(hs - as_)
        recent_gd[b].append(as_ - hs)

        # 更新累计进球与场次
        goals_for[a] += hs
        goals_for[b] += as_
        games_played[a] += 1
        games_played[b] += 1

        # 更新世界杯出场次数
        wc_exp[a] += 1
        wc_exp[b] += 1

    feat_df = pd.DataFrame(rows)

    print(f"[特征] 共构造 {len(feat_df)} 场比赛 × {feat_df.shape[1] - 4} 个特征")
    print(f"[特征] 特征列表:")
    feature_cols = [c for c in feat_df.columns
                    if c not in ("date", "home_team", "away_team", "result")]
    for c in feature_cols:
        print(f"        - {c}")

    # ------------------------------------------------------------------
    # 导出各队"最新状态"——供 Streamlit 预测页为任意两队构造特征向量。
    # 注意:这是遍历完所有历史比赛后的最终状态,代表"当前(截至数据末尾)"
    # 每支球队的实力与近况,用于对【未来】假想对阵做预测,不存在泄露问题。
    # ------------------------------------------------------------------
    team_state = {}
    for team in elo.keys():
        rr = recent_results[team]
        rg = recent_gd[team]
        team_state[team] = {
            "elo": round(elo[team], 2),
            "recent_winrate": round(sum(rr) / len(rr), 4) if rr else 0.5,
            "recent_goal_diff": round(sum(rg) / len(rg), 4) if rg else 0.0,
            "avg_goals": round(goals_for[team] / games_played[team], 4)
                         if games_played[team] > 0 else 1.0,
            "wc_exp": wc_exp[team],
            "games_played": games_played[team],
        }

    if save:
        feat_df.to_csv(FEATURE_CSV, index=False, encoding="utf-8-sig")
        print(f"[保存] 特征数据已写入:{FEATURE_CSV}")

        # 各队最终状态 + 历史交锋表,存为 JSON 供界面预测使用
        import json
        state_path = os.path.join(DATA_DIR, "team_state.json")
        # h2h 转成普通 dict(只保留有交锋记录的)
        h2h_plain = {a: dict(opp) for a, opp in h2h.items() if opp}
        payload = {
            "team_state": team_state,
            "h2h": h2h_plain,
            "data_end_year": int(df["date"].max().year),
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[保存] 各队最新状态已写入:{state_path}")

    print("=" * 60)
    print("特征工程完成 —— 所有特征均仅使用赛前历史数据,无未来信息泄露")
    print("=" * 60)
    return feat_df


# 供其他模块引用:特征列名(不含标识列和标签)
FEATURE_COLUMNS = [
    "home_elo", "away_elo", "elo_diff", "elo_abs_diff",
    "home_recent_winrate", "away_recent_winrate",
    "home_recent_goal_diff", "away_recent_goal_diff",
    "home_avg_goals", "away_avg_goals",
    "h2h_diff",
    "home_wc_exp", "away_wc_exp",
    "neutral", "is_world_cup_final", "match_year",
]


if __name__ == "__main__":
    build_features()
