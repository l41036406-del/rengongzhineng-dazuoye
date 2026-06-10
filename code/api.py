from __future__ import annotations

import json
import os
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
WEB_DIST = BASE_DIR / "web" / "dist"

LABELS = ["home_win", "draw", "away_win"]
LABEL_NAMES = {"home_win": "队伍A胜", "draw": "平局", "away_win": "队伍B胜"}


class MatchRequest(BaseModel):
    home_team: str = Field(min_length=1)
    away_team: str = Field(min_length=1)
    neutral: bool = True
    is_world_cup_final: bool = True
    model_name: str | None = None
    date: str | None = None
    tournament: str | None = None


class BatchRequest(BaseModel):
    matches: list[MatchRequest] = Field(min_length=1, max_length=500)


app = FastAPI(
    title="World Cup Intelligence API",
    version="1.0.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if FIGURE_DIR.exists():
    app.mount("/figures", StaticFiles(directory=FIGURE_DIR), name="figures")


@lru_cache
def clean_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "world_cup_clean.csv", parse_dates=["date"])


@lru_cache
def feature_data() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "world_cup_features.csv", parse_dates=["date"])


@lru_cache
def team_data() -> dict:
    with (DATA_DIR / "team_state.json").open(encoding="utf-8") as file:
        return json.load(file)


@lru_cache
def evaluation_data() -> dict:
    with (OUTPUT_DIR / "eval_results.json").open(encoding="utf-8") as file:
        return json.load(file)


@lru_cache
def model_bundle():
    return {
        "best": joblib.load(MODEL_DIR / "best_model.pkl"),
        "scaler": joblib.load(MODEL_DIR / "scaler.pkl"),
        "models": joblib.load(MODEL_DIR / "all_models.pkl"),
        "features": joblib.load(MODEL_DIR / "feature_columns.pkl"),
    }


def build_feature_vector(request: MatchRequest) -> dict[str, float]:
    state = team_data()
    teams = state["team_state"]
    h2h = state["h2h"]
    year = state["data_end_year"]
    default = {
        "elo": 1500.0,
        "recent_winrate": 0.5,
        "recent_goal_diff": 0.0,
        "avg_goals": 1.0,
        "wc_exp": 0,
    }
    home = teams.get(request.home_team, default)
    away = teams.get(request.away_team, default)
    home_h2h = h2h.get(request.home_team, {}).get(request.away_team, 0)
    away_h2h = h2h.get(request.away_team, {}).get(request.home_team, 0)
    return {
        "home_elo": home["elo"],
        "away_elo": away["elo"],
        "elo_diff": home["elo"] - away["elo"],
        "elo_abs_diff": home["elo"] + away["elo"],
        "home_recent_winrate": home["recent_winrate"],
        "away_recent_winrate": away["recent_winrate"],
        "home_recent_goal_diff": home["recent_goal_diff"],
        "away_recent_goal_diff": away["recent_goal_diff"],
        "home_avg_goals": home["avg_goals"],
        "away_avg_goals": away["avg_goals"],
        "h2h_diff": home_h2h - away_h2h,
        "home_wc_exp": home["wc_exp"],
        "away_wc_exp": away["wc_exp"],
        "neutral": int(request.neutral),
        "is_world_cup_final": int(request.is_world_cup_final),
        "match_year": int(year),
    }


def run_prediction(request: MatchRequest) -> dict:
    if request.home_team == request.away_team:
        raise HTTPException(status_code=400, detail="请选择两支不同的球队")

    bundle = model_bundle()
    model = bundle["best"]
    if request.model_name:
        model = bundle["models"].get(request.model_name)
        if model is None:
            raise HTTPException(status_code=400, detail="未找到指定模型")

    vector = build_feature_vector(request)
    features = bundle["features"]
    values = np.array([[vector[column] for column in features]], dtype=float)
    scaled = bundle["scaler"].transform(values)
    prediction = model.predict(scaled)[0]
    probabilities = {LABEL_NAMES[label]: 0.0 for label in LABELS}
    if hasattr(model, "predict_proba"):
        predicted = model.predict_proba(scaled)[0]
        classes = list(model.classes_)
        for label in LABELS:
            if label in classes:
                probabilities[LABEL_NAMES[label]] = float(predicted[classes.index(label)])
    else:
        probabilities[LABEL_NAMES[prediction]] = 1.0

    states = team_data()["team_state"]
    warnings = [
        team for team in (request.home_team, request.away_team) if team not in states
    ]
    return {
        "home_team": request.home_team,
        "away_team": request.away_team,
        "date": request.date,
        "tournament": request.tournament,
        "neutral": request.neutral,
        "is_world_cup_final": request.is_world_cup_final,
        "prediction": LABEL_NAMES.get(prediction, prediction),
        "probabilities": probabilities,
        "home_elo": states.get(request.home_team, {}).get("elo", 1500),
        "away_elo": states.get(request.away_team, {}).get("elo", 1500),
        "model": request.model_name or evaluation_data()["best_model_name"],
        "warnings": warnings,
    }


def team_history_summary(team: str) -> dict:
    data = clean_data()
    matches = data[(data["home_team"] == team) | (data["away_team"] == team)].copy()
    if matches.empty:
        return {
            "matches": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "win_rate": 0.0,
            "goals_for": 0.0,
            "goals_against": 0.0,
        }

    matches = matches.sort_values("date", ascending=False).head(10)
    is_home = matches["home_team"] == team
    goals_for = np.where(is_home, matches["home_score"], matches["away_score"])
    goals_against = np.where(is_home, matches["away_score"], matches["home_score"])
    wins = int((goals_for > goals_against).sum())
    draws = int((goals_for == goals_against).sum())
    losses = int((goals_for < goals_against).sum())
    count = len(matches)
    return {
        "matches": count,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": round(wins / count, 4),
        "goals_for": round(float(goals_for.mean()), 2),
        "goals_against": round(float(goals_against.mean()), 2),
    }


def team_recommendations(team: dict, fixtures: list[dict]) -> list[str]:
    recommendations = []
    strong_opponents = [item for item in fixtures if item["opponent_elo"] - team["elo"] >= 80]
    close_matches = [item for item in fixtures if item["confidence"] < 0.5]
    weaker_opponents = [item for item in fixtures if team["elo"] - item["opponent_elo"] >= 100]

    if team["recent_goal_diff"] < 0:
        recommendations.append(
            f"近期净胜球为 {team['recent_goal_diff']:.1f}，防线承压明显；面对高位压迫时应减少后场横传，并优先保护禁区弧顶。"
        )
    elif team["recent_goal_diff"] >= 1:
        recommendations.append(
            f"近期场均净胜球达到 {team['recent_goal_diff']:.1f}，进攻状态突出；建议延续主动压迫，在领先后降低无效对攻。"
        )

    if strong_opponents:
        names = "、".join(item["opponent"] for item in strong_opponents[:3])
        recommendations.append(
            f"对阵 {names} 时处于明显实力劣势，建议采用更紧凑的中低位防守，把进攻重点放在定位球和快速反击。"
        )
    if weaker_opponents:
        names = "、".join(item["opponent"] for item in weaker_opponents[:3])
        recommendations.append(
            f"对阵 {names} 时拥有较大 ELO 优势，应提高前场压迫强度，但需预留防守宽度以防对手反击。"
        )
    if close_matches:
        names = "、".join(item["opponent"] for item in close_matches[:3])
        recommendations.append(
            f"与 {names} 的预测分布较接近，临场阵容和场地因素可能改变结果，建议重点准备替补调整与最后 30 分钟策略。"
        )
    if team["recent_winrate"] < 40:
        recommendations.append(
            f"近期胜率仅 {team['recent_winrate']:.0f}%，应优先提升比赛稳定性，避免在均势阶段过早投入过多进攻人数。"
        )
    elif team["recent_winrate"] >= 70:
        recommendations.append(
            f"近期胜率为 {team['recent_winrate']:.0f}%，状态处于高位；轮换应保持中轴线连续性，避免破坏现有攻防节奏。"
        )
    return recommendations[:4]


def build_dynamic_report(matches: list[MatchRequest]) -> dict:
    predictions = [run_prediction(match) for match in matches]
    states = team_data()["team_state"]
    team_names = sorted(
        {name for item in predictions for name in (item["home_team"], item["away_team"])}
    )
    team_reports = []

    for name in team_names:
        state = states.get(
            name,
            {
                "elo": 1500.0,
                "recent_winrate": 0.5,
                "recent_goal_diff": 0.0,
                "avg_goals": 1.0,
                "wc_exp": 0,
                "games_played": 0,
            },
        )
        fixtures = []
        expected_wins = expected_draws = expected_losses = expected_points = 0.0
        for index, item in enumerate(predictions, start=1):
            if name not in (item["home_team"], item["away_team"]):
                continue
            is_home = item["home_team"] == name
            win_probability = item["probabilities"]["队伍A胜" if is_home else "队伍B胜"]
            draw_probability = item["probabilities"]["平局"]
            loss_probability = item["probabilities"]["队伍B胜" if is_home else "队伍A胜"]
            opponent = item["away_team"] if is_home else item["home_team"]
            opponent_elo = float(states.get(opponent, {}).get("elo", 1500))
            confidence = max(win_probability, draw_probability, loss_probability)
            expected_wins += win_probability
            expected_draws += draw_probability
            expected_losses += loss_probability
            expected_points += 3 * win_probability + draw_probability
            fixtures.append(
                {
                    "match": index,
                    "date": item["date"],
                    "opponent": opponent,
                    "side": "队伍A" if is_home else "队伍B",
                    "opponent_elo": round(opponent_elo, 1),
                    "win": round(win_probability * 100, 1),
                    "draw": round(draw_probability * 100, 1),
                    "loss": round(loss_probability * 100, 1),
                    "confidence": round(confidence, 4),
                    "prediction": (
                        "胜"
                        if item["prediction"] == ("队伍A胜" if is_home else "队伍B胜")
                        else "负"
                        if item["prediction"] == ("队伍B胜" if is_home else "队伍A胜")
                        else "平"
                    ),
                }
            )

        history = team_history_summary(name)
        report = {
            "team": name,
            "elo": round(float(state.get("elo", 1500)), 1),
            "recent_winrate": round(float(state.get("recent_winrate", 0.5)) * 100, 1),
            "recent_goal_diff": round(float(state.get("recent_goal_diff", 0)), 2),
            "avg_goals": round(float(state.get("avg_goals", 1)), 2),
            "wc_exp": int(state.get("wc_exp", 0)),
            "fixtures_count": len(fixtures),
            "expected_wins": round(expected_wins, 2),
            "expected_draws": round(expected_draws, 2),
            "expected_losses": round(expected_losses, 2),
            "expected_points": round(expected_points, 2),
            "points_per_match": round(expected_points / len(fixtures), 2),
            "history": history,
            "fixtures": fixtures,
        }
        report["recommendations"] = team_recommendations(report, fixtures)
        team_reports.append(report)

    outcome_counts = {"队伍A胜": 0, "平局": 0, "队伍B胜": 0}
    for item in predictions:
        outcome_counts[item["prediction"]] += 1
        probabilities = item["probabilities"]
        ordered = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)
        item["confidence"] = round(ordered[0][1] * 100, 1)
        item["probability_gap"] = round((ordered[0][1] - ordered[1][1]) * 100, 1)
        item["elo_gap"] = round(float(item["home_elo"]) - float(item["away_elo"]), 1)

    average_confidence = float(np.mean([item["confidence"] for item in predictions]))
    uncertain_matches = sum(item["confidence"] < 50 for item in predictions)
    generated_at = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")
    summary = {
        "matches": len(predictions),
        "teams": len(team_names),
        "average_confidence": round(average_confidence, 1),
        "uncertain_matches": uncertain_matches,
        "strongest_team": max(team_reports, key=lambda item: item["elo"])["team"],
        "best_schedule_outlook": max(
            team_reports, key=lambda item: item["points_per_match"]
        )["team"],
    }

    markdown_lines = [
        "# 上传赛程实时分析报告",
        "",
        f"- 生成时间：{generated_at}",
        f"- 分析比赛：{summary['matches']} 场",
        f"- 涉及队伍：{summary['teams']} 支",
        f"- 平均预测置信度：{summary['average_confidence']}%",
        "",
        "## 总体判断",
        "",
        f"本批数据中，ELO 最高的队伍是 **{summary['strongest_team']}**；按每场预期积分计算，赛程前景最佳的是 **{summary['best_schedule_outlook']}**。"
        f"共有 **{summary['uncertain_matches']}** 场比赛的最高结果概率低于 50%，这些对局更容易受到首发、伤停和临场战术影响。",
        "",
        "## 队伍分析",
    ]
    for team in sorted(team_reports, key=lambda item: item["expected_points"], reverse=True):
        markdown_lines.extend(
            [
                "",
                f"### {team['team']}",
                "",
                f"- ELO：{team['elo']}；近期胜率：{team['recent_winrate']}%；近期净胜球：{team['recent_goal_diff']}",
                f"- {team['fixtures_count']} 场比赛预期积分：{team['expected_points']}，场均 {team['points_per_match']}",
                f"- 预期胜/平/负：{team['expected_wins']} / {team['expected_draws']} / {team['expected_losses']}",
                "",
            ]
        )
        markdown_lines.extend(f"- 建议：{text}" for text in team["recommendations"])

    return {
        "generated_at": generated_at,
        "summary": summary,
        "outcome_distribution": [
            {"name": name, "value": value} for name, value in outcome_counts.items()
        ],
        "matches": predictions,
        "teams": sorted(team_reports, key=lambda item: item["expected_points"], reverse=True),
        "markdown": "\n".join(markdown_lines),
    }


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/overview")
def overview():
    data = clean_data()
    result_counts = (
        data["result"].map(LABEL_NAMES).value_counts().rename_axis("name").reset_index(name="value")
    )
    recent = data.sort_values("date", ascending=False).head(10).copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent["result"] = recent["result"].map(LABEL_NAMES)
    records = recent[
        ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "result"]
    ].to_dict(orient="records")
    return {
        "metrics": {
            "matches": len(data),
            "finals": int(data["is_world_cup_final"].sum()),
            "qualifiers": int((1 - data["is_world_cup_final"]).sum()),
            "year_min": int(data["date"].dt.year.min()),
            "year_max": int(data["date"].dt.year.max()),
        },
        "recent_matches": records,
        "result_distribution": result_counts.to_dict(orient="records"),
        "fields": [
            {"field": "date", "meaning": "比赛日期", "type": "日期"},
            {"field": "home_team / away_team", "meaning": "记录中的对阵双方", "type": "类别"},
            {"field": "home_score / away_score", "meaning": "双方进球数", "type": "数值"},
            {"field": "tournament", "meaning": "世界杯正赛或预选赛", "type": "类别"},
            {"field": "neutral", "meaning": "是否为中立场地", "type": "布尔"},
            {"field": "result", "meaning": "队伍A胜 / 平局 / 队伍B胜", "type": "目标标签"},
        ],
    }


@app.get("/api/models")
def models():
    evaluation = evaluation_data()
    metrics = [
        {"name": name, **values}
        for name, values in evaluation["metrics"].items()
    ]
    tuning = [
        {"k": int(k), **values}
        for k, values in evaluation.get("knn_tuning", {}).items()
    ]
    return {
        "best_model": evaluation["best_model_name"],
        "best_macro_f1_model": evaluation["best_macro_f1_name"],
        "best_k": evaluation.get("best_k"),
        "metrics": metrics,
        "knn_tuning": sorted(tuning, key=lambda item: item["k"]),
        "model_names": list(model_bundle()["models"].keys()),
    }


@app.get("/api/teams")
def teams():
    states = team_data()["team_state"]
    return {
        "teams": [
            {"name": name, **values}
            for name, values in sorted(states.items())
        ]
    }


@app.post("/api/predict")
def predict(request: MatchRequest):
    return run_prediction(request)


@app.post("/api/batch")
def batch_predict(request: BatchRequest):
    return {"results": [run_prediction(match) for match in request.matches]}


@app.get("/api/report")
def report():
    path = OUTPUT_DIR / "reports" / "analysis_report.md"
    if not path.exists():
        raise HTTPException(status_code=404, detail="分析报告尚未生成")
    return {"markdown": path.read_text(encoding="utf-8")}


@app.post("/api/report/analyze")
def analyze_report(request: BatchRequest):
    return build_dynamic_report(request.matches)


@app.get("/api/visualizations")
def visualizations():
    clean = clean_data().copy()
    features = feature_data().copy()
    evaluation = evaluation_data()

    clean["year"] = clean["date"].dt.year
    clean["total_goals"] = clean["home_score"] + clean["away_score"]
    trend = (
        clean.groupby("year")
        .agg(matches=("result", "size"), avg_goals=("total_goals", "mean"))
        .reset_index()
    )

    result_distribution = (
        clean["result"].value_counts().reindex(LABELS).fillna(0).astype(int)
    )

    goal_rows = []
    for goals in range(0, 9):
        goal_rows.append(
            {
                "goals": str(goals),
                "队伍A": int((clean["home_score"] == goals).sum()),
                "队伍B": int((clean["away_score"] == goals).sum()),
                "总进球": int((clean["total_goals"] == goals).sum()),
            }
        )
    goal_rows.append(
        {
            "goals": "9+",
            "队伍A": int((clean["home_score"] >= 9).sum()),
            "队伍B": int((clean["away_score"] >= 9).sum()),
            "总进球": int((clean["total_goals"] >= 9).sum()),
        }
    )

    team_records: dict[str, dict[str, int]] = {}
    for row in clean[["home_team", "away_team", "result"]].itertuples(index=False):
        team_records.setdefault(row.home_team, {"matches": 0, "wins": 0})
        team_records.setdefault(row.away_team, {"matches": 0, "wins": 0})
        team_records[row.home_team]["matches"] += 1
        team_records[row.away_team]["matches"] += 1
        if row.result == "home_win":
            team_records[row.home_team]["wins"] += 1
        elif row.result == "away_win":
            team_records[row.away_team]["wins"] += 1
    strong_teams = sorted(
        (
            {
                "team": team,
                "matches": values["matches"],
                "win_rate": round(values["wins"] / values["matches"] * 100, 1),
            }
            for team, values in team_records.items()
        ),
        key=lambda item: item["matches"],
        reverse=True,
    )[:12]
    strong_teams.sort(key=lambda item: item["win_rate"])

    venue_rows = []
    for neutral, label in ((False, "非中立场"), (True, "中立场")):
        subset = clean[clean["neutral"].astype(bool) == neutral]
        distribution = subset["result"].value_counts(normalize=True).reindex(LABELS).fillna(0)
        venue_rows.append(
            {
                "venue": label,
                **{
                    LABEL_NAMES[result]: round(float(distribution[result]) * 100, 1)
                    for result in LABELS
                },
            }
        )

    elo_edges = np.arange(1000, 2001, 50)
    home_counts, _ = np.histogram(features["home_elo"], bins=elo_edges)
    away_counts, _ = np.histogram(features["away_elo"], bins=elo_edges)
    elo_distribution = [
        {
            "elo": f"{int(elo_edges[index])}",
            "队伍A": int(home_counts[index]),
            "队伍B": int(away_counts[index]),
        }
        for index in range(len(home_counts))
    ]

    corr_columns = [
        "elo_diff",
        "home_recent_winrate",
        "away_recent_winrate",
        "home_recent_goal_diff",
        "away_recent_goal_diff",
        "home_avg_goals",
        "away_avg_goals",
        "h2h_diff",
        "neutral",
        "is_world_cup_final",
    ]
    corr = features[corr_columns].corr().round(2)
    correlations = [
        {"x": x, "y": y, "value": float(corr.loc[y, x])}
        for y in corr_columns
        for x in corr_columns
    ]

    model_metrics = [
        {
            "model": name,
            "accuracy": round(values["accuracy"] * 100, 2),
            "macro_f1": round(values["macro_f1"] * 100, 2),
        }
        for name, values in evaluation["metrics"].items()
    ]

    bundle = model_bundle()
    importance_model = bundle["best"]
    if not hasattr(importance_model, "feature_importances_"):
        importance_model = bundle["models"].get("随机森林")
    importances = getattr(importance_model, "feature_importances_", np.zeros(len(bundle["features"])))
    feature_importance = sorted(
        (
            {"feature": feature, "importance": round(float(value) * 100, 2)}
            for feature, value in zip(bundle["features"], importances)
        ),
        key=lambda item: item["importance"],
    )

    return {
        "trend": trend.round({"avg_goals": 2}).to_dict(orient="records"),
        "result_distribution": [
            {"name": LABEL_NAMES[name], "value": int(result_distribution[name])}
            for name in LABELS
        ],
        "goal_distribution": goal_rows,
        "strong_teams": strong_teams,
        "venue_comparison": venue_rows,
        "elo_distribution": elo_distribution,
        "correlation": {"columns": corr_columns, "values": correlations},
        "model_metrics": model_metrics,
        "feature_importance": feature_importance,
    }


if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="web-assets")

    @app.get("/{path:path}", include_in_schema=False)
    def serve_spa(path: str):
        requested = WEB_DIST / path
        if path and requested.is_file():
            return FileResponse(requested)
        return FileResponse(WEB_DIST / "index.html")
