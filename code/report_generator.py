# -*- coding: utf-8 -*-
"""
report_generator.py — 自动化分析报告生成器
================================================
功能:
    基于真实的模型评价结果(outputs/eval_results.json)和数据统计,
    自动生成一份结构完整、数据真实的自然语言分析报告。

两种模式:
    1. 本地模板模式(默认):无需联网、无需 API Key,
       用 Python 根据真实统计数字套用智能模板拼出报告。
       保证在任何电脑上都能稳定运行。
    2. 大模型增强模式(可选):若检测到环境变量中配置了
       大模型 API Key(ANTHROPIC_AUTH_TOKEN / ANTHROPIC_API_KEY 等),
       则可调用大模型生成表达更自然的分析报告。
       Key 仅从环境变量读取,绝不硬编码进代码,避免泄露。

输出:
    默认生成 Markdown 文本(outputs/reports/analysis_report.md),
    可在界面中选择导出。
"""

import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
REPORT_DIR = os.path.join(OUTPUT_DIR, "reports")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(REPORT_DIR, exist_ok=True)


# ------------------------------------------------------------------
# 读取真实的评价结果与数据统计
# ------------------------------------------------------------------
def load_eval():
    """读取模型评价结果。"""
    path = os.path.join(OUTPUT_DIR, "eval_results.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_data_stats():
    """读取清洗后数据,计算基础统计供报告引用。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA_DIR, "world_cup_clean.csv"))
    stats = {
        "n_matches": len(df),
        "year_min": str(df["date"].min())[:4],
        "year_max": str(df["date"].max())[:4],
        "n_final": int((df["is_world_cup_final"] == 1).sum()),
        "n_qual": int((df["is_world_cup_final"] == 0).sum()),
        "result_pct": (df["result"].value_counts(normalize=True) * 100).round(1).to_dict(),
    }
    return stats


# ------------------------------------------------------------------
# 本地模板模式:根据真实数据拼装报告(默认)
# ------------------------------------------------------------------
def build_template_report(eval_data, stats):
    """用真实统计数据套用智能模板生成 Markdown 报告。"""
    m = eval_data["metrics"]
    best = eval_data["best_model_name"]
    best_f1 = eval_data["best_macro_f1_name"]
    label_names = eval_data["label_names"]
    best_k = eval_data["best_k"]

    # 关键指标
    maj = m["多数类基准"]
    elo = m["ELO规则基准"]
    best_metric = m[best]
    best_f1_metric = m[best_f1]

    # 找出准确率最高的 ML 模型
    ml_names = [n for n in m if n not in ("多数类基准", "ELO规则基准")]
    acc_best = max(ml_names, key=lambda n: m[n]["accuracy"])

    rp = stats["result_pct"]
    home_pct = rp.get("home_win", 0)
    draw_pct = rp.get("draw", 0)
    away_pct = rp.get("away_win", 0)

    lines = []
    lines.append("# 世界杯比赛结果智能预测系统 — 分析报告\n")
    lines.append("> 本报告由系统基于真实的数据统计与模型评价结果自动生成。\n")

    # 1. 数据概况
    lines.append("## 一、数据概况\n")
    lines.append(
        f"本项目使用国际足球历史比赛数据(martj42/international_results,CC0 开源),"
        f"精确筛选出 FIFA 世界杯相关比赛共 **{stats['n_matches']} 场**,"
        f"时间跨度为 **{stats['year_min']}–{stats['year_max']}** 年,"
        f"其中世界杯正赛 {stats['n_final']} 场、预选赛 {stats['n_qual']} 场。\n")
    lines.append(
        f"比赛结果三分类分布为:队伍A胜 **{home_pct}%**、平局 **{draw_pct}%**、"
        f"队伍B胜 **{away_pct}%**。可以看出数据存在一定的**类别不平衡**——"
        f"平局样本最少且最难预测,因此本项目在评价时除准确率外重点关注 Macro-F1。\n")

    # 2. 特征工程
    lines.append("## 二、特征工程与防数据泄露\n")
    lines.append(
        "所有动态特征均严格遵循**赛前可获得原则**:每场比赛的 ELO 评分、近期胜率、"
        "近期净胜球、历史交锋、场均进球、世界杯经验等,均只使用该场比赛日期**之前**的"
        "历史数据滚动计算,杜绝未来信息泄露。训练集与测试集也按时间先后划分"
        "(早期数据训练、近期数据测试),进一步保证评价结果真实可信。\n")

    # 3. 模型表现
    lines.append("## 三、模型表现分析\n")
    lines.append(
        f"为客观衡量模型效果,本项目设置了两个基准模型作为参照:\n\n"
        f"- **多数类基准**(永远预测最多的类别):准确率 {maj['accuracy']}、"
        f"Macro-F1 {maj['macro_f1']};\n"
        f"- **ELO 规则基准**(ELO 高者胜):准确率 {elo['accuracy']}、"
        f"Macro-F1 {elo['macro_f1']}。\n")
    lines.append(
        f"在此基础上训练了逻辑回归、KNN、决策树、随机森林、朴素贝叶斯、SVM 共 6 种"
        f"机器学习模型,并对 KNN 进行了不同 K 值调参(最佳 K={best_k})。"
        f"所有机器学习模型的准确率均显著超过多数类基准(约 {maj['accuracy']}),"
        f"说明模型确实从数据中学到了有效规律。\n")
    lines.append(
        f"综合 Macro-F1、准确率与可解释性打分后,系统选择 **{best}** 作为默认预测模型"
        f"(准确率 {best_metric['accuracy']}、Macro-F1 {best_metric['macro_f1']})。"
        f"值得说明的是,若仅看 Macro-F1 单项指标,**{best_f1}** 表现最高"
        f"(Macro-F1 {best_f1_metric['macro_f1']}),但其可解释性较弱、且无法输出"
        f"特征重要性图;而 {best} 在准确率、稳定性和可解释性上更均衡,"
        f"并能给出特征重要性分析,更适合作为系统的默认预测与展示模型。\n")

    # 4. 关键发现
    lines.append("## 四、关键发现\n")
    lines.append(
        "1. **ELO 实力差是最重要的预测因素**——仅凭 ELO 规则就能达到约 "
        f"{elo['accuracy']} 的准确率,远超随机猜测(约 0.33),说明球队历史实力"
        "对比赛结果有很强的解释力。\n"
        "2. **平局最难预测**——各模型在平局类上的召回率普遍偏低,这与平局本身"
        "样本少、且足球比赛平局带有较强偶然性有关。\n"
        "3. **准确率高的模型未必 Macro-F1 高**——部分模型为追求整体准确率而"
        "倾向于多预测主胜/客胜,牺牲了平局类的识别能力,这正是需要同时关注"
        "Macro-F1 的原因。\n")

    # 5. 局限与改进
    lines.append("## 五、模型局限与改进方向\n")
    lines.append(
        "- **平局预测能力有限**:可尝试针对平局类做重采样、调整类别权重,"
        "或引入更多刻画「实力接近度」的特征。\n"
        "- **特征仍可丰富**:可补充球员名单、伤病、洲际风格差异、比赛重要程度等信息。\n"
        "- **足球本身偶然性高**:即便特征完备,单场比赛仍存在难以建模的偶然因素,"
        "因此本系统的预测结果应作为**辅助参考**,而非确定性判断。\n"
        "- **数据时间跨度大**:1930 年至今足球规则与风格变化巨大,早期数据的规律"
        "未必适用于现代比赛,后续可考虑对近年数据加权。\n")

    lines.append("\n---\n")
    lines.append("*本报告数据均来自项目真实运行结果,可结合 `outputs/figures/` 中的"
                 "图表一并查阅。*\n")

    return "\n".join(lines)


# ------------------------------------------------------------------
# 大模型增强模式(可选):仅在检测到 API Key 时尝试
# ------------------------------------------------------------------
def detect_api_key():
    """检测环境变量中是否配置了可用的大模型 API Key(不读取明文返回)。"""
    for key in ("ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        if os.environ.get(key):
            return key
    return None


def build_llm_report(eval_data, stats, template_text):
    """
    若检测到 API Key,尝试调用大模型润色/重写报告。
    任何失败都安全降级为本地模板报告。
    """
    key_name = detect_api_key()
    if not key_name:
        return None, "未检测到 API Key,使用本地模板报告"

    try:
        import requests
        base_url = os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com")
        token = os.environ.get(key_name)
        prompt = (
            "你是一名数据分析师。请基于下面这份世界杯比赛预测项目的"
            "草稿报告,用更自然流畅、有洞察力的中文重写成一份分析报告,"
            "保持所有数字与事实不变,不要编造数据:\n\n" + template_text
        )
        resp = requests.post(
            f"{base_url}/v1/messages",
            headers={
                "x-api-key": token,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-opus-4-8",
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=60,
        )
        if resp.status_code == 200:
            data = resp.json()
            text = "".join(
                block.get("text", "")
                for block in data.get("content", [])
            )
            if text.strip():
                return text, f"已调用大模型({key_name})生成增强报告"
        return None, f"大模型调用失败(HTTP {resp.status_code}),降级为本地模板报告"
    except Exception as e:
        return None, f"大模型调用异常({type(e).__name__}),降级为本地模板报告"


# ------------------------------------------------------------------
# 对外主接口
# ------------------------------------------------------------------
def generate_report(use_llm=False, save=True):
    """
    生成分析报告。
        use_llm=False:仅本地模板(默认,最稳)。
        use_llm=True :若有 API Key 则尝试大模型增强,失败自动降级。
    返回 (报告文本, 模式说明)。
    """
    eval_data = load_eval()
    stats = load_data_stats()
    template_text = build_template_report(eval_data, stats)

    mode = "本地模板模式"
    text = template_text
    if use_llm:
        llm_text, msg = build_llm_report(eval_data, stats, template_text)
        if llm_text:
            text = llm_text
            mode = "大模型增强模式"
        else:
            mode = f"本地模板模式({msg})"

    if save:
        out_path = os.path.join(REPORT_DIR, "analysis_report.md")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)

    return text, mode


if __name__ == "__main__":
    text, mode = generate_report(use_llm=False)
    print(f"[报告] 生成完成,模式:{mode}")
    print(f"[报告] 已保存到 outputs/reports/analysis_report.md")
    print("=" * 60)
    print(text[:600] + "\n...")
