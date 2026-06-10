====================================================================
        世界杯比赛结果智能预测与分析系统
        World Cup Result Predictor
====================================================================


一、项目名称
--------------------------------------------------------------------
    世界杯比赛结果智能预测与分析系统


二、项目简介
--------------------------------------------------------------------
    本项目基于国际足球历史比赛数据,聚焦 FIFA 世界杯(正赛 + 预选赛),
    将"预测一场比赛结果"建模为三分类问题:

        队伍A胜  /  平局  /  队伍B胜

    项目完整覆盖机器学习应用的全流程:
        数据清洗 -> 赛前滚动特征工程 -> 探索性可视化 ->
        多模型训练与对比 -> 指标评价 -> 交互式预测 -> 自动分析报告

    最大亮点:所有动态特征(ELO、近期状态、历史交锋等)均严格遵循
    "赛前可获得信息"原则逐场滚动计算,从机制上杜绝数据泄露。


三、运行环境
--------------------------------------------------------------------
    操作系统    : Windows 10 / 11(其它系统亦可)
    Python      : 3.10 及以上(开发使用 3.14)
    依赖库      : 见 requirements.txt
                  (pandas / numpy / scikit-learn / matplotlib /
                   seaborn / streamlit / plotly / streamlit-option-menu /
                   streamlit-lottie / joblib / requests)


四、运行方式（推荐 React 主站）
--------------------------------------------------------------------
    【方式一】最简单:双击 run.bat
        会自动安装依赖、构建 React 主站并启动 FastAPI。
        浏览器访问 http://127.0.0.1:8000

    【方式二】命令行手动运行
        1) 安装 Python 依赖:
               pip install -r requirements.txt

        2) 安装并构建前端:
               cd web
               npm install
               npm run build
               cd ..

        3) 启动网站:
               python -m uvicorn api:app --app-dir code --host 127.0.0.1 --port 8000

        4) (可选)若想从零复现数据与模型,在 code 目录依次运行:
               python data_prep.py            # 数据清洗
               python feature_engineering.py  # 赛前滚动特征工程
               python train_models.py         # 训练模型
               python evaluate.py             # 生成图表与评价
               python report_generator.py     # 生成分析报告(可选)

           注:项目已附带训练好的模型与产物,可跳过本步直接启动。

    【保留的 Streamlit 分析版】
               streamlit run code/app.py


五、网页应用功能(React 主站共 6 个页面)
--------------------------------------------------------------------
    1. 数据概览    : 数据规模、字段说明、结果类别分布
    2. 可视化分析  : 9 张核心图表(数据理解 / 足球规律 / 建模分析)
    3. 模型对比    : 8 模型 + 2 基准的指标汇总、KNN 调参曲线、混淆矩阵
    4. 单场预测    : 下拉选择两支球队,输出胜平负预测与概率(卡片式展示)
    5. 批量预测    : 上传 CSV 批量预测,可下载结果(附模板下载)
    6. 分析报告    : 一键生成自然语言分析报告,支持下载


六、主要文件说明
--------------------------------------------------------------------
    code/
        data_prep.py            数据清洗:筛选世界杯比赛、生成三分类标签
        feature_engineering.py  赛前滚动特征工程(项目核心,防数据泄露)
        train_models.py         训练 2 基准 + 6 机器学习模型 + KNN 调参
        evaluate.py             生成 9 张核心图 + 混淆矩阵 + 指标汇总
        report_generator.py     基于真实结果生成分析报告(本地模板,可选大模型)
        app.py                  Streamlit 网页主程序(绿茵主题 + 交互预测)

    data/
        results.csv             原始数据(martj42/international_results,CC0)
        world_cup_clean.csv     清洗后的世界杯比赛数据
        world_cup_features.csv  含 16 个赛前特征的建模数据
        team_state.json         各队最新状态(供预测页构造特征)

    models/
        best_model.pkl          系统默认预测模型(随机森林)
        all_models.pkl          全部训练好的模型(供对比)
        scaler.pkl              特征标准化器
        feature_columns.pkl     特征列名
        labels.pkl              标签顺序

    outputs/
        figures/                全部图表(9 核心图 + 9 混淆矩阵)
        reports/                生成的分析报告
        predictions/            批量预测结果
        eval_results.json       模型评价数据
        metrics_summary.csv     指标汇总表

    .streamlit/config.toml      网页绿茵主题配置
    requirements.txt            依赖库清单
    run.bat                     一键运行脚本


七、数据来源
--------------------------------------------------------------------
    martj42/international_results
    https://github.com/martj42/international_results
    许可:CC0-1.0(公共领域,可自由使用)
    内容:1872-2026 年国际足球比赛结果,本项目精确筛选其中的
          FIFA 世界杯正赛与预选赛共约 9700 场。


八、特别说明
--------------------------------------------------------------------
    1. 在中立场(neutral=True)比赛中,home_team / away_team 仅表示数据
       记录中的双方位置,不代表真实主客场;模型通过 neutral 字段区分
       是否存在真实主场因素。因此预测结果统一表述为"队伍A / 队伍B"。

    2. 足球比赛存在较强偶然性,本系统预测结果应作为辅助参考,
       而非确定性判断。

    3. 分析报告默认使用本地模板生成,保证无网络、无 API Key 时
       也能稳定运行;若环境变量中配置了大模型 API Key,可选调用
       大模型生成更自然的报告(API Key 不写入代码,仅从环境变量读取)。

    4. 网页标题使用 Noto Serif SC 字体子集,正文与控件使用
       Noto Sans SC 字体子集。字体文件随项目本地提供,不依赖外部
       字体服务;许可证见 web/public/fonts/OFL.txt。


====================================================================
