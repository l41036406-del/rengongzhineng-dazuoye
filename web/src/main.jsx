import React, { useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  BarChart3,
  Bot,
  ChevronDown,
  Database,
  FileChartColumn,
  FileDown,
  Gauge,
  Menu,
  PanelLeftClose,
  PanelLeftOpen,
  Sparkles,
  Table2,
  Target,
  Trophy,
  Upload,
} from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import "./styles.css";

const NAV_ITEMS = [
  { id: "overview", label: "数据概览", icon: Database },
  { id: "visuals", label: "可视化分析", icon: BarChart3 },
  { id: "models", label: "模型对比", icon: Bot },
  { id: "predict", label: "单场预测", icon: Target },
  { id: "batch", label: "批量预测", icon: Upload },
  { id: "report", label: "分析报告", icon: FileChartColumn },
];

const RESULT_COLORS = {
  队伍A胜: "#8FA35B",
  平局: "#C7A45A",
  队伍B胜: "#C96B4B",
};

const CHART_COLORS = {
  olive: "#8FA35B",
  moss: "#526C4F",
  gold: "#C7A45A",
  clay: "#C96B4B",
  stone: "#87918A",
  cream: "#E9E0CC",
  grid: "rgba(73, 94, 84, .16)",
  axis: "#53675F",
};

const FEATURE_LABELS = {
  home_elo: "队伍A ELO",
  away_elo: "队伍B ELO",
  elo_diff: "ELO差值",
  elo_abs_diff: "ELO总和",
  home_recent_winrate: "队伍A近期胜率",
  away_recent_winrate: "队伍B近期胜率",
  home_recent_goal_diff: "队伍A近期净胜球",
  away_recent_goal_diff: "队伍B近期净胜球",
  home_avg_goals: "队伍A场均进球",
  away_avg_goals: "队伍B场均进球",
  h2h_diff: "历史交锋差值",
  home_wc_exp: "队伍A世界杯经验",
  away_wc_exp: "队伍B世界杯经验",
  neutral: "中立场地",
  is_world_cup_final: "世界杯正赛",
  match_year: "比赛年份",
};

const FEATURE_SHORT_LABELS = {
  elo_diff: "ELO差值",
  home_recent_winrate: "队伍A胜率",
  away_recent_winrate: "队伍B胜率",
  home_recent_goal_diff: "A近期净胜球",
  away_recent_goal_diff: "B近期净胜球",
  home_avg_goals: "A场均进球",
  away_avg_goals: "B场均进球",
  h2h_diff: "交锋差值",
  neutral: "中立场地",
  is_world_cup_final: "世界杯正赛",
};

const featureLabel = (name) => FEATURE_LABELS[name] || name;
const featureShortLabel = (name) => FEATURE_SHORT_LABELS[name] || featureLabel(name);

const CSV_FIELD_ALIASES = {
  home_team: "home_team",
  away_team: "away_team",
  neutral: "neutral",
  is_world_cup_final: "is_world_cup_final",
  date: "date",
  tournament: "tournament",
  队伍A: "home_team",
  球队A: "home_team",
  主队: "home_team",
  队伍B: "away_team",
  球队B: "away_team",
  客队: "away_team",
  中立场地: "neutral",
  世界杯正赛: "is_world_cup_final",
  比赛日期: "date",
  日期: "date",
  赛事名称: "tournament",
  赛事: "tournament",
};

const decodeCsvFile = async (file) => {
  const bytes = await file.arrayBuffer();
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return new TextDecoder("gb18030").decode(bytes);
  }
};

const UPLOAD_FIELDS = [
  { field: "队伍A", alias: "home_team", meaning: "记录中的第一支队伍", type: "必填" },
  { field: "队伍B", alias: "away_team", meaning: "记录中的第二支队伍", type: "必填" },
  { field: "中立场地", alias: "neutral", meaning: "填写“是/否”或“1/0”", type: "可选" },
  {
    field: "世界杯正赛",
    alias: "is_world_cup_final",
    meaning: "填写“是/否”或“1/0”",
    type: "可选",
  },
  { field: "比赛日期", alias: "date", meaning: "推荐格式 YYYY-MM-DD", type: "可选" },
  { field: "赛事名称", alias: "tournament", meaning: "例如 World Cup", type: "可选" },
];

async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail || "请求失败");
  }
  return response.json();
}

function useApi(path) {
  const [state, setState] = useState({ data: null, error: "", loading: true });
  useEffect(() => {
    let active = true;
    api(path)
      .then((data) => active && setState({ data, error: "", loading: false }))
      .catch((error) =>
        active && setState({ data: null, error: error.message, loading: false }),
      );
    return () => {
      active = false;
    };
  }, [path]);
  return state;
}

function App() {
  const initialPage = window.location.hash.replace("#", "");
  const [page, setPage] = useState(
    NAV_ITEMS.some((item) => item.id === initialPage) ? initialPage : "overview",
  );
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(
    () => window.localStorage.getItem("sidebarCollapsed") === "true",
  );
  const [uploadedMatches, setUploadedMatches] = useState(() => {
    try {
      return JSON.parse(window.localStorage.getItem("uploadedMatches") || "[]");
    } catch {
      return [];
    }
  });
  const item = NAV_ITEMS.find((nav) => nav.id === page);

  const changePage = (id) => {
    setPage(id);
    window.history.replaceState(null, "", `#${id}`);
    setSidebarOpen(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };
  const saveUploadedMatches = (matches) => {
    setUploadedMatches(matches);
    window.localStorage.setItem("uploadedMatches", JSON.stringify(matches));
  };
  const toggleSidebar = () => {
    setSidebarCollapsed((current) => {
      window.localStorage.setItem("sidebarCollapsed", String(!current));
      return !current;
    });
  };

  return (
    <div className={`app-shell ${sidebarCollapsed ? "sidebar-collapsed" : ""}`}>
      <Sidebar
        page={page}
        open={sidebarOpen}
        collapsed={sidebarCollapsed}
        onCollapse={toggleSidebar}
        onNavigate={changePage}
      />
      {sidebarOpen ? (
        <button
          className="sidebar-scrim"
          aria-label="关闭导航"
          onClick={() => setSidebarOpen(false)}
        />
      ) : null}
      <main className="main-area">
        <header className="mobile-header">
          <button className="icon-button" onClick={() => setSidebarOpen(true)}>
            <Menu size={20} />
          </button>
          <span>{item?.label}</span>
          <div className="mobile-mark">◉</div>
        </header>
        <Page
          page={page}
          uploadedMatches={uploadedMatches}
          onUploadedMatches={saveUploadedMatches}
        />
        <footer>
          世界杯比赛结果智能预测与分析系统 · 数据来源
          martj42/international_results (CC0)
        </footer>
      </main>
    </div>
  );
}

function Sidebar({ page, open, collapsed, onCollapse, onNavigate }) {
  return (
    <aside className={`sidebar ${open ? "is-open" : ""} ${collapsed ? "is-collapsed" : ""}`}>
      <div className="brand">
        <div className="brand-mark">◉</div>
        <div className="brand-copy">
          <strong>世界杯预测系统</strong>
          <span>WORLD CUP INTELLIGENCE</span>
        </div>
        <button
          className="sidebar-toggle"
          aria-label={collapsed ? "展开侧边栏" : "收起侧边栏"}
          title={collapsed ? "展开侧边栏" : "收起侧边栏"}
          onClick={onCollapse}
        >
          {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
        </button>
      </div>
      <nav>
        {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            className={page === id ? "active" : ""}
            onClick={() => onNavigate(id)}
            title={collapsed ? label : undefined}
          >
            <Icon size={17} strokeWidth={1.8} />
            <span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-note">
        <Activity size={17} />
        <div>
          <span>模型状态</span>
          <strong>服务正常</strong>
        </div>
      </div>
    </aside>
  );
}

function Page({ page, uploadedMatches, onUploadedMatches }) {
  if (page === "visuals") return <Visualizations />;
  if (page === "models") return <Models />;
  if (page === "predict") return <Predict />;
  if (page === "batch") {
    return <Batch onUploadedMatches={onUploadedMatches} />;
  }
  if (page === "report") return <Report matches={uploadedMatches} />;
  return <Overview />;
}

function PageHeader({ title, subtitle, action, visual = false }) {
  return (
    <div className={`page-header ${visual ? "has-visual" : ""}`}>
      <div>
        <h1>{title}</h1>
        <p>{subtitle}</p>
      </div>
      {visual ? <div className="header-visual" aria-hidden="true" /> : null}
      {action}
    </div>
  );
}

function Loading() {
  return (
    <div className="loading">
      <span />
      正在读取项目数据
    </div>
  );
}

function ErrorState({ message }) {
  return <div className="error-state">{message}</div>;
}

function Overview() {
  const { data, error, loading } = useApi("/api/overview");
  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;
  const metrics = [
    ["01", data.metrics.matches.toLocaleString(), "世界杯相关比赛"],
    ["02", data.metrics.finals.toLocaleString(), "正赛场次"],
    ["03", data.metrics.qualifiers.toLocaleString(), "预选赛场次"],
    ["04", `${data.metrics.year_min}–${data.metrics.year_max}`, "时间跨度"],
  ];
  const analysisFlow = [
    {
      icon: Database,
      title: "数据接入",
      description: "9,735 场历史比赛、ELO 与上传赛程",
    },
    {
      icon: Activity,
      title: "特征工程",
      description: "16 维赛前特征，严格避免未来信息泄漏",
    },
    {
      icon: Bot,
      title: "模型计算",
      description: "9 类模型对比、概率预测与性能评估",
    },
    {
      icon: FileChartColumn,
      title: "实时报告",
      description: "逐队分析、赛程难度与针对性建议",
    },
  ];

  return (
    <>
      <PageHeader
        title={
          <>
            世界杯比赛结果
            <br />
            智能预测与分析系统
          </>
        }
        subtitle="融合历史比赛、ELO 评分与赛前状态的大数据分析平台。系统覆盖数据治理、特征工程、多模型评估、概率预测和上传赛程实时报告。"
        visual
      />
      <section className="home-transition">
        <div>
        <span>从历史比赛到赛前决策</span>
        <h2>
          <span>让每一次预测都能追溯到</span>
          <span>数据、特征与模型依据</span>
        </h2>
        </div>
        <p>
          平台不是简单展示比赛统计，而是把历史状态、对手强度和模型概率组织成可解释的分析流程，
          并在上传赛程后生成逐队报告。
        </p>
      </section>
      <section className="analysis-pipeline">
        <div className="pipeline-heading">
          <div>
            <span>ANALYTICS PIPELINE</span>
            <h2>从原始赛程到可执行洞察</h2>
          </div>
          <p>每一步都可追溯到真实数据、赛前特征与模型概率。</p>
        </div>
        <div className="pipeline-flow">
          {analysisFlow.map(({ icon: Icon, title, description }, index) => (
            <article key={title}>
              <div className="pipeline-icon">
                <Icon size={21} />
              </div>
              <div>
                <small>0{index + 1}</small>
                <strong>{title}</strong>
                <p>{description}</p>
              </div>
            </article>
          ))}
        </div>
      </section>
      <section className="metric-row">
        {metrics.map(([index, value, label]) => (
          <div className="metric" key={label}>
            <span>{index}</span>
            <strong>{value}</strong>
            <small>{label}</small>
          </div>
        ))}
      </section>
      <SectionTitle title="数据概览" note="最近更新：训练数据末次比赛" />
      <div className="overview-grid">
        <Panel title="最近比赛" icon={Table2}>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>日期</th>
                  <th>队伍 A</th>
                  <th>比分</th>
                  <th>队伍 B</th>
                  <th>结果</th>
                </tr>
              </thead>
              <tbody>
                {data.recent_matches.slice(0, 8).map((match, index) => (
                  <tr key={`${match.date}-${index}`}>
                    <td>{match.date}</td>
                    <td>{match.home_team}</td>
                    <td className="score">
                      {match.home_score} - {match.away_score}
                    </td>
                    <td>{match.away_team}</td>
                    <td className="accent-text">{match.result}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
        <Panel title="结果分布" icon={Gauge}>
          <div className="donut-wrap">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={data.result_distribution}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={70}
                  outerRadius={100}
                  paddingAngle={2}
                >
                  {data.result_distribution.map((entry) => (
                    <Cell key={entry.name} fill={RESULT_COLORS[entry.name]} />
                  ))}
                </Pie>
                <Tooltip content={<ChartTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div className="donut-center">
              <strong>{data.metrics.matches.toLocaleString()}</strong>
              <span>总场次</span>
            </div>
          </div>
          <div className="legend-list">
            {data.result_distribution.map((entry) => (
              <div key={entry.name}>
                <i style={{ background: RESULT_COLORS[entry.name] }} />
                <span>{entry.name}</span>
                <strong>{((entry.value / data.metrics.matches) * 100).toFixed(1)}%</strong>
              </div>
            ))}
          </div>
        </Panel>
      </div>
    </>
  );
}

function Visualizations() {
  const { data, error, loading } = useApi("/api/visualizations");
  const [group, setGroup] = useState(0);
  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;
  const groups = [
    [
      {
        id: "trend",
        title: "世界杯比赛数量与进球趋势",
        description: "按年份观察赛事规模与场均进球变化",
        chart: (
          <ComposedChart data={data.trend}>
            <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
            <XAxis dataKey="year" stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <YAxis yAxisId="matches" stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <YAxis
              yAxisId="goals"
              orientation="right"
              stroke={CHART_COLORS.clay}
              tick={{ fontSize: 9 }}
            />
            <Tooltip content={<ChartTooltip />} />
            <Legend />
            <Bar
              yAxisId="matches"
              dataKey="matches"
              name="比赛场数"
              fill={CHART_COLORS.olive}
              opacity={0.72}
            />
            <Line
              yAxisId="goals"
              dataKey="avg_goals"
              name="场均进球"
              stroke={CHART_COLORS.clay}
              strokeWidth={2.5}
              dot={false}
            />
          </ComposedChart>
        ),
      },
      {
        id: "results",
        title: "比赛结果类别分布",
        description: "三分类样本数量与类别结构",
        chart: (
          <PieChart>
            <Pie
              data={data.result_distribution}
              dataKey="value"
              nameKey="name"
              innerRadius="48%"
              outerRadius="76%"
              paddingAngle={3}
            >
              {data.result_distribution.map((item) => (
                <Cell key={item.name} fill={RESULT_COLORS[item.name]} />
              ))}
            </Pie>
            <Tooltip content={<ChartTooltip suffix=" 场" />} />
            <Legend />
          </PieChart>
        ),
      },
      {
        id: "goals",
        title: "进球数分布",
        description: "队伍 A、队伍 B 与总进球的频数分布",
        chart: (
          <BarChart data={data.goal_distribution}>
            <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
            <XAxis dataKey="goals" stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <YAxis stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <Tooltip content={<ChartTooltip suffix=" 场" />} />
            <Legend />
            <Bar dataKey="队伍A" fill={CHART_COLORS.olive} />
            <Bar dataKey="队伍B" fill={CHART_COLORS.clay} />
            <Bar dataKey="总进球" fill={CHART_COLORS.gold} />
          </BarChart>
        ),
      },
    ],
    [
      {
        id: "strong-teams",
        title: "传统强队历史胜率",
        description: "出场次数最多的十二支球队长期胜率",
        chart: (
          <BarChart data={data.strong_teams} layout="vertical" margin={{ left: 30, right: 30 }}>
            <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
            <XAxis type="number" unit="%" stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <YAxis
              type="category"
              dataKey="team"
              width={88}
              stroke={CHART_COLORS.axis}
              tick={{ fontSize: 9 }}
            />
            <Tooltip content={<ChartTooltip suffix="%" />} />
            <Bar dataKey="win_rate" name="历史胜率" fill={CHART_COLORS.olive} />
          </BarChart>
        ),
      },
      {
        id: "venue",
        title: "主场与中立场差异",
        description: "不同场地条件下的胜平负占比",
        chart: (
          <BarChart data={data.venue_comparison}>
            <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
            <XAxis dataKey="venue" stroke={CHART_COLORS.axis} tick={{ fontSize: 10 }} />
            <YAxis unit="%" stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <Tooltip content={<ChartTooltip suffix="%" />} />
            <Legend />
            <Bar dataKey="队伍A胜" fill={CHART_COLORS.olive} />
            <Bar dataKey="平局" fill={CHART_COLORS.gold} />
            <Bar dataKey="队伍B胜" fill={CHART_COLORS.clay} />
          </BarChart>
        ),
      },
      {
        id: "elo",
        title: "ELO 评分分布",
        description: "双方赛前实力评分的区间分布",
        chart: (
          <BarChart data={data.elo_distribution}>
            <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
            <XAxis
              dataKey="elo"
              interval={2}
              stroke={CHART_COLORS.axis}
              tick={{ fontSize: 9 }}
            />
            <YAxis stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <Tooltip content={<ChartTooltip suffix=" 场" />} />
            <Legend />
            <Bar dataKey="队伍A" fill={CHART_COLORS.olive} opacity={0.82} />
            <Bar dataKey="队伍B" fill={CHART_COLORS.clay} opacity={0.72} />
          </BarChart>
        ),
      },
    ],
    [
      {
        id: "correlation",
        title: "特征相关性",
        description: "核心赛前变量之间的相关关系",
        custom: <CorrelationHeatmap data={data.correlation} />,
      },
      {
        id: "models",
        title: "模型表现对比",
        description: "准确率与 Macro-F1 的综合比较",
        chart: (
          <BarChart data={data.model_metrics} margin={{ bottom: 45 }}>
            <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
            <XAxis
              dataKey="model"
              angle={-20}
              textAnchor="end"
              interval={0}
              height={72}
              stroke={CHART_COLORS.axis}
              tick={{ fontSize: 9 }}
            />
            <YAxis unit="%" stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <Tooltip content={<ChartTooltip suffix="%" />} />
            <Legend />
            <Bar dataKey="accuracy" name="准确率" fill={CHART_COLORS.moss} />
            <Bar dataKey="macro_f1" name="Macro-F1" fill={CHART_COLORS.gold} />
          </BarChart>
        ),
      },
      {
        id: "importance",
        title: "特征重要性",
        description: "随机森林对各赛前特征的使用权重",
        chart: (
          <BarChart
            data={data.feature_importance.map((item) => ({
              ...item,
              feature_label: featureLabel(item.feature),
            }))}
            layout="vertical"
            margin={{ left: 25, right: 35 }}
          >
            <CartesianGrid stroke={CHART_COLORS.grid} strokeDasharray="3 3" />
            <XAxis type="number" unit="%" stroke={CHART_COLORS.axis} tick={{ fontSize: 9 }} />
            <YAxis
              type="category"
              dataKey="feature_label"
              width={145}
              stroke={CHART_COLORS.axis}
              tick={{ fontSize: 8 }}
            />
            <Tooltip content={<ChartTooltip suffix="%" />} />
            <Bar dataKey="importance" name="重要性" fill={CHART_COLORS.olive} />
          </BarChart>
        ),
      },
    ],
  ];
  const labels = ["数据理解", "足球规律", "建模分析"];
  return (
    <>
      <PageHeader
        title="数据可视化分析"
        subtitle="九张核心图表从数据结构、足球规律和建模结果三个层面解释预测依据。"
      />
      <div className="segmented">
        {labels.map((label, index) => (
          <button
            key={label}
            className={group === index ? "active" : ""}
            onClick={() => setGroup(index)}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="visual-grid">
        {groups[group].map((item) => (
          <article
            className={`visual-card interactive-chart visual-card-${item.id}`}
            key={item.id}
          >
            <div className="visual-card-heading">
              <h3>{item.title}</h3>
              <p>{item.description}</p>
            </div>
            <div className="interactive-chart-body">
              {item.custom || (
                <ResponsiveContainer width="100%" height="100%">
                  {item.chart}
                </ResponsiveContainer>
              )}
            </div>
          </article>
        ))}
      </div>
    </>
  );
}

function CorrelationHeatmap({ data }) {
  const [active, setActive] = useState(null);
  const color = (value) => {
    const strength = Math.min(Math.abs(value), 1);
    if (value >= 0) return `rgba(143, 163, 91, ${0.12 + strength * 0.78})`;
    return `rgba(201, 107, 75, ${0.12 + strength * 0.78})`;
  };
  return (
    <div className="correlation-wrap">
      <div
        className="correlation-grid"
        style={{ gridTemplateColumns: `110px repeat(${data.columns.length}, minmax(36px, 1fr))` }}
      >
        <span />
        {data.columns.map((column) => (
          <strong className="correlation-x" key={column}>{featureShortLabel(column)}</strong>
        ))}
        {data.columns.map((row) => (
          <React.Fragment key={row}>
            <strong className="correlation-y">{featureLabel(row)}</strong>
            {data.columns.map((column) => {
              const item = data.values.find((value) => value.x === column && value.y === row);
              return (
                <button
                  key={`${row}-${column}`}
                  style={{
                    background: color(item.value),
                    color: Math.abs(item.value) >= 0.58 ? "#fffaf0" : "#29483e",
                  }}
                  onMouseEnter={() => setActive(item)}
                  onFocus={() => setActive(item)}
                  title={`${featureLabel(row)} × ${featureLabel(column)}：${item.value}`}
                >
                  {item.value.toFixed(2)}
                </button>
              );
            })}
          </React.Fragment>
        ))}
      </div>
      <div className="correlation-readout">
        {active
          ? `${featureLabel(active.y)} × ${featureLabel(active.x)}：相关系数 ${active.value.toFixed(2)}`
          : "悬停任意色块查看变量关系"}
      </div>
    </div>
  );
}

function Models() {
  const { data, error, loading } = useApi("/api/models");
  if (loading) return <Loading />;
  if (error) return <ErrorState message={error} />;
  const chartData = data.metrics.filter((item) => item.accuracy);
  return (
    <>
      <PageHeader
        title="模型训练与对比"
        subtitle="两个基准模型、六个机器学习模型与 KNN 参数搜索，以准确率和 Macro-F1 综合评价。"
      />
      <div className="highlight">
        <Trophy size={22} />
        <div>
          <span>系统默认模型</span>
          <strong>{data.best_model}</strong>
        </div>
        <p>综合性能、稳定性与可解释性最佳</p>
      </div>
      <div className="model-grid">
        <Panel title="指标对比" icon={BarChart3}>
          <ResponsiveContainer width="100%" height={360}>
            <BarChart data={chartData} margin={{ top: 20, right: 10, left: 0, bottom: 50 }}>
              <CartesianGrid stroke="rgba(145,163,157,.12)" vertical={false} />
              <XAxis dataKey="name" stroke="#70827c" angle={-28} textAnchor="end" height={80} />
              <YAxis
                stroke="#718176"
                domain={[0, 1]}
                width={42}
                tickFormatter={(value) => Number(value).toFixed(2)}
              />
              <Tooltip content={<ChartTooltip />} />
              <Legend />
              <Bar dataKey="accuracy" name="准确率" fill="#0d3b31" />
              <Bar dataKey="macro_f1" name="Macro-F1" fill="#b99a5d" />
            </BarChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="KNN 参数搜索" icon={Activity}>
          <ResponsiveContainer width="100%" height={360}>
            <LineChart data={data.knn_tuning} margin={{ top: 20, right: 15, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="rgba(145,163,157,.12)" vertical={false} />
              <XAxis dataKey="k" stroke="#718176" />
              <YAxis
                stroke="#718176"
                width={42}
                domain={["dataMin - 0.03", "dataMax + 0.03"]}
                tickFormatter={(value) => Number(value).toFixed(2)}
              />
              <Tooltip content={<ChartTooltip />} />
              <Line type="monotone" dataKey="accuracy" name="准确率" stroke="#0d3b31" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="macro_f1" name="Macro-F1" stroke="#74382e" strokeWidth={2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <p className="panel-note">最佳 K = {data.best_k}，用于平衡局部噪声与多数类偏置。</p>
        </Panel>
      </div>
      <Panel title="模型指标明细" icon={Table2}>
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                <th>模型</th>
                <th>准确率</th>
                <th>Macro-F1</th>
                <th>综合分</th>
              </tr>
            </thead>
            <tbody>
              {chartData.map((model) => (
                <tr key={model.name}>
                  <td>{model.name}</td>
                  <td>{model.accuracy}</td>
                  <td>{model.macro_f1}</td>
                  <td className="accent-text">{model.composite ?? "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Panel>
    </>
  );
}

function Predict() {
  const teamsState = useApi("/api/teams");
  const modelsState = useApi("/api/models");
  const [form, setForm] = useState({
    home_team: "Brazil",
    away_team: "Argentina",
    neutral: true,
    is_world_cup_final: true,
    model_name: "",
  });
  const [result, setResult] = useState(null);
  const [status, setStatus] = useState({ loading: false, error: "" });
  if (teamsState.loading || modelsState.loading) return <Loading />;
  if (teamsState.error || modelsState.error) {
    return <ErrorState message={teamsState.error || modelsState.error} />;
  }
  const teams = teamsState.data.teams;
  const submit = async (event) => {
    event.preventDefault();
    setStatus({ loading: true, error: "" });
    try {
      const prediction = await api("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...form, model_name: form.model_name || null }),
      });
      setResult(prediction);
      setStatus({ loading: false, error: "" });
    } catch (error) {
      setStatus({ loading: false, error: error.message });
    }
  };
  return (
    <>
      <PageHeader
        title="单场比赛预测"
        subtitle="选择对阵双方、比赛环境和模型，查看胜平负概率及模型置信度。"
      />
      <form className="predict-layout" onSubmit={submit}>
        <Panel title="比赛设置" icon={Target}>
          <div className="form-grid">
            <Select
              label="队伍 A"
              value={form.home_team}
              options={teams.map((team) => team.name)}
              onChange={(home_team) => setForm((current) => ({ ...current, home_team }))}
            />
            <Select
              label="队伍 B"
              value={form.away_team}
              options={teams.map((team) => team.name)}
              onChange={(away_team) => setForm((current) => ({ ...current, away_team }))}
            />
            <Select
              label="预测模型"
              value={form.model_name}
              options={["", ...modelsState.data.model_names]}
              optionLabel={(value) => value || `最佳模型（${modelsState.data.best_model}）`}
              onChange={(model_name) => setForm((current) => ({ ...current, model_name }))}
            />
          </div>
          <div className="toggle-row">
            <Toggle
              label="中立场地"
              checked={form.neutral}
              onChange={(neutral) => setForm((current) => ({ ...current, neutral }))}
            />
            <Toggle
              label="世界杯正赛"
              checked={form.is_world_cup_final}
              onChange={(is_world_cup_final) =>
                setForm((current) => ({ ...current, is_world_cup_final }))
              }
            />
          </div>
          <button className="primary-button" disabled={status.loading}>
            {status.loading ? "模型计算中…" : "开始预测"}
          </button>
          {status.error ? <p className="form-error">{status.error}</p> : null}
        </Panel>
        <PredictionResult result={result} />
      </form>
    </>
  );
}

function PredictionResult({ result }) {
  if (!result) {
    return (
      <Panel title="预测结果" icon={Sparkles}>
        <div className="empty-result">
          <Target size={42} />
          <strong>等待输入比赛条件</strong>
          <p>提交后将展示预测结果、三分类概率与双方 ELO。</p>
        </div>
      </Panel>
    );
  }
  const chart = Object.entries(result.probabilities).map(([name, value]) => ({
    name,
    probability: Math.round(value * 1000) / 10,
  }));
  return (
    <Panel title="预测结果" icon={Sparkles}>
      <div className="versus">
        <div>
          <span>队伍 A</span>
          <strong>{result.home_team}</strong>
          <small>ELO {result.home_elo}</small>
        </div>
        <b>VS</b>
        <div>
          <span>队伍 B</span>
          <strong>{result.away_team}</strong>
          <small>ELO {result.away_elo}</small>
        </div>
      </div>
      <div className="prediction-label">{result.prediction}</div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chart} layout="vertical" margin={{ left: 15, right: 20 }}>
          <XAxis type="number" domain={[0, 100]} hide />
          <YAxis type="category" dataKey="name" stroke="#91a39d" width={64} />
          <Tooltip content={<ChartTooltip suffix="%" />} />
          <Bar dataKey="probability" radius={[0, 6, 6, 0]}>
            {chart.map((entry) => (
              <Cell key={entry.name} fill={RESULT_COLORS[entry.name]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <p className="panel-note">使用模型：{result.model}。预测仅作课程分析与辅助参考。</p>
    </Panel>
  );
}

function Batch({ onUploadedMatches }) {
  const [rows, setRows] = useState([]);
  const [results, setResults] = useState([]);
  const [status, setStatus] = useState({ loading: false, error: "" });
  const loadFile = async (event) => {
    const file = event.target.files?.[0];
    if (!file) return;
    const text = await decodeCsvFile(file);
    const lines = text.split(/\r?\n/).filter(Boolean);
    const headers = lines[0]
      .replace(/^\uFEFF/, "")
      .split(",")
      .map((value) => CSV_FIELD_ALIASES[value.trim()] || value.trim());
    const parsed = lines.slice(1).map((line) => {
      const values = line.split(",").map((value) => value.trim());
      return Object.fromEntries(headers.map((header, index) => [header, values[index]]));
    });
    if (!headers.includes("home_team") || !headers.includes("away_team")) {
      setRows([]);
      setResults([]);
      setStatus({ loading: false, error: "CSV 必须包含“队伍A”和“队伍B”两列。" });
      return;
    }
    setRows(parsed);
    setResults([]);
    setStatus({ loading: false, error: "" });
  };
  const parseBoolean = (value, defaultValue = true) => {
    if (value === undefined || value === null || value === "") return defaultValue;
    return !["0", "false", "否", "不是", "no"].includes(String(value).trim().toLowerCase());
  };
  const buildPayload = () =>
    rows.map((row) => ({
      home_team: row.home_team,
      away_team: row.away_team,
      neutral: parseBoolean(row.neutral),
      is_world_cup_final: parseBoolean(row.is_world_cup_final),
      date: row.date || null,
      tournament: row.tournament || null,
    }));
  const runBatch = async () => {
    if (!rows.length) return;
    setStatus({ loading: true, error: "" });
    try {
      const payload = buildPayload();
      const data = await api("/api/batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matches: payload }),
      });
      setResults(data.results);
      onUploadedMatches(
        data.results.map((item, index) => ({
          ...payload[index],
          home_team: item.home_team,
          away_team: item.away_team,
        })),
      );
      setStatus({ loading: false, error: "" });
    } catch (error) {
      setStatus({ loading: false, error: error.message });
    }
  };
  const download = async () => {
    setStatus((current) => ({ ...current, error: "" }));
    try {
      const response = await fetch("/api/batch/export/xlsx", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matches: buildPayload() }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "Excel 预测结果生成失败");
      }
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "世界杯批量预测结果.xlsx";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setStatus((current) => ({ ...current, error: error.message }));
    }
  };
  const downloadTemplate = () => {
    const content = [
      "队伍A,队伍B,中立场地,世界杯正赛,比赛日期,赛事名称",
      "Brazil,Argentina,是,是,2026-06-15,World Cup",
      "France,Germany,否,是,2026-06-18,World Cup",
    ].join("\n");
    const url = URL.createObjectURL(
      new Blob([`\uFEFF${content}`], { type: "text/csv;charset=utf-8" }),
    );
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = "世界杯批量预测模板.csv";
    anchor.click();
    URL.revokeObjectURL(url);
  };
  return (
    <>
      <PageHeader
        title="批量预测"
        subtitle="上传标准 CSV，一次预测多场比赛并导出包含分类概率的结果文件。"
      />
      <div className="batch-grid">
        <Panel title="上传比赛文件" icon={Upload}>
          <label className="dropzone">
            <Upload size={30} />
            <strong>选择 CSV 文件</strong>
            <span>必须包含“队伍A”与“队伍B”，支持中文字段</span>
            <input type="file" accept=".csv" onChange={loadFile} />
          </label>
          <div className="template">
            <code>队伍A,队伍B,中立场地,世界杯正赛,比赛日期,赛事名称</code>
            <code>Brazil,Argentina,是,是,2026-06-15,World Cup</code>
          </div>
          <button className="secondary-button template-download" onClick={downloadTemplate}>
            <FileDown size={17} />
            下载中文 CSV 模板
          </button>
          <button className="primary-button" disabled={!rows.length || status.loading} onClick={runBatch}>
            {status.loading ? "批量计算中…" : `预测 ${rows.length || 0} 场比赛`}
          </button>
          {status.error ? <p className="form-error">{status.error}</p> : null}
        </Panel>
        <Panel title="预测结果" icon={Table2}>
          {!results.length ? (
            <div className="empty-result compact">
              <Table2 size={36} />
              <p>完成上传和计算后，结果将在此显示。</p>
            </div>
          ) : (
            <>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>队伍 A</th>
                      <th>队伍 B</th>
                      <th>结果</th>
                    </tr>
                  </thead>
                  <tbody>
                    {results.map((item, index) => (
                      <tr key={`${item.home_team}-${item.away_team}-${index}`}>
                        <td>{item.home_team}</td>
                        <td>{item.away_team}</td>
                        <td className="accent-text">{item.prediction}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <button className="secondary-button" onClick={download}>
                <FileDown size={17} />
                下载预测结果 XLSX
              </button>
              <p className="panel-note">
                本次上传数据已同步到“分析报告”，可按队伍查看实时图表与建议。
              </p>
            </>
          )}
        </Panel>
      </div>
      <Panel title="CSV 字段说明" icon={Database}>
        <p className="panel-note field-note">
          建议直接使用中文模板。系统同时兼容下列英文别名，便于已有数据文件继续使用。
        </p>
        <div className="field-grid upload-field-grid">
          {UPLOAD_FIELDS.map((field) => (
            <div key={field.field}>
              <strong>{field.field}</strong>
              <code>{field.alias}</code>
              <span>{field.meaning}</span>
              <small>{field.type}</small>
            </div>
          ))}
        </div>
      </Panel>
    </>
  );
}

function Report({ matches }) {
  const [state, setState] = useState({ data: null, error: "", loading: false });
  const [selectedTeam, setSelectedTeam] = useState("");
  const [exportError, setExportError] = useState("");
  useEffect(() => {
    if (!matches.length) {
      setState({ data: null, error: "", loading: false });
      return;
    }
    let active = true;
    setState({ data: null, error: "", loading: true });
    api("/api/report/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ matches }),
    })
      .then((data) => {
        if (!active) return;
        setState({ data, error: "", loading: false });
        setSelectedTeam((current) =>
          data.teams.some((team) => team.team === current) ? current : data.teams[0]?.team || "",
        );
      })
      .catch((error) => {
        if (active) setState({ data: null, error: error.message, loading: false });
      });
    return () => {
      active = false;
    };
  }, [matches]);
  const { data, error, loading } = state;
  const team = data?.teams.find((item) => item.team === selectedTeam);
  const download = async () => {
    if (!matches.length) return;
    setExportError("");
    try {
      const response = await fetch("/api/report/export/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ matches }),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body.detail || "PDF 报告生成失败");
      }
      const url = URL.createObjectURL(await response.blob());
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = "世界杯预测分析报告.pdf";
      anchor.click();
      URL.revokeObjectURL(url);
    } catch (error) {
      setExportError(error.message);
    }
  };
  return (
    <>
      <PageHeader
        title="上传数据实时分析"
        subtitle="报告严格基于最近一次上传赛程，逐队计算胜平负概率、预期积分、赛程难度与针对性建议。"
        action={
          data ? (
            <button className="secondary-button" onClick={download}>
              <FileDown size={17} />
              下载 PDF 报告
            </button>
          ) : null
        }
      />
      {!matches.length ? (
        <div className="report-empty">
          <Upload size={40} />
          <strong>尚未发现可分析的上传数据</strong>
          <p>请先在“批量预测”上传 CSV 并完成预测。报告不会使用固定模板代替真实数据。</p>
        </div>
      ) : null}
      {loading ? <Loading /> : null}
      {error ? <ErrorState message={error} /> : null}
      {exportError ? <ErrorState message={exportError} /> : null}
      {data ? (
        <div className="report-dashboard">
          <div className="report-callout">
            <div>
              <span>实时分析批次</span>
              <strong>{data.generated_at}</strong>
            </div>
            <p>
              ELO 最高：<b>{data.summary.strongest_team}</b> · 赛程前景最佳：
              <b>{data.summary.best_schedule_outlook}</b>
            </p>
          </div>

          <div className="report-summary">
            <ReportMetric label="上传比赛" value={data.summary.matches} suffix="场" />
            <ReportMetric label="涉及队伍" value={data.summary.teams} suffix="支" />
            <ReportMetric
              label="平均置信度"
              value={data.summary.average_confidence}
              suffix="%"
            />
            <ReportMetric
              label="高不确定对局"
              value={data.summary.uncertain_matches}
              suffix="场"
            />
          </div>

          <div className="report-chart-grid">
            <Panel title="预测结果构成" icon={BarChart3}>
              <ResponsiveContainer width="100%" height={270}>
                <PieChart>
                  <Pie
                    data={data.outcome_distribution}
                    dataKey="value"
                    nameKey="name"
                    innerRadius={62}
                    outerRadius={94}
                    paddingAngle={3}
                  >
                    {data.outcome_distribution.map((item) => (
                      <Cell key={item.name} fill={RESULT_COLORS[item.name]} />
                    ))}
                  </Pie>
                  <Tooltip content={<ChartTooltip suffix=" 场" />} />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </Panel>
            <Panel title="队伍预期积分排名" icon={Trophy}>
              <ResponsiveContainer width="100%" height={270}>
                <BarChart
                  data={data.teams.slice(0, 10)}
                  layout="vertical"
                  margin={{ left: 20, right: 25 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#d8d0bf" />
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="team" width={90} tick={{ fontSize: 10 }} />
                  <Tooltip content={<ChartTooltip suffix=" 分" />} />
                  <Bar dataKey="expected_points" name="预期积分" fill="#0d3b31" />
                </BarChart>
              </ResponsiveContainer>
            </Panel>
          </div>

          <div className="team-report-toolbar">
            <div>
              <span>队伍专项报告</span>
              <strong>选择队伍查看独立分析</strong>
            </div>
            <Select
              label="当前队伍"
              value={selectedTeam}
              options={data.teams.map((item) => item.team)}
              onChange={setSelectedTeam}
            />
          </div>

          {team ? (
            <>
              <div className="team-kpis">
                <ReportMetric label="ELO" value={team.elo} />
                <ReportMetric label="近期胜率" value={team.recent_winrate} suffix="%" />
                <ReportMetric label="场均进球" value={team.avg_goals} />
                <ReportMetric label="赛程预期积分" value={team.expected_points} />
                <ReportMetric label="场均预期积分" value={team.points_per_match} />
              </div>

              <div className="report-chart-grid">
                <Panel title={`${team.team} · 逐场结果概率`} icon={Gauge}>
                  <ResponsiveContainer width="100%" height={310}>
                    <BarChart data={team.fixtures} margin={{ top: 10, right: 10, left: 0, bottom: 35 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#d8d0bf" />
                      <XAxis
                        dataKey="opponent"
                        angle={-24}
                        textAnchor="end"
                        interval={0}
                        height={65}
                        tick={{ fontSize: 10 }}
                      />
                      <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 10 }} />
                      <Tooltip content={<ChartTooltip suffix="%" />} />
                      <Legend />
                      <Bar dataKey="win" name="胜" stackId="result" fill="#0d3b31" />
                      <Bar dataKey="draw" name="平" stackId="result" fill="#b99a5d" />
                      <Bar dataKey="loss" name="负" stackId="result" fill="#74382e" />
                    </BarChart>
                  </ResponsiveContainer>
                </Panel>
                <Panel title={`${team.team} · 状态画像`} icon={Activity}>
                  <div className="profile-list">
                    <ProfileBar label="近期胜率" value={team.recent_winrate} max={100} suffix="%" />
                    <ProfileBar
                      label="ELO 强度"
                      value={Math.max(team.elo - 1200, 0)}
                      max={800}
                      display={team.elo}
                    />
                    <ProfileBar
                      label="场均进球"
                      value={team.avg_goals}
                      max={3}
                    />
                    <ProfileBar
                      label="近 10 场历史胜率"
                      value={team.history.win_rate * 100}
                      max={100}
                      suffix="%"
                    />
                  </div>
                  <div className="history-strip">
                    <span>近 {team.history.matches} 场</span>
                    <b>{team.history.wins} 胜</b>
                    <b>{team.history.draws} 平</b>
                    <b>{team.history.losses} 负</b>
                    <span>
                      进 {team.history.goals_for} / 失 {team.history.goals_against}
                    </span>
                  </div>
                </Panel>
              </div>

              <section className="recommendation-section">
                <div className="section-title">
                  <h2>{team.team} 的针对性建议</h2>
                  <span>由本批对手强度与球队近期指标共同生成</span>
                </div>
                <div className="recommendation-grid">
                  {team.recommendations.map((text, index) => (
                    <article key={text}>
                      <span>{String(index + 1).padStart(2, "0")}</span>
                      <p>{text}</p>
                    </article>
                  ))}
                </div>
              </section>

              <Panel title={`${team.team} · 赛程明细`} icon={Table2}>
                <div className="table-scroll report-table">
                  <table>
                    <thead>
                      <tr>
                        <th>对手</th>
                        <th>预测</th>
                        <th>胜</th>
                        <th>平</th>
                        <th>负</th>
                        <th>对手 ELO</th>
                      </tr>
                    </thead>
                    <tbody>
                      {team.fixtures.map((fixture) => (
                        <tr key={`${fixture.match}-${fixture.opponent}`}>
                          <td>{fixture.opponent}</td>
                          <td className="accent-text">{fixture.prediction}</td>
                          <td>{fixture.win}%</td>
                          <td>{fixture.draw}%</td>
                          <td>{fixture.loss}%</td>
                          <td>{fixture.opponent_elo}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Panel>
            </>
          ) : null}
        </div>
      ) : null}
    </>
  );
}

function ReportMetric({ label, value, suffix = "" }) {
  return (
    <div className="report-metric">
      <span>{label}</span>
      <strong>
        {value}
        <small>{suffix}</small>
      </strong>
    </div>
  );
}

function ProfileBar({ label, value, max, suffix = "", display }) {
  const width = Math.max(0, Math.min(100, (value / max) * 100));
  return (
    <div className="profile-bar">
      <div>
        <span>{label}</span>
        <strong>
          {display ?? Math.round(value * 10) / 10}
          {suffix}
        </strong>
      </div>
      <i>
        <b style={{ width: `${width}%` }} />
      </i>
    </div>
  );
}

function Panel({ title, icon: Icon, children }) {
  return (
    <section className="panel">
      <div className="panel-title">
        <div>
          {Icon ? <Icon size={17} /> : null}
          <h2>{title}</h2>
        </div>
      </div>
      {children}
    </section>
  );
}

function SectionTitle({ title, note }) {
  return (
    <div className="section-title">
      <h2>{title}</h2>
      <span>{note}</span>
    </div>
  );
}

function Select({ label, value, options, onChange, optionLabel = (option) => option }) {
  return (
    <label className="field">
      <span>{label}</span>
      <div>
        <select value={value} onChange={(event) => onChange(event.target.value)}>
          {options.map((option) => (
            <option key={option || "default"} value={option}>
              {optionLabel(option)}
            </option>
          ))}
        </select>
        <ChevronDown size={16} />
      </div>
    </label>
  );
}

function Toggle({ label, checked, onChange }) {
  return (
    <label className="toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <i />
      <span>{label}</span>
    </label>
  );
}

function ChartTooltip({ active, payload, label, suffix = "" }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      {label !== undefined ? <strong>{label}</strong> : null}
      {payload.map((item) => (
        <span key={item.name} style={{ color: item.color }}>
          {item.name}: {typeof item.value === "number" ? item.value.toFixed(2) : item.value}
          {suffix}
        </span>
      ))}
    </div>
  );
}

createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
