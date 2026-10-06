"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { AnalyzeReportData, ProximityInfo } from "@/lib/reportTypes";
import { fmtNum } from "@/lib/reportTypes";

function Badge({ info }: { info?: ProximityInfo }) {
  if (!info) return <span className="report-badge">—</span>;
  const near = info.status === "near_bound";
  return (
    <span className={`report-badge${near ? " is-warn" : " is-ok"}`}>
      {info.label}
    </span>
  );
}

export default function Optimization({ data }: { data: AnalyzeReportData }) {
  const o = data.optimization;
  const trials = o.trials ?? [];
  const chartData = trials
    .filter((t) => t.trial != null && t.objective != null)
    .map((t) => ({
      trial: t.trial as number,
      objective: t.objective as number,
    }));

  const proxOrder = ["R_major", "K_inter", "K_minor", "Nmin", "Nmax"] as const;

  return (
    <section id="report-optimization" className="report-section">
      <h2 className="report-section-title">Оптимизация</h2>

      <dl className="report-grid">
        <div>
          <dt>Best trial</dt>
          <dd>{fmtNum(o.best_trial_number, 0)}</dd>
        </div>
        <div>
          <dt>Trials</dt>
          <dd>{fmtNum(o.n_trials, 0)}</dd>
        </div>
        <div>
          <dt>Best objective</dt>
          <dd>{fmtNum(o.objective)}</dd>
        </div>
      </dl>

      <h3 className="report-subheading">Параметры best trial</h3>
      <dl className="report-grid">
        {Object.entries(o.best_params ?? {}).map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{fmtNum(v, k.startsWith("N") ? 0 : 4)}</dd>
          </div>
        ))}
      </dl>

      <h3 className="report-subheading">Положение относительно bounds</h3>
      <ul className="report-prox-list">
        {proxOrder.map((key) => {
          const info = o.proximity?.[key];
          return (
            <li key={key}>
              <span className="report-prox-key">{key}</span>
              <span className="report-prox-val">
                {fmtNum(info?.value, key.startsWith("N") ? 0 : 4)}
                {info?.lo != null && info?.hi != null
                  ? ` ∈ [${fmtNum(info.lo, key.startsWith("N") ? 0 : 4)}, ${fmtNum(info.hi, key.startsWith("N") ? 0 : 4)}]`
                  : ""}
              </span>
              <Badge info={info} />
            </li>
          );
        })}
      </ul>

      {chartData.length > 0 ? (
        <div className="report-chart">
          <h3 className="report-subheading">Objective vs Trial</h3>
          <div className="report-chart-inner">
            <ResponsiveContainer width="100%" height={260}>
              <LineChart data={chartData} margin={{ top: 8, right: 12, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#e4e8ed" strokeDasharray="3 3" />
                <XAxis
                  dataKey="trial"
                  tick={{ fontSize: 11, fill: "#5c6570" }}
                  label={{ value: "Trial", position: "insideBottom", offset: -2, fontSize: 11 }}
                />
                <YAxis
                  tick={{ fontSize: 11, fill: "#5c6570" }}
                  width={48}
                  domain={["auto", "auto"]}
                />
                <Tooltip
                  contentStyle={{
                    fontSize: 12,
                    border: "1px solid #d8dde3",
                    borderRadius: 0,
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="objective"
                  stroke="#0c1c33"
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      ) : null}

      {(o.top10 ?? []).length > 0 ? (
        <div className="report-table-wrap">
          <h3 className="report-subheading">Top 10 trials</h3>
          <table className="report-table">
            <thead>
              <tr>
                <th>Trial</th>
                <th>Objective</th>
                <th>RMSE</th>
                <th>MAE</th>
                <th>Coverage</th>
                <th>R maj</th>
                <th>Nmin</th>
                <th>Nmax</th>
              </tr>
            </thead>
            <tbody>
              {(o.top10 ?? []).map((t) => (
                <tr
                  key={String(t.trial)}
                  className={
                    t.trial === o.best_trial_number ? "is-best" : undefined
                  }
                >
                  <td>{fmtNum(t.trial, 0)}</td>
                  <td>{fmtNum(t.objective)}</td>
                  <td>{fmtNum(t.CV_RMSE)}</td>
                  <td>{fmtNum(t.CV_MAE)}</td>
                  <td>{fmtNum(t.prediction_coverage)}</td>
                  <td>{fmtNum(t.R_major)}</td>
                  <td>{fmtNum(t.Nmin, 0)}</td>
                  <td>{fmtNum(t.Nmax, 0)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </section>
  );
}
