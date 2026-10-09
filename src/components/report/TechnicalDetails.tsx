"use client";

import { useState, type ReactNode } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { PARAM_LABELS, paramLabel } from "@/lib/reportLabels";
import type { AnalyzeReportData, ProximityInfo } from "@/lib/reportTypes";
import { fmtNum, fmtPct } from "@/lib/reportTypes";
import Hint from "./Hint";

function Badge({ info }: { info?: ProximityInfo }) {
  if (!info) return <span className="report-badge">—</span>;
  const near = info.status === "near_bound";
  return (
    <span className={`report-badge${near ? " is-warn" : " is-ok"}`}>
      {info.label}
    </span>
  );
}

function LabelWithHint({
  children,
  hint,
}: {
  children: ReactNode;
  hint: string;
}) {
  return (
    <>
      {children} <Hint text={hint} />
    </>
  );
}

export default function TechnicalDetails({ data }: { data: AnalyzeReportData }) {
  const [open, setOpen] = useState(false);
  const o = data.optimization;
  const p = data.search_params;
  const trials = o.trials ?? [];
  const chartData = trials
    .filter((t) => t.trial != null && t.objective != null)
    .map((t) => ({
      trial: t.trial as number,
      objective: t.objective as number,
    }));

  const proxOrder = ["R_major", "K_inter", "K_minor", "Nmin", "Nmax"] as const;

  return (
    <section
      id="report-tech"
      className={`report-tech${open ? " is-open" : ""}`}
    >
      <button
        type="button"
        className="report-tech-toggle"
        aria-expanded={open}
        onClick={() => setOpen((v) => !v)}
      >
        <span className="report-tech-toggle-main">
          <span className="report-tech-toggle-title">Технические детали</span>
          <span className="report-tech-toggle-hint">
            коэффициенты осей, подбор, график
          </span>
        </span>
        <span className="report-tech-toggle-action">
          {open ? "Свернуть" : "Развернуть"}
          <span className="report-tech-chevron" aria-hidden>
            {open ? "▴" : "▾"}
          </span>
        </span>
      </button>

      {open ? (
        <div className="report-tech-body">
          <p className="report-section-note">
            Служебные параметры машинного обучения: ход автоматического подбора
            области поиска и внутренние коэффициенты оптимизации.
          </p>

          <h3 className="report-subheading">
            <LabelWithHint hint="Отношения радиусов осей 2 и 3 к радиусу оси 1. В подборе варьируются эти коэффициенты; итоговые радиусы R₂ = R₁·K₂, R₃ = R₁·K₃.">
              Коэффициенты осей
            </LabelWithHint>
          </h3>
          <dl className="report-grid">
            <div>
              <dt>
                <LabelWithHint hint="K₂ = R₂ / R₁. Задаёт вытянутость области поиска по оси 2 относительно оси 1.">
                  {PARAM_LABELS.K_inter}
                </LabelWithHint>
              </dt>
              <dd>{fmtNum(p.K_inter, 2)}</dd>
            </div>
            <div>
              <dt>
                <LabelWithHint hint="K₃ = R₃ / R₁. Задаёт вытянутость области поиска по оси 3 относительно оси 1.">
                  {PARAM_LABELS.K_minor}
                </LabelWithHint>
              </dt>
              <dd>{fmtNum(p.K_minor, 2)}</dd>
            </div>
          </dl>

          <h3 className="report-subheading">
            <LabelWithHint hint="Автоматический перебор вариантов области поиска по кросс-валидации. Выбирается вариант с наименьшей ошибкой при достаточном покрытии.">
              Подбор параметров
            </LabelWithHint>
          </h3>
          <dl className="report-grid">
            <div>
              <dt>
                <LabelWithHint hint="Номер итерации, на которой получены итоговые параметры области поиска.">
                  Лучший вариант (№ итерации)
                </LabelWithHint>
              </dt>
              <dd>{fmtNum(o.best_trial_number, 0)}</dd>
            </div>
            <div>
              <dt>
                <LabelWithHint hint="Сколько вариантов области поиска было проверено.">
                  Число итераций подбора
                </LabelWithHint>
              </dt>
              <dd>{fmtNum(o.n_trials, 0)}</dd>
            </div>
            <div>
              <dt>
                <LabelWithHint hint="Значение целевой функции лучшего варианта (штрафованный RMSE). Чем меньше, тем лучше.">
                  Минимум целевой функции
                </LabelWithHint>
              </dt>
              <dd>{fmtNum(o.objective, 2)}</dd>
            </div>
          </dl>

          <h3 className="report-subheading">
            <LabelWithHint hint="Для каждого параметра: диапазон, в котором искали оптимальное значение, и значение, выбранное по итогам подбора.">
              Диапазоны поиска и найденные значения
            </LabelWithHint>
          </h3>
          <ul className="report-prox-list">
            {proxOrder.map((key) => {
              const info = o.proximity?.[key];
              const digits = key.startsWith("N")
                ? 0
                : key.startsWith("K")
                  ? 2
                  : 0;
              return (
                <li key={key} className="report-prox-item">
                  <div className="report-prox-head">
                    <span className="report-prox-key">{paramLabel(key)}</span>
                    <Badge info={info} />
                  </div>
                  <span className="report-prox-val">
                    {fmtNum(info?.value, digits)}
                    {info?.lo != null && info?.hi != null
                      ? ` ∈ [${fmtNum(info.lo, digits)}, ${fmtNum(info.hi, digits)}]`
                      : ""}
                  </span>
                </li>
              );
            })}
          </ul>

          {chartData.length > 0 ? (
            <div className="report-chart">
              <h3 className="report-subheading">
                <LabelWithHint hint="Как менялась целевая функция (ошибка) по ходу перебора. Резкие пики — неудачные варианты; нижняя полка — лучшие.">
                  Целевая функция по итерациям
                </LabelWithHint>
              </h3>
              <div className="report-chart-inner">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart
                    data={chartData}
                    margin={{ top: 8, right: 8, left: 0, bottom: 4 }}
                  >
                    <CartesianGrid stroke="#e4e8ed" strokeDasharray="3 3" />
                    <XAxis
                      dataKey="trial"
                      tick={{ fontSize: 10, fill: "#5c6570" }}
                      label={{
                        value: "№ итерации",
                        position: "insideBottom",
                        offset: -2,
                        fontSize: 10,
                      }}
                    />
                    <YAxis
                      tick={{ fontSize: 10, fill: "#5c6570" }}
                      width={40}
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
              <h3 className="report-subheading">
                <LabelWithHint hint="Десять вариантов с наименьшей целевой функцией. Строка лучшего варианта выделена.">
                  10 лучших вариантов
                </LabelWithHint>
              </h3>
              <table className="report-table">
                <thead>
                  <tr>
                    <th>№</th>
                    <th>Целевая функция</th>
                    <th>СКО</th>
                    <th>MAE</th>
                    <th>%</th>
                    <th>R₁</th>
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
                      <td>{fmtNum(t.objective, 2)}</td>
                      <td>{fmtNum(t.CV_RMSE, 2)}</td>
                      <td>{fmtNum(t.CV_MAE, 2)}</td>
                      <td>{fmtPct(t.prediction_coverage)}</td>
                      <td>{fmtNum(t.R_major, 0)}</td>
                      <td>{fmtNum(t.Nmin, 0)}</td>
                      <td>{fmtNum(t.Nmax, 0)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : null}
        </div>
      ) : null}
      </section>
  );
}
