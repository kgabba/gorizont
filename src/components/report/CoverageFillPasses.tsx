"use client";

import { PARAM_LABELS, sectorCountLabel } from "@/lib/reportLabels";
import type { AnalyzeReportData } from "@/lib/reportTypes";
import { fmtNum, fmtPct } from "@/lib/reportTypes";

function lossLabel(loss: string | null | undefined, isIdw: boolean): string {
  if (loss === "j") return "J (ошибка + штраф за неоценённые)";
  if (loss === "rmse") return "RMSE с условием покрытия";
  return isIdw ? "J / RMSE" : "RMSE";
}

export default function CoverageFillPasses({
  data,
}: {
  data: AnalyzeReportData;
}) {
  const fill = data.coverage_fill;
  if (!fill?.enabled) return null;

  const isIdw = (data.header.method || "") === "anisotropic_idw";
  const passes = fill.passes ?? [];
  const triggered = Boolean(fill.triggered);

  return (
    <section id="report-passes" className="report-section">
      <h2 className="report-section-title">Проходы подбора области поиска</h2>

      <div
        className={`report-pass-banner${triggered ? " is-two" : " is-one"}`}
        role="status"
      >
        <p className="report-pass-banner-title">
          {triggered
            ? "Два прохода: основной подбор + дозаполнение покрытия"
            : "Один проход: покрытие после основного подбора достаточное"}
        </p>
        <p className="report-pass-banner-text">
          {fill.summary_ru ||
            (triggered
              ? "Сначала подбирается локальная область поиска, затем неоценённые точки добиваются вторым проходом с RMSE и требованием покрытия 100%."
              : "Второй проход не запускался.")}
        </p>
        {triggered ? (
          <dl className="report-pass-banner-meta">
            <div>
              <dt>Покрытие после прохода 1</dt>
              <dd>{fmtPct(fill.pass1_coverage, 1)}</dd>
            </div>
            <div>
              <dt>Итоговое покрытие</dt>
              <dd>{fmtPct(fill.pass2_coverage ?? data.kpi.prediction_coverage, 1)}</dd>
            </div>
            <div>
              <dt>Точек по проходам</dt>
              <dd>
                {fmtNum(fill.n_by_pass?.["1"], 0)} / {fmtNum(fill.n_by_pass?.["2"], 0)}
              </dd>
            </div>
          </dl>
        ) : null}
      </div>

      <div className={`report-pass-grid${triggered ? " is-two" : ""}`}>
        {passes.map((p) => {
          const isFill = p.role === "fill" || p.pass === 2;
          return (
            <article
              key={String(p.pass ?? p.label)}
              className={`report-pass-card${isFill ? " is-fill" : " is-primary"}`}
            >
              <header className="report-pass-card-head">
                <h3 className="report-pass-card-title">
                  {p.label || `Проход ${p.pass}`}
                </h3>
                <span className="report-pass-pill">
                  {isFill ? "дозаполнение" : "основной"}
                </span>
              </header>
              <p className="report-pass-card-loss">
                Целевая функция: {lossLabel(p.loss, isIdw)}
              </p>
              {(() => {
                const sectorOn = Boolean(p.sector_search ?? p.octant_search);
                return (
                  <p className="report-pass-card-loss">
                    Секторный поиск:{" "}
                    <strong>
                      {sectorOn
                        ? `включён (${sectorCountLabel(p.n_sectors)})`
                        : "выключен"}
                    </strong>
                  </p>
                );
              })()}
              <dl className="report-grid report-pass-dl">
                {isIdw ? (
                  <div>
                    <dt>{PARAM_LABELS.power}</dt>
                    <dd>{fmtNum(p.power, 3)}</dd>
                  </div>
                ) : null}
                <div>
                  <dt>{PARAM_LABELS.R_major}</dt>
                  <dd>{fmtNum(p.R_major, 0)}</dd>
                </div>
                <div>
                  <dt>{PARAM_LABELS.R_inter}</dt>
                  <dd>{fmtNum(p.R_inter, 0)}</dd>
                </div>
                <div>
                  <dt>{PARAM_LABELS.R_minor}</dt>
                  <dd>{fmtNum(p.R_minor, 0)}</dd>
                </div>
                {Boolean(p.sector_search ?? p.octant_search) ? (
                  <>
                    <div>
                      <dt>{PARAM_LABELS.MINOCT}</dt>
                      <dd>{fmtNum(p.MINOCT, 0)}</dd>
                    </div>
                    <div>
                      <dt>{PARAM_LABELS.MINPEROC}</dt>
                      <dd>{fmtNum(p.MINPEROC, 0)}</dd>
                    </div>
                    <div>
                      <dt>{PARAM_LABELS.MAXPEROC}</dt>
                      <dd>{fmtNum(p.MAXPEROC, 0)}</dd>
                    </div>
                  </>
                ) : null}
                <div>
                  <dt>{PARAM_LABELS.Nmin}</dt>
                  <dd>{fmtNum(p.Nmin, 0)}</dd>
                </div>
                <div>
                  <dt>{PARAM_LABELS.Nmax}</dt>
                  <dd>{fmtNum(p.Nmax, 0)}</dd>
                </div>
                <div>
                  <dt>Покрытие</dt>
                  <dd>{fmtPct(p.prediction_coverage, 1)}</dd>
                </div>
                <div>
                  <dt>{isFill ? "RMSE (новые точки)" : "RMSE (OOF)"}</dt>
                  <dd>
                    {fmtNum(
                      isFill
                        ? (p.local_CV_RMSE ?? p.CV_RMSE)
                        : p.CV_RMSE,
                      3,
                    )}
                  </dd>
                </div>
                {isFill && p.n_new != null ? (
                  <div>
                    <dt>Добавлено точек</dt>
                    <dd>{fmtNum(p.n_new, 0)}</dd>
                  </div>
                ) : null}
              </dl>
            </article>
          );
        })}
      </div>
    </section>
  );
}
