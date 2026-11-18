"use client";

import {
  METHOD_LABELS,
  PARAM_LABELS,
  sectorCountLabel,
} from "@/lib/reportLabels";
import type { AnalyzeReportData } from "@/lib/reportTypes";
import { fmtNum } from "@/lib/reportTypes";

export default function SearchParams({ data }: { data: AnalyzeReportData }) {
  const p = data.search_params;
  const isIdw = (p.method || data.header.method) === "anisotropic_idw";
  const methodLabel =
    data.header.method_label ||
    METHOD_LABELS[p.method || data.header.method || ""] ||
    "—";
  const sectorOn = Boolean(p.sector_search ?? p.octant_search);

  return (
    <section id="report-search" className="report-section">
      <h2 className="report-section-title">Область поиска</h2>

      <div className="report-compare">
        <div className="report-compare-item">Эллипсоид вариограммы</div>
        <div className="report-compare-neq">≠</div>
        <div className="report-compare-item">Область поиска</div>
      </div>
      <p className="report-section-note">
        Параметры пространственной структуры фиксируются на этапе анализа
        анизотропии. Размер области поиска
        {isIdw ? " и степень IDW" : ""} подбирается отдельно по
        пространственной кросс-валидации. Метод: <strong>{methodLabel}</strong>.
        {data.coverage_fill?.triggered
          ? " Ниже — параметры основного (1-го) прохода; параметры дозаполнения см. в разделе «Проходы подбора»."
          : null}
      </p>

      <div
        className={`report-pass-banner${sectorOn ? " is-two" : " is-one"}`}
        role="status"
      >
        <p className="report-pass-banner-title">
          {sectorOn
            ? `Секторный поиск включён — ${sectorCountLabel(p.n_sectors)}`
            : "Секторный поиск выключен"}
        </p>
        <p className="report-pass-banner-text">
          {sectorOn
            ? "Пробы отбираются по секторам внутри эллипсоида поиска (как в Micromine: 4 / 8 / 16). Общие Nmin/Nmax заданы секторными ограничениями."
            : "Используется обычный эллипсоид поиска с глобальными Nmin / Nmax без секторных лимитов."}
        </p>
      </div>

      <dl className="report-grid">
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
        {sectorOn ? (
          <>
            <div>
              <dt>{PARAM_LABELS.n_sectors}</dt>
              <dd>{sectorCountLabel(p.n_sectors)}</dd>
            </div>
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
            <div>
              <dt>{PARAM_LABELS.Nmin}</dt>
              <dd>
                {fmtNum(p.Nmin, 0)}
                <span className="report-section-note"> (из секторов)</span>
              </dd>
            </div>
            <div>
              <dt>{PARAM_LABELS.Nmax}</dt>
              <dd>
                {fmtNum(p.Nmax, 0)}
                <span className="report-section-note"> (из секторов)</span>
              </dd>
            </div>
          </>
        ) : (
          <>
            <div>
              <dt>{PARAM_LABELS.Nmin}</dt>
              <dd>{fmtNum(p.Nmin, 0)}</dd>
            </div>
            <div>
              <dt>{PARAM_LABELS.Nmax}</dt>
              <dd>{fmtNum(p.Nmax, 0)}</dd>
            </div>
          </>
        )}
        {isIdw && p.mean_n_used != null ? (
          <div>
            <dt>{PARAM_LABELS.mean_n_used}</dt>
            <dd>{fmtNum(p.mean_n_used, 1)}</dd>
          </div>
        ) : null}
      </dl>
    </section>
  );
}
