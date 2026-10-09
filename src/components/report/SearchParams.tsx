"use client";

import { PARAM_LABELS } from "@/lib/reportLabels";
import type { AnalyzeReportData } from "@/lib/reportTypes";
import { fmtNum } from "@/lib/reportTypes";

export default function SearchParams({ data }: { data: AnalyzeReportData }) {
  const p = data.search_params;
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
        анизотропии. Размер области поиска подбирается отдельно по
        пространственной кросс-валидации.
      </p>

      <dl className="report-grid">
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
        <div>
          <dt>{PARAM_LABELS.Nmin}</dt>
          <dd>{fmtNum(p.Nmin, 0)}</dd>
        </div>
        <div>
          <dt>{PARAM_LABELS.Nmax}</dt>
          <dd>{fmtNum(p.Nmax, 0)}</dd>
        </div>
      </dl>
    </section>
  );
}
