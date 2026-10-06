"use client";

import type { AnalyzeReportData } from "@/lib/reportTypes";
import { fmtNum } from "@/lib/reportTypes";

export default function SearchParams({ data }: { data: AnalyzeReportData }) {
  const p = data.search_params;
  return (
    <section id="report-search" className="report-section">
      <h2 className="report-section-title">Параметры поиска</h2>

      <div className="report-compare">
        <div className="report-compare-item">Variogram ellipsoid</div>
        <div className="report-compare-neq">≠</div>
        <div className="report-compare-item">Search neighbourhood</div>
      </div>
      <p className="report-section-note">
        Параметры пространственной структуры фиксируются на этапе анализа
        анизотропии. Размер поискового соседства подбирается отдельно по
        пространственной кросс-валидации.
      </p>

      <dl className="report-grid">
        <div>
          <dt>R major</dt>
          <dd>{fmtNum(p.R_major)}</dd>
        </div>
        <div>
          <dt>R inter</dt>
          <dd>{fmtNum(p.R_inter)}</dd>
        </div>
        <div>
          <dt>R minor</dt>
          <dd>{fmtNum(p.R_minor)}</dd>
        </div>
        <div>
          <dt>K inter</dt>
          <dd>{fmtNum(p.K_inter)}</dd>
        </div>
        <div>
          <dt>K minor</dt>
          <dd>{fmtNum(p.K_minor)}</dd>
        </div>
        <div>
          <dt>Nmin</dt>
          <dd>{fmtNum(p.Nmin, 0)}</dd>
        </div>
        <div>
          <dt>Nmax</dt>
          <dd>{fmtNum(p.Nmax, 0)}</dd>
        </div>
      </dl>
    </section>
  );
}
