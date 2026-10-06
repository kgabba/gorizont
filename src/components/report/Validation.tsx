"use client";

import type { AnalyzeReportData } from "@/lib/reportTypes";
import { fmtNum } from "@/lib/reportTypes";

export default function Validation({ data }: { data: AnalyzeReportData }) {
  const v = data.validation;
  return (
    <section id="report-validation" className="report-section">
      <h2 className="report-section-title">Качество валидации</h2>

      <dl className="report-grid">
        <div>
          <dt>CV RMSE</dt>
          <dd>{fmtNum(v.CV_RMSE)}</dd>
        </div>
        <div>
          <dt>CV MAE</dt>
          <dd>{fmtNum(v.CV_MAE)}</dd>
        </div>
        <div>
          <dt>Coverage</dt>
          <dd>{fmtNum(v.prediction_coverage)}</dd>
        </div>
        <div>
          <dt>Target predictions</dt>
          <dd>{fmtNum(v.n_tgt, 0)}</dd>
        </div>
        <div>
          <dt>Valid predictions</dt>
          <dd>{fmtNum(v.n_valid, 0)}</dd>
        </div>
        {v.n_invalid != null ? (
          <div>
            <dt>Invalid predictions</dt>
            <dd>{fmtNum(v.n_invalid, 0)}</dd>
          </div>
        ) : null}
      </dl>

      {v.coverage_below_threshold ? (
        <p className="report-warn">
          Prediction coverage ниже порога{" "}
          {fmtNum(v.coverage_threshold ?? 0.97, 2)}. Рекомендуется проверить
          геометрию поиска и покрытие пространства.
        </p>
      ) : null}

      {!v.oof_available ? (
        <p className="report-section-note">
          Диаграммы Observed vs Predicted и Residual histogram недоступны:
          OOF-предсказания не сохраняются в артефактах текущего прогона.
        </p>
      ) : null}
    </section>
  );
}
