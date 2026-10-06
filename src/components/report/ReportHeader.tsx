"use client";

import type { AnalyzeReportData } from "@/lib/reportTypes";
import { fmtDate, fmtNum } from "@/lib/reportTypes";

export default function ReportHeader({ data }: { data: AnalyzeReportData }) {
  const { header, kpi } = data;
  return (
    <header id="report-summary" className="report-header">
      <p className="report-brand">ГОРИЗОНТ</p>
      <h1 className="report-title">Отчёт по настройке пространственной оценки</h1>

      <dl className="report-meta">
        <div>
          <dt>Run ID</dt>
          <dd className="report-mono">{header.run_id}</dd>
        </div>
        <div>
          <dt>Файл</dt>
          <dd>{header.filename ?? "—"}</dd>
        </div>
        <div>
          <dt>Точек</dt>
          <dd>{fmtNum(header.n_points, 0)}</dd>
        </div>
        <div>
          <dt>HoleID</dt>
          <dd>
            {header.has_hole_id === true
              ? "present"
              : header.has_hole_id === false
                ? "absent"
                : "—"}
          </dd>
        </div>
        <div>
          <dt>Тип CV</dt>
          <dd>{header.cv_method ?? "—"}</dd>
        </div>
        <div>
          <dt>Дата расчёта</dt>
          <dd>{fmtDate(header.created_at)}</dd>
        </div>
      </dl>

      <div className="report-kpi">
        <div className="report-kpi-item">
          <span className="report-kpi-label">CV RMSE</span>
          <span className="report-kpi-value">{fmtNum(kpi.CV_RMSE)}</span>
        </div>
        <div className="report-kpi-item">
          <span className="report-kpi-label">CV MAE</span>
          <span className="report-kpi-value">{fmtNum(kpi.CV_MAE)}</span>
        </div>
        <div className="report-kpi-item">
          <span className="report-kpi-label">Prediction coverage</span>
          <span className="report-kpi-value">
            {fmtNum(kpi.prediction_coverage)}
          </span>
        </div>
      </div>
    </header>
  );
}
