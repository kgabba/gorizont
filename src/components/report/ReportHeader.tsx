"use client";

import type { AnalyzeReportData } from "@/lib/reportTypes";
import { fmtDate, fmtNum, fmtPct } from "@/lib/reportTypes";

export default function ReportHeader({ data }: { data: AnalyzeReportData }) {
  const { header, kpi } = data;
  return (
    <header id="report-summary" className="report-header">
      <h1 className="report-title">
        Отчёт по настройке интерполяции блочной модели
      </h1>

      <dl className="report-meta">
        <div>
          <dt>Идентификатор расчёта</dt>
          <dd className="report-mono">{header.run_id}</dd>
        </div>
        <div>
          <dt>Файл</dt>
          <dd>{header.filename ?? "—"}</dd>
        </div>
        <div>
          <dt>Количество точек</dt>
          <dd>{fmtNum(header.n_points, 0)}</dd>
        </div>
        <div>
          <dt>Идентификатор скважины (HoleID)</dt>
          <dd>
            {header.has_hole_id === true
              ? "есть"
              : header.has_hole_id === false
                ? "нет"
                : "—"}
          </dd>
        </div>
        <div>
          <dt>Схема кросс-валидации</dt>
          <dd>{header.cv_method ?? "—"}</dd>
        </div>
        <div>
          <dt>Дата расчёта</dt>
          <dd>{fmtDate(header.created_at)}</dd>
        </div>
      </dl>

      <div className="report-kpi">
        <div className="report-kpi-item">
          <span className="report-kpi-label">RMSE на валидационных данных</span>
          <span className="report-kpi-value">{fmtNum(kpi.CV_RMSE, 2)}</span>
        </div>
        <div className="report-kpi-item">
          <span className="report-kpi-label">MAE на валидационных данных</span>
          <span className="report-kpi-value">{fmtNum(kpi.CV_MAE, 2)}</span>
        </div>
        <div className="report-kpi-item">
          <span className="report-kpi-label">Процент успешных оценок</span>
          <span className="report-kpi-value">
            {fmtPct(kpi.prediction_coverage)}
          </span>
        </div>
      </div>
    </header>
  );
}
