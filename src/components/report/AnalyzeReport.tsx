"use client";

import { useEffect, useState } from "react";
import type { AnalyzeReportData } from "@/lib/reportTypes";
import Conclusion from "./Conclusion";
import Optimization from "./Optimization";
import ReportHeader from "./ReportHeader";
import SearchParams from "./SearchParams";
import SpatialStructure from "./SpatialStructure";
import Validation from "./Validation";

const NAV = [
  { id: "report-summary", label: "Итог" },
  { id: "report-spatial", label: "Пространственная структура" },
  { id: "report-search", label: "Параметры поиска" },
  { id: "report-validation", label: "Валидация" },
  { id: "report-optimization", label: "Оптимизация" },
  { id: "report-conclusion", label: "Заключение" },
] as const;

export default function AnalyzeReport({ runId }: { runId: string }) {
  const [data, setData] = useState<AnalyzeReportData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`/api/analysis/${encodeURIComponent(runId)}`)
      .then(async (res) => {
        const body = await res.json().catch(() => null);
        if (!res.ok) {
          const detail =
            body && typeof body === "object" && "detail" in body
              ? String(body.detail)
              : `Ошибка ${res.status}`;
          throw new Error(detail);
        }
        return body as AnalyzeReportData;
      })
      .then((report) => {
        if (!cancelled) setData(report);
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Не удалось загрузить отчёт");
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [runId]);

  if (loading) {
    return (
      <div className="report-page">
        <p className="report-loading">Загрузка отчёта…</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="report-page">
        <p className="report-error">{error ?? "Отчёт недоступен"}</p>
        <a className="report-back" href="/#analyze">
          Вернуться к загрузке
        </a>
      </div>
    );
  }

  return (
    <div className="report-page">
      <nav className="report-nav" aria-label="Разделы отчёта">
        <ul>
          {NAV.map((item) => (
            <li key={item.id}>
              <a href={`#${item.id}`}>{item.label}</a>
            </li>
          ))}
        </ul>
      </nav>

      <div className="report-body">
        <ReportHeader data={data} />
        <SpatialStructure data={data} />
        <SearchParams data={data} />
        <Validation data={data} />
        <Optimization data={data} />
        <Conclusion data={data} />

        <p className="report-footer-link">
          <a href="/#analyze">Новый анализ</a>
        </p>
      </div>
    </div>
  );
}
