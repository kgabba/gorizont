"use client";

import { useEffect, useState } from "react";
import type { AnalyzeReportData } from "@/lib/reportTypes";
import Conclusion from "./Conclusion";
import CoverageFillPasses from "./CoverageFillPasses";
import ReportHeader from "./ReportHeader";
import SearchParams from "./SearchParams";
import SpatialStructure from "./SpatialStructure";
import TechnicalDetails from "./TechnicalDetails";
import ValidationPlots from "./ValidationPlots";

const NAV = [
  { id: "report-summary", label: "Сводка", short: "Сводка" },
  { id: "report-validation", label: "Проверка OOF", short: "OOF" },
  {
    id: "report-spatial",
    label: "Вариограмма и анизотропия",
    short: "Вариограмма",
  },
  { id: "report-passes", label: "Проходы подбора", short: "Проходы" },
  { id: "report-search", label: "Область поиска", short: "Поиск" },
  { id: "report-tech", label: "Технические детали", short: "Детали" },
  {
    id: "report-conclusion",
    label: "Рекомендуемые параметры",
    short: "Параметры",
  },
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

  const navItems = NAV.filter(
    (item) => item.id !== "report-passes" || Boolean(data.coverage_fill?.enabled),
  );

  return (
    <div className="report-page">
      <nav className="report-nav" aria-label="Разделы отчёта">
        <div className="report-nav-inner">
          <a href="/" className="report-nav-brand">
            Горизонт
          </a>
          <ul>
            {navItems.map((item) => (
              <li key={item.id}>
                <a href={`#${item.id}`}>
                  <span className="report-nav-label-full">{item.label}</span>
                  <span className="report-nav-label-short">{item.short}</span>
                </a>
              </li>
            ))}
          </ul>
        </div>
      </nav>

      <div className="report-body">
        <ReportHeader data={data} />
        <ValidationPlots data={data} />
        <SpatialStructure data={data} />
        <CoverageFillPasses data={data} />
        <SearchParams data={data} />
        <TechnicalDetails data={data} />
        <Conclusion data={data} />

        <p className="report-footer-link">
          <a href="/#analyze" className="report-footer-btn">
            Новый анализ
          </a>
        </p>
      </div>
    </div>
  );
}
