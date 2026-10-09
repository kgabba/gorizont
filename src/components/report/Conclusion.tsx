"use client";

import { buildConclusion } from "@/lib/reportConclusion";
import type { AnalyzeReportData } from "@/lib/reportTypes";

export default function Conclusion({ data }: { data: AnalyzeReportData }) {
  const model = buildConclusion(data.conclusion_inputs);
  return (
    <section id="report-conclusion" className="report-section">
      <h2 className="report-section-title">Рекомендуемые параметры</h2>
      <p className="report-rec-lead">{model.lead}</p>

      <div className="report-rec-grid">
        {model.groups.map((g) => (
          <article key={g.title} className="report-rec-card">
            <h3 className="report-rec-card-title">{g.title}</h3>
            <dl className="report-rec-list">
              {g.items.map((it) => (
                <div key={it.label} className="report-rec-row">
                  <dt>{it.label}</dt>
                  <dd>{it.value}</dd>
                </div>
              ))}
            </dl>
          </article>
        ))}
      </div>

      <aside
        className={`report-rec-notes${model.hasWarn ? " is-warn" : ""}`}
      >
        <h3 className="report-rec-notes-title">Примечание</h3>
        <ul>
          {model.notes.map((n) => (
            <li key={n}>{n}</li>
          ))}
        </ul>
      </aside>
    </section>
  );
}
