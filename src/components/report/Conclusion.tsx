"use client";

import { buildConclusion } from "@/lib/reportConclusion";
import type { AnalyzeReportData } from "@/lib/reportTypes";

export default function Conclusion({ data }: { data: AnalyzeReportData }) {
  const blocks = buildConclusion(data.conclusion_inputs);
  return (
    <section id="report-conclusion" className="report-section">
      <h2 className="report-section-title">Инженерное заключение</h2>
      <div className="report-conclusion">
        {blocks.map((b) => (
          <article
            key={b.title}
            className={`report-conclusion-block${b.tone === "warn" ? " is-warn" : ""}`}
          >
            <h3>{b.title}</h3>
            <p>{b.body}</p>
          </article>
        ))}
      </div>
    </section>
  );
}
