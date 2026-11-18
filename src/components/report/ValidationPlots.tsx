"use client";

import { useState } from "react";
import type { AnalyzeReportData } from "@/lib/reportTypes";
import { artifactUrl } from "@/lib/reportTypes";

function ArtifactImage({
  runId,
  path,
  label,
}: {
  runId: string;
  path: string;
  label: string;
}) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return (
    <figure className="report-figure">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={artifactUrl(runId, path)}
        alt={label}
        className="report-figure-img"
        onError={() => setFailed(true)}
      />
      <figcaption>{label}</figcaption>
    </figure>
  );
}

export default function ValidationPlots({ data }: { data: AnalyzeReportData }) {
  const images = data.artifacts?.validation_images ?? [];
  if (images.length === 0) return null;

  return (
    <section id="report-validation" className="report-section">
      <h2 className="report-section-title">Проверка качества (OOF)</h2>
      <p className="report-section-note">
        Графики по out-of-fold прогнозам лучшей модели: scatter / QQ и swath
        (средние содержания вдоль осей координат).
        {data.coverage_fill?.triggered
          ? " При двух проходах на графиках объединены оценки pass 1 и pass 2 (точки pass 1 не перезаписывались)."
          : null}
      </p>
      <div className="report-figures">
        {images.map((img) => (
          <ArtifactImage
            key={img.path}
            runId={data.run_id}
            path={img.path}
            label={img.label}
          />
        ))}
      </div>
    </section>
  );
}
