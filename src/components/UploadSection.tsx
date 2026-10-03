"use client";

import { useCallback, useState } from "react";
import FileUpload from "./FileUpload";

type TuneStatus = "idle" | "uploading" | "done" | "error";

type TuneResponse = {
  report_markdown?: string;
  detail?: string;
};

export default function UploadSection() {
  const [status, setStatus] = useState<TuneStatus>("idle");
  const [report, setReport] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleFile = useCallback(async (file: File) => {
    setStatus("uploading");
    setReport(null);
    setError(null);

    const body = new FormData();
    body.append("file", file);
    body.append("cv_method", "spatial_block");

    try {
      const res = await fetch("/api/tune/csv", {
        method: "POST",
        body,
      });

      let data: TuneResponse | null = null;
      try {
        data = (await res.json()) as TuneResponse;
      } catch {
        data = null;
      }

      if (!res.ok) {
        const detail =
          typeof data?.detail === "string"
            ? data.detail
            : `Ошибка ${res.status}`;
        throw new Error(detail);
      }

      if (!data?.report_markdown) {
        throw new Error("Ответ API без report_markdown");
      }

      setReport(data.report_markdown);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Неизвестная ошибка");
      setStatus("error");
    }
  }, []);

  return (
    <section id="analyze" className="upload-section">
      <div className="upload-section-inner">
        <h2 className="upload-section-title">Загрузите данные</h2>
        <p className="upload-section-lead">
          Пространственные данные для анализа и настройки параметров кригинга
        </p>

        <div className="upload-section-drop">
          <FileUpload
            onFileSelect={handleFile}
            disabled={status === "uploading"}
          />
        </div>

        <p className="upload-section-hint">X · Y · Z · Grade · Domain</p>

        {status === "uploading" && (
          <p className="upload-section-status">Идёт анализ…</p>
        )}

        {status === "error" && error && (
          <p className="upload-section-error">{error}</p>
        )}

        {status === "done" && report && (
          <pre className="upload-section-report">{report}</pre>
        )}
      </div>
    </section>
  );
}
