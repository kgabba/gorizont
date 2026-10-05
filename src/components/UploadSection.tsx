"use client";

import { useCallback, useEffect, useState } from "react";
import FileUpload from "./FileUpload";

type AnalyzeStatus = "idle" | "ready" | "running" | "done" | "error";

type AnalyzeResponse = {
  run_id: string;
  status: string;
  input: { n_points: number; has_hole_id: boolean };
  anisotropy: {
    type?: string | null;
    variogram_model?: string | null;
    nugget?: number | null;
    sill?: number | null;
    range_major?: number | null;
    range_intermediate?: number | null;
    range_minor?: number | null;
    orientation_matrix?: number[][] | null;
    major_axis_xyz?: number[] | null;
    intermediate_axis_xyz?: number[] | null;
    minor_axis_xyz?: number[] | null;
    moi_strength?: number | null;
  };
  optimization: {
    R_major?: number | null;
    R_inter?: number | null;
    R_minor?: number | null;
    K_inter?: number | null;
    K_minor?: number | null;
    Nmin?: number | null;
    Nmax?: number | null;
    CV_RMSE?: number | null;
    CV_MAE?: number | null;
    prediction_coverage?: number | null;
    best_trial_number?: number | null;
  };
  detail?: string;
};

const STEPS = [
  "Загрузка данных",
  "Анализ пространственной структуры",
  "Анизотропия и вариография",
  "Оптимизация параметров",
  "Готово",
] as const;

function fmt(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  return Number(v).toLocaleString("ru-RU", {
    maximumFractionDigits: digits,
  });
}

export default function UploadSection() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<AnalyzeStatus>("idle");
  const [stepIdx, setStepIdx] = useState(0);
  const [result, setResult] = useState<AnalyzeResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (status !== "running") return;
    setStepIdx(0);
    const timers = [
      window.setTimeout(() => setStepIdx(1), 400),
      window.setTimeout(() => setStepIdx(2), 1200),
      window.setTimeout(() => setStepIdx(3), 2400),
    ];
    return () => timers.forEach(clearTimeout);
  }, [status]);

  const onFileSelect = useCallback((f: File) => {
    setFile(f);
    setStatus("ready");
    setResult(null);
    setError(null);
    setStepIdx(0);
  }, []);

  const runAnalyze = useCallback(async () => {
    if (!file) return;
    setStatus("running");
    setError(null);
    setResult(null);

    const body = new FormData();
    body.append("file", file);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body,
      });

      let data: AnalyzeResponse | { detail?: string } | null = null;
      try {
        data = (await res.json()) as AnalyzeResponse;
      } catch {
        data = null;
      }

      if (!res.ok) {
        const detail =
          data && typeof data === "object" && "detail" in data
            ? String(data.detail)
            : `Ошибка ${res.status}`;
        throw new Error(detail);
      }

      const ok = data as AnalyzeResponse;
      if (!ok?.run_id || !ok.anisotropy || !ok.optimization) {
        throw new Error("Неполный ответ сервера");
      }

      setStepIdx(4);
      setResult(ok);
      setStatus("done");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Неизвестная ошибка");
      setStatus("error");
    }
  }, [file]);

  return (
    <section id="analyze" className="upload-section">
      <div className="upload-section-inner">
        <h2 className="upload-section-title">Загрузите данные</h2>
        <p className="upload-section-lead">
          CSV для анизотропии и подбора параметров Ordinary Kriging
        </p>

        <div className="upload-section-drop">
          <FileUpload
            onFileSelect={onFileSelect}
            disabled={status === "running"}
          />
        </div>

        <p className="upload-section-hint">
          X · Y · Z · Value
          <span className="upload-section-hint-opt"> · HoleID optional</span>
        </p>

        {file && status !== "running" ? (
          <div className="upload-section-actions">
            <button
              type="button"
              className="upload-run-btn"
              onClick={runAnalyze}
            >
              Запустить анализ
            </button>
          </div>
        ) : null}

        {status === "running" ? (
          <ol className="upload-steps">
            {STEPS.map((label, i) => (
              <li
                key={label}
                className={`upload-step${
                  i < stepIdx ? " is-done" : i === stepIdx ? " is-active" : ""
                }`}
              >
                {label}
              </li>
            ))}
          </ol>
        ) : null}

        {status === "error" && error ? (
          <p className="upload-section-error">{error}</p>
        ) : null}

        {status === "done" && result ? (
          <div className="analyze-result">
            <p className="analyze-result-meta">
              Точек: {result.input.n_points}
              {result.input.has_hole_id ? " · HoleID" : ""}
            </p>

            <div className="analyze-result-block">
              <h3 className="analyze-result-heading">Анизотропия</h3>
              <dl className="analyze-result-grid">
                <div>
                  <dt>Major range</dt>
                  <dd>{fmt(result.anisotropy.range_major)}</dd>
                </div>
                <div>
                  <dt>Intermediate range</dt>
                  <dd>{fmt(result.anisotropy.range_intermediate)}</dd>
                </div>
                <div>
                  <dt>Minor range</dt>
                  <dd>{fmt(result.anisotropy.range_minor)}</dd>
                </div>
                <div>
                  <dt>Nugget</dt>
                  <dd>{fmt(result.anisotropy.nugget)}</dd>
                </div>
                <div>
                  <dt>Sill</dt>
                  <dd>{fmt(result.anisotropy.sill)}</dd>
                </div>
                <div>
                  <dt>Variogram model</dt>
                  <dd>{result.anisotropy.variogram_model ?? "—"}</dd>
                </div>
              </dl>
            </div>

            <div className="analyze-result-block">
              <h3 className="analyze-result-heading">Оптимизация</h3>
              <dl className="analyze-result-grid">
                <div>
                  <dt>R major</dt>
                  <dd>{fmt(result.optimization.R_major)}</dd>
                </div>
                <div>
                  <dt>R inter</dt>
                  <dd>{fmt(result.optimization.R_inter)}</dd>
                </div>
                <div>
                  <dt>R minor</dt>
                  <dd>{fmt(result.optimization.R_minor)}</dd>
                </div>
                <div>
                  <dt>Nmin</dt>
                  <dd>{fmt(result.optimization.Nmin, 0)}</dd>
                </div>
                <div>
                  <dt>Nmax</dt>
                  <dd>{fmt(result.optimization.Nmax, 0)}</dd>
                </div>
              </dl>
            </div>

            <div className="analyze-result-block">
              <h3 className="analyze-result-heading">Валидация</h3>
              <dl className="analyze-result-grid">
                <div>
                  <dt>CV RMSE</dt>
                  <dd>{fmt(result.optimization.CV_RMSE)}</dd>
                </div>
                <div>
                  <dt>CV MAE</dt>
                  <dd>{fmt(result.optimization.CV_MAE)}</dd>
                </div>
                <div>
                  <dt>Coverage</dt>
                  <dd>{fmt(result.optimization.prediction_coverage)}</dd>
                </div>
              </dl>
            </div>
          </div>
        ) : null}
      </div>
    </section>
  );
}
