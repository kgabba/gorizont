"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState } from "react";
import FileUpload from "./FileUpload";

type AnalyzeStatus =
  | "idle"
  | "inspecting"
  | "confirm"
  | "running"
  | "error";

type InspectResponse = {
  columns: string[];
  mapping: {
    X: string | null;
    Y: string | null;
    Z: string | null;
    Value: string | null;
    HoleID: string | null;
  };
  confidence: Record<string, string>;
  value_candidates: string[];
  warnings: string[];
  ready: boolean;
  n_points: number;
  detail?: string;
};

type AnalyzeStartResponse = {
  run_id: string;
  status: string;
  detail?: string;
};

type ProgressStatus = {
  run_id: string;
  status: string;
  stage?: string | null;
  stage_label?: string | null;
  trial?: number | null;
  n_trials?: number | null;
  best_objective?: number | null;
  error?: string | null;
};

type MappingState = {
  X: string;
  Y: string;
  Z: string;
  Value: string;
  HoleID: string;
};

const STEPS = [
  { id: "upload", label: "Загрузка данных" },
  { id: "anisotropy", label: "Анализ пространственной структуры" },
  { id: "variogram", label: "Вариограмма и анизотропия" },
  { id: "search", label: "Подбор области поиска" },
  { id: "done", label: "Готово" },
] as const;

const NONE = "";

function stageToStepIdx(stage: string | null | undefined): number {
  switch (stage) {
    case "upload":
      return 0;
    case "anisotropy":
      return 1;
    case "search":
      return 3;
    case "done":
      return 4;
    case "error":
      return 3;
    default:
      return 1;
  }
}

function MappingSelect({
  label,
  value,
  options,
  allowEmpty,
  emptyLabel,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  allowEmpty?: boolean;
  emptyLabel?: string;
  onChange: (v: string) => void;
}) {
  return (
    <label className="upload-map-field">
      <span className="upload-map-label">{label}</span>
      <select
        className="upload-map-select"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {allowEmpty ? (
          <option value={NONE}>{emptyLabel ?? "— не использовать —"}</option>
        ) : (
          <option value={NONE} disabled>
            Выберите колонку
          </option>
        )}
        {options.map((c) => (
          <option key={c} value={c}>
            {c}
          </option>
        ))}
      </select>
    </label>
  );
}

export default function UploadSection() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<AnalyzeStatus>("idle");
  const [stepIdx, setStepIdx] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [inspect, setInspect] = useState<InspectResponse | null>(null);
  const [mapping, setMapping] = useState<MappingState | null>(null);
  const [progress, setProgress] = useState<ProgressStatus | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current != null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => () => stopPolling(), [stopPolling]);

  const onFileSelect = useCallback(async (f: File) => {
    setFile(f);
    setError(null);
    setInspect(null);
    setMapping(null);
    setProgress(null);
    setStepIdx(0);
    setStatus("inspecting");

    const body = new FormData();
    body.append("file", f);
    try {
      const res = await fetch("/api/inspect", { method: "POST", body });
      const data = (await res.json().catch(() => null)) as
        | InspectResponse
        | { detail?: string }
        | null;
      if (!res.ok) {
        const detail =
          data && typeof data === "object" && "detail" in data
            ? String(data.detail)
            : `Ошибка ${res.status}`;
        throw new Error(detail);
      }
      const ok = data as InspectResponse;
      setInspect(ok);
      setMapping({
        X: ok.mapping.X ?? NONE,
        Y: ok.mapping.Y ?? NONE,
        Z: ok.mapping.Z ?? NONE,
        Value: ok.mapping.Value ?? NONE,
        HoleID: ok.mapping.HoleID ?? NONE,
      });
      setStatus("confirm");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Неизвестная ошибка");
      setStatus("error");
    }
  }, []);

  const pollStatus = useCallback(
    async (runId: string) => {
      try {
        const res = await fetch(
          `/api/analysis/${encodeURIComponent(runId)}/status`,
        );
        const data = (await res.json().catch(() => null)) as
          | ProgressStatus
          | { detail?: string }
          | null;
        if (!res.ok) {
          const detail =
            data && typeof data === "object" && "detail" in data
              ? String(data.detail)
              : `Ошибка ${res.status}`;
          throw new Error(detail);
        }
        const prog = data as ProgressStatus;
        setProgress(prog);
        setStepIdx(stageToStepIdx(prog.stage));

        if (prog.status === "completed") {
          stopPolling();
          setStepIdx(4);
          router.push(`/report/${runId}`);
          return;
        }
        if (prog.status === "failed") {
          stopPolling();
          setError(prog.error || "Не удалось выполнить анализ.");
          setStatus("error");
        }
      } catch (err) {
        stopPolling();
        setError(err instanceof Error ? err.message : "Неизвестная ошибка");
        setStatus("error");
      }
    },
    [router, stopPolling],
  );

  const runAnalyze = useCallback(async () => {
    if (!file || !mapping) return;
    if (!mapping.X || !mapping.Y || !mapping.Z || !mapping.Value) {
      setError("Укажите колонки X, Y, Z и Value.");
      setStatus("confirm");
      return;
    }

    stopPolling();
    setStatus("running");
    setError(null);
    setProgress(null);
    setStepIdx(0);

    const body = new FormData();
    body.append("file", file);
    body.append("col_x", mapping.X);
    body.append("col_y", mapping.Y);
    body.append("col_z", mapping.Z);
    body.append("col_value", mapping.Value);
    if (mapping.HoleID) body.append("col_holeid", mapping.HoleID);

    try {
      const res = await fetch("/api/analyze", {
        method: "POST",
        body,
      });

      let data: AnalyzeStartResponse | { detail?: string } | null = null;
      try {
        data = (await res.json()) as AnalyzeStartResponse;
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

      const ok = data as AnalyzeStartResponse;
      if (!ok?.run_id) {
        throw new Error("Неполный ответ сервера");
      }

      setProgress({
        run_id: ok.run_id,
        status: "running",
        stage: "upload",
        stage_label: "Загрузка данных",
      });
      await pollStatus(ok.run_id);
      pollRef.current = window.setInterval(() => {
        void pollStatus(ok.run_id);
      }, 1500);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Неизвестная ошибка");
      setStatus("error");
    }
  }, [file, mapping, pollStatus, stopPolling]);

  const columns = inspect?.columns ?? [];
  const canConfirm =
    !!mapping?.X && !!mapping?.Y && !!mapping?.Z && !!mapping?.Value;

  const trial = progress?.trial ?? 0;
  const nTrials = progress?.n_trials ?? 0;
  const searchActive = progress?.stage === "search" && nTrials > 0;
  const barPct = searchActive
    ? Math.min(100, Math.round((Number(trial) / Number(nTrials)) * 100))
    : progress?.stage === "done"
      ? 100
      : 0;

  return (
    <section id="analyze" className="upload-section">
      <div className="upload-section-inner">
        <h2 className="upload-section-title">Загрузите данные</h2>

        <div className="upload-section-drop">
          <FileUpload
            onFileSelect={onFileSelect}
            disabled={status === "running" || status === "inspecting"}
          />
        </div>

        <p className="upload-section-hint">
          X · Y · Z · Содержание
          <span className="upload-section-hint-opt">
            {" "}
            · HoleID (необяз.)
          </span>
        </p>

        {status === "inspecting" ? (
          <p className="upload-section-status">Определение колонок…</p>
        ) : null}

        {status === "confirm" && mapping && inspect ? (
          <div className="upload-confirm">
            <h3 className="upload-confirm-title">
              Проверьте корректность данных
            </h3>
            <p className="upload-confirm-lead">
              Найдено точек: {inspect.n_points}. Сопоставьте колонки файла с
              полями анализа.
            </p>

            <div className="upload-map-grid">
              <MappingSelect
                label="X (координата)"
                value={mapping.X}
                options={columns}
                onChange={(v) => setMapping({ ...mapping, X: v })}
              />
              <MappingSelect
                label="Y (координата)"
                value={mapping.Y}
                options={columns}
                onChange={(v) => setMapping({ ...mapping, Y: v })}
              />
              <MappingSelect
                label="Z (координата)"
                value={mapping.Z}
                options={columns}
                onChange={(v) => setMapping({ ...mapping, Z: v })}
              />
              <MappingSelect
                label="Содержание (Value)"
                value={mapping.Value}
                options={
                  inspect.value_candidates.length > 0
                    ? [
                        ...inspect.value_candidates,
                        ...columns.filter(
                          (c) => !inspect.value_candidates.includes(c),
                        ),
                      ]
                    : columns
                }
                onChange={(v) => setMapping({ ...mapping, Value: v })}
              />
              <MappingSelect
                label="HoleID (скважина)"
                value={mapping.HoleID}
                options={columns}
                allowEmpty
                emptyLabel="— без HoleID —"
                onChange={(v) => setMapping({ ...mapping, HoleID: v })}
              />
            </div>

            {inspect.warnings.length > 0 ? (
              <ul className="upload-confirm-warnings">
                {inspect.warnings.map((w) => (
                  <li key={w}>{w}</li>
                ))}
              </ul>
            ) : null}

            <div className="upload-section-actions">
              <button
                type="button"
                className="upload-run-btn"
                disabled={!canConfirm}
                onClick={runAnalyze}
              >
                Подтвердить и запустить анализ
              </button>
            </div>
          </div>
        ) : null}

        {status === "running" ? (
          <div className="upload-progress">
            <ol className="upload-steps">
              {STEPS.map((step, i) => (
                <li
                  key={step.id}
                  className={`upload-step${
                    i < stepIdx
                      ? " is-done"
                      : i === stepIdx
                        ? " is-active"
                        : ""
                  }`}
                >
                  {step.label}
                </li>
              ))}
            </ol>

            {searchActive ? (
              <div className="upload-trial-progress" aria-live="polite">
                <div className="upload-trial-meta">
                  <span>
                    Итерация {Math.min(Number(trial), Number(nTrials))} из{" "}
                    {nTrials}
                  </span>
                  <span>{barPct}%</span>
                </div>
                <div
                  className="upload-trial-bar"
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={barPct}
                >
                  <div
                    className="upload-trial-bar-fill"
                    style={{ width: `${barPct}%` }}
                  />
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {status === "error" && error ? (
          <p className="upload-section-error">{error}</p>
        ) : null}
      </div>
    </section>
  );
}
