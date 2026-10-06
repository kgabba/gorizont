"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
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

type AnalyzeResponse = {
  run_id: string;
  status: string;
  anisotropy?: object;
  optimization?: object;
  detail?: string;
};

type MappingState = {
  X: string;
  Y: string;
  Z: string;
  Value: string;
  HoleID: string;
};

const STEPS = [
  "Загрузка данных",
  "Анализ пространственной структуры",
  "Анизотропия и вариография",
  "Оптимизация параметров",
  "Готово",
] as const;

const NONE = "";

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

  const onFileSelect = useCallback(async (f: File) => {
    setFile(f);
    setError(null);
    setInspect(null);
    setMapping(null);
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

  const runAnalyze = useCallback(async () => {
    if (!file || !mapping) return;
    if (!mapping.X || !mapping.Y || !mapping.Z || !mapping.Value) {
      setError("Укажите колонки X, Y, Z и Value.");
      setStatus("confirm");
      return;
    }

    setStatus("running");
    setError(null);

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
      router.push(`/report/${ok.run_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Неизвестная ошибка");
      setStatus("error");
    }
  }, [file, mapping, router]);

  const columns = inspect?.columns ?? [];
  const canConfirm =
    !!mapping?.X && !!mapping?.Y && !!mapping?.Z && !!mapping?.Value;

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
            disabled={status === "running" || status === "inspecting"}
          />
        </div>

        <p className="upload-section-hint">
          X · Y · Z · Value / Grade / Au…
          <span className="upload-section-hint-opt">
            {" "}
            · HoleID optional
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
                label="Value (содержание)"
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
      </div>
    </section>
  );
}
