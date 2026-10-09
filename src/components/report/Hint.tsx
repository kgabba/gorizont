"use client";

import { useEffect, useId, useRef, useState } from "react";

type Props = {
  text: string;
};

/** Small «?» — shows explanation on click (and hover on desktop). */
export default function Hint({ text }: Props) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLSpanElement>(null);
  const tipId = useId();

  useEffect(() => {
    if (!open) return;
    const onDoc = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <span
      ref={rootRef}
      className={`report-hint-wrap${open ? " is-open" : ""}`}
    >
      <button
        type="button"
        className="report-hint"
        aria-label="Пояснение"
        aria-expanded={open}
        aria-describedby={open ? tipId : undefined}
        onClick={(e) => {
          e.preventDefault();
          e.stopPropagation();
          setOpen((v) => !v);
        }}
      >
        ?
      </button>
      {open ? (
        <span id={tipId} className="report-hint-tip" role="tooltip">
          {text}
        </span>
      ) : null}
    </span>
  );
}
