"use client";

import { useCallback, useRef, useState } from "react";

type FileUploadProps = {
  onFileSelect?: (file: File) => void;
  disabled?: boolean;
  variant?: "light" | "dark";
};

const ACCEPT = ".csv,text/csv";

function isAllowedFile(file: File) {
  return file.name.toLowerCase().endsWith(".csv");
}

export default function FileUpload({
  onFileSelect,
  disabled,
  variant = "light",
}: FileUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const dark = variant === "dark";

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (disabled || !file || !isAllowedFile(file)) return;
      setFileName(file.name);
      onFileSelect?.(file);
    },
    [disabled, onFileSelect],
  );

  const surface = dark
    ? isDragging
      ? "border-white/55 bg-white/10"
      : "border-white/30 bg-white/[0.04] hover:border-white/50 hover:bg-white/[0.07]"
    : isDragging
      ? "border-neutral-800 bg-neutral-50"
      : "border-neutral-400 bg-white hover:border-neutral-600";

  return (
    <div
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled || undefined}
      onClick={() => {
        if (!disabled) inputRef.current?.click();
      }}
      onKeyDown={(e) => {
        if (disabled) return;
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          inputRef.current?.click();
        }
      }}
      onDragEnter={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!disabled) setIsDragging(true);
      }}
      onDragOver={(e) => {
        e.preventDefault();
        e.stopPropagation();
        if (!disabled) setIsDragging(true);
      }}
      onDragLeave={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
      }}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        setIsDragging(false);
        handleFile(e.dataTransfer.files?.[0]);
      }}
      className={`flex min-h-56 flex-col items-center justify-center border border-dashed px-6 py-16 text-center transition-colors ${
        disabled ? "cursor-wait opacity-60" : "cursor-pointer"
      } ${surface}`}
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPT}
        className="hidden"
        disabled={disabled}
        onChange={(e) => {
          handleFile(e.target.files?.[0]);
          e.target.value = "";
        }}
      />

      {fileName ? (
        <p className={`text-sm ${dark ? "text-white" : "text-neutral-800"}`}>
          {fileName}
        </p>
      ) : (
        <>
          <p
            className={`text-base ${dark ? "text-white" : "text-neutral-800"}`}
          >
            Перетащите CSV сюда
          </p>
          <p
            className={`mt-2 text-sm ${dark ? "text-white/55" : "text-neutral-500"}`}
          >
            или выберите файл
          </p>
        </>
      )}
    </div>
  );
}
