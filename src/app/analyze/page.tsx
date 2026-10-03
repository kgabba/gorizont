"use client";

import { useEffect } from "react";

/** Старый URL — на блок загрузки на главной */
export default function AnalyzePage() {
  useEffect(() => {
    window.location.replace("/#analyze");
  }, []);

  return null;
}
