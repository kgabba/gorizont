/** Display labels for report UI — Micromine-oriented Russian terminology. */

export const AXIS = {
  major: "Ось 1 (главная)",
  intermediate: "Ось 2 (промежуточная)",
  minor: "Ось 3 (малая)",
} as const;

export const ANISO_TYPE_RU: Record<string, string> = {
  ISOTROPIC: "Изотропная",
  AXIAL_PROLATE: "Осевая (пролатная)",
  AXIAL_OBLATE: "Осевая (облатная)",
  TRIAXIAL: "Трехосная",
};

export const VARIOGRAM_MODEL_RU: Record<string, string> = {
  spherical: "сферическая",
  exponential: "экспоненциальная",
  gaussian: "гауссова",
};

export const PARAM_LABELS: Record<string, string> = {
  R_major: "Радиус оси 1 (R₁)",
  R_inter: "Радиус оси 2 (R₂)",
  R_minor: "Радиус оси 3 (R₃)",
  K_inter: "Коэфф. R₂/R₁",
  K_minor: "Коэфф. R₃/R₁",
  Nmin: "Мин. кол-во точек (общее)",
  Nmax: "Макс. кол-во точек (общее)",
};

export function anisoTypeLabel(raw: string | null | undefined): string {
  if (!raw) return "—";
  return ANISO_TYPE_RU[raw] ?? raw;
}

export function variogramModelLabel(raw: string | null | undefined): string {
  if (!raw) return "—";
  return VARIOGRAM_MODEL_RU[raw.toLowerCase()] ?? raw;
}

export function paramLabel(key: string): string {
  return PARAM_LABELS[key] ?? key;
}
