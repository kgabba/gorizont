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
  power: "Степень IDW (p)",
  R_major: "Радиус оси 1 (R₁)",
  R_inter: "Радиус оси 2 (R₂)",
  R_minor: "Радиус оси 3 (R₃)",
  K_inter: "Коэфф. R₂/R₁",
  K_minor: "Коэфф. R₃/R₁",
  Nmin: "Мин. кол-во точек (общее)",
  Nmax: "Макс. кол-во точек (общее)",
  mean_n_used: "Среднее число соседей",
  sector_search: "Секторный поиск",
  octant_search: "Секторный поиск",
  n_sectors: "Число секторов",
  MINOCT: "Мин. заполненных секторов",
  MINPEROC: "Мин. проб в секторе",
  MAXPEROC: "Макс. проб в секторе",
  neighbour_cap: "Потолок соседей (секторы × макс.)",
};

export const SECTOR_COUNT_RU: Record<number, string> = {
  4: "4 (квадранты)",
  8: "8 (октанты)",
  16: "16 (восемь клиньев × верх/низ)",
};

export function sectorCountLabel(n: number | null | undefined): string {
  if (n == null || Number.isNaN(Number(n))) return "—";
  const k = Number(n);
  return SECTOR_COUNT_RU[k] ?? String(k);
}

export const METHOD_LABELS: Record<string, string> = {
  ordinary_kriging: "Ordinary Kriging",
  anisotropic_idw: "Анизотропный IDW",
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
