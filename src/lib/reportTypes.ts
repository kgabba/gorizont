export type ProximityStatus = "inside" | "near_bound" | "unknown";

export type ProximityInfo = {
  status: ProximityStatus;
  label: string;
  near_bound?: "min" | "max" | "both" | null;
  value?: number | null;
  lo?: number | null;
  hi?: number | null;
};

export type TrialRow = {
  trial: number | null;
  R_major?: number | null;
  K_inter?: number | null;
  K_minor?: number | null;
  R_inter?: number | null;
  R_minor?: number | null;
  Nmin?: number | null;
  Nmax?: number | null;
  objective?: number | null;
  CV_RMSE?: number | null;
  CV_MAE?: number | null;
  prediction_coverage?: number | null;
  n_valid?: number | null;
  n_invalid?: number | null;
  n_tgt?: number | null;
};

export type ReportImage = {
  name: string;
  path: string;
  label: string;
};

export type AnalyzeReportData = {
  run_id: string;
  status: string;
  missing_artifacts?: string[];
  header: {
    run_id: string;
    filename?: string | null;
    n_points?: number | null;
    has_hole_id?: boolean | null;
    cv_method?: string | null;
    created_at?: string | null;
  };
  kpi: {
    CV_RMSE?: number | null;
    CV_MAE?: number | null;
    prediction_coverage?: number | null;
  };
  spatial_structure: {
    anisotropy_type?: string | null;
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
    major_azimuth_deg?: number | null;
    major_dip_deg?: number | null;
    intermediate_azimuth_deg?: number | null;
    intermediate_dip_deg?: number | null;
    minor_azimuth_deg?: number | null;
    minor_dip_deg?: number | null;
    moi_strength?: number | null;
    ellipsoid_kind?: string | null;
  };
  search_params: {
    R_major?: number | null;
    R_inter?: number | null;
    R_minor?: number | null;
    K_inter?: number | null;
    K_minor?: number | null;
    Nmin?: number | null;
    Nmax?: number | null;
  };
  validation: {
    CV_RMSE?: number | null;
    CV_MAE?: number | null;
    prediction_coverage?: number | null;
    n_valid?: number | null;
    n_tgt?: number | null;
    n_invalid?: number | null;
    oof_available?: boolean;
    coverage_below_threshold?: boolean;
    coverage_threshold?: number;
  };
  optimization: {
    best_trial_number?: number | null;
    n_trials?: number | null;
    objective?: number | null;
    best_params?: Record<string, number | null | undefined>;
    search_space?: Record<string, number[]>;
    proximity?: Record<string, ProximityInfo>;
    any_near_bound?: boolean;
    trials?: TrialRow[];
    top10?: TrialRow[];
  };
  artifacts?: {
    images?: ReportImage[];
  };
  conclusion_inputs: {
    anisotropy_type?: string | null;
    range_major?: number | null;
    range_intermediate?: number | null;
    range_minor?: number | null;
    CV_RMSE?: number | null;
    CV_MAE?: number | null;
    prediction_coverage?: number | null;
    any_near_bound?: boolean;
    coverage_below_threshold?: boolean;
    coverage_threshold?: number;
    Nmin?: number | null;
    Nmax?: number | null;
    R_major?: number | null;
    R_inter?: number | null;
    R_minor?: number | null;
  };
};

export function artifactUrl(runId: string, relPath: string): string {
  return `/api/analysis/${encodeURIComponent(runId)}/artifacts/${relPath}`;
}

export function fmtNum(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  return Number(v).toLocaleString("ru-RU", {
    maximumFractionDigits: digits,
  });
}

export function fmtAxis(v: number[] | null | undefined): string {
  if (!v || v.length < 3) return "—";
  return v.map((x) => fmtNum(x, 4)).join(", ");
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("ru-RU", {
    timeZone: "UTC",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }) + " UTC";
}
