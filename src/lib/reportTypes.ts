export type ProximityStatus = "inside" | "near_bound" | "unknown";

export type ProximityInfo = {
  status: ProximityStatus;
  label: string;
  near_bound?: "min" | "max" | "both" | null;
  value?: number | null;
  lo?: number | null;
  hi?: number | null;
};

export type SectorSearchFields = {
  octant_search?: boolean | null;
  sector_search?: boolean | null;
  n_sectors?: number | null;
  MINOCT?: number | null;
  MINPEROC?: number | null;
  MAXPEROC?: number | null;
  discard_underfilled?: boolean | null;
  neighbour_cap?: number | null;
};

export type TrialRow = {
  trial: number | null;
  method?: string | null;
  power?: number | null;
  J?: number | null;
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
} & SectorSearchFields;

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
    method?: string | null;
    method_label?: string | null;
  };
  kpi: {
    CV_RMSE?: number | null;
    CV_MAE?: number | null;
    prediction_coverage?: number | null;
    J?: number | null;
    coverage_fill_triggered?: boolean | null;
    n_passes?: number | null;
  };
  coverage_fill?: {
    enabled?: boolean | null;
    triggered?: boolean | null;
    n_passes?: number | null;
    trigger_below?: number | null;
    pass1_coverage?: number | null;
    pass2_coverage?: number | null;
    pass2_coverage_min?: number | null;
    r_expand_factor?: number | null;
    n_by_pass?: { "1"?: number | null; "2"?: number | null };
    residual_failure_reasons?: Record<string, number>;
    summary_ru?: string | null;
    passes?: Array<
      {
        pass?: number | null;
        label?: string | null;
        role?: string | null;
        loss?: string | null;
        power?: number | null;
        R_major?: number | null;
        R_inter?: number | null;
        R_minor?: number | null;
        K_inter?: number | null;
        K_minor?: number | null;
        Nmin?: number | null;
        Nmax?: number | null;
        CV_RMSE?: number | null;
        CV_MAE?: number | null;
        local_CV_RMSE?: number | null;
        local_CV_MAE?: number | null;
        prediction_coverage?: number | null;
        n_new?: number | null;
        n_trials?: number | null;
      } & SectorSearchFields
    >;
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
    method?: string | null;
    power?: number | null;
    mean_n_used?: number | null;
    R_major?: number | null;
    R_inter?: number | null;
    R_minor?: number | null;
    K_inter?: number | null;
    K_minor?: number | null;
    Nmin?: number | null;
    Nmax?: number | null;
  } & SectorSearchFields;
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
    coverage_fill_triggered?: boolean;
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
    validation_images?: ReportImage[];
    files?: ReportImage[];
  };
  conclusion_inputs: {
    method?: string | null;
    power?: number | null;
    mean_n_used?: number | null;
    anisotropy_type?: string | null;
    variogram_model?: string | null;
    nugget?: number | null;
    sill?: number | null;
    range_major?: number | null;
    range_intermediate?: number | null;
    range_minor?: number | null;
    major_azimuth_deg?: number | null;
    major_dip_deg?: number | null;
    intermediate_azimuth_deg?: number | null;
    intermediate_dip_deg?: number | null;
    minor_azimuth_deg?: number | null;
    minor_dip_deg?: number | null;
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
    coverage_fill_triggered?: boolean | null;
    coverage_fill?: AnalyzeReportData["coverage_fill"];
  } & SectorSearchFields;
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

/** Fraction 0..1 → percentage string, e.g. 1 → «100%». */
export function fmtPct(v: number | null | undefined, digits = 0): string {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "—";
  const pct = Number(v) * 100;
  return (
    pct.toLocaleString("ru-RU", {
      maximumFractionDigits: digits,
      minimumFractionDigits: digits,
    }) + "%"
  );
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
