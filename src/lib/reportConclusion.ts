import { anisoTypeLabel, variogramModelLabel } from "./reportLabels";
import type { AnalyzeReportData } from "./reportTypes";
import { fmtNum, fmtPct } from "./reportTypes";

export type RecommendedGroup = {
  title: string;
  items: { label: string; value: string }[];
};

export type ConclusionModel = {
  lead: string;
  groups: RecommendedGroup[];
  notes: string[];
  hasWarn: boolean;
};

function axisAngle(
  az: number | null | undefined,
  dip: number | null | undefined,
): string {
  if (az == null && dip == null) return "—";
  return `азимут ${fmtNum(az, 1)}°, погружение ${fmtNum(dip, 1)}°`;
}

export function buildConclusion(
  inputs: AnalyzeReportData["conclusion_inputs"],
): ConclusionModel {
  const type = anisoTypeLabel(inputs.anisotropy_type);

  const groups: RecommendedGroup[] = [
    {
      title: "Вариограмма",
      items: [
        { label: "Тип анизотропии", value: type },
        {
          label: "Модель вариограммы",
          value: variogramModelLabel(inputs.variogram_model),
        },
        {
          label: "Эффект самородка (nugget)",
          value: fmtNum(inputs.nugget, 3),
        },
        {
          label: "Порог (sill)",
          value: fmtNum(inputs.sill, 3),
        },
        {
          label: "Диапазоны по осям 1 / 2 / 3",
          value: `${fmtNum(inputs.range_major, 0)} / ${fmtNum(inputs.range_intermediate, 0)} / ${fmtNum(inputs.range_minor, 0)}`,
        },
        {
          label: "Ось 1 (главная)",
          value: axisAngle(inputs.major_azimuth_deg, inputs.major_dip_deg),
        },
        {
          label: "Ось 2 (промежуточная)",
          value: axisAngle(
            inputs.intermediate_azimuth_deg,
            inputs.intermediate_dip_deg,
          ),
        },
        {
          label: "Ось 3 (малая)",
          value: axisAngle(inputs.minor_azimuth_deg, inputs.minor_dip_deg),
        },
      ],
    },
    {
      title: "Область поиска",
      items: [
        {
          label: "Радиусы осей 1 / 2 / 3",
          value: `${fmtNum(inputs.R_major, 0)} / ${fmtNum(inputs.R_inter, 0)} / ${fmtNum(inputs.R_minor, 0)}`,
        },
        {
          label: "Мин. / макс. кол-во точек",
          value: `${fmtNum(inputs.Nmin, 0)} / ${fmtNum(inputs.Nmax, 0)}`,
        },
      ],
    },
  ];

  const notes: string[] = [];
  if (inputs.any_near_bound) {
    notes.push(
      "Один или несколько подобранных параметров находятся у края заданного диапазона поиска.",
    );
  }
  if (inputs.coverage_below_threshold) {
    const thr = inputs.coverage_threshold ?? 0.97;
    notes.push(
      `Процент успешных оценок ниже порога ${fmtPct(thr)} — проверьте геометрию области поиска.`,
    );
  }
  notes.push(
    "Параметры носят рекомендательный характер и не заменяют экспертизу по месторождению.",
  );

  return {
    lead:
      "Рекомендуемые параметры пространственной оценки получены по результатам автоматизированного анализа и пространственной кросс-валидации. Ниже приведены параметры, рекомендуемые для проверки и последующего использования при настройке интерполяции в ГГИС.",
    groups,
    notes,
    hasWarn: Boolean(
      inputs.any_near_bound || inputs.coverage_below_threshold,
    ),
  };
}
