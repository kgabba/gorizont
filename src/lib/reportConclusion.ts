import type { AnalyzeReportData } from "./reportTypes";
import { fmtNum } from "./reportTypes";

export type ConclusionBlock = {
  title: string;
  body: string;
  tone?: "default" | "warn";
};

export function buildConclusion(
  inputs: AnalyzeReportData["conclusion_inputs"],
): ConclusionBlock[] {
  const blocks: ConclusionBlock[] = [];

  const rMaj = inputs.range_major;
  const rInt = inputs.range_intermediate;
  const rMin = inputs.range_minor;
  const type = inputs.anisotropy_type ?? "—";

  let ratioText = "";
  if (
    rMaj != null &&
    rMin != null &&
    Number(rMin) > 0 &&
    Number.isFinite(Number(rMaj) / Number(rMin))
  ) {
    const ratio = Number(rMaj) / Number(rMin);
    ratioText = ` Отношение основных диапазонов (major/minor) составляет ${fmtNum(ratio, 2)}.`;
  }

  blocks.push({
    title: "Пространственная структура",
    body:
      `Для набора данных определён тип анизотропии ${type} ` +
      `с ranges ${fmtNum(rMaj)} / ${fmtNum(rInt)} / ${fmtNum(rMin)} ` +
      `(variogram / continuity ellipsoid).` +
      ratioText +
      ` Параметры пространственной структуры фиксируются на этапе анализа анизотропии и не изменяются при оптимизации поискового соседства.`,
  });

  blocks.push({
    title: "Настройка поискового соседства",
    body:
      `Параметры поискового соседства подобраны по пространственной кросс-валидации: ` +
      `R = ${fmtNum(inputs.R_major)} / ${fmtNum(inputs.R_inter)} / ${fmtNum(inputs.R_minor)}, ` +
      `Nmin–Nmax = ${fmtNum(inputs.Nmin, 0)}–${fmtNum(inputs.Nmax, 0)}. ` +
      `Размер search neighbourhood определяется отдельно от variogram ellipsoid.`,
  });

  blocks.push({
    title: "Результаты cross-validation",
    body:
      `Полученный набор параметров обеспечивает CV RMSE ${fmtNum(inputs.CV_RMSE)}, ` +
      `CV MAE ${fmtNum(inputs.CV_MAE)} при prediction coverage ${fmtNum(inputs.prediction_coverage)}. ` +
      `Значения RMSE интерпретируются только в контексте масштаба данных и постановки задачи; ` +
      `отдельная качественная оценка «хорошо/плохо» без этого контекста не приводится.`,
  });

  const limitations: string[] = [];
  limitations.push(
    "Отчёт основан на сохранённых артефактах текущего прогона и не заменяет инженерную экспертизу по месторождению.",
  );
  if (inputs.any_near_bound) {
    limitations.push(
      "Один или несколько оптимизированных параметров расположены вблизи границы заданного диапазона поиска. Рекомендуется дополнительная проверка расширенного диапазона.",
    );
  }
  if (inputs.coverage_below_threshold) {
    const thr = inputs.coverage_threshold ?? 0.97;
    limitations.push(
      `Prediction coverage ниже порога ${fmtNum(thr, 2)}. Рекомендуется проверить геометрию поиска и покрытие пространства наблюдениями.`,
    );
  }

  blocks.push({
    title: "Ограничения",
    body: limitations.join(" "),
    tone: inputs.any_near_bound || inputs.coverage_below_threshold ? "warn" : "default",
  });

  return blocks;
}
