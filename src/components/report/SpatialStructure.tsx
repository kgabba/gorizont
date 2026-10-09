"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import {
  AXIS,
  anisoTypeLabel,
  variogramModelLabel,
} from "@/lib/reportLabels";
import type { AnalyzeReportData } from "@/lib/reportTypes";
import { artifactUrl, fmtNum } from "@/lib/reportTypes";
import Hint from "./Hint";

const ContinuityEllipsoidView = dynamic(
  () => import("./ContinuityEllipsoidView"),
  { ssr: false },
);

function ArtifactImage({
  runId,
  path,
  label,
}: {
  runId: string;
  path: string;
  label: string;
}) {
  const [failed, setFailed] = useState(false);
  if (failed) return null;
  return (
    <figure className="report-figure">
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img
        src={artifactUrl(runId, path)}
        alt={label}
        className="report-figure-img"
        onError={() => setFailed(true)}
      />
      <figcaption>{label}</figcaption>
    </figure>
  );
}

function AxisAngles({ data }: { data: AnalyzeReportData["spatial_structure"] }) {
  const rows = [
    {
      name: AXIS.major,
      az: data.major_azimuth_deg,
      dip: data.major_dip_deg,
    },
    {
      name: AXIS.intermediate,
      az: data.intermediate_azimuth_deg,
      dip: data.intermediate_dip_deg,
    },
    {
      name: AXIS.minor,
      az: data.minor_azimuth_deg,
      dip: data.minor_dip_deg,
    },
  ];
  if (rows.every((r) => r.az == null && r.dip == null)) return null;
  return (
    <div className="report-angles">
      <table className="report-angles-table">
        <thead>
          <tr>
            <th>Ось</th>
            <th>Азимут °</th>
            <th>Погружение °</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>{fmtNum(r.az, 1)}</td>
              <td>{fmtNum(r.dip, 1)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="report-angles-note">
        Конвенция: азимут 0° = +X, против часовой к +Y; погружение 0° =
        горизонталь, 90° = +Z.
      </p>
    </div>
  );
}

export default function SpatialStructure({ data }: { data: AnalyzeReportData }) {
  const s = data.spatial_structure;
  // Hide principal axes (3D ellipsoid instead) and directional VGs for now
  const HIDDEN_IMAGES = new Set([
    "principal_axes.png",
    "major_variogram.png",
    "intermediate_variogram.png",
    "minor_variogram.png",
  ]);
  const images = (data.artifacts?.images ?? []).filter(
    (i) => !HIDDEN_IMAGES.has(i.name),
  );

  return (
    <section id="report-spatial" className="report-section">
      <h2 className="report-section-title">Вариограмма и анизотропия</h2>
      <p className="report-section-note">
        Параметры пространственной непрерывности (эллипсоид вариограммы). Может
        отличаться от области поиска при интерполяции!
      </p>

      <dl className="report-grid">
        <div>
          <dt>Тип анизотропии</dt>
          <dd>{anisoTypeLabel(s.anisotropy_type)}</dd>
        </div>
        <div>
          <dt>Модель вариограммы</dt>
          <dd>{variogramModelLabel(s.variogram_model)}</dd>
        </div>
        <div>
          <dt>Эффект самородка (nugget)</dt>
          <dd>{fmtNum(s.nugget, 3)}</dd>
        </div>
        <div>
          <dt>Порог (sill)</dt>
          <dd>{fmtNum(s.sill, 3)}</dd>
        </div>
        <div>
          <dt>Диапазон (Range) по оси 1</dt>
          <dd>{fmtNum(s.range_major, 0)}</dd>
        </div>
        <div>
          <dt>Диапазон (Range) по оси 2</dt>
          <dd>{fmtNum(s.range_intermediate, 0)}</dd>
        </div>
        <div>
          <dt>Диапазон (Range) по оси 3</dt>
          <dd>{fmtNum(s.range_minor, 0)}</dd>
        </div>
        {s.moi_strength != null ? (
          <div>
            <dt>
              Сила анизотропии (MOI){" "}
              <Hint text="Внутренний показатель выраженности направленной непрерывности по тензору моментов инерции. Чем выше, тем сильнее отличие направлений." />
            </dt>
            <dd>{fmtNum(s.moi_strength, 2)}</dd>
          </div>
        ) : null}
      </dl>

      <h3 className="report-subheading">Эллипсоид вариограммы (3D)</h3>
      <div className="report-ellipsoid-block">
        <ContinuityEllipsoidView
          orientationMatrix={s.orientation_matrix}
          rangeMajor={s.range_major}
          rangeIntermediate={s.range_intermediate}
          rangeMinor={s.range_minor}
          majorAxis={s.major_axis_xyz}
          intermediateAxis={s.intermediate_axis_xyz}
          minorAxis={s.minor_axis_xyz}
        />
        <AxisAngles data={s} />
      </div>

      {images.length > 0 ? (
        <div className="report-figures">
          {images.map((img) => (
            <ArtifactImage
              key={img.path}
              runId={data.run_id}
              path={img.path}
              label={img.label}
            />
          ))}
        </div>
      ) : null}
    </section>
  );
}
