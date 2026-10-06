"use client";

import dynamic from "next/dynamic";
import { useState } from "react";
import type { AnalyzeReportData } from "@/lib/reportTypes";
import { artifactUrl, fmtAxis, fmtNum } from "@/lib/reportTypes";

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
      name: "Major",
      az: data.major_azimuth_deg,
      dip: data.major_dip_deg,
    },
    {
      name: "Intermediate",
      az: data.intermediate_azimuth_deg,
      dip: data.intermediate_dip_deg,
    },
    {
      name: "Minor",
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
            <th>Azimuth °</th>
            <th>Dip °</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.name}>
              <td>{r.name}</td>
              <td>{fmtNum(r.az, 2)}</td>
              <td>{fmtNum(r.dip, 2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="report-angles-note">
        Конвенция: azimuth 0 = +X, против часовой к +Y; dip 0 = горизонталь, 90
        = +Z. Углы от осей aniso_candidate.
      </p>
    </div>
  );
}

export default function SpatialStructure({ data }: { data: AnalyzeReportData }) {
  const s = data.spatial_structure;
  // principal_axes.png excluded — replaced by interactive continuity ellipsoid
  const images = (data.artifacts?.images ?? []).filter(
    (i) => i.name !== "principal_axes.png",
  );

  return (
    <section id="report-spatial" className="report-section">
      <h2 className="report-section-title">Пространственная структура</h2>
      <p className="report-section-note">
        Variogram / continuity ellipsoid — параметры пространственной
        непрерывности. Не путать с search neighbourhood.
      </p>

      <dl className="report-grid">
        <div>
          <dt>Тип</dt>
          <dd>{s.anisotropy_type ?? "—"}</dd>
        </div>
        <div>
          <dt>Variogram model</dt>
          <dd>{s.variogram_model ?? "—"}</dd>
        </div>
        <div>
          <dt>Nugget</dt>
          <dd>{fmtNum(s.nugget)}</dd>
        </div>
        <div>
          <dt>Sill</dt>
          <dd>{fmtNum(s.sill)}</dd>
        </div>
        <div>
          <dt>Range major</dt>
          <dd>{fmtNum(s.range_major)}</dd>
        </div>
        <div>
          <dt>Range intermediate</dt>
          <dd>{fmtNum(s.range_intermediate)}</dd>
        </div>
        <div>
          <dt>Range minor</dt>
          <dd>{fmtNum(s.range_minor)}</dd>
        </div>
        {s.moi_strength != null ? (
          <div>
            <dt>MOI strength</dt>
            <dd>{fmtNum(s.moi_strength)}</dd>
          </div>
        ) : null}
      </dl>

      <h3 className="report-subheading">Orientation</h3>
      <dl className="report-grid report-grid-wide">
        <div>
          <dt>Major axis XYZ</dt>
          <dd className="report-mono">{fmtAxis(s.major_axis_xyz)}</dd>
        </div>
        <div>
          <dt>Intermediate axis XYZ</dt>
          <dd className="report-mono">{fmtAxis(s.intermediate_axis_xyz)}</dd>
        </div>
        <div>
          <dt>Minor axis XYZ</dt>
          <dd className="report-mono">{fmtAxis(s.minor_axis_xyz)}</dd>
        </div>
      </dl>

      <h3 className="report-subheading">Continuity ellipsoid</h3>
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
