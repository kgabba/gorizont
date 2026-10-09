"use client";

import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

type Props = {
  orientationMatrix?: number[][] | null;
  rangeMajor?: number | null;
  rangeIntermediate?: number | null;
  rangeMinor?: number | null;
  majorAxis?: number[] | null;
  intermediateAxis?: number[] | null;
  minorAxis?: number[] | null;
};

function isVec3(v: number[] | null | undefined): v is [number, number, number] {
  return (
    Array.isArray(v) &&
    v.length >= 3 &&
    v.slice(0, 3).every((x) => typeof x === "number" && Number.isFinite(x))
  );
}

/** Grid on an arbitrary plane: origin = plane center, u/v = in-plane axes. */
function makePlaneGrid(
  origin: THREE.Vector3,
  uAxis: THREE.Vector3,
  vAxis: THREE.Vector3,
  halfU: number,
  halfV: number,
  divU: number,
  divV: number,
  color: number,
): THREE.LineSegments {
  const u = uAxis.clone().normalize();
  const v = vAxis.clone().normalize();
  const positions: number[] = [];

  for (let i = 0; i <= divV; i++) {
    const tv = -1 + (2 * i) / divV;
    const a = origin
      .clone()
      .addScaledVector(u, -halfU)
      .addScaledVector(v, tv * halfV);
    const b = origin
      .clone()
      .addScaledVector(u, halfU)
      .addScaledVector(v, tv * halfV);
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }
  for (let i = 0; i <= divU; i++) {
    const tu = -1 + (2 * i) / divU;
    const a = origin
      .clone()
      .addScaledVector(u, tu * halfU)
      .addScaledVector(v, -halfV);
    const b = origin
      .clone()
      .addScaledVector(u, tu * halfU)
      .addScaledVector(v, halfV);
    positions.push(a.x, a.y, a.z, b.x, b.y, b.z);
  }

  const geo = new THREE.BufferGeometry();
  geo.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(positions, 3),
  );
  return new THREE.LineSegments(
    geo,
    new THREE.LineBasicMaterial({
      color,
      transparent: true,
      opacity: 0.5,
      depthWrite: false,
    }),
  );
}

function axisPair(
  dir: THREE.Vector3,
  halfLen: number,
  color: number,
): THREE.Group {
  const g = new THREE.Group();
  const d = dir.clone().normalize();
  const pts = new Float32Array([
    -d.x * halfLen,
    -d.y * halfLen,
    -d.z * halfLen,
    d.x * halfLen,
    d.y * halfLen,
    d.z * halfLen,
  ]);
  const lineGeo = new THREE.BufferGeometry();
  lineGeo.setAttribute("position", new THREE.BufferAttribute(pts, 3));
  g.add(
    new THREE.Line(
      lineGeo,
      new THREE.LineBasicMaterial({ color }),
    ),
  );
  g.add(
    new THREE.ArrowHelper(
      d,
      new THREE.Vector3(0, 0, 0),
      halfLen,
      color,
      Math.max(halfLen * 0.1, 0.04),
      Math.max(halfLen * 0.06, 0.025),
    ),
  );
  return g;
}

/**
 * Continuity ellipsoid from stored ranges + orientation — display only.
 * Axes inscribed (length = semi-axes). Three outer face-grids share a corner
 * of a padded box and do not cut the ellipsoid.
 */
export default function ContinuityEllipsoidView({
  orientationMatrix,
  rangeMajor,
  rangeIntermediate,
  rangeMinor,
  majorAxis,
  intermediateAxis,
  minorAxis,
}: Props) {
  const mountRef = useRef<HTMLDivElement | null>(null);

  const canRender =
    rangeMajor != null &&
    rangeIntermediate != null &&
    rangeMinor != null &&
    rangeMajor > 0 &&
    rangeIntermediate > 0 &&
    rangeMinor > 0 &&
    (isVec3(majorAxis) ||
      (Array.isArray(orientationMatrix) &&
        orientationMatrix.length === 3 &&
        orientationMatrix.every((r) => Array.isArray(r) && r.length === 3)));

  useEffect(() => {
    if (!canRender || !mountRef.current) return;

    const el = mountRef.current;
    const width = el.clientWidth || 640;
    const height = 400;

    const a = Number(rangeMajor);
    const b = Number(rangeIntermediate);
    const c = Number(rangeMinor);
    const scale = 1 / Math.max(a, b, c);
    const sa = a * scale;
    const sb = b * scale;
    const sc = c * scale;

    let maj: [number, number, number];
    let inter: [number, number, number];
    let min: [number, number, number];
    if (isVec3(majorAxis) && isVec3(intermediateAxis) && isVec3(minorAxis)) {
      maj = [majorAxis[0], majorAxis[1], majorAxis[2]];
      inter = [
        intermediateAxis[0],
        intermediateAxis[1],
        intermediateAxis[2],
      ];
      min = [minorAxis[0], minorAxis[1], minorAxis[2]];
    } else {
      const Q = orientationMatrix as number[][];
      maj = [Q[0][0], Q[1][0], Q[2][0]];
      inter = [Q[0][1], Q[1][1], Q[2][1]];
      min = [Q[0][2], Q[1][2], Q[2][2]];
    }

    const e1 = new THREE.Vector3(...maj).normalize();
    const e2 = new THREE.Vector3(...inter).normalize();
    const e3 = new THREE.Vector3(...min).normalize();
    const basis = new THREE.Matrix4().makeBasis(e1, e2, e3);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xffffff);

    const buf = 1.2;
    const A = sa * buf;
    const B = sb * buf;
    const C = sc * buf;
    const extent = Math.max(A, B, C);

    const camera = new THREE.PerspectiveCamera(42, width / height, 0.01, 100);
    // Distance scales with padded box; ~2.15×extent ≈ closer than previous ~2.8–3.4
    const dist = extent * 2.15;
    camera.position
      .set(1, 0.82, 1)
      .normalize()
      .multiplyScalar(dist);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(width, height);
    el.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 0, 0);
    controls.minDistance = extent * 0.9;
    controls.maxDistance = extent * 8;

    scene.add(new THREE.AmbientLight(0xffffff, 0.9));
    const light = new THREE.DirectionalLight(0xffffff, 0.5);
    light.position.set(2, 3, 1);
    scene.add(light);

    const root = new THREE.Group();
    root.setRotationFromMatrix(basis);
    scene.add(root);

    const geo = new THREE.SphereGeometry(1, 48, 32);
    const mat = new THREE.MeshPhysicalMaterial({
      color: 0x6ea8e0,
      transparent: true,
      opacity: 0.3,
      roughness: 0.45,
      metalness: 0.05,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const ellipsoid = new THREE.Mesh(geo, mat);
    ellipsoid.scale.set(sa, sb, sc);
    root.add(ellipsoid);

    const wire = new THREE.LineSegments(
      new THREE.WireframeGeometry(geo),
      new THREE.LineBasicMaterial({
        color: 0x3a6a96,
        transparent: true,
        opacity: 0.18,
      }),
    );
    wire.scale.set(sa, sb, sc);
    root.add(wire);

    // Inscribed axes in world (same dirs as e1/e2/e3)
    scene.add(axisPair(e1, sa, 0xc0392b));
    scene.add(axisPair(e2, sb, 0x27ae60));
    scene.add(axisPair(e3, sc, 0x2980b9));

    // Three outer faces of padded AABB in principal frame — no center cut
    // Local axes: X=maj, Y=inter, Z=min
    const div = 8;
    const gridColor = 0xb8c2cc;
    const xHat = new THREE.Vector3(1, 0, 0);
    const yHat = new THREE.Vector3(0, 1, 0);
    const zHat = new THREE.Vector3(0, 0, 1);

    // Face z = -C (span X,Y)
    root.add(
      makePlaneGrid(
        new THREE.Vector3(0, 0, -C),
        xHat,
        yHat,
        A,
        B,
        div,
        div,
        gridColor,
      ),
    );
    // Face y = -B (span X,Z)
    root.add(
      makePlaneGrid(
        new THREE.Vector3(0, -B, 0),
        xHat,
        zHat,
        A,
        C,
        div,
        div,
        gridColor,
      ),
    );
    // Face x = -A (span Y,Z)
    root.add(
      makePlaneGrid(
        new THREE.Vector3(-A, 0, 0),
        yHat,
        zHat,
        B,
        C,
        div,
        div,
        gridColor,
      ),
    );

    let frame = 0;
    const tick = () => {
      frame = requestAnimationFrame(tick);
      controls.update();
      renderer.render(scene, camera);
    };
    tick();

    const onResize = () => {
      const w = el.clientWidth || width;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener("resize", onResize);
      controls.dispose();
      geo.dispose();
      mat.dispose();
      renderer.dispose();
      if (renderer.domElement.parentNode === el) {
        el.removeChild(renderer.domElement);
      }
    };
  }, [
    canRender,
    orientationMatrix,
    rangeMajor,
    rangeIntermediate,
    rangeMinor,
    majorAxis,
    intermediateAxis,
    minorAxis,
  ]);

  if (!canRender) {
    return (
      <p className="report-section-note">
        3D-вид эллипсоида вариограммы недоступен: нет полных диапазонов или
        ориентации.
      </p>
    );
  }

  return (
    <div className="report-ellipsoid">
      <div className="report-ellipsoid-canvas" ref={mountRef} />
      <ul className="report-ellipsoid-legend">
        <li>
          <span className="swatch is-maj" /> Ось 1 — направление наибольшей
          непрерывности
        </li>
        <li>
          <span className="swatch is-int" /> Ось 2 — промежуточное направление
        </li>
        <li>
          <span className="swatch is-min" /> Ось 3 — направление наименьшей
          непрерывности
        </li>
      </ul>
      <p className="report-section-note">
        Эллипсоид показывает, насколько далеко по каждому направлению сохраняется
        связь содержаний (диапазоны вариограммы).
      </p>
    </div>
  );
}
