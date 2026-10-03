import type { SVGProps } from "react";

type IconProps = SVGProps<SVGSVGElement>;

const base = {
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.35,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
};

export function IconDatabase(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <ellipse cx="12" cy="6" rx="7" ry="2.6" {...base} />
      <path d="M5 6v4c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6V6" {...base} />
      <path d="M5 10v4c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6v-4" {...base} />
      <path d="M5 14v4c0 1.4 3.1 2.6 7 2.6s7-1.2 7-2.6v-4" {...base} />
    </svg>
  );
}

export function IconValidate(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <rect x="5.5" y="3.5" width="13" height="17" rx="1.2" {...base} />
      <path d="M8.5 8.5h7M8.5 12h7M8.5 15.5h4.5" {...base} />
      <path d="M14.2 17.8l1.4 1.4 2.8-3" {...base} />
    </svg>
  );
}

export function IconTerrain(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M3.5 16.5l4.2-5.2 3.3 3.1 4-6.4 5.5 8.5" {...base} />
      <path d="M3.5 19.5h17" {...base} />
      <path d="M7.5 8.2c1.2-.9 2.4-.4 3 .6" {...base} />
    </svg>
  );
}

export function IconStats(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M4.5 19.5h15" {...base} />
      <rect x="6" y="11" width="2.4" height="6" rx="0.4" {...base} />
      <rect x="10.8" y="7.5" width="2.4" height="9.5" rx="0.4" {...base} />
      <rect x="15.6" y="9.5" width="2.4" height="7.5" rx="0.4" {...base} />
    </svg>
  );
}

export function IconLayers(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M12 4.5l8 4-8 4-8-4 8-4z" {...base} />
      <path d="M4 12.2l8 4 8-4" {...base} />
      <path d="M4 16l8 4 8-4" {...base} />
    </svg>
  );
}

export function IconCube(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M12 3.8l7.2 4.1v8.2L12 20.2l-7.2-4.1V7.9L12 3.8z" {...base} />
      <path d="M12 12v8.2M12 12l7.2-4.1M12 12L4.8 7.9" {...base} />
    </svg>
  );
}

export function IconPyramid(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M12 4.2L20.2 18.5H3.8L12 4.2z" {...base} />
      <path d="M12 4.2v14.3M7.2 14.2h9.6" {...base} />
    </svg>
  );
}

export function IconVariogram(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M4 18.5h16" {...base} />
      <path d="M4.5 18V6" {...base} />
      <path d="M6.5 15.5c2.2-4.2 4.6-6.8 8.8-8.2 2.2-.7 4.2-.5 5.7.2" {...base} />
      <circle cx="8.2" cy="14.2" r="1" fill="currentColor" stroke="none" />
      <circle cx="11.4" cy="11.4" r="1" fill="currentColor" stroke="none" />
      <circle cx="14.8" cy="9.6" r="1" fill="currentColor" stroke="none" />
      <circle cx="18.2" cy="8.4" r="1" fill="currentColor" stroke="none" />
    </svg>
  );
}

export function IconEllipsoid(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <ellipse cx="12" cy="12" rx="8.2" ry="4.6" {...base} />
      <ellipse cx="12" cy="12" rx="4.6" ry="8.2" {...base} />
      <path d="M3.8 12h16.4M12 3.8v16.4" {...base} />
    </svg>
  );
}

export function IconSliders(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M5 8h14M5 16h14" {...base} />
      <circle cx="9" cy="8" r="2.1" {...base} />
      <circle cx="15" cy="16" r="2.1" {...base} />
    </svg>
  );
}

export function IconReport(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <rect x="5" y="3.5" width="14" height="17" rx="1.2" {...base} />
      <path d="M8.5 8h7M8.5 11.5h7" {...base} />
      <path d="M8.5 15.5v2.5M11.5 14v4M14.5 16v2" {...base} />
    </svg>
  );
}

export function IconArrowDown(props: IconProps) {
  return (
    <svg viewBox="0 0 24 24" aria-hidden {...props}>
      <path d="M12 5v12.5M7.5 13.5L12 18l4.5-4.5" {...base} />
    </svg>
  );
}
