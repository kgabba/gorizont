"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { EXAMPLE_REPORT_PATH } from "@/lib/exampleReport";

export default function SiteHeader() {
  const [hidden, setHidden] = useState(false);

  useEffect(() => {
    const update = () => {
      const maxScroll = Math.max(
        document.documentElement.scrollHeight - window.innerHeight,
        1,
      );
      // Hide around the midpoint of the page scroll
      setHidden(window.scrollY > maxScroll * 0.5);
    };

    update();
    window.addEventListener("scroll", update, { passive: true });
    window.addEventListener("resize", update);
    return () => {
      window.removeEventListener("scroll", update);
      window.removeEventListener("resize", update);
    };
  }, []);

  return (
    <header className={`site-header${hidden ? " is-hidden" : ""}`}>
      <Link href="/" className="site-header-brand">
        Горизонт
      </Link>
      <nav className="site-header-nav" aria-label="Основная навигация">
        <a href="#analyze" className="site-header-link">
          Анализ
        </a>
        <Link
          href={EXAMPLE_REPORT_PATH}
          className="site-header-link site-header-link-example"
        >
          <span className="site-header-link-full">Пример отчёта</span>
          <span className="site-header-link-short">Пример</span>
        </Link>
        <a href="#contacts" className="site-header-link">
          Контакты
        </a>
      </nav>
    </header>
  );
}
