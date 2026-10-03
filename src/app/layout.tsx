import type { Metadata } from "next";
import { Manrope, Onest } from "next/font/google";
import "./globals.css";

const body = Manrope({
  subsets: ["latin", "cyrillic"],
  variable: "--font-body",
  display: "swap",
});

const display = Onest({
  subsets: ["latin", "cyrillic"],
  variable: "--font-display",
  display: "swap",
});

export const metadata: Metadata = {
  title: "ГОРИЗОНТ — пространственная оценка",
  description:
    "Автоматизация подготовки параметров пространственной оценки месторождений: вариография, анизотропия, кригинг и отчёт по качеству модели.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru" className={`${body.variable} ${display.variable}`}>
      <body className="min-h-full antialiased">{children}</body>
    </html>
  );
}
