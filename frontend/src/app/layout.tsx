import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Nav } from "@/components/ui";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "H-XAI-NIDS — Explainable Network Intrusion Detection",
  description:
    "A Hierarchical Contextual Stacking Network Intrusion Detection System: " +
    "binary detection, attack-family classification, zero-day anomaly " +
    "detection, and a TreeSHAP explanation for every alert.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full`}
    >
      <body className="flex min-h-full flex-col bg-canvas text-ink antialiased">
        <Nav />
        <main className="flex-1">
          <div className="mx-auto w-full max-w-7xl px-5 py-6">{children}</div>
        </main>
        <footer className="border-t border-line px-5 py-6">
          <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-3 text-xs text-ink-faint">
            <p>
              H-XAI-NIDS · Hierarchical Contextual Stacking NIDS · M.Sc. AIML,
              VIT Vellore
            </p>
            <p className="font-mono">
              CICIDS-2017 · 451,380 flows · 30 selected features
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}

