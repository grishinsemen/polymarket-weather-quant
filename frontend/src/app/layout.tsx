import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Polymarket Weather Quant | Pro Terminal & Replay Lab",
  description: "Autonomous ICAO METAR temperature quant terminal, 11-bucket X-Ray, and Experiment Replay lab",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#070b12] text-slate-200 antialiased">
        {children}
      </body>
    </html>
  );
}
