import "./globals.css";
import type { Metadata } from "next";

export const metadata: Metadata = {
  title: "Hyperion | Inference Optimization Lab",
  description: "Compare Vidur inference configurations against latency SLOs and inspect the performance trade-offs.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return <html lang="en"><body>{children}</body></html>;
}
