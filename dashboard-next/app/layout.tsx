import "./globals.css";
import type { Metadata } from "next";
export const metadata: Metadata = { title: "Hyperion Control Room", description: "Inference OS simulator" };
export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) { return <html lang="en"><body>{children}</body></html>; }
