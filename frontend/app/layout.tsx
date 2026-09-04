import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Revora",
  description: "AI-Powered Revenue Recovery & Payment Intelligence",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
