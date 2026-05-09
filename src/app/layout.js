import { Outfit } from "next/font/google";
import "./globals.css";

const outfit = Outfit({
  variable: "--font-outfit",
  subsets: ["latin"],
});

export const metadata = {
  title: "CropGuard AI — Plant Disease Detection for Farmers",
  description:
    "Scan your crops with your phone camera. Instantly detect diseases on apple, tomato, grape, potato and 10+ more crops. Simple, accurate, free.",
  keywords: "plant disease detection, crop scanner, farming AI, apple scab, agriculture app",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" data-scroll-behavior="smooth" className={`${outfit.variable}`}>
      <body className="min-h-screen antialiased bg-background text-text font-sans">
        {children}
      </body>
    </html>
  );
}
