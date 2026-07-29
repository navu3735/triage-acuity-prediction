"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import "./globals.css";

const links = [
  { href: "/", label: "Operations" },
  { href: "/intake", label: "Emergency Intake" },
  { href: "/patients", label: "Active Cases" },
  { href: "/ai", label: "AI Workflows" },
];

export default function RootLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <html lang="en">
      <body>
        <div className="shell">
          <aside className="sidebar">
            <div className="brand">
              <strong>Smartflow</strong>
              <span>AI Care</span>
            </div>
            <nav className="nav">
              {links.map((link) => (
                <Link
                  key={link.href}
                  href={link.href}
                  className={pathname === link.href ? "active" : undefined}
                >
                  {link.label}
                </Link>
              ))}
            </nav>
          </aside>
          <main className="content">{children}</main>
        </div>
      </body>
    </html>
  );
}
