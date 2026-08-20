import type { Metadata } from 'next';
import SiteHeader from '@/components/SiteHeader';
import FooterSitemap from '@/components/FooterSitemap';
import { SITE } from '@/lib/site';
import './globals.css';

export const metadata: Metadata = {
  title: `${SITE.title} code companion`,
  description: SITE.lead,
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="dpmm-sheet flex min-h-screen flex-col">
        <SiteHeader />
        <main className="w-full flex-1">{children}</main>
        <FooterSitemap />
      </body>
    </html>
  );
}
