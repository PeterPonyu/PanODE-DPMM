'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';
import { badgeEntries, isExternalHref } from '@/lib/badges';
import { ROUTES, SITE } from '@/lib/site';

function navClass(active: boolean): string {
  return ['dpmm-nav', active ? 'is-active' : ''].filter(Boolean).join(' ');
}

function isActive(pathname: string, href: string): boolean {
  if (href === '/') {
    return pathname === '/';
  }
  return pathname === href || pathname.startsWith(`${href}/`);
}

export default function DpmmCodeHeader() {
  const pathname = usePathname() || '/';
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="dpmm-head">
      <div className="dpmm-head-row">
        <Link href="/" className="dpmm-mark" onClick={() => setMenuOpen(false)}>
          <span className="dpmm-mark-box" aria-hidden="true">
            dp
          </span>
          <span>{SITE.navTitle}</span>
        </Link>

        <nav className="dpmm-nav-row" aria-label="Primary">
          {ROUTES.map((item) => (
            <Link key={item.href} href={item.href} className={navClass(isActive(pathname, item.href))}>
              {item.label}
            </Link>
          ))}
        </nav>

        <div className="dpmm-head-links">
          {badgeEntries()
            .filter(({ badge }) => badge.enabled && badge.href)
            .map(({ key, badge }) => (
              <a
                key={key}
                href={badge.href}
                className="dpmm-chip"
                {...(isExternalHref(badge.href ?? '')
                  ? { target: '_blank', rel: 'noopener noreferrer' }
                  : {})}
              >
                {badge.label}
              </a>
            ))}
        </div>

        <button
          type="button"
          className="dpmm-menu-btn"
          aria-expanded={menuOpen}
          aria-controls="dpmm-mobile-nav"
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span>{menuOpen ? 'Close' : 'Menu'}</span>
        </button>
      </div>

      <div id="dpmm-mobile-nav" className={menuOpen ? 'dpmm-mobile is-open' : 'dpmm-mobile'}>
        <nav className="dpmm-mobile-nav" aria-label="Mobile">
          {ROUTES.map((item) => (
            <Link
              key={item.href}
              href={item.href}
              className={navClass(isActive(pathname, item.href))}
              onClick={() => setMenuOpen(false)}
            >
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
