import Link from 'next/link';
import { ROUTES } from '@/lib/site';

export default function RouteCards() {
  return (
    <nav className="route-list" aria-label="Companion pages">
      {ROUTES.map((route) => (
        <Link key={route.href} href={route.href} className="route-link">
          <span className="route-id">{route.number}</span>
          <span>
            <strong>{route.label}</strong>
            <span className="route-blurb">{route.blurb}</span>
          </span>
        </Link>
      ))}
    </nav>
  );
}
