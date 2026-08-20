import { ROUTES, SITE } from '@/lib/site';

export default function DpmmCodeFooter() {
  return (
    <footer className="dpmm-foot">
      <div className="dpmm-foot-row">
        <span>{SITE.title} code companion</span>
        <div className="dpmm-foot-links">
          {ROUTES.map((route) => (
            <a key={route.href} href={route.href}>
              {route.label}
            </a>
          ))}
          <a href={SITE.github} target="_blank" rel="noopener noreferrer">
            GitHub
          </a>
        </div>
      </div>
    </footer>
  );
}
