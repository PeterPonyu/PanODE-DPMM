import Link from 'next/link';
import { PACKAGE_ROWS, PAGE_BINDINGS, ROUTES, SITE } from '@/lib/site';

/** Unique page module for the PanODE-DPMM code companion home. */
const HOME = PAGE_BINDINGS.home;

export default function DpmmCodeHomePage() {
  return (
    <div className="sheet-wrap">
      <p className="kicker">{SITE.kicker}</p>
      <h1 className="sheet-title">{SITE.title}</h1>
      <p className="lead">{SITE.lead}</p>

      <aside className="scope-box">
        <h2>What this page is</h2>
        <p>
          A companion for the public GitHub tree. It lists install commands and module paths. It
          does not host manuscript figures, result tables, or an article DOI.
        </p>
      </aside>

      <section className="sheet-grid">
        <div className="sheet-card">
          <h2>Package map</h2>
          <dl className="pkg-list">
            {PACKAGE_ROWS.map((row) => (
              <div key={row.path} className="pkg-row">
                <dt>
                  <code>{row.path}</code>
                </dt>
                <dd>{row.note}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="sheet-card">
          <h2>Install</h2>
          <pre className="cmd">
            <code>pip install -e &quot;.[dev]&quot;</code>
          </pre>
          <p className="fine">
            Optional extras: <code>dev,bio</code> and <code>dev,graph</code>. Point runners at a
            local <code>.h5ad</code> with <code>--data-path</code> or{' '}
            <code>PANODE_DATASETS_ROOT</code>.
          </p>
          <p className="fine">
            Source:{' '}
            <a href={SITE.github} target="_blank" rel="noopener noreferrer">
              github.com/PeterPonyu/PanODE-DPMM
            </a>
          </p>
        </div>
      </section>

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

      <p className="sr-only" data-page-id={HOME.pageId}>
        {HOME.pageId} {HOME.runnerId} {HOME.dataId} {HOME.lawId} {HOME.sharedRunner}
      </p>
    </div>
  );
}
