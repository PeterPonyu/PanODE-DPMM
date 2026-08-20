import PageShell from '@/components/PageShell';
import { PACKAGE_ROWS, PAGE_BINDINGS } from '@/lib/site';

/** Unique page module: tracked package inventory, no hosted result panels. */
const RESULTS = PAGE_BINDINGS.results;

export default function DpmmPackagePage() {
  return (
    <PageShell title="Package" kicker="Tracked tree">
      <p>
        The public repository is a Python package plus local runners. Figure scripts stay in the
        tree so a clone can render locally; this site does not publish those outputs.
      </p>
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
      <p className="sr-only" data-page-id={RESULTS.pageId}>
        {RESULTS.pageId} {RESULTS.runnerId} {RESULTS.dataId} {RESULTS.lawId} {RESULTS.sharedRunner}
      </p>
    </PageShell>
  );
}
