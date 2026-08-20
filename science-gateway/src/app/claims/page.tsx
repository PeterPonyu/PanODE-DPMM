import PageShell from '@/components/PageShell';
import { PAGE_BINDINGS } from '@/lib/site';

/** Unique page module: site boundary, not scientific claims. */
const CLAIMS = PAGE_BINDINGS.claims;

export default function DpmmScopePage() {
  return (
    <PageShell title="Scope" kicker="Public companion">
      <section className="scope-box">
        <h2>On this page</h2>
        <ul>
          <li>How to install the Python package from the public clone</li>
          <li>Which directories the repository tracks</li>
          <li>Where tests and CI live</li>
        </ul>
      </section>
      <section className="scope-box">
        <h2>Not on this page</h2>
        <ul>
          <li>Manuscript figures or captioned result panels</li>
          <li>Benchmark tables, rank lists, or archive/article identifiers</li>
          <li>A journal submission kit</li>
        </ul>
      </section>
      <p>
        Cite software metadata in <code>CITATION.cff</code> if the clone is useful. Cite a
        manuscript only after that manuscript is public.
      </p>
      <p className="sr-only" data-page-id={CLAIMS.pageId}>
        {CLAIMS.pageId} {CLAIMS.runnerId} {CLAIMS.dataId} {CLAIMS.lawId} {CLAIMS.sharedRunner}
      </p>
    </PageShell>
  );
}
