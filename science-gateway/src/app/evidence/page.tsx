import PageShell from '@/components/PageShell';
import { PAGE_BINDINGS } from '@/lib/site';

/** Unique page module: tests and CI, not a result table. */
const EVIDENCE = PAGE_BINDINGS.evidence;

export default function DpmmTestsPage() {
  return (
    <PageShell title="Tests" kicker="Automated checks">
      <p>
        The public workflow <code>.github/workflows/ci.yml</code> runs the repository test suite.
        That badge reports job status. It is not a scientific scoreboard.
      </p>
      <pre className="cmd">
        <code>{`pre-commit install
pytest`}</code>
      </pre>
      <p>
        An optional local architecture viewer lives under <code>model-arch-viewer/</code> and is
        started with <code>npm install</code> then <code>npm run dev</code> in that directory.
      </p>
      <p className="sr-only" data-page-id={EVIDENCE.pageId}>
        {EVIDENCE.pageId} {EVIDENCE.runnerId} {EVIDENCE.dataId} {EVIDENCE.lawId}{' '}
        {EVIDENCE.sharedRunner}
      </p>
    </PageShell>
  );
}
