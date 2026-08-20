import PageShell from '@/components/PageShell';
import { PAGE_BINDINGS, SITE } from '@/lib/site';

/** Unique page module: editable install and runner flags. */
const METHODS = PAGE_BINDINGS.methods;

export default function DpmmInstallPage() {
  return (
    <PageShell title="Install" kicker="Local setup">
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Editable extras</h2>
        <pre className="cmd">
          <code>{`pip install -e ".[dev]"
pip install -e ".[dev,bio]"
pip install -e ".[dev,graph]"`}</code>
        </pre>
      </section>
      <section className="space-y-3">
        <h2 className="text-lg font-semibold">Runner inputs</h2>
        <p>
          Benchmark entry points take a caller-supplied <code>--data-path</code>. Optional
          environment variables: <code>PANODE_DATASETS_ROOT</code>,{' '}
          <code>PANODE_DEFAULT_DATASET</code>.
        </p>
        <pre className="cmd">
          <code>{`python -m benchmarks.runners.benchmark_base \\
  --data-path /path/to/data.h5ad \\
  --epochs 5 \\
  --no-early-stopping \\
  --series dpmm`}</code>
        </pre>
      </section>
      <p>
        Clone:{' '}
        <a href={SITE.github} className="underline" target="_blank" rel="noopener noreferrer">
          {SITE.github.replace('https://', '')}
        </a>
      </p>
      <p className="sr-only" data-page-id={METHODS.pageId}>
        {METHODS.pageId} {METHODS.runnerId} {METHODS.dataId} {METHODS.lawId} {METHODS.sharedRunner}
      </p>
    </PageShell>
  );
}
