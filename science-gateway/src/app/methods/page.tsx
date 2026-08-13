import PageShell from '@/components/PageShell';

export default function MethodsPage() {
  return (
    <PageShell title="Methods" kicker="Protocol and definitions">
      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Model</h2>
        <p>
          Online Dirichlet process mixture (DPMM) priors refit latent component count during
          autoencoder training. Five model families are compared under a shared preprocessing and
          evaluation protocol.
        </p>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Data scope</h2>
        <ul className="list-disc space-y-2 pl-5">
          <li>Partition datasets: setty, endoderm, dentate</li>
          <li>External benchmarking across named Wilcoxon-matched baselines</li>
          <li>Biological grounding limited to Fig. 6 dentate GO programs</li>
        </ul>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Exclusions</h2>
        <ul className="list-disc space-y-2 pl-5">
          <li>No ODE training jobs from this Site leaf</li>
          <li>No universal biological validation claim from metric panels alone</li>
          <li>F06 GO list is complete text transcription — not a 56×18 enrichment matrix</li>
        </ul>
      </section>

      <section className="space-y-4">
        <h2 className="text-lg font-semibold text-slate-900">Reproducibility</h2>
        <p>
          Public code:{' '}
          <a
            href="https://github.com/PeterPonyu/PanODE-DPMM"
            className="text-brand hover:underline"
            target="_blank"
            rel="noopener noreferrer"
          >
            github.com/PeterPonyu/PanODE-DPMM
          </a>
          . Archive DOI not yet assigned.
        </p>
      </section>
    </PageShell>
  );
}
