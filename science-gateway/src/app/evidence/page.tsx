import PageShell from '@/components/PageShell';
import StatTile from '@/components/StatTile';

export default function EvidencePage() {
  return (
    <PageShell title="Evidence" kicker="Metrics and controls">
      <p>
        Verifier-gated numbers from matched external benchmarks. Metric panels are labeled as
        evidence, not marketing stats.
      </p>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <StatTile value="11" label="Named Wilcoxon externals" note="Matched comparison set" />
        <StatTile value="37" label="Metrics per external" note="Full ranking grid" />
        <StatTile value="407" label="Wilcoxon rows" note="11 × 37 matched pairs" />
      </div>

      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Negative controls and scope</h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-slate-700">
          <li>F06 GO programs are qualitative dentate grounding — not pan-tissue validation</li>
          <li>F07–F10 report selection, provenance, utility, and decision metrics separately</li>
          <li>Sensitivity and training dynamics (F04–F05) bound hyperparameter stability</li>
        </ul>
      </section>

      <section className="rounded-2xl border border-amber-200 bg-amber-50/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">What would refute</h2>
        <p className="mt-2 text-slate-700">
          Dentate GO terms absent under reproduced F06D pipeline, or partition structure not stable
          across matched reruns on setty/endoderm/dentate.
        </p>
      </section>
    </PageShell>
  );
}
