import { ClaimBlock } from '@/components/PageShell';
import RouteCards from '@/components/RouteCards';
import FigurePanel from '@/components/FigurePanel';
import StatTile from '@/components/StatTile';
import { GO_TERMS, SITE } from '@/lib/site';

export default function HomePage() {
  return (
    <div className="mx-auto max-w-5xl px-4 py-10 sm:px-6">
      <p className="text-[11px] font-semibold uppercase tracking-[0.16em] text-teal-700">
        {SITE.kicker}
      </p>
      <h1 className="mt-2 text-2xl font-bold tracking-tight text-slate-900 sm:text-3xl">
        {SITE.title}
      </h1>
      <p className="mt-4 max-w-3xl text-lg text-slate-700">{SITE.lead}</p>

      <section className="mt-10 rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500">
          Physical object
        </h2>
        <p className="mt-2 text-slate-800">{SITE.physicalObject}</p>
      </section>

      <div className="mt-8">
        <ClaimBlock />
      </div>

      <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatTile value="5" label="Model families" />
        <StatTile value="3" label="Partition datasets" note="setty · endoderm · dentate" />
        <StatTile value="10" label="Result figures" />
        <StatTile value="MIT" label="Public repository" />
      </div>

      <section className="mt-10 space-y-6">
        <FigurePanel
          src="/figures/F06.png"
          alt="UMAP partitions for setty, endoderm, and dentate plus dentate GO enrichment bars"
          kicker="Fig. 6 · Partition object"
          caption="Saved UMAP embeddings for three partition datasets. Complete GO:####### strings are listed below from regenerated F06D — not a 56×18 heatmap grid. Qualitative grounding, not universal biological validation."
        />
        <div className="rounded-2xl border border-slate-200 bg-white/80 p-6">
          <h2 className="text-lg font-semibold text-slate-900">Dentate GO programs</h2>
          <p className="mt-2 text-sm text-slate-600">
            Strongest term per selected dentate component, transcribed from regenerated F06D. Physical
            biology on this leaf is F06 only.
          </p>
          <ul className="mt-4 space-y-2 font-mono text-sm">
            {GO_TERMS.map((term) => (
              <li key={term.id} className="flex flex-wrap gap-x-3 gap-y-1 text-slate-700">
                <span className="font-semibold text-brand">{term.id}</span>
                <span className="font-sans">{term.name}</span>
              </li>
            ))}
          </ul>
        </div>
      </section>

      <section className="mt-10">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Explore
        </h2>
        <RouteCards />
      </section>
    </div>
  );
}
