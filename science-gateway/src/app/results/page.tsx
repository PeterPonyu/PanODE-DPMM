import PageShell from '@/components/PageShell';
import FigurePanel from '@/components/FigurePanel';

const METRIC_FIGURES = [
  { file: 'F07', caption: 'External selection panel — tagged metric, follows F06 biological object.' },
  { file: 'F08', caption: 'External provenance panel — dataset scope and win-rate context.' },
  { file: 'F09', caption: 'Utility and cost trade-off panel.' },
  { file: 'F10', caption: 'Decision map — variant ranking summary.' },
] as const;

export default function ResultsPage() {
  return (
    <PageShell title="Results" kicker="Outcome figures">
      <p>
        Primary outcomes start with the partition object (Fig. 6). F07–F10 are tagged metric panels
        and sit after the biological grounding figure — not a win-rate theatre.
      </p>

      <FigurePanel
        src="/figures/F06.png"
        alt="Biological exploration partitions and GO enrichment"
        kicker="Fig. 6 · Biological exploration"
        caption="Partition UMAPs and dentate GO programs. Complete GO identifiers are transcribed on Home — not truncated in a dense heatmap."
      />

      <div className="grid gap-6">
        {METRIC_FIGURES.map((fig) => (
          <FigurePanel
            key={fig.file}
            src={`/figures/${fig.file}.png`}
            alt={`${fig.file} results panel`}
            kicker={`${fig.file} · metrics`}
            caption={fig.caption}
          />
        ))}
      </div>

      <p className="text-sm text-slate-500">
        Architecture (F01) and sensitivity (F04–F05) panels are referenced on Methods and Evidence.
        The in-repo model-arch-viewer is an internal export tool, not this Site.
      </p>
    </PageShell>
  );
}
