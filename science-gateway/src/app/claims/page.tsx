import PageShell from '@/components/PageShell';
import { SITE } from '@/lib/site';

export default function ClaimsPage() {
  return (
    <PageShell title="Claims" kicker="Falsifiable statements">
      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Claim 1 — adaptive partitions</h2>
        <p className="mt-3 text-slate-700">{SITE.primaryClaim}</p>
        <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Would refute
        </h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>DPMM variants fail to recover stable dentate GO programs under reproduced F06 pipeline</li>
          <li>Online component refit collapses to fixed-k behavior on all three partition datasets</li>
        </ul>
      </section>

      <section className="rounded-2xl border border-slate-200 bg-white/80 p-6">
        <h2 className="text-lg font-semibold text-slate-900">Claim 2 — honest metric scope</h2>
        <p className="mt-3 text-slate-700">
          External win-rate panels (F07–F10) describe selection and utility context; they do not
          substitute for Fig. 6 biological grounding.
        </p>
        <h3 className="mt-6 text-sm font-semibold uppercase tracking-wide text-slate-500">
          Out of scope
        </h3>
        <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-slate-600">
          <li>Universal single-cell atlas validation</li>
          <li>Journal venue packaging or invented article DOI</li>
          <li>56×18 GO heatmap as a shipped Site artifact (F06 uses complete term transcription)</li>
        </ul>
      </section>
    </PageShell>
  );
}
