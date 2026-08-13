/**
 * PanODE-DPMM science gateway site config.
 * Fail-closed: Article DOI disabled until real assignment.
 */
export const SITE = {
  slug: 'PanODE-DPMM',
  navTitle: 'PanODE-DPMM',
  title:
    'Online Dirichlet process mixture priors adapt latent cluster count during autoencoder training',
  kicker: 'ZF Lab · latent partitions · GO programs',
  lead:
    'The physical object is a single-cell latent partition on setty, endoderm, and dentate, plus dentate GO gene programs from Fig. 6 — not a win-rate dashboard.',
  physicalObject:
    'Saved UMAP partitions and dentate GO enrichment bars (Fig. 6). F07–F10 are tagged metric panels and follow the biological object.',
  primaryClaim:
    'DPMM-family autoencoders recover interpretable dentate GO programs while refitting component count online during training.',
  homepage: 'https://peterponyu.github.io/',
  scportal: 'https://peterponyu.github.io/scportal/',
  github: 'https://github.com/PeterPonyu/PanODE-DPMM',
} as const;

export type BadgeConfig = {
  label: string;
  href?: string;
  enabled: boolean;
  disabledReason?: string;
};

export const BADGES = {
  code: {
    label: 'Code',
    href: SITE.github,
    enabled: true,
  } satisfies BadgeConfig,
  site: {
    label: 'Site',
    href: 'https://peterponyu.github.io/PanODE-DPMM/',
    enabled: true,
  } satisfies BadgeConfig,
  archive: {
    label: 'Archive',
    enabled: false,
    disabledReason: 'No Zenodo record yet',
  } satisfies BadgeConfig,
  articleDoi: {
    label: 'Article DOI',
    enabled: false,
    disabledReason: 'On acceptance',
  } satisfies BadgeConfig,
} as const;

export const ROUTES = [
  { href: '/results', label: 'Results', number: '01', blurb: 'Fig. 6 partitions and GO programs; metric panels F07–F10.' },
  { href: '/methods', label: 'Methods', number: '02', blurb: 'DPMM prior, training protocol, dataset scope.' },
  { href: '/evidence', label: 'Evidence', number: '03', blurb: 'Wilcoxon externals, sensitivity, runtime.' },
  { href: '/claims', label: 'Claims', number: '04', blurb: 'Falsifiable statements and refutation hooks.' },
] as const;

export const GO_TERMS = [
  { id: 'GO:0030182', name: 'neuron differentiation' },
  { id: 'GO:0099504', name: 'synaptic vesicle cycle' },
  { id: 'GO:0021782', name: 'glial cell development' },
  { id: 'GO:0045666', name: 'positive regulation of neuron differentiation' },
  { id: 'GO:0007411', name: 'axon guidance' },
  { id: 'GO:0140059', name: 'dendrite arborization' },
] as const;
