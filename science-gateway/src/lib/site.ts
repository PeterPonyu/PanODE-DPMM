/**
 * PanODE-DPMM public companion page.
 * Code description only — no manuscript figures or result tables.
 */
export const SITE = {
  slug: 'PanODE-DPMM',
  navTitle: 'PanODE-DPMM',
  title: 'PanODE-DPMM',
  kicker: 'Public Python package',
  lead:
    'Research code for Dirichlet-process-mixture regularised autoencoders, ablation variants, and local evaluation utilities.',
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
    label: 'GitHub',
    href: SITE.github,
    enabled: true,
  } satisfies BadgeConfig,
  site: {
    label: 'This page',
    href: 'https://peterponyu.github.io/PanODE-DPMM/',
    enabled: true,
  } satisfies BadgeConfig,
  archive: {
    label: 'Archive',
    enabled: false,
    disabledReason: 'No archive record on this page',
  } satisfies BadgeConfig,
  articleDoi: {
    label: 'Article DOI',
    enabled: false,
    disabledReason: 'No article DOI on this page',
  } satisfies BadgeConfig,
} as const;

export const ROUTES = [
  {
    href: '/results',
    label: 'Package',
    number: '01',
    blurb: 'Tracked Python modules, benchmarks, and local figure tooling.',
  },
  {
    href: '/methods',
    label: 'Install',
    number: '02',
    blurb: 'Editable install extras and optional dataset path variables.',
  },
  {
    href: '/evidence',
    label: 'Tests',
    number: '03',
    blurb: 'pytest, pre-commit, and the public CI workflow.',
  },
  {
    href: '/claims',
    label: 'Scope',
    number: '04',
    blurb: 'This page describes the repository. It is not a journal article.',
  },
] as const;

export const PACKAGE_ROWS = [
  { path: 'models/', note: 'DPMM-regularised autoencoder variants and shared layers' },
  { path: 'eval_lib/', note: 'Portable evaluation helpers used by local runners' },
  { path: 'benchmarks/', note: 'Smoke and series runners for a caller-supplied .h5ad path' },
  { path: 'experiments/', note: 'Orchestration entry points for optional local jobs' },
  { path: 'refined_figures/', note: 'Scripts that emit figures on the machine that has the data' },
  { path: 'model-arch-viewer/', note: 'Optional local Next.js preview; not this public page' },
  { path: 'tests/', note: 'Automated checks run by pytest and CI' },
] as const;

/** Unique infra binding per route. Shared chrome is chrome.dpmm-code-sheet only. */
export type PageBinding = {
  pageId: string;
  runnerId: string;
  dataId: string;
  lawId: string;
  sharedRunner: 'chrome.dpmm-code-sheet';
};

export const PAGE_BINDINGS = {
  home: {
    pageId: 'dpmm.page.code-home',
    runnerId: 'dpmm.runner.package-sheet',
    dataId: 'dpmm.data.repo-layout',
    lawId: 'dpmm.law.code-companion-not-article',
    sharedRunner: 'chrome.dpmm-code-sheet',
  },
  results: {
    pageId: 'dpmm.page.package',
    runnerId: 'dpmm.runner.module-inventory',
    dataId: 'dpmm.data.tracked-python-tree',
    lawId: 'dpmm.law.no-hosted-result-panels',
    sharedRunner: 'chrome.dpmm-code-sheet',
  },
  methods: {
    pageId: 'dpmm.page.install',
    runnerId: 'dpmm.runner.editable-extras',
    dataId: 'dpmm.data.pip-extras-dev-bio-graph',
    lawId: 'dpmm.law.caller-supplies-h5ad',
    sharedRunner: 'chrome.dpmm-code-sheet',
  },
  evidence: {
    pageId: 'dpmm.page.tests',
    runnerId: 'dpmm.runner.pytest-ci',
    dataId: 'dpmm.data.workflow-ci-yml',
    lawId: 'dpmm.law.ci-is-not-a-result-table',
    sharedRunner: 'chrome.dpmm-code-sheet',
  },
  claims: {
    pageId: 'dpmm.page.scope',
    runnerId: 'dpmm.runner.site-boundary',
    dataId: 'dpmm.data.companion-copy',
    lawId: 'dpmm.law.no-manuscript-on-pages',
    sharedRunner: 'chrome.dpmm-code-sheet',
  },
} as const satisfies Record<string, PageBinding>;
