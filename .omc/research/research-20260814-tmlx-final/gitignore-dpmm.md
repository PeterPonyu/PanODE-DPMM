# PanODE-DPMM — Final Hygiene: .gitignore + README Leak Audit

Date: 2026-08-14 · Repo: `PeterPonyu/PanODE-DPMM` (local: `~/Desktop/labs/PanODE-DPMM`) · Branch: `main`

## 1. Tracked AI/process artifacts on public tree?

**No.** Scanned all 7 remote branches (`main`, `chore/public-path-scrub`, 5× `dependabot/*`) recursively, via both `git ls-tree` and the live GitHub Trees API, for:

- `.cursor/`, `.cursor.cloud`, `.cursorignore`
- `.omc/`, `.omx/`, `.claude/`, `.codex/`
- `agent-transcripts/`, `*.canvas.tsx`
- `.env`, `.env.*`

Zero matches on every branch. Nothing to `git rm --cached`.

### `.cursor.cloud` — user-reported exposure: NOT confirmed

- Not present in any branch's git tree (local or live GitHub API).
- Not present in the local working tree (`ls -la` at repo root).
- Not served by the deployed Pages site (`/.cursor.cloud`, `/.cursorignore`, `/.cursor/` all return 404 on `peterponyu.github.io/PanODE-DPMM/`).
- `.cursor/` dirs *do* exist locally in **other** repos under `~/Desktop/labs/` (`active/PeterPonyu.github.io`, `active/HetCLOP`, etc.) — out of scope (PanODE-DPMM only), but the report may have originated there.

**Verdict: not leaked, not tracked. Report appears stale or refers to a different repo.**

## 2. .gitignore coverage — FIXED

Before: covered `.omc/`, `.omx/`, `.claude/worktrees/` (partial), `.env`, `.vscode/`, `.idea/`. Missing: `.cursor/`, `.cursor.cloud`, `.cursorignore`, `.codex/`, full `.claude/`, `agent-transcripts/`, `*.canvas.tsx`, `.env.*`.

Commit `9ce40d1` (pushed to `origin/main`, hooks ran, no `--no-verify`, no force-push) adds:

```
.claude/          # broadened from .claude/worktrees/
.cursor/
.cursor.cloud
.cursorignore
.codex/
agent-transcripts/
*.canvas.tsx
.env.*            # with !.env.example exception
.aider*  .continue/  .codeium/  .windsurf/
.credentials/  credentials/  credentials.*  *.pem  *.key
```

Verified with `git check-ignore`: all listed patterns ignore local AI/editor state and credential material; `.env.example` correctly *not* ignored. `git ls-files` re-checked post-commit: still zero tracked matches.

## 3. README / root docs packaging leak?

**No leak found.** Audited `README.md`, `CITATION.cff`, and the `docs/` static export (the only tracked root doc set) for: venue, manuscript, submission, bibliography-kit, "under review", product-splash, invented IDs.

- `README.md` citation block: `@article{Fu2026PanODEDPMM, ... note = {Preprint}, doi = {10.64898/2026.03.26.714611}}`.
  - DOI **resolves**: `doi.org` 302 → `biorxiv.org/lookup/doi/10.64898/2026.03.26.714611`; Crossref confirms a real bioRxiv record (posted 2026-03-30, group "Bioinformatics"). **Not an invented ID.** Public preprint pointer = science content, kept.
- `CITATION.cff`: "…the associated manuscript once it is public." Mild, generic pointer; no venue/submission detail. Kept (science-content wording).
- `docs/claims/index.*`: only match is an explicit *disclaimer* — "Out of scope: Journal venue packaging or invented article DOI". Anti-leak statement, not a leak. Kept.
- No "under review", no venue name, no submission/packaging references, no product-splash anywhere in README or docs.

## 4. Hard stops respected

No GPU/ODE/CPTAC/build_pdf/Tessera-PDF/Option-B-55-row/DOI changes. No mega-portal. Science-content wording untouched. Scope kept to `PanODE-DPMM`.

## Summary

| Check | Result |
|---|---|
| `.cursor.cloud` tracked/leaked? | **No** — absent from all branches, working tree, and Pages site |
| AI artifacts tracked anywhere? | **No** |
| .gitignore gaps | **Fixed** — commit `9ce40d1`, pushed to `origin/main` |
| `git rm --cached` needed? | No (nothing tracked) |
| README/docs packaging leak? | **No** — DOI is a real resolving bioRxiv preprint; only anti-leak disclaimer found |
