#!/usr/bin/env node
/**
 * Checks for the static export: required routes, no journal chrome, no unpublished-result leak.
 */
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs';
import { join, relative } from 'node:path';

const out = join(process.cwd(), 'out');
const required = [
  'index.html',
  'results/index.html',
  'methods/index.html',
  'evidence/index.html',
  'claims/index.html',
  '.nojekyll',
];
const forbidden = ['abstract', 'cite', 'team'];
const denylist = ['PEERJ_REVIEWER_FAQ.md', 'PEERJ_PORTAL_INPUTS.txt', 'superpowers'];
const leakPatterns = [
  /unpublished results/i,
  /Science Gateway/i,
  /Wilcoxon/i,
  /GO:\d{7}/,
  /Fig\.?\s*6/i,
  /\bF0[6-9]\b/,
  /\bF10\b/,
  /Primary claim/i,
  /neuron differentiation/i,
  /synaptic vesicle/i,
  /win-rate/i,
  /ZF Lab/i,
];

let failed = 0;

for (const rel of required) {
  const p = join(out, rel);
  if (!existsSync(p)) {
    console.error(`FAIL G1: missing ${rel}`);
    failed += 1;
  }
}

for (const dir of forbidden) {
  if (existsSync(join(out, dir))) {
    console.error(`FAIL G3: forbidden route directory out/${dir}/`);
    failed += 1;
  }
}

function walk(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (denylist.includes(entry.name)) {
        console.error(`FAIL G9: denylist dir ${p}`);
        failed += 1;
      }
      walk(p);
    } else if (denylist.some((d) => entry.name.includes(d))) {
      console.error(`FAIL G9: denylist file ${p}`);
      failed += 1;
    }
  }
}

function scanLeaks(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) {
      if (entry.name === 'figures') {
        console.error(`FAIL leak: exported figures directory ${relative(out, p)}`);
        failed += 1;
      }
      scanLeaks(p);
      continue;
    }
    if (/\.(png|pdf|svg|jpe?g|webp)$/i.test(entry.name)) {
      console.error(`FAIL leak: exported binary ${relative(out, p)}`);
      failed += 1;
      continue;
    }
    if (!/\.(html|txt)$/i.test(entry.name)) {
      continue;
    }
    const text = readFileSync(p, 'utf8');
    for (const re of leakPatterns) {
      if (re.test(text)) {
        console.error(`FAIL leak: ${re} in ${relative(out, p)}`);
        failed += 1;
      }
    }
  }
}

if (existsSync(out)) {
  walk(out);
  scanLeaks(out);
  const html = readFileSync(join(out, 'index.html'), 'utf8');
  if (/github\.com\/PeterPonyu\/HetCLOP/i.test(html)) {
    console.error('FAIL G6: private HetCLOP Code href in index.html');
    failed += 1;
  }
  for (const label of ['Abstract', 'Cite', 'Team']) {
    if (new RegExp(`>${label}<`, 'i').test(html)) {
      console.error(`FAIL G3: journal nav label "${label}" in index.html`);
      failed += 1;
    }
  }
  if (/Get started|Try now|Launch/i.test(html)) {
    console.error('FAIL G7: product headline pattern in index.html');
    failed += 1;
  }
  if (statSync(join(out, 'index.html')).size < 200) {
    console.error('FAIL G1: index.html too small');
    failed += 1;
  }
}

if (failed) {
  process.exit(1);
}

console.log(`verify-export: ok (${required.length} required paths, leak-scan clean)`);
