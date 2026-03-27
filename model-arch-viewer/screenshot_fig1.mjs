import { chromium } from "playwright";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const REPO_ROOT = path.resolve(__dirname, "..");
const BASE = `${process.env.PANODE_VIEWER_BASE_URL ?? "http://localhost:3099"}/figure1`;
const OUT = process.env.PANODE_PAPER_FIGURES_DIR ?? path.join(REPO_ROOT, "benchmarks", "paper_figures");

for (const series of ["dpmm", "topic"]) {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 700, height: 700 }, deviceScaleFactor: 3 });
  await page.goto(`${BASE}/${series}`, { waitUntil: "networkidle" });
  await page.waitForTimeout(2000);

  const el = await page.$("#figure1-root");
  if (el) {
    fs.mkdirSync(path.join(OUT, series), { recursive: true });
    const outPath = path.join(OUT, series, `Fig1_arch_${series}.png`);
    await el.screenshot({ path: outPath, type: "png" });
    console.log(`Saved: ${outPath}`);
  } else {
    console.error(`#figure1-root not found for ${series}`);
  }
  await browser.close();
}
