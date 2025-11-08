// Synthetic workload + evaluation driver for AdaptiveIVF (Node.js)
import { AdaptiveIVF } from './adaptive_ivf.mjs';

function makeDataset(n = 40000, d = 64, nClusters = 60) {
  // Gaussian clusters
  const centers = new Array(nClusters);
  for (let i = 0; i < nClusters; i++) {
    const c = new Float32Array(d);
    for (let j = 0; j < d; j++) c[j] = randn() * 4.0;
    centers[i] = c;
  }
  // even-ish allocation
  const per = Math.floor(n / nClusters);
  const X = [];
  const ids = new Int32Array(n);
  let idx = 0;
  for (let i = 0; i < nClusters; i++) {
    const size = i === nClusters - 1 ? (n - idx) : per;
    for (let s = 0; s < size; s++) {
      const v = new Float32Array(d);
      for (let j = 0; j < d; j++) v[j] = centers[i][j] + randn();
      X.push(v);
      ids[idx] = idx;
      idx++;
    }
  }
  return { X, ids };
}

function randn() {
  let u = 0, v = 0;
  while (!u) u = Math.random();
  while (!v) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function zipfSampler(P, alpha = 1.1) {
  // weights ~ 1 / r^alpha
  const ranks = Array.from({ length: P }, (_, i) => i + 1);
  let w = ranks.map(r => 1 / Math.pow(r, alpha));
  const s = w.reduce((a, c) => a + c, 0);
  w = w.map(v => v / s);
  return () => {
    let r = Math.random(), cum = 0;
    for (let i = 0; i < P; i++) { cum += w[i]; if (r <= cum) return i; }
    return P - 1;
  };
}

function percentile(arr, q) {
  const a = [...arr].sort((x, y) => x - y);
  const idx = Math.floor((q / 100) * (a.length - 1));
  return a[idx];
}

async function main() {
  const n = 4000, d = 64, kCoarse = 16, kBase = 4, queries = 500, k = 10, targetRecall = 0.9;

  console.log(`Building dataset n=${n}, d=${d}`);
  const { X, ids } = makeDataset(n, d, 60);

  console.log(`Building index (k_coarse=${kCoarse}, k_base=${kBase})`);
  const idx = new AdaptiveIVF(d, kCoarse, kBase);
  const t0 = Date.now();
  idx.build(X, ids);
  const buildMs = (Date.now() - t0) / 1000;
  console.log(`Build time: ${buildMs.toFixed(2)}s, base partitions: ${idx.baseParts.length}`);

  const P = idx.baseParts.length;
  const sampleP = zipfSampler(P, 1.1);

  const qlat = [], qrec = [], qnprobe = [], qscan = [];

  console.log("Running queries with APS + maintenance...");
  for (let t = 1; t <= queries; t++) {
    const p = sampleP();
    const bp = idx.baseParts[p];
    let v;
    if (!bp || bp.vecs.length === 0) {
      v = X[Math.floor(Math.random() * X.length)];
    } else {
      v = bp.vecs[Math.floor(Math.random() * bp.vecs.length)];
    }
    const q = new Float32Array(d);
    for (let j = 0; j < d; j++) q[j] = v[j] + randn() * 0.1;

    const exact = idx.exactTopK(q, X, Array.from(ids), k);
    const t1 = performance.now();
    const { ids: found, meta } = idx.search(q, k, targetRecall, exact);
    const dt = performance.now() - t1;

    qlat.push(dt);
    qnprobe.push(meta.nprobe);
    qscan.push(meta.scanned);
    qrec.push(meta.recallAtK ?? 0);

    // occasional updates
    if (t % 20 === 0) {
      for (let i = 0; i < 10; i++) {
        const nv = new Float32Array(d);
        for (let j = 0; j < d; j++) nv[j] = randn() * 0.5 + randn();
        const nid = ids.length + Math.floor(Math.random() * 100000);
        idx.insert(nv, nid);
      }
      for (let i = 0; i < 10; i++) {
        const delId = Math.floor(Math.random() * ids.length);
        idx.delete(delId);
      }
    }

    if (t % 50 === 0) idx.maintain();
    if (t % 100 === 0) {
      console.log(`.. q${t}: latency ${dt.toFixed(2)} ms, nprobe ${meta.nprobe}, scanned ${meta.scanned}, recall@${k}=${(meta.recallAtK ?? 0).toFixed(2)}`);
    }
  }

  const mean = a => a.reduce((x, y) => x + y, 0) / a.length;

  console.log("\nAdaptive IVF Demo — Summary");
  console.log(`Queries: ${queries}`);
  console.log(`Avg latency (ms): ${mean(qlat).toFixed(2)}`);
  console.log(`p50 / p95 latency (ms): ${percentile(qlat, 50).toFixed(2)} / ${percentile(qlat, 95).toFixed(2)}`);
  console.log(`Avg nprobe: ${mean(qnprobe).toFixed(2)}`);
  console.log(`Avg vectors scanned: ${mean(qscan).toFixed(0)}`);
  console.log(`Avg recall@${k}: ${mean(qrec).toFixed(3)}`);
}

main().catch(err => { console.error(err); process.exit(1); });
