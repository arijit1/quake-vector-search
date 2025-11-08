
function randn() {
  // Box–Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function zeros(n, d = null) {
  if (d === null) return new Float32Array(n);
  const out = new Array(n);
  for (let i = 0; i < n; i++) out[i] = new Float32Array(d);
  return out;
}

function copyRow(src) {
  const out = new Float32Array(src.length);
  out.set(src);
  return out;
}

function l2(a, b) {
  let s = 0.0;
  for (let i = 0; i < a.length; i++) {
    const t = a[i] - b[i];
    s += t * t;
  }
  return s;
}

function l2Batch(q, Y) {
  // q: Float32Array(d); Y: Float32Array[] length m
  const m = Y.length;
  const out = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    out[i] = l2(q, Y[i]);
  }
  return out;
}

function argmin(arr) {
  let best = 0, bv = arr[0];
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] < bv) { bv = arr[i]; best = i; }
  }
  return best;
}

function topkIndices(arr, k) {
  // returns indices of k smallest values
  const idx = Array.from(arr, (_, i) => i);
  if (k >= arr.length) return idx.sort((i, j) => arr[i] - arr[j]);
  idx.sort((i, j) => arr[i] - arr[j]);
  return idx.slice(0, k);
}

function meanRows(X) {
  const n = X.length, d = X[0].length;
  const out = new Float32Array(d);
  if (n === 0) return out;
  for (let i = 0; i < n; i++) {
    const r = X[i];
    for (let j = 0; j < d; j++) out[j] += r[j];
  }
  for (let j = 0; j < d; j++) out[j] /= n;
  return out;
}

function kmeans(X, k, iters = 15, seed = 0) {
  // very small k-means for demo
  // init by sampling k distinct points
  const n = X.length, d = X[0].length;
  // simple seeded-ish sampler
  function sRand(i) {
    // deterministic-ish but lightweight
    const a = Math.sin((i + 1) * 9301 + seed * 49297) * 233280;
    return a - Math.floor(a);
  }
  const chosen = new Set();
  const centroids = zeros(k, d);
  let c = 0;
  while (c < k) {
    const idx = Math.floor(sRand(c) * n);
    if (!chosen.has(idx)) {
      chosen.add(idx);
      centroids[c] = copyRow(X[idx]);
      c++;
    }
  }

  let assign = new Int32Array(n);
  for (let it = 0; it < iters; it++) {
    // assign
    for (let i = 0; i < n; i++) {
      const dists = new Float64Array(k);
      for (let j = 0; j < k; j++) dists[j] = l2(X[i], centroids[j]);
      assign[i] = argmin(dists);
    }
    // recompute centroids
    const sums = zeros(k, d);
    const counts = new Int32Array(k);
    for (let i = 0; i < n; i++) {
      const a = assign[i];
      const row = X[i];
      counts[a]++;
      const s = sums[a];
      for (let j = 0; j < d; j++) s[j] += row[j];
    }
    for (let j = 0; j < k; j++) {
      if (counts[j] > 0) {
        for (let t = 0; t < d; t++) centroids[j][t] = sums[j][t] / counts[j];
      } else {
        // reseed to a random point
        const ridx = Math.floor(Math.random() * n);
        centroids[j] = copyRow(X[ridx]);
      }
    }
  }
  return { centroids, assign };
}

class BasePartition {
  constructor(vecs, ids, centroid) {
    this.vecs = vecs;          // Float32Array[] of length m
    this.ids = ids;            // Int32Array of length m
    this.centroid = centroid;  // Float32Array(d)
    this.hits = 0;
    this.lastSplitAt = 0;
  }
}

class CoarseCell {
  constructor(centroid) {
    this.centroid = centroid; // Float32Array(d)
    this.baseIds = [];        // indices into index.baseParts
  }
}

export class AdaptiveIVF {
  constructor(dim, kCoarse = 16, kBase = 4) {
    this.dim = dim;
    this.kCoarse = kCoarse;
    this.kBase = kBase;
    this.coarse = [];
    this.baseParts = [];
    this.id2loc = new Map(); // id -> [baseIdx, offset]
    this.queryCounter = 0;

    // thresholds (toy)
    this.splitSize = 3000;
    this.mergeSize = 300;
    this.hotSplitMultiplier = 1.5;
  }

  build(X, ids = null, coarseK = null, baseK = null) {
    if (coarseK) this.kCoarse = coarseK;
    if (baseK) this.kBase = baseK;
    if (!ids) {
      ids = new Int32Array(X.length);
      for (let i = 0; i < ids.length; i++) ids[i] = i;
    }

    const { centroids: coarseC, assign: coarseA } = kmeans(X, this.kCoarse, 12, 42);
    this.coarse = coarseC.map(c => new CoarseCell(c));

    // per coarse cell, split into base clusters
    for (let cId = 0; cId < this.kCoarse; cId++) {
      const group = [];
      const gid = [];
      for (let i = 0; i < X.length; i++) if (coarseA[i] === cId) { group.push(X[i]); gid.push(ids[i]); }
      if (group.length === 0) continue;

      const kb = Math.min(this.kBase, Math.max(1, Math.floor(group.length / 50)));
      const { assign: baseA } = kmeans(group, kb, 10, 123 + cId);

      for (let b = 0; b < kb; b++) {
        const partVecs = [];
        const partIds = [];
        for (let i = 0; i < group.length; i++) if (baseA[i] === b) { partVecs.push(group[i]); partIds.push(gid[i]); }
        if (partVecs.length === 0) continue;
        const centroid = meanRows(partVecs);
        const bp = new BasePartition(partVecs, Int32Array.from(partIds), centroid);
        this.baseParts.push(bp);
        const bidx = this.baseParts.length - 1;
        this.coarse[cId].baseIds.push(bidx);
        for (let off = 0; off < partIds.length; off++) {
          this.id2loc.set(partIds[off], [bidx, off]);
        }
      }
    }
  }

  insert(v, id) {
    // route to nearest coarse, then nearest base
    const cd = this.coarse.map(c => l2(v, c.centroid));
    const cIdx = argmin(cd);
    const baseIdxs = this.coarse[cIdx].baseIds;
    if (baseIdxs.length === 0) {
      const bp = new BasePartition([v], Int32Array.from([id]), copyRow(v));
      this.baseParts.push(bp);
      const bidx = this.baseParts.length - 1;
      this.coarse[cIdx].baseIds.push(bidx);
      this.id2loc.set(id, [bidx, 0]);
      return;
    }
    let best = -1, bd = Infinity;
    for (const b of baseIdxs) {
      const d = l2(v, this.baseParts[b].centroid);
      if (d < bd) { bd = d; best = b; }
    }
    const bp = this.baseParts[best];
    bp.vecs.push(v);
    const newIds = new Int32Array(bp.ids.length + 1);
    newIds.set(bp.ids, 0); newIds[newIds.length - 1] = id;
    bp.ids = newIds;
    bp.centroid = meanRows(bp.vecs);
    this.id2loc.set(id, [best, bp.vecs.length - 1]);
  }

  delete(id) {
    const loc = this.id2loc.get(id);
    if (!loc) return;
    const [bidx, off] = loc;
    const bp = this.baseParts[bidx];
    const last = bp.vecs.length - 1;
    // swap-remove
    [bp.vecs[off], bp.vecs[last]] = [bp.vecs[last], bp.vecs[off]];
    const ids = bp.ids;
    const tmp = ids[off]; ids[off] = ids[last]; ids[last] = tmp;
    bp.vecs.pop();
    bp.ids = ids.slice(0, ids.length - 1);
    if (bp.vecs.length > 0) bp.centroid = meanRows(bp.vecs);
    if (off < bp.ids.length) this.id2loc.set(bp.ids[off], [bidx, off]);
    this.id2loc.delete(id);
  }

  _partitionScores(q) {
    const centroids = this.baseParts.map(bp => bp.centroid);
    const d2 = l2Batch(q, centroids); // Float64Array
    const size = this.baseParts.map(bp => bp.vecs.length);
    // temperature ~ median distance
    const sqrtD = Array.from(d2, Math.sqrt);
    const sorted = [...sqrtD].sort((a, b) => a - b);
    const mid = sorted.length ? sorted[Math.floor(sorted.length / 2)] : 1e-6;
    const tau = mid + 1e-6;

    // logits = -sqrt(d)/tau + 0.5*log(size+1)
    const logits = sqrtD.map((sd, i) => -sd / tau + 0.5 * Math.log(size[i] + 1.0));
    const m = Math.max(...logits);
    const exps = logits.map(v => Math.exp(v - m));
    const sum = exps.reduce((a, c) => a + c, 0);
    const p = exps.map(v => v / sum);

    const arr = p.map((pi, i) => ({ bidx: i, p: pi, sz: size[i] }));
    arr.sort((a, b) => b.p - a.p);
    return arr;
  }

  _chooseNProbe(probs, targetRecall = 0.9, maxProbe = 64) {
    let cum = 0.0;
    for (let i = 0; i < probs.length && i < maxProbe; i++) {
      cum += probs[i].p;
      if (cum >= targetRecall) return i + 1;
    }
    return Math.min(probs.length, maxProbe);
  }

  search(q, k = 10, targetRecall = 0.9, exactRef = null) {
    this.queryCounter++;
    const probs = this._partitionScores(q);
    const nprobe = this._chooseNProbe(probs, targetRecall, 64);

    const candIds = [];
    const candVecs = [];
    for (let i = 0; i < nprobe; i++) {
      const bidx = probs[i].bidx;
      const bp = this.baseParts[bidx];
      bp.hits++;
      if (bp.vecs.length === 0) continue;
      candIds.push(...bp.ids);
      candVecs.push(...bp.vecs);
    }
    if (candVecs.length === 0) {
      return { ids: [], d2: [], meta: { nprobe: 0, scanned: 0, recallAtK: null } };
    }

    // distances
    const dists = new Float64Array(candVecs.length);
    for (let i = 0; i < candVecs.length; i++) dists[i] = l2(q, candVecs[i]);

    const tk = topkIndices(dists, Math.min(k, dists.length));
    const foundIds = tk.map(i => candIds[i]);
    const foundD2 = tk.map(i => dists[i]);

    let rec = null;
    if (exactRef && exactRef.length > 0) {
      const setRef = new Set(exactRef);
      let inter = 0;
      for (const id of foundIds) if (setRef.has(id)) inter++;
      rec = inter / Math.min(k, exactRef.length);
    }
    return { ids: foundIds, d2: foundD2, meta: { nprobe, scanned: candVecs.length, recallAtK: rec } };
  }

  maintain(hotWindow = 2000) {
    // SPLIT hot/large partitions
    for (let bidx = 0; bidx < this.baseParts.length; bidx++) {
      const bp = this.baseParts[bidx];
      const size = bp.vecs.length;
      const hot = (bp.hits - bp.lastSplitAt);
      let splitThresh = this.splitSize / Math.max(1.0, (hot / hotWindow));
      splitThresh = Math.max(this.splitSize / this.hotSplitMultiplier, Math.min(this.splitSize * 2, splitThresh));
      if (size >= splitThresh && size >= 16) {
        const { assign } = kmeans(bp.vecs, 2, 8, 17 + bidx);
        const g0 = [], g1 = [], i0 = [], i1 = [];
        for (let i = 0; i < bp.vecs.length; i++) {
          if (assign[i] === 0) { g0.push(bp.vecs[i]); i0.push(bp.ids[i]); }
          else { g1.push(bp.vecs[i]); i1.push(bp.ids[i]); }
        }
        if (g0.length && g1.length) {
          const bp0 = new BasePartition(g0, Int32Array.from(i0), meanRows(g0));
          const bp1 = new BasePartition(g1, Int32Array.from(i1), meanRows(g1));
          this.baseParts[bidx] = bp0;
          this.baseParts.push(bp1);
          const newIdx = this.baseParts.length - 1;
          for (let off = 0; off < i0.length; off++) this.id2loc.set(i0[off], [bidx, off]);
          for (let off = 0; off < i1.length; off++) this.id2loc.set(i1[off], [newIdx, off]);
          this.baseParts[bidx].lastSplitAt = this.queryCounter;
          this.baseParts[newIdx].lastSplitAt = this.queryCounter;
        }
      }
    }

    // MERGE tiny partitions (nearest-centroid pairing)
    const tiny = [];
    for (let i = 0; i < this.baseParts.length; i++) {
      if (this.baseParts[i].vecs.length <= this.mergeSize) tiny.push(i);
    }
    const used = new Set();
    for (const i of tiny) {
      if (used.has(i)) continue;
      const ci = this.baseParts[i].centroid;
      let bestJ = -1, bd = Infinity;
      for (const j of tiny) {
        if (j === i || used.has(j)) continue;
        const cj = this.baseParts[j].centroid;
        const d = l2(ci, cj);
        if (d < bd) { bd = d; bestJ = j; }
      }
      if (bestJ < 0) continue;
      const bpi = this.baseParts[i], bpj = this.baseParts[bestJ];
      const vecs = bpi.vecs.concat(bpj.vecs);
      const ids = new Int32Array(bpi.ids.length + bpj.ids.length);
      ids.set(bpi.ids, 0); ids.set(bpj.ids, bpi.ids.length);
      bpi.vecs = vecs; bpi.ids = ids; bpi.centroid = meanRows(vecs);
      for (let off = 0; off < ids.length; off++) this.id2loc.set(ids[off], [i, off]);
      // mark j as empty
      bpj.vecs = []; bpj.ids = new Int32Array(0);
      used.add(i); used.add(bestJ);
    }
  }

  exactTopK(q, allVecs, allIds, k) {
    const dists = new Float64Array(allVecs.length);
    for (let i = 0; i < allVecs.length; i++) dists[i] = l2(q, allVecs[i]);
    const tk = topkIndices(dists, Math.min(k, dists.length));
    return tk.map(i => allIds[i]);
  }
}
