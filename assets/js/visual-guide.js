/* Small, deterministic mathematical experiments; no plotting dependencies. */
(() => {
  'use strict';
  const svgNS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs = {}, text) => {
    const node = document.createElementNS(svgNS, tag);
    Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, value));
    if (text !== undefined) node.textContent = text;
    return node;
  };
  const line = (x1, y1, x2, y2, cls = 'line') => el('line', {x1, y1, x2, y2, class: cls});
  const text = (x, y, value, cls = 'small', anchor = 'start') => el('text', {x, y, class: cls, 'text-anchor': anchor}, value);
  const path = (points, cls) => el('path', {d: points.map(([x, y], i) => `${i ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)}`).join(' '), class: cls});
  const dot = (x, y, cls = 'teal', r = 5) => el('circle', {cx: x, cy: y, r, class: cls});
  const range = (n) => Array.from({length: n + 1}, (_, i) => i / n);
  function axes(g, xlabel, ylabel) {
    g.append(line(48, 20, 48, 190, 'axis'), line(48, 190, 615, 190, 'axis'));
    g.append(text(48, 13, ylabel), text(615, 224, xlabel, 'small', 'end'));
  }
  function entropy(p) {
    return p === 0 || p === 1 ? 0 : -p * Math.log2(p) - (1 - p) * Math.log2(1 - p);
  }
  // Box–Muller with deterministic inputs so moving a slider does not resample noise.
  const noise = range(100).map((_, i) => {
    const u = ((i * 73 + 19) % 997 + 1) / 999;
    const v = ((i * 137 + 31) % 991 + 1) / 993;
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  });
  const signal = range(100).map((t) => 0.8 * Math.sin(2 * Math.PI * t) + 0.35 * Math.sin(6 * Math.PI * t));
  function render(lab) {
    const type = lab.dataset.lab;
    const input = lab.querySelector('input');
    const value = Number(input.value);
    const g = lab.querySelector('[data-plot]');
    const result = lab.querySelector('.vg-lab-result');
    const output = lab.querySelector('output');
    output.value = type === 'rvq' ? String(value) : value.toFixed(2);
    g.replaceChildren();
    if (type === 'initialization') {
      axes(g, 'layer depth →', 'log₁₀(second moment / input)');
      [-6, -3, 0, 3, 6].forEach((v) => {
        const y = 105 - v * 12;
        g.append(line(48, y, 615, y), text(39, y + 4, String(v), 'tiny', 'end'));
      });
      g.append(path(range(20).map((t) => [48 + 567 * t, 105 - 12 * (20 * t * Math.log10(value))]), 'stroke-teal'));
      g.append(text(48, 210, '0'), text(615, 210, '20', 'small', 'end'));
      const ratio = Math.pow(value, 20);
      result.textContent = `After 20 layers: ${ratio < 0.01 || ratio > 100 ? ratio.toExponential(2) : ratio.toFixed(3)}× the input second moment. ${value < 1 ? 'The signal shrinks with depth.' : value > 1 ? 'The signal grows with depth.' : 'The approximate scale is preserved.'}`;
    } else if (type === 'entropy') {
      axes(g, 'probability of heads →', 'entropy (bits)');
      [0, 0.5, 1].forEach((v) => g.append(text(48 + 567 * v, 210, String(v), 'small', v === 1 ? 'end' : 'start')));
      g.append(text(35, 35, '1', 'small', 'end'));
      g.append(path(range(100).map((p) => [48 + 567 * p, 190 - 160 * entropy(p)]), 'stroke-blue'));
      g.append(line(48 + 567 * value, 190, 48 + 567 * value, 190 - 160 * entropy(value), 'stroke-teal dash'), dot(48 + 567 * value, 190 - 160 * entropy(value)));
      result.textContent = `H(X) = ${entropy(value).toFixed(3)} bits per outcome. ${value === 0 || value === 1 ? 'The outcome is certain.' : value === 0.5 ? 'A fair coin has maximum binary entropy.' : 'Bias makes the average outcome more predictable.'}`;
    } else if (type === 'diffusion') {
      axes(g, 'signal position →', 'toy amplitude');
      const scale = (values) => values.map((y, i) => [48 + 567 * i / 100, 105 - 26 * y]);
      g.append(path(scale(signal), 'stroke-blue dash'));
      const mixed = signal.map((s, i) => Math.sqrt(value) * s + Math.sqrt(1 - value) * noise[i]);
      g.append(path(scale(mixed), 'stroke-teal'));
      result.textContent = `Signal coefficient: ${Math.sqrt(value).toFixed(3)}. Noise coefficient: ${Math.sqrt(1 - value).toFixed(3)}. Dashed blue is the clean signal; solid teal is the mixture.`;
    } else if (type === 'rvq') {
      axes(g, 'x coordinate →', 'y coordinate');
      const toPoint = ([x, y]) => [48 + x * 112, 190 - y * 61];
      const stages = [[0, 0], [4, 2], [4.5, 2.1], [4.55, 2.12]];
      const target = [4.6, 2.3];
      for (let i = 1; i <= value; i++) {
        const [x1, y1] = toPoint(stages[i - 1]);
        const [x2, y2] = toPoint(stages[i]);
        g.append(line(x1, y1, x2, y2, i === 1 ? 'stroke-blue' : 'stroke-teal'), dot(x2, y2));
      }
      const [tx, ty] = toPoint(target);
      const [px, py] = toPoint(stages[value]);
      g.append(line(px, py, tx, ty, 'stroke-orange dash'), dot(tx, ty, 'orange'), text(tx - 12, ty - 14, 'target (4.6, 2.3)', 'small', 'end'));
      const residual = target.map((v, i) => v - stages[value][i]);
      result.textContent = `Reconstruction: (${stages[value].join(', ')}). Residual: (${residual.map((v) => v.toFixed(2)).join(', ')}). Error length: ${Math.hypot(...residual).toFixed(3)}.`;
    } else if (type === 'matrix') {
      const cx = 320, cy = 113, s = 56;
      g.append(line(105, cy, 535, cy, 'axis'), line(cx, 25, cx, 203, 'axis'));
      g.append(el('circle', {cx, cy, r: s, class: 'stroke-blue dash'}));
      g.append(el('ellipse', {cx, cy, rx: Math.max(0.1, value * s), ry: s, class: 'stroke-teal'}));
      g.append(line(cx, cy, cx + value * s, cy, 'stroke-orange'), dot(cx + value * s, cy, 'orange'));
      g.append(text(105, 230, 'Blue: unit circle'), text(535, 230, 'Teal: transformed circle', 'small', 'end'));
      result.textContent = `Horizontal radius: ${value.toFixed(2)}. Vertical radius: 1. Maximum length gain: ${Math.max(1, value).toFixed(2)}.${value === 0 ? ' One direction collapses: this matrix is singular.' : ''}`;
    } else if (type === 'kl') {
      axes(g, 'β →', 'optimal rate R* (nats), clipped at 2');
      g.append(path(range(100).map((t) => {
        const beta = t * 0.8;
        const rate = beta === 0 ? 2 : Math.min(2, Math.max(0, 0.5 * Math.log(0.4 / beta)));
        return [48 + 567 * t, 190 - 75 * rate];
      }), 'stroke-blue'));
      g.append(line(331.5, 20, 331.5, 190, 'stroke-orange dash'), text(337, 38, 'βc = 0.4'));
      const rate = value === 0 ? Infinity : Math.max(0, 0.5 * Math.log(0.4 / value));
      g.append(dot(48 + 567 * value / 0.8, 190 - 75 * Math.min(2, rate)));
      result.textContent = value === 0 ? 'At β = 0 there is no finite rate penalty: the ideal optimum approaches zero distortion as rate grows without bound.' : `Optimal rate: ${rate.toFixed(3)} nats. Distortion: ${(4 * Math.exp(-2 * rate)).toFixed(3)}. ${value >= 0.4 ? 'This final objective favors zero information.' : 'A positive information rate is optimal in this idealized model.'}`;
    } else if (type === 'convex') {
      axes(g, 'x →', 'f(x) = x²');
      const sx = (x) => 48 + (x + 1.2) / 3.4 * 567;
      const sy = (y) => 190 - 35 * y;
      g.append(path(range(100).map((t) => {const x = -1.1 + 3.2 * t; return [sx(x), sy(x * x)];}), 'stroke-blue'));
      g.append(line(sx(-1), sy(1), sx(2), sy(4), 'stroke-orange'));
      const x = -1 * value + 2 * (1 - value), chord = value + 4 * (1 - value);
      g.append(line(sx(x), sy(x * x), sx(x), sy(chord), 'stroke-teal dash'), dot(sx(x), sy(x * x)), dot(sx(x), sy(chord), 'orange'));
      result.textContent = `At x = ${x.toFixed(2)}: f(x) = ${(x * x).toFixed(3)} ≤ chord = ${chord.toFixed(3)}. The gap is ${(chord - x * x).toFixed(3)}.`;
    }
  }
  function revealHash() {
    let id;
    try { id = decodeURIComponent(window.location.hash.slice(1)); } catch { return; }
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    let parent = target.closest('details');
    let opened = false;
    while (parent) { parent.open = true; opened = true; parent = parent.parentElement.closest('details'); }
    if (id === 'full-notes') {
      const details = target.nextElementSibling;
      if (details?.matches('details')) { details.open = true; opened = true; }
    }
    if (opened) requestAnimationFrame(() => target.scrollIntoView({block: 'start'}));
  }
  function init() {
    document.querySelectorAll('.vg-lab').forEach((lab) => {
      render(lab);
      lab.hidden = false;
      lab.querySelector('input').addEventListener('input', () => render(lab));
    });
    revealHash();
    window.addEventListener('hashchange', revealHash);
    // Also handle a repeated click on the current fragment.
    document.addEventListener('click', (event) => {
      const anchor = event.target.closest('a[href]');
      if (!anchor) return;
      const url = new URL(anchor.href, window.location.href);
      if (url.origin === location.origin && url.pathname === location.pathname && url.hash === location.hash) revealHash();
    });
    let closedForPrint = [];
    window.addEventListener('beforeprint', () => {
      closedForPrint = [...document.querySelectorAll('details:not([open])')];
      closedForPrint.forEach((details) => { details.open = true; });
    });
    window.addEventListener('afterprint', () => closedForPrint.forEach((details) => { details.open = false; }));
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
