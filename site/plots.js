// Plot helpers shared by the recording pages.

var GUT = 46, PAD = 8;   // time lanes' left gutter and right margin

// Opened from the filesystem the browser keeps a page's data scripts even after they are regenerated, so ask afresh there.
function fresh(url) { return location.protocol === 'file:' ? url + '?t=' + Date.now() : url; }

// A static x-y plot as inline SVG. o: W, H, x and y as [lo, hi], xl and yl axis titles, pts [{x, y, label, now, faint, miss, cls}],
// diag for the y = x line, join for a line through consecutive points not separated by a `cut`, marks [{x, y}] for short level bars.
function xy(o) {
  var W = o.W, H = o.H, L = 62, B = 52, T = 12, pw = W - L - (o.right || 18), ph = H - T - B, g = '';
  function X(v) { return L + pw * (v - o.x[0]) / (o.x[1] - o.x[0]); }
  function Y(v) { return T + ph * (1 - (v - o.y[0]) / (o.y[1] - o.y[0])); }
  nice(o.x[0], o.x[1], o.nx || 6).v.forEach(function (v) {
    g += '<line class="grid" x1="' + X(v) + '" y1="' + T + '" x2="' + X(v) + '" y2="' + (T + ph) + '"/><text x="' + X(v) + '" y="' + (T + ph + 18) + '" text-anchor="middle">' + v + '</text>';
  });
  nice(o.y[0], o.y[1], 6).v.forEach(function (v) {
    g += '<line class="grid" x1="' + L + '" y1="' + Y(v) + '" x2="' + (L + pw) + '" y2="' + Y(v) + '"/><text x="' + (L - 8) + '" y="' + (Y(v) + 4) + '" text-anchor="end">' + v + '</text>';
  });
  g += '<line class="axis" x1="' + L + '" y1="' + T + '" x2="' + L + '" y2="' + (T + ph) + '"/><line class="axis" x1="' + L + '" y1="' + (T + ph) + '" x2="' + (L + pw) + '" y2="' + (T + ph) + '"/>' +
       '<text class="name" x="' + (L + pw / 2) + '" y="' + (H - 10) + '" text-anchor="middle">' + o.xl + '</text>' +
       '<text class="name" transform="translate(16 ' + (T + ph / 2) + ') rotate(-90)" text-anchor="middle">' + o.yl + '</text>';
  if (o.diag) {
    var a = Math.max(o.x[0], o.y[0]), b = Math.min(o.x[1], o.y[1]);
    g += '<line class="same" x1="' + X(a) + '" y1="' + Y(a) + '" x2="' + X(b) + '" y2="' + Y(b) + '"/>';
  }
  if (o.join) o.pts.forEach(function (p, i) {
    var q = o.pts[i + 1];
    if (q && !p.miss && !q.miss && !p.cut) g += '<line class="join" x1="' + X(p.x) + '" y1="' + Y(p.y) + '" x2="' + X(q.x) + '" y2="' + Y(q.y) + '"/>';
  });
  o.pts.forEach(function (p) {
    var cls = (p.miss ? 'miss' : 'pt') + (p.now ? ' now' : '') + (p.faint ? ' faint' : '') + (p.cls ? ' ' + p.cls : '');
    g += '<circle class="' + cls + '" cx="' + X(p.x) + '" cy="' + Y(p.miss ? o.y[0] : p.y) + '" r="' + (p.miss ? 4 : 5.5) + '"><title>' + (p.tip || '') + '</title></circle>';
    var flip = X(p.x) > L + pw * 0.6;   // a label past here would run off the right edge
    if (p.label) g += '<text class="lab' + (p.now ? ' now' : '') + '" x="' + (X(p.x) + (flip ? -11 : 11)) + '" y="' + (Y(p.y) + 4) + '"' + (flip ? ' text-anchor="end"' : '') + '>' + p.label + '</text>';
  });
  (o.marks || []).forEach(function (m) {   // after the dots, so a median is never hidden
    g += '<line class="mark" x1="' + (X(m.x) - 16) + '" y1="' + Y(m.y) + '" x2="' + (X(m.x) + 16) + '" y2="' + Y(m.y) + '"/>';
  });
  return '<svg class="xy" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' + o.yl + ' against ' + o.xl + '">' + g + '</svg>';
}
function span(v, floor) {   // axis limits a little outside the data
  var lo = Math.min.apply(null, v), hi = Math.max.apply(null, v), p = Math.max((hi - lo) * 0.1, floor);
  return [lo - p, hi + p];
}

function nice(lo, hi, n) {
  var raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  var st = [1, 2, 5, 10].map(function (m) { return m * mag; }).filter(function (v) { return v >= raw; })[0], out = [];
  for (var v = Math.ceil(lo / st) * st; v <= hi + 1e-9; v += st) out.push(+v.toFixed(10));
  return { v: out, st: st };
}

function fit(c) {
  var cw = c.clientWidth, ch = c.clientHeight, d = devicePixelRatio || 1, w = Math.round(cw * d), h = Math.round(ch * d);
  if (c.width !== w || c.height !== h) { c.width = w; c.height = h; }
  c.W = cw; c.H = ch;
  var g = c.getContext('2d'); g.setTransform(d, 0, 0, d, 0, 0); return g;
}

function text(g, k, s, x, y, align, base) { g.fillStyle = k.ink3; g.font = '11px ' + k.sans; g.textAlign = align || 'right'; g.textBaseline = base || 'middle'; g.fillText(s, x, y); }
function pct(v) { v *= 100; return (v >= 10 ? v.toFixed(0) : v >= 1 ? v.toFixed(1) : v.toFixed(2)) + '%'; }
function boxcar(a, n) {   // centred moving average over prefix sums, shrinking at the ends
  if (n < 2) return a;
  var out = new Float32Array(a.length), cum = new Float64Array(a.length + 1), h = n >> 1, i;
  for (i = 0; i < a.length; i++) cum[i + 1] = cum[i] + a[i];
  for (i = 0; i < a.length; i++) { var lo = Math.max(0, i - h), hi = Math.min(a.length, i + h + 1); out[i] = (cum[hi] - cum[lo]) / (hi - lo); }
  return out;
}
function unpack(b64) { var s = atob(b64), a = new Int8Array(s.length); for (var i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; }
// A spectrogram PNG stores loudness as alpha; paint it in a colour once, cached on the image.
function tint(im, col) {
  if (!im.complete || !im.naturalWidth) return null;
  if (!im.tinted || im.tintCol !== col) {
    var c = document.createElement('canvas'); c.width = im.naturalWidth; c.height = im.naturalHeight;
    var g = c.getContext('2d'); g.drawImage(im, 0, 0); g.globalCompositeOperation = 'source-in'; g.fillStyle = col; g.fillRect(0, 0, c.width, c.height);
    im.tinted = c; im.tintCol = col;
  }
  return im.tinted;
}
function grid(g, k, tk, X, yt, top, h, pw) {
  g.strokeStyle = k.rule; g.lineWidth = 1; g.beginPath();
  tk.v.forEach(function (tt) { var x = Math.round(X(tt)) + 0.5; g.moveTo(x, top); g.lineTo(x, top + h); });
  yt.forEach(function (y) { var yy = Math.round(y[0]) + 0.5; g.moveTo(GUT, yy); g.lineTo(GUT + pw, yy); });
  g.stroke();
}

// One time lane of a recording, everything left of the playhead already heard. kind: wave, loud, spec or drops.
// S: wave (square-root companded max/min int8 pairs), wave_rate, loud, loud_rate, spec (tinted canvas or null), spec_cps,
// spec_fmax, hits (drop times) and heard. v: t (playhead), x0 (window start), win, big, hover, playing, gain, smoothMs, k (colours),
// and ticks to label the time axis under a lane other than the drops lane.
function drawLane(kind, c, S, v) {
  var g = fit(c), k = v.k, top = v.big ? 8 : 24, axis = (v.big || v.ticks || kind === 'drops') ? 20 : 4, x0 = v.x0, WIN = v.win;
  var pw = c.W - GUT - PAD, h = c.H - top - axis, yt = [];
  function X(tt) { return GUT + (tt - x0) / WIN * pw; }
  var tk = nice(x0, x0 + WIN, v.big ? 12 : 6);
  g.clearRect(0, 0, c.W, c.H);
  g.save(); g.beginPath(); g.rect(GUT, top, pw, h); g.clip();
  if (kind === 'wave') laneWave(g, c, S, v, pw, top, h, X, tk, yt);
  else if (kind === 'loud') laneLoud(g, S, v, pw, top, h, X, tk, yt);
  else if (kind === 'spec') {
    nice(0, S.spec_fmax / 1000, v.big ? 8 : 3).v.forEach(function (f) { yt.push([top + h - f * 1000 / S.spec_fmax * h, f ? f + ' kHz' : '0']); });
    grid(g, k, tk, X, yt, top, h, pw);
    var im = S.spec, cps = S.spec_cps;
    if (im) {
      var a = Math.max(0, Math.floor(x0 * cps)), b = Math.min(im.width, Math.ceil((x0 + WIN) * cps) + 1);
      g.imageSmoothingEnabled = false;   // one column per analysis slice, drawn as measured
      if (b > a) g.drawImage(im, a, 0, b - a, im.height, X(a / cps), top, (b - a) / cps / WIN * pw, h);
    }
  } else {
    grid(g, k, tk, X, [], top, h, pw);
    S.hits.forEach(function (tt, i) {
      var x = X(tt); if (x < GUT - 5 || x > c.W) return;
      g.fillStyle = k.ink; g.fillRect(x - 0.75, top + h * 0.4, 1.5, h * 0.5);
      if (S.heard[i]) { g.fillStyle = k.chan; g.beginPath(); g.arc(x, top + h * 0.24, 3.2, 0, 7); g.fill(); }
    });
  }
  if (kind !== 'drops') {   // drops as faint guides on every lane, so events line up by eye
    g.setLineDash([2, 4]); g.lineWidth = 1;
    S.hits.forEach(function (tt, i) {
      var x = Math.round(X(tt)) + 0.5; if (x < GUT || x > c.W) return;
      g.strokeStyle = S.heard[i] ? k.chan : k.ink3; g.globalAlpha = S.heard[i] ? 0.55 : 0.3;
      g.beginPath(); g.moveTo(x, top); g.lineTo(x, top + h); g.stroke();
    });
    g.setLineDash([]); g.globalAlpha = 1;
  }
  var phx = X(v.t);
  g.globalAlpha = 0.5; g.fillStyle = k.surface; g.fillRect(phx, top, c.W, h); g.globalAlpha = 1;
  g.fillStyle = k.hydro; g.fillRect(phx - 0.75, top, 1.5, h);
  if (v.hover != null) { g.fillStyle = k.ink2; g.fillRect(X(v.hover) - 0.5, top, 1, h); }
  g.restore();
  g.strokeStyle = k.rule; g.lineWidth = 1; g.strokeRect(GUT + 0.5, top + 0.5, pw - 1, h - 1);
  yt.forEach(function (y) { if (y[1]) text(g, k, y[1], GUT - 6, y[0]); });
  if (axis > 4) {
    var dec = Math.max(0, -Math.floor(Math.log10(tk.st)));
    tk.v.forEach(function (tt) { var x = X(tt); if (x > GUT + 16 && x < c.W - 24) text(g, k, tt.toFixed(dec) + ' s', x, top + h + 6, 'center', 'top'); });
  }
  return { x0: x0, pw: pw };
}
function laneWave(g, c, S, v, pw, top, h, X, tk, yt) {
  var k = v.k, w = S.wave, n = w.length / 2, dt = v.win / pw, c0 = Math.floor(v.x0 / dt), cols = Math.ceil(pw) + 2;
  var hi = new Float32Array(cols), lo = new Float32Array(cols), ok = new Uint8Array(cols), target = 0.02, j;
  for (j = 0; j < cols; j++) {
    var a = Math.floor((c0 + j) * dt * S.wave_rate), b = Math.max(a + 1, Math.floor((c0 + j + 1) * dt * S.wave_rate)), mx = 0, mn = 0;
    for (var q = Math.max(0, a); q < Math.min(n, b); q++) { if (w[2 * q] > mx) mx = w[2 * q]; if (w[2 * q + 1] < mn) mn = w[2 * q + 1]; }
    ok[j] = b > 0 && a < n; hi[j] = mx / 127; lo[j] = mn / 127;   // stored square-root companded, which is the drawing scale
    if (ok[j]) target = Math.max(target, hi[j], -lo[j]);
  }
  // Eased while playing so auto gain does not pump as loud drops scroll in and out.
  c.peak = v.gain === 'full' ? 1 : c.peak && v.playing ? c.peak + (target - c.peak) * 0.2 : target;
  var peak = c.peak, m = Math.round(v.smoothMs / 1000 / dt), H = boxcar(hi, m), L = boxcar(lo, m), mid = top + h / 2;
  function Y(s) { return mid - Math.max(-1, Math.min(1, s / peak)) * (h / 2 - 4); }
  yt.push([Y(peak), pct(peak * peak)], [Y(peak / 2), pct(peak * peak / 4)], [mid, '0'], [Y(-peak / 2), ''], [Y(-peak), '']);
  grid(g, k, tk, X, yt, top, h, pw);
  g.fillStyle = k.chan;
  if (m < 2) {   // sharp: one min-to-max bar per pixel column, the way audio editors draw it
    for (j = 0; j < cols; j++) if (ok[j]) { var y0 = Y(H[j]); g.fillRect(Math.round(X((c0 + j) * dt)), y0, 1, Math.max(1, Y(L[j]) - y0)); }
  } else {
    g.beginPath();
    for (j = 0; j < cols; j++) if (ok[j]) g.lineTo(X((c0 + j + 0.5) * dt), Y(H[j]));
    for (j = cols - 1; j >= 0; j--) if (ok[j]) g.lineTo(X((c0 + j + 0.5) * dt), Y(L[j]));
    g.globalAlpha = 0.4; g.fill(); g.globalAlpha = 1;
  }
}
function laneLoud(g, S, v, pw, top, h, X, tk, yt) {
  var k = v.k, Lr = S.loud_rate, arr = S.loud, x0 = v.x0;
  function Y(db) { return top + 4 + (-db / 60) * (h - 8); }
  [0, -20, -40, -60].forEach(function (db) { yt.push([Y(db), db ? String(db) : '0 dB']); });
  grid(g, k, tk, X, yt, top, h, pw);
  var k0 = Math.max(0, Math.floor(x0 * Lr) - 1), k1 = Math.min(arr.length, Math.ceil((x0 + v.win) * Lr) + 2);
  var pad = Math.round(v.smoothMs / 1000 * Lr), s0 = Math.max(0, k0 - pad), seg = boxcar(Float32Array.from(arr.slice(s0, k1 + pad)), pad);
  if (k1 - k0 < 2) return;
  g.beginPath(); for (var q = k0; q < k1; q++) g.lineTo(X(q / Lr), Y(seg[q - s0]));
  g.strokeStyle = k.chan; g.lineWidth = 1.25; g.lineJoin = 'miter'; g.stroke();
  g.lineTo(X((k1 - 1) / Lr), Y(-60)); g.lineTo(X(k0 / Lr), Y(-60));
  g.globalAlpha = 0.12; g.fillStyle = k.chan; g.fill(); g.globalAlpha = 1;
}

// The (i) buttons: hover or focus shows its text from INFO in the tip element, a click pins it (touch screens have no hover).
// host() is where the tip is attached. Returns the function that hides it.
function infoTips(INFO, tip, host) {
  var pinned = null;
  function showTip(btn) {
    var d = INFO[btn.dataset.info];
    tip.innerHTML = '<b>' + d[0] + '</b>' + d.slice(1).map(function (p) { return '<p>' + p + '</p>'; }).join('');
    host().appendChild(tip);
    tip.hidden = false;
    var r = btn.getBoundingClientRect(), w = tip.offsetWidth, h = tip.offsetHeight;
    tip.style.left = Math.max(16, Math.min(innerWidth - w - 16, r.left - 12)) + 'px';
    tip.style.top = (r.bottom + 8 + h < innerHeight ? r.bottom + 8 : Math.max(8, r.top - 8 - h)) + 'px';
  }
  function hideTip() { tip.hidden = true; if (pinned) { pinned.setAttribute('aria-expanded', 'false'); pinned = null; } }
  document.addEventListener('pointerover', function (e) { var b = e.target.closest('.info'); if (b && !pinned) showTip(b); });
  document.addEventListener('pointerout', function (e) { if (e.target.closest('.info') && !pinned) tip.hidden = true; });
  document.addEventListener('focusin', function (e) { var b = e.target.closest('.info'); if (b && !pinned) showTip(b); });
  document.addEventListener('focusout', function (e) { if (e.target.closest('.info') && !pinned) tip.hidden = true; });
  document.addEventListener('click', function (e) {
    var b = e.target.closest('.info');
    if (!b) { if (pinned && !e.target.closest('.tip')) hideTip(); return; }
    e.stopPropagation();
    var same = pinned === b;
    hideTip();
    if (!same) { pinned = b; b.setAttribute('aria-expanded', 'true'); showTip(b); }
  }, true);
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape' && !tip.hidden) hideTip(); });
  addEventListener('scroll', function () { if (pinned) showTip(pinned); else tip.hidden = true; }, { passive: true });
  return hideTip;
}

// Media streams from the network so playback starts at once; once the browser holds the whole file, a copy goes into its
// Cache Storage (never earlier, so it cannot slow playback), and later visits play that copy while the server's file keeps its size.
var MEDIA_CACHE = 'recordings-v1';
function mediaUrl(url) {   // resolves to the cached copy as a blob URL, or to the network URL
  if (!window.caches) return Promise.resolve(url);   // opened from the filesystem, or an old browser
  return caches.open(MEDIA_CACHE).then(function (c) {
    return c.match(url).then(function (hit) {
      if (!hit) return url;
      var copy = function () { return hit.blob().then(function (b) { return URL.createObjectURL(b); }); };
      return fetch(url, { method: 'HEAD', cache: 'no-store' }).then(function (h) {
        if (h.ok && h.headers.get('Content-Length') !== hit.headers.get('Content-Length')) { c.delete(url); return url; }
        return copy();
      }, copy);   // offline: the copy is all there is
    });
  }).catch(function () { return url; });
}
function keepMedia(el, url) {   // copy url into the cache once el has all of it, unless it is already there
  if (!window.caches || /^blob:/.test(el.src)) return;
  function check() {
    var b = el.buffered;
    if (!el.duration || !b.length || b.start(0) > 0.1 || b.end(b.length - 1) < el.duration - 0.1 || b.length > 1) return;
    el.removeEventListener('progress', check);
    caches.open(MEDIA_CACHE).then(function (c) { return c.match(url).then(function (hit) { if (!hit) return c.add(url); }); }).catch(function () {});
  }
  el.addEventListener('progress', check);
}
