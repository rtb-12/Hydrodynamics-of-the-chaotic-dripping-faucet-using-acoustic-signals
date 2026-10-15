// Plot helpers shared by the recording pages.

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
  (o.marks || []).forEach(function (m) {
    g += '<line class="mark" x1="' + (X(m.x) - 16) + '" y1="' + Y(m.y) + '" x2="' + (X(m.x) + 16) + '" y2="' + Y(m.y) + '"/>';
  });
  o.pts.forEach(function (p) {
    var cls = (p.miss ? 'miss' : 'pt') + (p.now ? ' now' : '') + (p.faint ? ' faint' : '') + (p.cls ? ' ' + p.cls : '');
    g += '<circle class="' + cls + '" cx="' + X(p.x) + '" cy="' + Y(p.miss ? o.y[0] : p.y) + '" r="' + (p.miss ? 4 : 5.5) + '"><title>' + (p.tip || '') + '</title></circle>';
    var flip = X(p.x) > L + pw * 0.6;   // a label past here would run off the right edge
    if (p.label) g += '<text class="lab' + (p.now ? ' now' : '') + '" x="' + (X(p.x) + (flip ? -11 : 11)) + '" y="' + (Y(p.y) + 4) + '"' + (flip ? ' text-anchor="end"' : '') + '>' + p.label + '</text>';
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
