#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "av", "noisereduce", "pillow", "imageio-ffmpeg"]
# ///
"""Turn a drip video (plus an optional separately recorded audio file) into a recording
for site/recording.html: levelled, cropped, tone-mapped video, aligned and cleaned audio,
detected drops and the plots' data.

    uv run analysis/recording.py exp/1.MOV --audio "exp/WhatsApp Audio.mp4"
    uv run analysis/recording.py --selftest

Everything it finds automatically (impact point, tilt, gate band) can be overridden;
check.jpg in the output folder shows what it found.
"""

import argparse, base64, json, math, pathlib, subprocess, sys, tempfile
import av, numpy as np, noisereduce as nr, imageio_ffmpeg
import scipy.ndimage as nd, scipy.signal as ss
from scipy.io import wavfile
from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'site' / 'recordings'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

FS = 48000                  # audio rate everything is resampled to
ALIGN_OK = 1.5              # audio alignment peak must beat the runner-up by this factor to be trusted
SCOUT_FRAMES = 600          # frames used to locate the impact point and water line
SCOUT_SCALE = 4             # downscale factor for that scouting pass
CROP_W = 0.6                # crop width as a fraction of frame width (16:9 crop)
GATE_HALF_W = 0.045         # gate half-width as a fraction of frame height
GATE_SIZES = (0.08, 0.15)   # candidate gate heights, fractions of frame height
GATE_STEP = 0.02            # candidate gate spacing along the drop column, fraction of frame height
MIN_DROPS = 10              # a gate that sees fewer drops than this is not a gate
RATE_AGREE = math.log(1.25) # a band agrees with the consensus drip rate if within 25% of it
TIE = 0.97                  # bands scoring within 3% of the best count as tied...
TIE_MIN = 2                 # ...or within this many intervals, since one stray pulse can land in-band
BELOW_SURFACE = 0.1         # gates may reach this far under the water line, where the impact shows
STREAK_K = 0.35             # the drop streak ends where its motion falls below this fraction of its peak
SURFACE_SEARCH = 0.15       # look for the water-line edge this far (fraction of height) below the streak's end
SURFACE_ABOVE = 0.05        # ...and this far above it, since a tilted line rises above it on one side
GATE_K = 3.0                # gate threshold in robust standard deviations
GATE_FLOOR = 0.25           # ...but never below this fraction of a typical crossing
MIN_GAP_S = 0.05            # two drops closer than this are one drop (20 Hz ceiling)
BAND = (2000, 15000)        # impact and plink energy lives here, below it is room noise
HIGHPASS = 800              # cleaning high-pass, Hz
ONSET_K = 8.0               # onset threshold as a multiple of the median envelope
HEARD_TOL = 0.015           # a drop is "heard" if an onset lands this close to the usual gate-to-sound delay
MAX_LAG = 0.5               # longest plausible gate-to-sound delay (gate far above the surface), seconds
LAG_BIN = 0.004             # histogram bin for finding that delay, seconds
WAVE_RATE = 4000            # waveform min/max pairs per second
LOUD_RATE = 200             # loudness points per second
SPEC_MAX_W = 16000          # spectrogram PNG width cap, browsers struggle beyond this
SPEC_FMAX = 16000           # spectrogram top frequency, Hz
HLG, PQ = 18, 16            # colour-transfer codes that mean HDR


# --- video -----------------------------------------------------------------

def probe(path):
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        return dict(w=v.width, h=v.height, fps=float(v.average_rate),
                    duration=float(v.duration * v.time_base) if v.duration else float(c.duration / 1e6),
                    hdr=v.codec_context.color_trc in (HLG, PQ), has_audio=bool(c.streams.audio),
                    created=c.metadata.get('creation_time', '')[:10])


def gray_frames(path, scale=1, limit=None):
    """Yield (time_s, gray uint8 array) per frame."""
    with av.open(str(path)) as c:
        v = c.streams.video[0]
        v.thread_type = 'AUTO'
        for i, f in enumerate(c.decode(v)):
            if limit and i >= limit:
                return
            yield f.time, f.reformat(f.width // scale, f.height // scale, format='gray').to_ndarray()


def find_geometry(path, meta):
    """Drop column, and the water line and its tilt, from the first few seconds."""
    fr = np.array([g for _, g in gray_frames(path, SCOUT_SCALE, SCOUT_FRAMES)], np.float32)
    mx, top, end = drop_column(np.abs(np.diff(fr, axis=0)).mean(0))
    h = fr.shape[1]
    edge = np.abs(nd.sobel(nd.gaussian_filter(np.median(fr, 0), 1), 0))
    lo, hi = max(0, end - int(SURFACE_ABOVE * h)), min(h, end + max(2, int(SURFACE_SEARCH * h)))
    xs = np.arange(4, fr.shape[2] - 4, 4)
    ys = np.array([lo + np.argmax(edge[lo:hi, x]) for x in xs])
    k, b = np.polyfit(xs, ys, 1)
    keep = np.abs(ys - (k * xs + b)) < 2 * np.median(np.abs(ys - (k * xs + b))) + 1
    k, b = np.polyfit(xs[keep], ys[keep], 1)
    s = SCOUT_SCALE
    return dict(x=int(mx * s), y=int(k * mx + b) * s, top=top * s, tilt=float(np.degrees(np.arctan(k))))


def drop_column(motion):
    """Column, first row and last row of the thin vertical streak the falling drops draw.
    Camera shake and people in the background move wide areas, so they do not make one."""
    streak = nd.gaussian_filter(np.clip(motion - nd.median_filter(motion, size=(1, 31)), 0, None), 1)
    x = int(np.argmax(nd.uniform_filter1d(streak.sum(0), 3)))
    prof = nd.uniform_filter1d(streak[:, max(0, x - 2):x + 3].mean(1), 3)
    rows = np.flatnonzero(prof > STREAK_K * prof.max())
    return x, int(rows.min()), int(rows.max())


def crop_box(meta, geo):
    """16:9 box centred on the impact point, pulled inside the frame and clear of rotation corners."""
    W, H = meta['w'], meta['h']
    cw = int(W * CROP_W) // 2 * 2
    ch = int(cw * 9 / 16) // 2 * 2
    margin = int(abs(math.tan(math.radians(geo['tilt']))) * W / 2) + 2
    cx = min(max(geo['x'] - cw // 2, 0), W - cw)
    cy = min(max(geo['y'] - ch // 2, margin), H - margin - ch)
    return cw, ch, cx, cy


def column_motion(path, meta, x):
    """Per-frame, per-row change in the drop column minus the same in a strip beside it,
    which cancels light flicker and camera shake since they move every column alike."""
    W, half = meta['w'], int(GATE_HALF_W * meta['h'])
    rx = x + 4 * half if x + 5 * half < W else x - 4 * half
    cols = (slice(max(0, x - half), x + half), slice(max(0, rx - half), rx + half))
    ts, rows, prev = [], [], None
    for t, g in gray_frames(path):
        cur = [g[:, c].astype(np.float32) for c in cols]
        if prev is not None:
            rows.append(np.abs(cur[0] - prev[0]).mean(1) - np.abs(cur[1] - prev[1]).mean(1))
            ts.append(t)
        prev = cur
    return np.array(ts), np.array(rows, np.float32)


def choose_gate(ts, M, top, surface, h):
    """Rows on the drop's path that best see one pulse per drop. The jet doubles the train and faint rows
    halve it, so agree on the drip rate across bands first, then take the most consistent band at that rate."""
    cands = []
    for size in (int(f * h) for f in GATE_SIZES):
        for y0 in range(top, min(h, surface + int(BELOW_SURFACE * h)) - size, max(1, int(GATE_STEP * h))):
            d = detect_drops(ts, M[:, y0:y0 + size].mean(1))
            if len(d) >= MIN_DROPS:
                iv = np.diff(d)
                m = float(np.median(iv))
                good = int(((iv > 0.6 * m) & (iv < 1.5 * m)).sum())
                cands.append((2 * good - len(iv), m, (y0, y0 + size), d))   # good intervals minus bad ones
    if not cands:
        g = (max(0, surface - int(GATE_SIZES[0] * h / 2)), surface + int(GATE_SIZES[0] * h / 2))
        return g, detect_drops(ts, M[:, g[0]:g[1]].mean(1))
    rate = weighted_median([math.log(c[1]) for c in cands], [max(1, c[0]) for c in cands])
    agree = [c for c in cands if abs(math.log(c[1]) - rate) < RATE_AGREE]
    top_score = max(c[0] for c in agree)
    # Near-ties go to the higher band: jet and splash contaminate the rows nearest the surface.
    best = min((c for c in agree if c[0] >= top_score - max(TIE_MIN, (1 - TIE) * top_score)), key=lambda c: c[2][0])
    return best[2], best[3]


def weighted_median(values, weights):
    order = np.argsort(values)
    cum = np.cumsum(np.asarray(weights, float)[order])
    return float(np.asarray(values)[order][np.searchsorted(cum, cum[-1] / 2)])


def detect_drops(ts, sig):
    """Gate crossings, each timed by the centroid of its motion pulse."""
    s = sig - ss.medfilt(sig, 31)
    noise = 1.4826 * np.median(np.abs(s - np.median(s)))
    # Codecs store a still background as exactly zero change, so noise alone can be 0.
    thr = max(GATE_K * noise, GATE_FLOOR * np.percentile(s[s > 0], 99)) if (s > 0).any() else np.inf
    dt = float(np.median(np.diff(ts)))
    peaks, _ = ss.find_peaks(s, height=thr, distance=max(1, int(MIN_GAP_S / dt)))
    if len(peaks) > 2:   # one drop can pulse over several frames: space peaks by half the typical interval
        gap = 0.5 * float(np.median(np.diff(ts[peaks])))
        peaks, _ = ss.find_peaks(s, height=thr, distance=max(1, int(gap / dt)))
    mid = np.r_[ts[0], 0.5 * (ts[1:] + ts[:-1])]   # change p happened between frames p-1 and p
    out = []
    for p in peaks[(peaks >= 2) & (peaks < len(s) - 2)]:
        w = np.clip(s[p - 2:p + 3], 0, None)
        out.append(float((w * mid[p - 2:p + 3]).sum() / w.sum()))
    return np.array(out)


def encode_video(src, dst, meta, box, tilt):
    cw, ch, cx, cy = box
    tone = ('zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=hable:desat=0,'
            'zscale=t=bt709:m=bt709:r=tv,format=yuv420p,') if meta['hdr'] else 'format=yuv420p,'
    vf = f'{tone}rotate={-tilt}*PI/180,crop={cw}:{ch}:{cx}:{cy},hqdn3d=1.5:1.5:3:3,unsharp=5:5:0.4'
    subprocess.run([FFMPEG, '-loglevel', 'error', '-y', '-i', str(src), '-an', '-vf', vf,
                    '-fps_mode', 'passthrough', '-c:v', 'libx264', '-crf', '18', '-preset', 'slow',
                    '-g', '30', '-pix_fmt', 'yuv420p', '-color_primaries', 'bt709', '-color_trc', 'bt709',
                    '-colorspace', 'bt709', '-movflags', '+faststart', str(dst)], check=True)


# --- audio -----------------------------------------------------------------

def read_audio(path):
    """Mono float32 at FS, zero-padded so sample 0 is the container's time zero."""
    with av.open(str(path)) as c:
        rs = av.AudioResampler(format='flt', layout='mono', rate=FS)
        chunks, start = [], None
        for f in c.decode(audio=0):
            for r in rs.resample(f):
                start = r.time if start is None else start
                chunks.append(r.to_ndarray().ravel())
        for r in rs.resample(None):
            chunks.append(r.to_ndarray().ravel())
    x = np.concatenate(chunks).astype(np.float32)
    return np.r_[np.zeros(int(round(max(0.0, start or 0.0) * FS)), np.float32), x]


def envelope(x, lp=300):
    y = ss.sosfiltfilt(ss.butter(4, BAND, 'bandpass', fs=FS, output='sos'), x)
    return ss.sosfiltfilt(ss.butter(2, lp, fs=FS, output='sos'), np.abs(ss.hilbert(y)))


def align(ref, other):
    """Seconds to add to a video time to find the same moment in `other`, and how clearly it won."""
    a, b = ss.resample_poly(envelope(ref), 1, 48), ss.resample_poly(envelope(other), 1, 48)
    c = ss.correlate(b - b.mean(), a - a.mean(), 'full', method='fft')
    k = int(np.argmax(c))
    runner = np.delete(c, np.s_[max(0, k - 50):k + 50]).max()
    return (k - (len(a) - 1)) / 1000.0, float(c[k] / runner) if runner > 0 else float('inf')


def on_timeline(x, offset, n):
    """Resample-free shift: out[i] = x[i + offset*FS], zero where x has no sample."""
    k = int(round(offset * FS))
    out = np.zeros(n, np.float32)
    lo, hi = max(0, -k), min(n, len(x) - k)
    if hi > lo:
        out[lo:hi] = x[lo + k:hi + k]
    return out


def clean(x):
    hp = ss.sosfiltfilt(ss.butter(4, HIGHPASS, 'highpass', fs=FS, output='sos'), x)
    blocks = hp[:len(hp) // 2400 * 2400].reshape(-1, 2400)
    quiet = blocks[np.argsort(np.abs(blocks).max(1))[:max(1, int(len(blocks) * 0.4))]]
    y = nr.reduce_noise(y=hp, sr=FS, y_noise=quiet.ravel()[:5 * FS], stationary=True, prop_decrease=0.9)
    return (y / (np.abs(y).max() or 1) * 0.9).astype(np.float32)


def gate_to_sound(drops, heard_t):
    """Delay from gate to sound, and which drops made one. The fall below the gate can take most of a drip
    period, so take the delay all drops share (the histogram peak) rather than pairing nearest sounds."""
    d = (heard_t[None, :] - drops[:, None]).ravel()
    d = d[(d > -MAX_LAG) & (d < MAX_LAG)]
    if not len(d):
        return None, [False] * len(drops)
    counts, edges = np.histogram(d, bins=np.arange(-MAX_LAG, MAX_LAG + LAG_BIN, LAG_BIN))
    k = int(np.argmax(counts))
    near = d[(d >= edges[k] - LAG_BIN) & (d < edges[k + 1] + LAG_BIN)]
    lag = float(np.median(near))
    return lag, [bool(np.min(np.abs(heard_t - t - lag)) < HEARD_TOL) for t in drops]


def onsets(x):
    e = envelope(x, 500)
    p, _ = ss.find_peaks(e, height=ONSET_K * np.median(e), distance=int(MIN_GAP_S * FS))
    return p / FS


# --- page data -------------------------------------------------------------

def minmax(x):
    """Base64 of int8 (max, min) pairs at WAVE_RATE, fine enough to zoom into one plink."""
    n = FS // WAVE_RATE
    r = x[:len(x) // n * n].reshape(-1, n) / (np.abs(x).max() or 1)
    pairs = np.round(np.c_[r.max(1), r.min(1)] * 127).astype(np.int8)
    return base64.b64encode(pairs.tobytes()).decode()


def loudness(x):
    n = FS // LOUD_RATE
    e = envelope(x, 500)
    e = e[:len(e) // n * n].reshape(-1, n).max(1)
    return np.clip(np.round(20 * np.log10(e / (e.max() or 1) + 1e-9)), -60, 0).astype(int).tolist()


def spectrogram_png(x, path, duration):
    cps = min(200, SPEC_MAX_W / duration)   # long recordings get coarser columns rather than tiling
    hop = min(512, int(FS / cps))
    f, _, Z = ss.spectrogram(x, FS, nperseg=512, noverlap=512 - hop)
    Z = 10 * np.log10(Z[f <= SPEC_FMAX] + 1e-14)
    lo, hi = np.percentile(Z, [50, 99.9])
    a = (np.clip((Z - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)[::-1]
    Image.fromarray(np.dstack([np.full_like(a, 255), a]), 'LA').save(path, optimize=True)
    return FS / hop


def write_check(path, meta, geo, gate_box, box):
    g = next(gray_frames(path, 1, 1))[1]
    im = Image.fromarray(g).convert('RGB')
    d = ImageDraw.Draw(im)
    W = meta['w']
    k = math.tan(math.radians(geo['tilt']))
    d.line([(0, geo['y'] - k * geo['x']), (W, geo['y'] + k * (W - geo['x']))], fill=(0, 160, 255), width=3)
    d.rectangle(gate_box, outline=(255, 140, 0), width=3)
    cw, ch, cx, cy = box
    d.rectangle((cx, cy, cx + cw, cy + ch), outline=(255, 255, 0), width=3)
    d.ellipse((geo['x'] - 8, geo['y'] - 8, geo['x'] + 8, geo['y'] + 8), outline=(255, 0, 0), width=3)
    return im


def write_index(out_root):
    names = sorted(p.parent.name for p in out_root.glob('*/data.js'))
    (out_root / 'index.js').write_text(f'window.RECORDING_LIST = {json.dumps(names)};\n')


def process(video, audio=None, name=None, channel='microphone', impact=None, tilt=None,
            gate=None, out_root=OUT, quiet=False):
    say = (lambda *a: None) if quiet else print
    video = pathlib.Path(video)
    name = name or video.stem
    out = out_root / name
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(video)
    say(f'{video.name}: {meta["w"]}x{meta["h"]} {meta["fps"]:.2f} fps {meta["duration"]:.1f} s'
        f'{" HDR" if meta["hdr"] else ""}')

    geo = find_geometry(video, meta)
    if impact:
        geo['x'], geo['y'] = impact
    if tilt is not None:
        geo['tilt'] = tilt
    box = crop_box(meta, geo)
    ts, M = column_motion(video, meta, geo['x'])
    if gate:
        drops = detect_drops(ts, M[:, gate[0]:gate[1]].mean(1))
    else:
        gate, drops = choose_gate(ts, M, geo.get('top', 0), geo['y'], meta['h'])
    half = int(GATE_HALF_W * meta['h'])
    say(f'drop column x={geo["x"]}, water line y={geo["y"]}, tilt {geo["tilt"]:+.2f} deg, gate rows {gate[0]}-{gate[1]},'
        f' crop {box[0]}x{box[1]}+{box[2]}+{box[3]}')
    write_check(video, meta, geo, (geo['x'] - half, gate[0], geo['x'] + half, gate[1]), box).save(
        out / 'check.jpg', quality=85)

    n = int(round(meta['duration'] * FS))
    if audio:
        if not meta['has_audio']:
            sys.exit('the video has no audio track to align the external audio against')
        offset, confidence = align(read_audio(video), ext := read_audio(audio))
        raw = on_timeline(ext, offset, n)
        say(f'audio offset {offset:+.3f} s (peak {confidence:.1f}x the next best'
            f'{", LOW - check by ear" if confidence < ALIGN_OK else ""})')
    elif meta['has_audio']:
        offset, confidence = 0.0, None
        raw = on_timeline(read_audio(video), 0.0, n)
    else:
        sys.exit('no audio: the video has no track and --audio was not given')

    cl = clean(raw)
    heard_t = onsets(cl)
    iv = np.diff(drops)
    unsure = confidence is not None and confidence < ALIGN_OK
    # Sounds from audio that may belong to another take say nothing about these drops.
    lag, heard = (None, [False] * len(drops)) if unsure else gate_to_sound(drops, heard_t)
    av_ms = lag * 1e3 if lag is not None else None

    wavfile.write(out / 'clean.wav', FS, (cl * 32767).astype(np.int16))
    wavfile.write(out / 'raw.wav', FS, (raw / (np.abs(raw).max() or 1) * 0.9 * 32767).astype(np.int16))
    spec_cps = spectrogram_png(cl, out / 'spec-clean.png', meta['duration'])
    spectrogram_png(raw, out / 'spec-raw.png', meta['duration'])
    encode_video(video, out / 'video.mp4', meta, box, geo['tilt'])

    data = dict(
        name=name, created=meta['created'], duration=round(meta['duration'], 3), fps=round(meta['fps'], 3),
        channel=channel, offset=round(offset, 4), confidence=confidence and round(confidence, 2),
        offset_unsure=unsure,
        wave_rate=WAVE_RATE, loud_rate=LOUD_RATE, spec_cps=spec_cps, spec_fmax=SPEC_FMAX,
        wave=dict(clean=minmax(cl), raw=minmax(raw)),
        loud=dict(clean=loudness(cl), raw=loudness(raw)),
        drops=np.round(drops, 4).tolist(), heard=heard,
        stats=dict(drops=len(drops), rate=round(1 / float(np.median(iv)), 2) if len(iv) else None,
                   mean_ms=round(float(iv.mean()) * 1e3, 1) if len(iv) else None,
                   sd_ms=round(float(iv.std()) * 1e3, 1) if len(iv) else None,
                   heard=int(sum(heard)), av_ms=av_ms and round(av_ms, 1)))
    (out / 'data.js').write_text(
        f'(window.RECORDINGS = window.RECORDINGS || {{}})[{json.dumps(name)}] = {json.dumps(data)};\n')
    write_index(out_root)
    s = data['stats']
    say(f'{s["drops"]} drops, {s["rate"]} Hz, {s["heard"]} heard, a/v {s["av_ms"]} ms -> {out}')
    return data


# --- self-test -------------------------------------------------------------

def synth(dir, hdr, tilt_deg=2.0, lag=0.4, fps=60, dur=6.0, W=640, H=360, seed=1):
    """A clip with known drops, a tilted water line, and clicks on an audio track and a lagged copy."""
    rng = np.random.default_rng(seed)
    t_imp = np.cumsum(rng.uniform(0.15, 0.25, 40))
    t_imp = t_imp[(t_imp > 0.5) & (t_imp < dur - 0.3)]
    x0, y0, k = 320, 250, math.tan(math.radians(tilt_deg))
    yy, xx = np.mgrid[0:H, 0:W]
    base = np.where(yy > y0 + k * (xx - x0), 200, 70).astype(np.float32)
    fall, jet = 0.12, 0.15   # seconds from nozzle to surface; how long the rebound jet stands
    raw = []
    for i in range(int(dur * fps)):
        t = i / fps
        # The things that broke detection on real footage: light flicker, a hanging drop that
        # wobbles at the nozzle, a rebound jet above the surface, and someone walking past.
        f = base + 15 * math.sin(2 * math.pi * 19 * t) + rng.normal(0, 3, base.shape)
        f[(yy - 12) ** 2 + (xx - x0) ** 2 < rng.uniform(30, 70)] = 255
        wx = int(40 + 30 * t)   # passes beside the drop column; through it would hide the drop itself
        f[60:200, wx:wx + 40] = 40
        for ti in t_imp:
            if ti - fall <= t <= ti:
                cy = 20 + (t - ti + fall) / fall * (y0 - 20)
                f[(yy - cy) ** 2 + (xx - x0) ** 2 < 36] = 255
            if ti < t < ti + jet:
                top = y0 - 60 * math.sin(math.pi * (t - ti) / jet)
                f[(yy > top) & (yy < y0) & (abs(xx - x0) < 4)] = 230
        raw.append(np.clip(f, 0, 255).astype(np.uint8))
    n = int(dur * FS)
    click = np.sin(2 * np.pi * 5000 * np.arange(480) / FS) * np.exp(-np.arange(480) / 96)
    track = rng.normal(0, 0.003, n)
    for ti in t_imp:
        j = int(ti * FS)
        track[j:j + 480] += 0.5 * click[:n - j]
    ext = track[int(lag * FS):] + rng.normal(0, 0.003, n - int(lag * FS))   # recorder started `lag` s late
    wavfile.write(dir / 'track.wav', FS, (track * 32767).astype(np.int16))
    wavfile.write(dir / 'ext.wav', FS, (ext * 32767).astype(np.int16))
    tags = ['-color_primaries', 'bt2020', '-color_trc', 'arib-std-b67', '-colorspace', 'bt2020nc',
            '-pix_fmt', 'yuv420p10le'] if hdr else ['-pix_fmt', 'yuv420p']
    subprocess.run([FFMPEG, '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'gray', '-s', f'{W}x{H}',
                    '-r', str(fps), '-i', '-', '-i', str(dir / 'track.wav'), *(['-c:v', 'libx265', '-x265-params', 'log-level=error'] if hdr else ['-c:v', 'libx264']),
                    *tags, '-c:a', 'aac', '-b:a', '256k', '-shortest', str(dir / 'clip.mov')],
                   input=b''.join(r.tobytes() for r in raw), check=True)
    return t_imp, fall


def selftest():
    fails = []

    def check(name, ok, detail=''):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            fails.append(name)

    for hdr in (False, True):
        print(f'\n{"HDR (HLG) clip, own audio" if hdr else "SDR clip, external audio started 0.4 s late"}')
        tmp = pathlib.Path(tempfile.mkdtemp())
        truth, fall = synth(tmp, hdr)
        d = process(tmp / 'clip.mov', None if hdr else tmp / 'ext.wav', name='t', out_root=tmp / 'out', quiet=True)
        got = np.array(d['drops'])
        check('every drop found once', len(got) == len(truth), f'{len(got)} of {len(truth)}')
        if len(got) == len(truth):
            err = np.diff(got) - np.diff(truth)
            check('intervals within one frame', np.abs(err).max() < 1 / 60,
                  f'worst {np.abs(err).max() * 1e3:.1f} ms, frame is 16.7 ms')
        if not hdr:
            check('audio offset recovered', abs(d['offset'] + 0.4) < 0.003, f'{d["offset"]:+.4f} s')
            check('and trusted', not d['offset_unsure'], f'{d["confidence"]}x')
        check('every drop heard', d['stats']['heard'] == len(truth), f'{d["stats"]["heard"]}')
        out = tmp / 'out' / 't'
        check('outputs written', all((out / f).exists() for f in
              ['video.mp4', 'clean.wav', 'raw.wav', 'spec-clean.png', 'data.js', 'check.jpg']))
        check('index lists it', 'window.RECORDING_LIST = ["t"]' in (tmp / 'out' / 'index.js').read_text())
        lvl = probe(out / 'video.mp4')
        check('output is SDR', not lvl['hdr'])
        geo = find_geometry(out / 'video.mp4', lvl)
        check('output is level', abs(geo['tilt']) < 0.5, f'residual {geo["tilt"]:+.2f} deg')

    print('\nexternal audio from another take')
    tmp = pathlib.Path(tempfile.mkdtemp())
    synth(tmp, False)
    rng = np.random.default_rng(5)
    other = rng.normal(0, 0.003, 6 * FS)   # another take: its own drops, at unrelated times
    for j in (rng.uniform(0, 5.9, 30) * FS).astype(int):
        other[j:j + 480] += 0.5 * np.sin(2 * np.pi * 5000 * np.arange(480) / FS) * np.exp(-np.arange(480) / 96)
    wavfile.write(tmp / 'other.wav', FS, (other * 32767).astype(np.int16))
    d = process(tmp / 'clip.mov', tmp / 'other.wav', name='t', out_root=tmp / 'out', quiet=True)
    check('flagged unsure', d['offset_unsure'], f'{d["confidence"]}x')
    check('claims no sounds and no delay', d['stats']['heard'] == 0 and d['stats']['av_ms'] is None)

    print('\nno drops anywhere')
    g, d = choose_gate(np.arange(600) / 60, np.zeros((600, 360), np.float32), 20, 250, 360)
    check('falls back to a band on the water line', g[0] < 250 < g[1] and len(d) == 0, f'rows {g}')
    rng = np.random.default_rng(0)
    _, c = align(rng.normal(size=10 * FS).astype(np.float32), rng.normal(size=10 * FS).astype(np.float32))
    check('unrelated audio is not trusted', c < ALIGN_OK, f'{c:.2f}x')
    check('no sounds means no delay and nothing heard', gate_to_sound(np.array([1.0]), np.array([])) == (None, [False]))

    print(f'\n{"all passed" if not fails else f"{len(fails)} failed: {fails}"}')
    return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('video', nargs='?')
    ap.add_argument('--audio', help='separately recorded audio to align and use instead of the video\'s own')
    ap.add_argument('--name', help='folder name under site/recordings (default: video file name)')
    ap.add_argument('--channel', choices=['microphone', 'hydrophone'], default='microphone')
    ap.add_argument('--impact', nargs=2, type=int, metavar=('X', 'Y'), help='impact point in pixels')
    ap.add_argument('--tilt', type=float, help='water-line tilt in degrees (positive: right side lower)')
    ap.add_argument('--gate', nargs=2, type=int, metavar=('Y0', 'Y1'),
                    help='pixel rows the falling drop crosses, between the nozzle and the jet')
    ap.add_argument('--selftest', action='store_true', help='run the pipeline on synthetic clips and check it')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.video:
        ap.error('give a video, or --selftest')
    process(a.video, a.audio, a.name, a.channel, a.impact, a.tilt, a.gate and tuple(a.gate))
    return 0


if __name__ == '__main__':
    sys.exit(main())
