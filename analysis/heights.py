#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "imageio-ffmpeg", "pillow", "av", "noisereduce"]
# ///
"""Compare the sound of one drop across fall heights, for site/heights.html: clean each phone
recording of background music, find every drop, measure its sound and write the page's data.

    uv run analysis/heights.py --clip 20 "audios /water drop 20 cm.mp4" --clip 40 "audios /water drop 40 cm .mp4"
    uv run analysis/heights.py --set sweep --same-tap --clip 35 a.mpeg --video 35 IMG_5388.mov 1000 500
    uv run analysis/heights.py --selftest

Each recording is cut to its steadiest stretch of --span seconds, so every height is judged on its cleanest drops.
A slow-motion video of the same drip can sit beside a recording; its drops are pinned onto the sound's.
"""

import argparse, base64, collections, json, pathlib, subprocess, sys, tempfile
import numpy as np, imageio_ffmpeg
import scipy.ndimage as nd, scipy.signal as ss
from scipy.io import wavfile
from lanes import FS, lanes, scale
from recording import probe, frames

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'site' / 'recordings'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

HP_HZ = 1000                # cleaning high-pass: music and hum sit below it, each drop keeps a 1.2-1.4 kHz burst above it
HP_ORDER = 8                # ...steep enough to keep that burst within 0.3 dB
NFFT, HOP = 1024, 256       # STFT used to subtract the steady noise left after the high-pass
NOISE_FLOOR_GAIN = 0.3      # subtraction never cuts a bin by more than this, which keeps faint drops and avoids warbling
BAND = (2000, 9000)         # drop sound is measured here; the phone's codec invents what lies above 9 kHz
DET_BAND = (4000, 9000)     # drops are found here, where the music is quietest
RATE = 2000                 # detection envelope rate, Hz
FLOOR_S = 3.0               # the detection floor is the running median over this long
PERIOD_S = (0.5, 6.0)       # drip periods searched
LOUD = 10.0                 # with --same-tap, drops this many times the floor in every clip set the shared period...
FIT, FIT_REL = 0.8, 0.06    # ...the longest fitting nearly this share of the best fit's gaps as whole periods, within this fraction of one
NEAR = 0.2                  # each clip then fits its own period within this fraction of the shared one, as the drip slows with the tank...
OWN_GAPS = 6                # ...when it has at least this many gaps between loud drops; a near-silent clip keeps the shared period
BETWEEN = 0.4               # the period is halved while midpoints hold drops this much more often than quarter points
SNR_CAP = 100.0             # one loud burst cannot drag the drip's beat grid beyond this
BEAT_STIFF = 150.0          # beat grid penalty on log(interval / period) squared
SEARCH_S = 0.04             # each drop is looked for this far either side of its beat
LOOKBACK_S = 0.2            # ...and moved to an earlier event this close before it, in case the beat sat on a rebound
THR = 3.0                   # a drop must rise this many times above the floor
CLUSTER = 0.25              # an earlier event counts if it reaches this fraction of the drop's peak
BUSY = 2.0                  # median level 30-300 ms either side of a drop above this many times the floor is speech or handling
ENV_MS = 1                  # feature envelope step, ms
PRE_MS, POST_MS = 10, 140   # stored sound around each drop
FLOOR_MS = (300, 20)        # each drop's own noise floor is measured over this stretch before it, ms
END_DB = 6.0                # a drop's sound has ended once it falls within this of the floor
RING_WIN, RING_HOP, RING_NFFT = 512, 96, 2048   # ring STFT: 10.7 ms frames every 2 ms
RING_HALF = 150             # a frame is tonal if at least RING_TONAL of what it adds to the floor lies this close to its peak...
RING_TONAL = 0.3
RING_SNR = 10.0             # ...and that peak is this many times its own frequency's floor
RING_FRAMES = 3             # a ring is this many tonal frames in a row, each within RING_DF of the last
RING_DF = 250
RING_START_MS = 20          # only a ring starting this soon after the impact belongs to it; later ones follow a rebound
VIEW_RATE = 24000           # stored sound rate; still above twice the 9 kHz the codec keeps
AUDIO_KBPS = '96k'          # page audio bitrate
SPAN_S = 60.0               # each recording is cut to its steadiest stretch this long
REGULAR = 0.15              # a gap within this fraction of the drip period is a clean one; any other is a missed or extra drop
CAPTURE_FPS = 240           # the videos' slow-motion capture rate; their drip must then match the sound's within RATE_TOL
RATE_TOL = 0.1
LEVEL_BLOCK_S = 0.1         # a slow-motion clip's slowed stretch is found from its own sound level in blocks this long...
SLOW_MARGIN_S = 0.5         # ...less this many clip seconds at each end, where the speed ramps
VID_SCALE = 2               # video frames are searched at this fraction of full size
VID_HALF_W = 120            # drops are looked for this many pixels either side of the given column, as the drip wanders
VID_TOP = 0.3               # ...from this fraction of the way down to the water line; above it is the faucet's stand
VID_LIT = 12                # a pixel lights when it is this many grey levels above the frames two either side
VID_ENTER = 0.25            # a drop is first seen in this top share of the searched rows
FALL_STEP = (10, 60)        # rows (searched size) a falling drop moves from one frame to the next...
FALL_FRAMES = 4             # ...for at least this many frames
VID_GAP_S = 0.5             # no second drop this soon after one, real seconds
VID_CROP = (720, 960)       # video crop around the drop column, pixels, with the water line VID_WATER of the way down
VID_WATER = 0.6
VID_OUT = (360, 480)        # page video size
VID_CRF = 26


def load(path):
    raw = subprocess.run([FFMPEG, '-v', 'error', '-i', str(path), '-ac', '1', '-ar', str(FS), '-f', 'f32le', '-'],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.float32).astype(float)


def clean(x):
    """Zero-phase high-pass, then mild subtraction of each frequency's median power, which is the between-drop noise."""
    y = ss.sosfiltfilt(ss.butter(HP_ORDER, HP_HZ, 'high', fs=FS, output='sos'), x)
    _, _, Y = ss.stft(y, FS, nperseg=NFFT, noverlap=NFFT - HOP)
    P = np.abs(Y) ** 2
    noise = np.median(P, axis=1, keepdims=True) / np.log(2)   # median of an exponential power is ln 2 times its mean
    gain = np.sqrt(np.maximum(1 - noise / (P + 1e-30), NOISE_FLOOR_GAIN ** 2))
    return ss.istft(Y * gain, FS, nperseg=NFFT, noverlap=NFFT - HOP)[1][:len(x)]


def bandpass(x, band):
    return ss.sosfiltfilt(ss.butter(4, band, 'bandpass', fs=FS, output='sos'), x)


def busy(snr, k):
    """Speech, handling or a knock around envelope sample k: a raised median just before or after it."""
    a, b = int(0.03 * RATE), int(0.3 * RATE)
    return max(np.median(snr[max(0, k - b):max(1, k - a)]), np.median(snr[k + a:k + b])) > BUSY


def period(snr):
    """Drip period in envelope samples, from the autocorrelation summed at one, two and three periods."""
    a = np.log(np.clip(snr, 1, None))
    a = nd.maximum_filter1d(a, int(0.015 * RATE)) - a.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(a, 2 * len(a))) ** 2)[:len(a)]
    lags = np.arange(int(PERIOD_S[0] * RATE), int(PERIOD_S[1] * RATE))
    lags = lags[3 * lags < len(ac)]
    return int(lags[np.argmax(sum(ac[k * lags] for k in (1, 2, 3)))])


def loud_gaps(snr):
    """Seconds between consecutive loud drops, leaving out any inside speech or handling noise."""
    pk, _ = ss.find_peaks(snr, height=LOUD, distance=int(PERIOD_S[0] * RATE))
    return np.diff([k for k in pk if not busy(snr, k)]) / RATE


def tap_period(gaps, lo=PERIOD_S[0], hi=PERIOD_S[1]):
    """Envelope samples of the longest period in lo-hi s fitting nearly as many gaps as whole periods as the best does: a
    missed drop doubles a gap but never halves one, so the period's fractions fit as well yet lose to it, and its double fits half."""
    cands = np.arange(hi, lo, -0.005)
    k = np.round(gaps[None, :] / cands[:, None])
    ok = (k >= 1) & (np.abs(gaps[None, :] - k * cands[:, None]) <= FIT_REL * cands[:, None])
    fit = ok.mean(1)
    i = int(np.argmax(fit >= FIT * fit.max()))   # the first, so the longest, candidate near the best
    return int(round(np.median(gaps[ok[i]] / k[i][ok[i]]) * RATE))


def drip(snr, o, p=None):
    """Beat grid for the drip. When only every other drop is loud, the period found is doubled, so it is halved while
    midpoints between beats hold drops more often than noise explains."""
    p, w = p or period(snr), int(SEARCH_S * RATE)
    beats = beat_grid(o, p)
    hit = lambda f: np.mean([snr[max(0, m - w):m + w].max() >= THR for m in (u + int(f * (v - u)) for u, v in zip(beats, beats[1:]))])
    while p // 2 >= PERIOD_S[0] * RATE and len(beats) > 2:
        if hit(0.5) - (hit(0.25) + hit(0.75)) / 2 < BETWEEN:
            break
        p //= 2
        beats = beat_grid(o, p)
    return beats


def beat_grid(o, p):
    """One beat per drip period, placed on the onset strength o by dynamic programming, so a run of faint
    drops or a drifting drip rate cannot knock the grid out of step."""
    lags = np.arange(int(0.6 * p), int(1.4 * p) + 1)
    pen = BEAT_STIFF * np.log(lags / p) ** 2
    C, back = o.copy(), -np.ones(len(o), int)
    for t in range(lags[0], len(o)):
        ok = lags <= t
        cand = C[t - lags[ok]] - pen[ok]
        j = int(np.argmax(cand))
        if cand[j] > 0:
            C[t] += cand[j]; back[t] = t - lags[ok][j]
    t, beats = len(o) - lags[-1] + int(np.argmax(C[-lags[-1]:])), []
    while t >= 0:
        beats.append(t); t = back[t]
    return beats[::-1]


def strength(y):
    """Detection envelope at RATE, as multiples of its running floor."""
    e = ss.sosfiltfilt(ss.butter(2, 400, fs=FS, output='sos'), np.abs(ss.hilbert(bandpass(y, DET_BAND))))[::FS // RATE]
    return e / nd.median_filter(e, int(FLOOR_S * RATE), mode='nearest')


def own_period(snr, p):
    """This clip's drip period, fitted to its own loud drops within NEAR of the shared period p (envelope samples)."""
    g = loud_gaps(snr)
    return tap_period(g, (1 - NEAR) * p / RATE, (1 + NEAR) * p / RATE) if len(g) >= OWN_GAPS else p


def find_drops(y, p=None):
    """Onset times, s: the drop nearest each beat, if it clears THR and is not inside speech or handling noise.
    p is the drip period in envelope samples when it is known; otherwise it is found from this clip alone."""
    snr = strength(y)
    p = p and own_period(snr, p)
    w, lb = int(SEARCH_S * RATE), int(LOOKBACK_S * RATE)
    found = []
    for beat in drip(snr, np.log(np.clip(snr, 1, SNR_CAP)), p):
        k = max(0, beat - w) + int(np.argmax(snr[max(0, beat - w):beat + w]))
        pk, _ = ss.find_peaks(snr[max(0, k - lb):k - int(0.01 * RATE)], height=max(THR, CLUSTER * snr[k]))
        k = max(0, k - lb) + int(pk[0]) if len(pk) else k
        if snr[k] >= THR and not busy(snr, k):
            found.append(k)
    return [onset(y, k / RATE) for k in found]


def onset(y, t):
    """Where the sound near t first climbs 30% of the way from its floor to its peak."""
    a, b = int((t - 0.03) * FS), int((t + 0.02) * FS)
    env = np.abs(ss.hilbert(bandpass(y[max(0, a - 4800):b + 4800], DET_BAND)))[a - max(0, a - 4800):][:b - a]
    env = ss.sosfiltfilt(ss.butter(2, 3000, fs=FS, output='sos'), env)
    k = int(np.argmax(env))
    fl = np.median(env[:int(0.02 * FS)])
    below = np.nonzero(env[:k + 1] < fl + 0.3 * (env[k] - fl))[0]
    return (a + (below[-1] + 1 if len(below) else 0)) / FS


def measure(yb, t):
    """Loudness over the drop's own floor, how long it lasts, and its ring (pitch, decay)."""
    s, m = int(t * FS), FS * ENV_MS // 1000
    blocks = lambda a, b: np.sqrt((yb[a:b][:(b - a) // m * m].reshape(-1, m) ** 2).mean(1))
    floor = np.median(blocks(s - FLOOR_MS[0] * FS // 1000, s - FLOOR_MS[1] * FS // 1000))
    env = nd.uniform_filter1d(blocks(s, s + POST_MS * FS // 1000), 3) / floor
    pk = int(np.argmax(env[:30]))
    gone = np.nonzero(env[pk:] < 10 ** (END_DB / 20))[0]
    end = pk + (int(gone[0]) if len(gone) else len(env) - pk)
    hz = ring(yb[s:s + POST_MS * FS // 1000], yb[s - FLOOR_MS[0] * FS // 1000:s - FLOOR_MS[1] * FS // 1000])
    dur, rings, tau = end * ENV_MS, hz is not None, None
    if rings and end - pk >= 5:
        tail = np.log(env[pk:end])
        tau = round(-1 / np.polyfit(np.arange(len(tail)) * ENV_MS, tail, 1)[0], 2)
    return dict(db=round(20 * np.log10(env[pk]), 2), dur=dur, ring=rings, hz=hz, tau=tau if tau and 0 < tau < POST_MS else None), env


def ring(seg, pre):
    """Pitch of a tone that stands out of the floor for RING_FRAMES frames in a row, starting near the impact; else None."""
    spec = lambda v: ss.stft(v, FS, nperseg=RING_WIN, noverlap=RING_WIN - RING_HOP, nfft=RING_NFFT, boundary=None, padded=False)
    f, t, Z = spec(seg)
    P, N = np.abs(Z) ** 2, (np.abs(spec(pre)[2]) ** 2).mean(1, keepdims=True)
    inb = ((f >= BAND[0]) & (f <= BAND[1]))[:, None]
    E = np.where(inb, np.maximum(P - N, 0), 0)
    j = E.argmax(0)
    near = abs(f[:, None] - f[j]) <= RING_HALF
    ok = ((E * near).sum(0) >= RING_TONAL * E.sum(0)) & (P[j, range(len(t))] >= RING_SNR * N[j, 0])
    run = []
    for i in range(len(t)):
        run = run + [i] if ok[i] and (not run or abs(f[j[i]] - f[j[run[-1]]]) < RING_DF) else ([i] if ok[i] else [])
        if len(run) >= RING_FRAMES:
            return round(float(np.median(f[j[run]]))) if t[run[0]] * 1000 <= RING_START_MS + RING_WIN / FS * 500 else None
    return None


def snippet(y, t):
    s = int(t * FS)
    seg = ss.resample_poly(y[s - PRE_MS * FS // 1000:s + POST_MS * FS // 1000], VIEW_RATE, FS)
    scale = float(np.abs(seg).max()) or 1.0
    q = np.round(seg / scale * 127).astype(np.int8)
    return base64.b64encode(q.tobytes()).decode(), scale


def slow_stretch(path):
    """Clip seconds of a phone slow-motion clip's slowed stretch: its sound is slowed with it into a low murmur, so the
    stretch is the quiet middle between the normal-speed ends."""
    x, b = load(path), int(LEVEL_BLOCK_S * FS)
    lv = 20 * np.log10(np.sqrt((x[:len(x) // b * b].reshape(-1, b) ** 2).mean(1)) + 1e-12)
    quiet = lv < (np.median(lv) + np.median(lv[:int(1 / LEVEL_BLOCK_S)])) / 2
    return int(np.argmax(quiet)) * LEVEL_BLOCK_S + SLOW_MARGIN_S, (len(quiet) - int(np.argmax(quiet[::-1]))) * LEVEL_BLOCK_S - SLOW_MARGIN_S


def lit_rows(path, meta, x, y):
    """Real times, and per frame which rows of the drop column light up against the frames two either side, with the
    water line's row. The column is the most lit one near x, as the drip wanders; y is the water line, pixels."""
    s, r0 = VID_SCALE, int(VID_TOP * y) // VID_SCALE
    c0, c1 = max(0, x - VID_HALF_W) // s, (x + VID_HALF_W) // s
    buf, ts, out = collections.deque(maxlen=5), [], []
    for t, g in frames(path, meta, s):
        buf.append((t, g[r0:(y + 10) // s, c0:c1].astype(np.int16)))
        if len(buf) == 5:
            d = buf[2][1] - np.maximum(buf[0][1], buf[4][1])
            L = d - np.median(d, axis=1, keepdims=True) > VID_LIT   # less each row's median, which is the lights' flicker
            c = int(np.argmax(nd.uniform_filter1d(L.mean(0), 3)))
            ts.append(buf[2][0]); out.append(L[:, max(0, c - 2):c + 3].any(1))
    return np.array(ts), np.array(out), y // s - r0


def impacts(ts, P, surf):
    """When each drop meets the water: a lit blob entering at the top is followed down, FALL_STEP rows a frame to the lit
    row nearest where its speed so far puts it, and the line through its fall is carried on to the water line's row surf."""
    out, t, top = [], 1, int(VID_ENTER * surf)
    while t < len(P):
        rows = np.nonzero(P[t][:top])[0]
        f, run = ([rows.max()], [t]) if len(rows) else ([], [])
        while run and run[-1] + 1 < len(P):
            nxt = np.nonzero(P[run[-1] + 1])[0]
            nxt = nxt[(nxt >= f[-1] + FALL_STEP[0]) & (nxt <= f[-1] + FALL_STEP[1]) & (nxt < surf)]
            if not len(nxt):
                break
            ahead = f[-1] + (f[-1] - f[-2] if len(f) > 1 else sum(FALL_STEP) / 2)
            f.append(nxt[np.argmin(np.abs(nxt - ahead))]); run.append(run[-1] + 1)
        if len(run) >= FALL_FRAMES:
            k, b = np.polyfit(ts[run], f, 1)
            out.append(float((surf - b) / k))
            t = int(np.searchsorted(ts, out[-1] + VID_GAP_S))
        else:
            t += 1
    return out


def fill(times, p):
    """Drop times with each missed drop put back at its share of the gap, as the page's beats() does."""
    out = []
    for t in times:
        if out:
            n = int(round((t - out[-1]) / p))
            out += [out[-1] + (t - out[-1]) * k / n for k in range(1, n)]
        out.append(t)
    return out


def same_drip(path, hits, beats):
    """The video's and the sound's drip periods, s, which must agree, or the capture rate assumed is wrong."""
    if len(hits) < 2:
        raise SystemExit(f'{path}: {len(hits)} drops seen in the video; check its drop column and water line')
    g, p = float(np.median(np.diff(hits))), float(np.median(np.diff(beats)))
    if abs(g / p - 1) > RATE_TOL:
        raise SystemExit(f'{path}: drops every {g:.2f} s in the video, {p:.2f} s in the sound; was it shot at {CAPTURE_FPS} fps?')
    return g, p


def trim(hits, g):
    """hits without a first or last drop whose gap to its neighbour is no whole number of periods g: the speed still ramps there."""
    off = lambda a, b: abs((b - a) / g - round((b - a) / g)) > REGULAR
    while len(hits) > 2 and off(hits[0], hits[1]):
        hits = hits[1:]
    while len(hits) > 2 and off(hits[-2], hits[-1]):
        hits = hits[:-1]
    return hits


def pins(beats, hits):
    """(sound s, video s) pairs putting each of the video's drops on one of the sound's, the video's run centred in it."""
    j = max(0, (len(beats) - len(hits)) // 2)
    return [[round(b, 4), round(v, 4)] for b, v in zip(beats[j:], hits)]


def video(path, x, y, out, beats):
    """A slow-motion clip's slowed stretch, cropped around the drop column, at real speed into out/video.mp4, and the pins
    that put its drops on the sound's. It was filmed apart from the sound, so its drops are other drops of the same drip."""
    meta = probe(path, CAPTURE_FPS, slow_stretch(path))
    hits = impacts(*lit_rows(path, meta, x, y))
    g, p = same_drip(path, hits, beats)
    W, H = VID_CROP
    cx, cy = min(max(x - W // 2, 0), meta['w'] - W), min(max(y - int(VID_WATER * H), 0), meta['h'] - H)
    t0, t1 = meta['span']
    subprocess.run([FFMPEG, '-v', 'error', '-y', '-ss', str(t0), '-to', str(t1), '-i', str(path), '-an',
                    '-vf', f'setpts=PTS/{meta["slow"]},crop={W}:{H}:{cx}:{cy},scale={VID_OUT[0]}:{VID_OUT[1]}',
                    '-fps_mode', 'passthrough', '-c:v', 'libx264', '-crf', str(VID_CRF), '-preset', 'slow', '-g', '60',
                    '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(out / 'video.mp4')], check=True)
    print(f'  video: {len(hits)} drops, one every {g:.3f} s against the sound\'s {p:.3f} s')
    return dict(duration=round(meta['duration'], 3), pins=pins(beats, fill(trim(hits, g), g)))


def encode(x, path):
    with tempfile.TemporaryDirectory() as tmp:
        wav = pathlib.Path(tmp) / 'a.wav'
        wavfile.write(wav, FS, np.clip(x * 32767, -32767, 32767).astype(np.int16))
        subprocess.run([FFMPEG, '-v', 'error', '-y', '-i', str(wav), '-c:a', 'aac', '-b:a', AUDIO_KBPS, str(path)], check=True)


def typical(rows):
    """The drop nearest the median loudness and length among its height's usual kind (ringing or not)."""
    ring = sum(r['ring'] for r in rows) * 2 >= len(rows)
    pool = [i for i, r in enumerate(rows) if r['ring'] == ring]
    md, mu = np.median([rows[i]['db'] for i in pool]), np.median([rows[i]['dur'] for i in pool])
    return min(pool, key=lambda i: abs(rows[i]['db'] - md) / 6 + abs(rows[i]['dur'] - mu) / 10)


def analyse(x, p=None):
    y = clean(x)   # finds faint drops, but its gain opens around each one, so drops are measured on the raw band
    yb = bandpass(x, BAND)
    times = [t for t in find_drops(y, p) if FLOOR_MS[0] / 1000 < t < len(y) / FS - POST_MS / 1000]
    rows, envs = [], []
    for t in times:
        r, env = measure(yb, t)
        rows.append(dict(t=round(t, 4), **r)); envs.append(20 * np.log10(np.maximum(env, 1e-3)))
    return y, rows, np.array(envs)


def steadiest(times, span, duration):
    """Start of the stretch `span` long whose drops come most evenly: the most clean gaps, each missed or extra drop
    costing two, ties to the least jitter."""
    t = np.array(times)
    if duration <= span or len(t) < 3:
        return 0.0
    iv, best = np.diff(t), None
    p = np.median(iv)
    for a in np.r_[np.clip(t - 0.5, 0, duration - span), duration - span]:
        g = iv[(t[:-1] >= a) & (t[1:] <= a + span)]
        ok = np.abs(g / p - 1) < REGULAR
        score = (int(ok.sum()) - 2 * int((~ok).sum()), -float(np.std(g[ok])) if ok.any() else 0.0)
        if best is None or score > best[0]:
            best = (score, float(a))
    return best[1]


def process(clips, out=OUT / 'heights', span=SPAN_S, same_tap=False, videos=()):
    """Every clip's steadiest stretch: audio, lanes and drops into out, scaled alike so the heights compare by eye.
    With same_tap the clips share one drip period, found from the loud drops of all of them together.
    videos holds (height, path, x, y) for slow-motion clips of the drip: x is the drop column and y the water line, pixels."""
    vids = {f'{float(h):g}cm': (path, int(x), int(y)) for h, path, x, y in videos}
    hs = [float(h) for h, _ in clips]
    if len(set(hs)) != len(hs):
        raise SystemExit('one clip per height: keep the take with the steadier drip')
    clips = [(float(h), load(path)) for h, path in sorted(clips, key=lambda c: float(c[0]))]
    p = tap_period(np.concatenate([loud_gaps(strength(clean(x))) for _, x in clips])) if same_tap else None
    if p:
        print(f'shared drip period {p / RATE:.3f} s')
    recs = []
    for h, x in clips:
        a = steadiest([r['t'] for r in analyse(x, p)[1]], span, len(x) / FS) if span else 0.0
        x = x[int(a * FS):int((a + span) * FS)] if span else x
        own = p and own_period(strength(clean(x)), p) / RATE
        recs.append(dict(slug=f'{h:g}cm', h=h, start=round(a, 2), x=x, a=analyse(x, p),
                         drip_s=own and round(own, 3), drips=own and int(round(len(x) / FS / own))))
    ref = dict(clean=scale(*[r['a'][0] for r in recs]), raw=scale(*[r['x'] for r in recs]))
    for r in recs:
        x, (y, rows, envs) = r.pop('x'), r.pop('a')
        d = out / r['slug']
        d.mkdir(parents=True, exist_ok=True)
        encode(x, d / 'raw.m4a'); encode(y, d / 'clean.m4a')
        snips = [snippet(y, row['t']) for row in rows]
        q = np.percentile(envs, [25, 50, 75], axis=0)
        r.update(duration=round(len(x) / FS, 2), period_ms=round(float(np.median(np.diff([w['t'] for w in rows]))) * 1000, 1),
                 typical=typical(rows), drops=rows, snip=[w for w, _ in snips], snip_scale=[round(v, 6) for _, v in snips],
                 env=[[round(float(v), 1) for v in row] for row in q], **lanes(y, x, d, len(x) / FS, ref))
        print(f'{r["slug"]}: {len(rows)} drops, period {r["period_ms"]} ms, {sum(w["ring"] for w in rows)} ring')
        if r['slug'] in vids:
            r['video'] = video(*vids[r['slug']], d, fill([w['t'] for w in rows], r['drip_s'] or r['period_ms'] / 1000))
    data = dict(rate=VIEW_RATE, pre_ms=PRE_MS, post_ms=POST_MS, env_ms=ENV_MS, band=BAND,
                amax=max(max(c['snip_scale']) for c in recs), clips=recs)
    (out / 'data.js').write_text('window.HEIGHTS = ' + json.dumps(data, separators=(',', ':')) + ';\n')
    return data


def synth(seed=0, n=40, period=1.05, ring_every=2, faint=0.04, rebound=0.0):
    """Hum, noise and a speech-like burst, plus drops that alternate a 4 kHz ringing plink and a faint click."""
    rng = np.random.default_rng(seed)
    t = np.arange(int((n + 1) * period * FS)) / FS
    x = 0.05 * np.sin(2 * np.pi * 220 * t) + 0.03 * np.sin(2 * np.pi * 440 * t) + 0.002 * rng.standard_normal(len(t))
    truth = []
    for i in range(n):
        s = int((0.6 + i * period + 0.01 * rng.standard_normal()) * FS)
        k = np.arange(int(0.08 * FS))
        click = np.zeros(len(k)); click[:24] = rng.standard_normal(24)
        if i % ring_every == 0:
            x[s:s + len(k)] += 0.3 * click + 0.4 * np.exp(-k / (0.012 * FS)) * np.sin(2 * np.pi * 4000 * k / FS)
        else:
            x[s:s + len(k)] += faint * click
        if rebound:   # a louder plink 120 ms later, from the jet's droplet falling back
            r = s + int(0.12 * FS)
            x[r:r + len(k)] += 2 * rebound * np.exp(-k / (0.008 * FS)) * np.sin(2 * np.pi * 6000 * k / FS)
        truth.append((s / FS, i % ring_every == 0))
    b = int((0.6 + 10.5 * period - 1 / 6) * FS)   # a third of a second of talking, centred between two drops
    x[b:b + FS // 3] += 0.05 * bandpass(rng.standard_normal(FS // 3), (3000, 8000))
    return x, truth


def selftest():
    x, truth = synth()
    y = clean(x)
    hum = np.sin(2 * np.pi * 220 * np.arange(len(x)) / FS)
    assert abs(np.dot(y, hum)) / len(x) < 1e-3, 'cleaning left the hum in'
    s = int(truth[0][0] * FS)
    a, b = bandpass(x, BAND)[s:s + 2400], bandpass(y, BAND)[s:s + 2400]
    assert np.corrcoef(a, b)[0, 1] > 0.98, 'cleaning changed the drop'

    _, rows, _ = analyse(x)
    assert len(rows) == len(truth), f'found {len(rows)} drops, made {len(truth)}'
    for r, (t, ring) in zip(rows, truth):
        assert abs(r['t'] - t) < 0.002, f'drop at {t:.3f} s timed at {r["t"]} s'
        assert r['ring'] == ring, f'drop at {t:.3f} s: ring {r["ring"]}, made {ring}'
        if ring:
            assert abs(r['hz'] - 4000) < 50, f'ring read as {r["hz"]} Hz'
            assert 8 < r['tau'] < 16, f'12 ms decay read as {r["tau"]} ms'
    rr = [r for r in rows if r['ring']]
    assert np.median([r['db'] for r in rr]) > np.median([r['db'] for r in rows if not r['ring']]) + 10

    x, truth = synth(n=12, ring_every=1)
    s = int((truth[6][0] - 0.1) * FS)   # talking over one drop
    x[s:s + FS // 2] += 0.1 * bandpass(np.random.default_rng(1).standard_normal(FS // 2), (3000, 8000))
    got = find_drops(clean(x))
    assert not [d for d in got if abs(d - truth[6][0]) < 0.2], 'speech counted as a drop'
    assert len(got) == 11, f'{len(got)} drops beside the talking, made 11'

    times = list(np.r_[np.arange(0, 10), np.arange(15, 60), np.arange(60, 70) + 0.3 * np.sin(np.arange(10)), np.arange(70, 100)])
    a = steadiest(times, 25, 100)   # a run of missed drops at 10-15 s and a jittery stretch at 60-70 s
    assert 14.4 <= a and a + 25 <= 60.1 or 69.4 <= a, f'steadiest stretch starts at {a} s'

    gaps = np.array([2.9, 2.95, 5.8, 2.85, 0.7, 2.92, 8.75, 3.0, 2.88])   # missed drops, a rebound, a slow drift
    assert abs(tap_period(gaps) / RATE - 2.9) < 0.05, f'gaps of a 2.9 s drip read as {tap_period(gaps) / RATE:.2f} s'
    x, truth = synth(n=14, period=2.9)
    got = find_drops(clean(x), tap_period(loud_gaps(strength(clean(x)))))
    assert len(got) == 14 and all(abs(d - t) < 0.002 for d, (t, _) in zip(got, truth)), f'{len(got)} drops on a 2.9 s drip, made 14'

    x, truth = synth(n=12, ring_every=1, rebound=0.4)
    got = find_drops(clean(x))
    assert len(got) == 12 and all(abs(d - t) < 0.002 for d, (t, _) in zip(got, truth)), 'drops timed on their rebound'

    with tempfile.TemporaryDirectory() as tmp:
        p, q = pathlib.Path(tmp) / 'a.wav', pathlib.Path(tmp) / 'b.wav'
        wavfile.write(p, FS, (x * 20000).astype(np.int16)); wavfile.write(q, FS, (x * 5000).astype(np.int16))
        d = process([('20', p), ('25', q), ('40', p)], pathlib.Path(tmp) / 'out')
        assert [c['slug'] for c in d['clips']] == ['20cm', '25cm', '40cm']
        assert d['clips'][0]['drops'][d['clips'][0]['typical']]['ring']
        assert len(base64.b64decode(d['clips'][0]['snip'][0])) == (PRE_MS + POST_MS) * VIEW_RATE // 1000
        assert all((pathlib.Path(tmp) / 'out' / '25cm' / f).stat().st_size > 0 for f in ('clean.m4a', 'raw.m4a', 'spec-clean.png'))
        top = [np.abs(np.frombuffer(base64.b64decode(c['wave']['raw']), np.int8).astype(int)).max() for c in d['clips']]
        assert top[0] == 127 and abs(top[1] - 127 / 2) <= 4, f'a quarter as loud drawn at {top[1]}, the loud one at {top[0]} of 127: not one scale'
        try:
            process([('20', p), ('20', q)], pathlib.Path(tmp) / 'two')
            raise AssertionError('two takes at one height were accepted')
        except SystemExit:
            pass
        d = process([('20', p)], pathlib.Path(tmp) / 'cut', span=6.0)
        c = d['clips'][0]
        assert c['duration'] == 6.0 and len(c['drops']) >= 4, f'cut to {c["duration"]} s with {len(c["drops"])} drops'
        tone = np.sin(2 * np.pi * 440 * np.arange(16 * FS) / FS) * np.r_[np.ones(2 * FS), 0.01 * np.ones(12 * FS), np.ones(2 * FS)]
        wavfile.write(p, FS, (tone * 20000).astype(np.int16))
        a, b = slow_stretch(p)
        assert abs(a - 2 - SLOW_MARGIN_S) < 0.11 and abs(b - 14 + SLOW_MARGIN_S) < 0.11, f'slowed stretch of 2-14 s read as {a}-{b} s'

    rng = np.random.default_rng(2)
    ts, surf, hit = np.arange(2400) / 240, 175, [1.0, 4.1, 7.3, 9.0]   # a drop at 9 s that the camera missed comes later
    P = rng.random((len(ts), 180)) < 0.003
    for k, h in enumerate(hit[:3]):
        for j in range(-6, -1):   # 30 rows a frame, lost from sight two frames before the water as in the real clips
            f = round(h * 240) + j
            P[f, max(0, int(surf + 30 * (f - h * 240))) - 4:int(surf + 30 * (f - h * 240))] = True
            P[f, int(surf + 30 * (f - h * 240)) + 25] = k == 1 and j == -3   # a stray pixel just below one drop
        P[round(h * 240) + 1:round(h * 240) + 30, surf - 15:] = True   # the splash
    got = impacts(ts, P, surf)
    assert len(got) == 3 and all(abs(g - h) < 0.002 for g, h in zip(got, hit)), f'impacts at {hit[:3]} s read as {got}'
    assert fill([0, 1, 3, 4], 1) == [0, 1, 2, 3, 4], 'a missed drop not put back'
    assert trim([0, 1.3, 2.3, 4.3, 5.6], 1) == [1.3, 2.3, 4.3], 'drops in the speed ramps kept'
    assert pins(list(range(10)), [0.5, 1.5]) == [[4, 0.5], [5, 1.5]], 'the video not centred in the sound'
    try:
        same_drip('clip', [0, 2, 4], [0, 1, 2])
        raise AssertionError('a video drip twice the sound\'s was accepted')
    except SystemExit:
        pass
    try:
        same_drip('clip', [3.0], [0, 1, 2])
        raise AssertionError('a video with one drop was accepted')
    except SystemExit:
        pass
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--clip', nargs=2, action='append', metavar=('HEIGHT_CM', 'AUDIO'), default=[])
    ap.add_argument('--video', nargs=4, action='append', metavar=('HEIGHT_CM', 'VIDEO', 'X', 'Y'), default=[],
                    help='slow-motion video of that height\'s drip, with its drop column X and water line Y in pixels')
    ap.add_argument('--span', type=float, default=SPAN_S, help='seconds of each recording to keep, its steadiest stretch; 0 keeps it all')
    ap.add_argument('--set', default='heights', help='name of the recording set, its folder under site/recordings')
    ap.add_argument('--same-tap', action='store_true', help='the tap was not touched between clips, so they share one drip period')
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.clip:
        ap.error('give at least one --clip')
    process(a.clip, OUT / a.set, span=a.span, same_tap=a.same_tap, videos=a.video)


if __name__ == '__main__':
    sys.exit(main())
