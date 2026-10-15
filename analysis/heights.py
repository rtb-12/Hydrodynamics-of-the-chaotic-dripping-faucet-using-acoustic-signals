#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "scipy", "imageio-ffmpeg", "pillow"]
# ///
"""Compare the sound of one drop across fall heights, for site/heights.html: clean each phone
recording of background music, find every drop, measure its sound and write the page's data.

    uv run analysis/heights.py --clip 20 "audios /water drop 20 cm.mp4" --clip 40 "audios /water drop 40 cm .mp4"
    uv run analysis/heights.py --selftest

A second clip at the same height is shown as another take of it.
"""

import argparse, base64, json, pathlib, subprocess, sys, tempfile
import numpy as np, imageio_ffmpeg
import scipy.ndimage as nd, scipy.signal as ss, scipy.stats as st
from scipy.io import wavfile
from lanes import FS, lanes, scale

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'site' / 'recordings' / 'heights'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

HP_HZ = 1000                # cleaning high-pass: music and hum sit below it, each drop keeps a 1.2-1.4 kHz burst above it
HP_ORDER = 8                # ...steep enough to keep that burst within 0.3 dB
NFFT, HOP = 1024, 256       # STFT used to subtract the steady noise left after the high-pass
NOISE_FLOOR_GAIN = 0.3      # subtraction never cuts a bin by more than this, which keeps faint drops and avoids warbling
BAND = (2000, 9000)         # drop sound is measured here; the phone's codec invents what lies above 9 kHz
DET_BAND = (4000, 9000)     # drops are found here, where the music is quietest
RATE = 2000                 # detection envelope rate, Hz
FLOOR_S = 3.0               # the detection floor is the running median over this long
PERIOD_S = (0.5, 2.0)       # drip periods searched
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
FLAT_MS = 30                # flatness is read from this much sound after the onset
VIEW_RATE = 24000           # stored sound rate; still above twice the 9 kHz the codec keeps
AUDIO_KBPS = '96k'          # page audio bitrate


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


def period(snr):
    """Drip period in envelope samples, from the autocorrelation summed at one, two and three periods."""
    a = np.log(np.clip(snr, 1, None))
    a = nd.maximum_filter1d(a, int(0.015 * RATE)) - a.mean()
    ac = np.fft.irfft(np.abs(np.fft.rfft(a, 2 * len(a))) ** 2)[:len(a)]
    lags = np.arange(int(PERIOD_S[0] * RATE), int(PERIOD_S[1] * RATE))
    lags = lags[3 * lags < len(ac)]
    return int(lags[np.argmax(sum(ac[k * lags] for k in (1, 2, 3)))])


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


def find_drops(y):
    """Onset times, s: the drop nearest each beat, if it clears THR and is not inside speech or handling noise."""
    e = ss.sosfiltfilt(ss.butter(2, 400, fs=FS, output='sos'), np.abs(ss.hilbert(bandpass(y, DET_BAND))))[::FS // RATE]
    snr = e / nd.median_filter(e, int(FLOOR_S * RATE), mode='nearest')
    w, lb, a, b = (int(v * RATE) for v in (SEARCH_S, LOOKBACK_S, 0.03, 0.3))
    found = []
    for beat in beat_grid(np.log(np.clip(snr, 1, SNR_CAP)), period(snr)):
        k = max(0, beat - w) + int(np.argmax(snr[max(0, beat - w):beat + w]))
        pk, _ = ss.find_peaks(snr[max(0, k - lb):k - int(0.01 * RATE)], height=max(THR, CLUSTER * snr[k]))
        k = max(0, k - lb) + int(pk[0]) if len(pk) else k
        busy = max(np.median(snr[max(0, k - b):max(1, k - a)]), np.median(snr[k + a:k + b]))
        if snr[k] >= THR and busy <= BUSY:
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
    """Loudness over the drop's own floor, how long it lasts, its ring (pitch, decay) and spectral flatness."""
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
    Pf = np.abs(np.fft.rfft(yb[s:s + FLAT_MS * FS // 1000])) ** 2
    Pf = Pf[(np.fft.rfftfreq(FLAT_MS * FS // 1000, 1 / FS) >= BAND[0]) & (np.fft.rfftfreq(FLAT_MS * FS // 1000, 1 / FS) <= BAND[1])] + 1e-30
    return dict(db=round(20 * np.log10(env[pk]), 2), dur=dur, ring=rings, hz=hz, tau=tau if tau and 0 < tau < POST_MS else None,
                flat=round(float(np.exp(np.log(Pf).mean()) / Pf.mean()), 4)), env


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


def analyse(x):
    y = clean(x)   # finds faint drops, but its gain opens around each one, so drops are measured on the raw band
    yb = bandpass(x, BAND)
    times = [t for t in find_drops(y) if FLOOR_MS[0] / 1000 < t < len(y) / FS - POST_MS / 1000]
    rows, envs = [], []
    for t in times:
        r, env = measure(yb, t)
        rows.append(dict(t=round(t, 4), **r)); envs.append(20 * np.log10(np.maximum(env, 1e-3)))
    return y, rows, np.array(envs)


def spearman(pairs):
    r = st.spearmanr([a for a, _ in pairs], [b for _, b in pairs])
    return dict(rho=round(float(r.statistic), 3), p=float(f'{r.pvalue:.2g}'), n=len(pairs))


def process(clips, out=OUT):
    """Every clip's audio, lanes and drops into out, scaled alike so the heights compare by eye."""
    seen, recs = {}, []
    for h, path in sorted(clips, key=lambda c: float(c[0])):
        h = float(h)
        seen[h] = seen.get(h, 0) + 1
        x = load(path)
        recs.append(dict(slug=f'{h:g}cm' + (f'-{seen[h]}' if seen[h] > 1 else ''), h=h, take=seen[h], x=x, a=analyse(x)))
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
    every = [(c['h'], r) for c in recs for r in c['drops']]
    rings = [(h, r) for h, r in every if r['ring']]
    stats = dict(ring=spearman([(h, r['ring']) for h, r in every]), db=spearman([(h, r['db']) for h, r in every]),
                 dur=spearman([(h, r['dur']) for h, r in every]), flat=spearman([(h, r['flat']) for h, r in every]),
                 hz=spearman([(h, r['hz']) for h, r in rings]),
                 tau=spearman([(h, r['tau']) for h, r in rings if r['tau'] is not None]))
    data = dict(rate=VIEW_RATE, pre_ms=PRE_MS, post_ms=POST_MS, env_ms=ENV_MS, band=BAND,
                amax=max(max(c['snip_scale']) for c in recs), stats=stats, clips=recs)
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
    assert np.median([r['flat'] for r in rr]) < np.median([r['flat'] for r in rows if not r['ring']])

    x, truth = synth(n=12, ring_every=1)
    s = int((truth[6][0] - 0.1) * FS)   # talking over one drop
    x[s:s + FS // 2] += 0.1 * bandpass(np.random.default_rng(1).standard_normal(FS // 2), (3000, 8000))
    got = find_drops(clean(x))
    assert not [d for d in got if abs(d - truth[6][0]) < 0.2], 'speech counted as a drop'
    assert len(got) == 11, f'{len(got)} drops beside the talking, made 11'

    x, truth = synth(n=12, ring_every=1, rebound=0.4)
    got = find_drops(clean(x))
    assert len(got) == 12 and all(abs(d - t) < 0.002 for d, (t, _) in zip(got, truth)), 'drops timed on their rebound'

    with tempfile.TemporaryDirectory() as tmp:
        p, q = pathlib.Path(tmp) / 'a.wav', pathlib.Path(tmp) / 'b.wav'
        wavfile.write(p, FS, (x * 20000).astype(np.int16)); wavfile.write(q, FS, (x * 5000).astype(np.int16))
        d = process([('20', p), ('20', q), ('40', p)], pathlib.Path(tmp) / 'out')
        assert [c['slug'] for c in d['clips']] == ['20cm', '20cm-2', '40cm']
        assert d['clips'][0]['drops'][d['clips'][0]['typical']]['ring']
        assert len(base64.b64decode(d['clips'][0]['snip'][0])) == (PRE_MS + POST_MS) * VIEW_RATE // 1000
        assert all((pathlib.Path(tmp) / 'out' / '20cm-2' / f).stat().st_size > 0 for f in ('clean.m4a', 'raw.m4a', 'spec-clean.png'))
        top = [np.abs(np.frombuffer(base64.b64decode(c['wave']['raw']), np.int8).astype(int)).max() for c in d['clips']]
        assert top[0] == 127 and abs(top[1] - 127 / 2) <= 4, f'a quarter as loud drawn at {top[1]}, the loud one at {top[0]} of 127: not one scale'
    print('selftest ok')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--clip', nargs=2, action='append', metavar=('HEIGHT_CM', 'AUDIO'), default=[])
    ap.add_argument('--selftest', action='store_true')
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.clip:
        ap.error('give at least one --clip')
    process(a.clip)


if __name__ == '__main__':
    sys.exit(main())
