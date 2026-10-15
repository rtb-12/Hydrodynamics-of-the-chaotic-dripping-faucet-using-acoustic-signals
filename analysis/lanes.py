"""The time lanes the recording pages draw (waveform, loudness, spectrogram), written once for every pipeline."""

import base64
import numpy as np, scipy.signal as ss
from PIL import Image

FS = 48000                  # audio rate everything is resampled to
WAVE_RATE = 4000            # waveform min/max pairs per second
LOUD_RATE = 200             # loudness points per second
SPEC_MAX_W = 16000          # spectrogram PNG width cap, browsers struggle beyond this
SPEC_FMAX = 16000           # spectrogram top frequency, Hz
BAND = (2000, 15000)        # impact and plink energy lives here, below it is room noise


def envelope(x, lp=300):
    y = ss.sosfiltfilt(ss.butter(4, BAND, 'bandpass', fs=FS, output='sos'), x)
    return ss.sosfiltfilt(ss.butter(2, lp, fs=FS, output='sos'), np.abs(ss.hilbert(y)))


def minmax(x, peak=None):
    """Base64 of int8 (max, min) pairs at WAVE_RATE, fine enough to zoom into one plink. Square-root
    companded, the scale the viewer draws on, so a plink at 1% of the loudest keeps 13 levels instead of 1."""
    n = FS // WAVE_RATE
    r = x[:len(x) // n * n].reshape(-1, n) / (peak or np.abs(x).max() or 1)
    r = np.c_[r.max(1), r.min(1)]
    pairs = np.round(np.sign(r) * np.sqrt(np.abs(r)) * 127).astype(np.int8)
    return base64.b64encode(pairs.tobytes()).decode()


def loudness(x, top=None):
    n = FS // LOUD_RATE
    e = envelope(x, 500)
    e = e[:len(e) // n * n].reshape(-1, n).max(1)
    return np.clip(np.round(20 * np.log10(e / (top or e.max() or 1) + 1e-9)), -60, 0).astype(int).tolist()


def spectrum(x, duration):
    """Power in dB up to SPEC_FMAX per analysis slice, and the slices per second."""
    cps = min(200, SPEC_MAX_W / duration)   # long recordings get coarser columns rather than tiling
    hop = min(512, int(FS / cps))
    f, _, Z = ss.spectrogram(x, FS, nperseg=512, noverlap=512 - hop)
    return 10 * np.log10(Z[f <= SPEC_FMAX] + 1e-14), FS / hop


def scale(*xs):
    """Full height and full colour for these tracks together: the loudest sample, the loudest envelope, and
    the spectrogram's median (clear) to its top 0.1% (solid). One scale lets recordings be compared by eye."""
    Z = np.concatenate([spectrum(x, len(x) / FS)[0].ravel() for x in xs])
    return dict(peak=max(float(np.abs(x).max()) for x in xs), top=max(float(envelope(x, 500).max()) for x in xs),
                levels=np.percentile(Z, [50, 99.9]))


def spectrogram_png(x, path, duration, levels=None):
    Z, cps = spectrum(x, duration)
    lo, hi = np.percentile(Z, [50, 99.9]) if levels is None else levels
    a = (np.clip((Z - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)[::-1]
    Image.fromarray(np.dstack([np.full_like(a, 255), a]), 'LA').save(path, optimize=True)
    return cps


def lanes(cl, raw, out, duration, ref=None):
    """The page data for both tracks, with the two spectrogram PNGs written into out. ref maps each track to
    a shared scale(); without it each track is scaled to itself."""
    ref = ref or {}
    tracks, data = dict(clean=cl, raw=raw), dict(wave_rate=WAVE_RATE, loud_rate=LOUD_RATE, spec_fmax=SPEC_FMAX, wave={}, loud={})
    for k, x in tracks.items():
        r = ref.get(k, {})
        data['spec_cps'] = spectrogram_png(x, out / f'spec-{k}.png', duration, r.get('levels'))
        data['wave'][k], data['loud'][k] = minmax(x, r.get('peak')), loudness(x, r.get('top'))
    return data
