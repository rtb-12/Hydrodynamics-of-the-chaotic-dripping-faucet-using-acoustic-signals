"""Acoustic drip analysis, standard library only.

Covers stages 1 to 3 of the pipeline: recording -> onsets -> validation -> intervals.
Written without numpy so it runs on any Python 3 with no install step; every function
here vectorises trivially if you later want the speed.

The synthesiser exists so the detector can be tested against known ground truth
before any real water is involved.
"""

import math, wave, array, random, statistics

# --- faucet model -----------------------------------------------------------
# D'Innocenzo & Renna 1996 Eq. 7, breakoff law (i), Table I parameters (cgs).
# Verified: period-1 -> period-2 between R = 0.605 and 0.610; paper reports 0.61.

P = dict(k=475.0, g=980.0, b=1.0, xc=0.19, alpha=0.25, rho=1.0, m0=0.01, v0=0.1)
DT = 1e-4


def simulate(R, n_drops, discard=150):
    """Return (intervals_s, drop_masses_g) for flow rate R in ml/s."""
    x, v, M, t, last = 0.0, P['v0'], P['m0'], 0.0, 0.0
    Ts, Ms = [], []
    def acc(x, v, M):
        return P['g'] - P['k']*x/M - (R + P['b'])*v/M
    while len(Ts) < n_drops + discard:
        k1x, k1v = v, acc(x, v, M)
        k2x, k2v = v + .5*DT*k1v, acc(x + .5*DT*k1x, v + .5*DT*k1v, M + .5*DT*R)
        k3x, k3v = v + .5*DT*k2v, acc(x + .5*DT*k2x, v + .5*DT*k2v, M + .5*DT*R)
        k4x, k4v = v + DT*k3v,    acc(x + DT*k3x,    v + DT*k3v,    M + DT*R)
        xn = x + DT/6*(k1x + 2*k2x + 2*k3x + k4x)
        vn = v + DT/6*(k1v + 2*k2v + 2*k3v + k4v)
        Mn = M + DT*R
        t += DT
        if xn >= P['xc']:
            f = (P['xc'] - x)/(xn - x) if xn != x else 0.0
            tc = t - DT + f*DT
            vc, Mc = v + f*(vn - v), M + f*(Mn - M)
            dM = max(1e-9, min(P['alpha']*Mc*vc, Mc - 1e-4))
            r = (3*dM/(4*math.pi*P['rho']))**(1/3)
            x, v, M, t = P['xc'] - r*dM/Mc, vc, Mc - dM, tc
            Ts.append(tc - last); Ms.append(dM); last = tc
        else:
            x, v, M = xn, vn, Mn
    return Ts[discard:], Ms[discard:]


# --- acoustics --------------------------------------------------------------

MINNAERT = 3.2866        # (1/2pi)*sqrt(3*gamma*P0/rho) for air in water at 1 atm
BUBBLE_RATIO = 0.1775    # bubble/drop diameter, from Phillips' 4.0 mm -> 0.71 mm pair


def plink_frequency(drop_mass_g):
    """Minnaert frequency of the bubble a drop of this mass would entrain."""
    d_drop = (6*drop_mass_g/math.pi)**(1/3)      # cm, water at 1 g/cm^3
    return MINNAERT/(BUBBLE_RATIO*d_drop/200)    # radius in metres


def fall_time(height_m):
    return math.sqrt(2*height_m/9.81)


# --- synthetic recording ----------------------------------------------------

def synthesise(path, intervals, masses, fs=96000, fall_h=0.086,
               tau=0.006, noise=0.004, silent=(), seed=1):
    """Write a 2-channel WAV: ch0 acoustic plinks, ch1 optical gate pulses.

    `silent` is a set of drop indices to leave acoustically silent, which is how
    the entrainment-window failure mode gets reproduced on demand.
    """
    rng = random.Random(seed)
    detach, t = [], 0.0
    for T in intervals:
        t += T; detach.append(t)
    delay = fall_time(fall_h)
    n = int((detach[-1] + delay + 0.25)*fs)
    ac = array.array('h', bytes(2*n))
    gt = array.array('h', bytes(2*n))

    for i, (td, m) in enumerate(zip(detach, masses)):
        # gate sees the drop just below the nozzle; a short bipolar blip
        g0 = int(td*fs)
        for k in range(int(0.0015*fs)):
            if g0 + k < n:
                gt[g0 + k] = int(18000*math.sin(2*math.pi*k/(0.0015*fs)))
        if i in silent:
            continue
        f = plink_frequency(m)
        a0 = int((td + delay)*fs)
        for k in range(int(tau*6*fs)):
            j = a0 + k
            if j >= n:
                break
            s = math.exp(-k/(tau*fs))*math.sin(2*math.pi*f*k/fs)
            ac[j] = max(-32767, min(32767, ac[j] + int(20000*s)))

    for i in range(n):                                    # broadband room noise
        ac[i] = max(-32767, min(32767, ac[i] + int(rng.gauss(0, noise*32767))))

    inter = array.array('h', bytes(4*n))
    inter[0::2] = ac
    inter[1::2] = gt
    with wave.open(path, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(fs)
        w.writeframes(inter.tobytes())
    return path


def read_wav(path):
    """Return (channels, fs) with each channel a list of floats in [-1, 1]."""
    with wave.open(path, 'rb') as w:
        nch, sw, fs, nfr = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        if sw != 2:
            raise ValueError('only 16-bit WAV supported')
        raw = array.array('h'); raw.frombytes(w.readframes(nfr))
    return [[raw[i]/32768.0 for i in range(c, len(raw), nch)] for c in range(nch)], fs


# --- signal conditioning ----------------------------------------------------

def _biquad(x, b0, b1, b2, a1, a2):
    y = [0.0]*len(x)
    x1 = x2 = y1 = y2 = 0.0
    for i, xi in enumerate(x):
        yi = b0*xi + b1*x1 + b2*x2 - a1*y1 - a2*y2
        y[i] = yi
        x2, x1 = x1, xi
        y2, y1 = y1, yi
    return y


def bandpass(x, fs, f_lo, f_hi, sections=2):
    """RBJ constant-peak-gain bandpass, cascaded. Kills room rumble and pump hum."""
    f0 = math.sqrt(f_lo*f_hi)
    Q = f0/(f_hi - f_lo)
    w0 = 2*math.pi*f0/fs
    alpha = math.sin(w0)/(2*Q)
    a0 = 1 + alpha
    coeffs = (alpha/a0, 0.0, -alpha/a0, -2*math.cos(w0)/a0, (1 - alpha)/a0)
    for _ in range(sections):
        x = _biquad(x, *coeffs)
    return x


def envelope(x, fs, tau=0.0015):
    """Rectify and one-pole smooth. tau must be short against the inter-drop gap."""
    a = 1 - math.exp(-1/(fs*tau))
    y, acc = [0.0]*len(x), 0.0
    for i, xi in enumerate(x):
        acc += a*(abs(xi) - acc)
        y[i] = acc
    return y


def detect(x, fs, f_lo=4000, f_hi=20000, k=6.0, min_gap=0.008, do_filter=True):
    """Onset times in seconds.

    Threshold is k times the median envelope, which is robust to a few loud
    events. The envelope smoothing adds a small constant latency; it cancels
    in the intervals, so onsets are consistent even if not absolute.
    """
    b = bandpass(x, fs, f_lo, f_hi) if do_filter else x
    e = envelope(b, fs)
    med = statistics.median(e)
    thr = max(k*med, 0.02*max(e)) if max(e) > 0 else 1.0
    out, guard, armed = [], int(min_gap*fs), True
    last = -guard
    for i in range(1, len(e)):
        if armed and e[i] >= thr and e[i-1] < thr and i - last >= guard:
            out.append(i/fs); last = i; armed = False
        elif not armed and e[i] < thr*0.5:
            armed = True
    return out


def intervals(times):
    return [times[i] - times[i-1] for i in range(1, len(times))]


# --- validation (pipeline stage 2) -----------------------------------------

def estimate_offset(acoustic, gate, max_offset=1.0, bin_w=0.004):
    """Fall-time delay between the two channels, by histogram of pairwise gaps.

    Comparing medians of the two event times looks simpler but breaks as soon as
    the acoustic channel drops events, which is precisely the case this whole
    validation exists to measure. The true delay shows up once per real pair and
    dominates the histogram; spurious gaps spread out.
    """
    if not acoustic or not gate:
        return 0.0
    bins = {}
    for t in acoustic:
        for g in gate:
            d = t - g
            if -max_offset <= d <= max_offset:
                bins.setdefault(round(d/bin_w), []).append(d)
    if not bins:
        return 0.0
    peak = max(bins, key=lambda b: len(bins[b]))
    near = bins.get(peak-1, []) + bins[peak] + bins.get(peak+1, [])
    return statistics.median(near)


def match(acoustic, gate, offset=None, tol=0.004):
    """Compare an acoustic event list against the optical ground truth."""
    if offset is None:
        offset = estimate_offset(acoustic, gate)
    shifted = [g + offset for g in gate]
    used, hits, resid = set(), 0, []
    for t in acoustic:
        best, bi = tol, None
        for i, s in enumerate(shifted):
            if i in used:
                continue
            d = abs(t - s)
            if d < best:
                best, bi = d, i
        if bi is not None:
            used.add(bi); hits += 1; resid.append(t - shifted[bi])
    return dict(
        n_gate=len(gate), n_acoustic=len(acoustic), matched=hits,
        missed=len(gate) - hits, false_pos=len(acoustic) - hits,
        detection_rate=hits/len(gate) if gate else 0.0,
        offset=offset,
        jitter_us=(statistics.pstdev(resid)*1e6 if len(resid) > 1 else 0.0),
    )


def classify(Ts, tol=2e-4, max_p=8):
    """Smallest period p with T[n] ~= T[n+p]; 0 means aperiodic."""
    for p in range(1, max_p + 1):
        if all(abs(Ts[i] - Ts[i+p]) < tol for i in range(len(Ts) - p)):
            return p
    return 0


def write_csv(path, rows, header):
    with open(path, 'w') as f:
        f.write(','.join(header) + '\n')
        for r in rows:
            f.write(','.join(f'{v:.6g}' if isinstance(v, float) else str(v) for v in r) + '\n')
    return path
