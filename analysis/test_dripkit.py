"""Self-check for dripkit. Run: python3 test_dripkit.py

Synthesises a recording with known drop times, runs the real detector over it,
and asserts the intervals come back. Also deliberately silences drops to prove
the validation stage catches the entrainment-window failure mode, which is the
whole point of Phase 1.
"""

import os, math, tempfile, statistics
import dripkit as dk

FS = 96000
fails = []


def check(name, ok, detail=''):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        fails.append(name)


tmp = tempfile.mkdtemp()

# --- 1. the model still reproduces the paper -------------------------------
print('\nmodel')
T1, _ = dk.simulate(0.50, 40)
T2, _ = dk.simulate(0.70, 40)
T4, _ = dk.simulate(0.82, 40)
Tc, _ = dk.simulate(1.05, 40)
check('R=0.50 is period-1', dk.classify(T1) == 1, f'got {dk.classify(T1)}')
check('R=0.70 is period-2', dk.classify(T2) == 2, f'got {dk.classify(T2)}')
check('R=0.82 is period-4', dk.classify(T4) == 4, f'got {dk.classify(T4)}')
check('R=1.05 is aperiodic', dk.classify(Tc) == 0, f'got {dk.classify(Tc)}')
check('intervals in the published band',
      0.02 < min(T1 + T2 + T4 + Tc) and max(T1 + T2 + T4 + Tc) < 0.09,
      f'{min(T1+T2+T4+Tc):.4f}..{max(T1+T2+T4+Tc):.4f} s vs paper 0.025..0.078')

# --- 2. acoustics mapping ---------------------------------------------------
print('\nacoustics')
m4 = math.pi/6*0.4**3          # a 4.0 mm water sphere, grams
f4 = dk.plink_frequency(m4)
check('4 mm drop lands near Phillips 8.66 kHz', 8000 < f4 < 10000, f'{f4/1000:.2f} kHz')
check('smaller drop rings higher', dk.plink_frequency(m4/8) > f4)
check('fall time from 86 mm', abs(dk.fall_time(0.086) - 0.1324) < 5e-4,
      f'{dk.fall_time(0.086)*1000:.1f} ms')

# --- 3. detector recovers known intervals -----------------------------------
print('\ndetection on a clean period-2 recording')
Ts, Ms = dk.simulate(0.70, 60)
wav = dk.synthesise(os.path.join(tmp, 'clean.wav'), Ts, Ms, fs=FS)
ch, fs = dk.read_wav(wav)
acoustic = dk.detect(ch[0], fs)
gate = dk.detect(ch[1], fs, f_lo=200, f_hi=3000, k=5.0)

check('every drop heard', len(acoustic) == len(Ts), f'{len(acoustic)} of {len(Ts)}')
check('every drop gated', len(gate) == len(Ts), f'{len(gate)} of {len(Ts)}')

rec = dk.intervals(acoustic)
true = Ts[1:]
if len(rec) == len(true):
    err = [abs(a - b) for a, b in zip(rec, true)]
    check('intervals recovered within 0.5 ms', max(err) < 5e-4,
          f'max error {max(err)*1e6:.0f} us, median {statistics.median(err)*1e6:.0f} us')
    check('recovered sequence still reads as period-2', dk.classify(rec, tol=6e-4) == 2,
          f'got {dk.classify(rec, tol=6e-4)}')
else:
    check('intervals recovered', False, f'{len(rec)} vs {len(true)}')

stats = dk.match(acoustic, gate)
check('validation reports 100% detection', stats['detection_rate'] > 0.999,
      f"{stats['detection_rate']*100:.1f}%")
check('validation finds no false positives', stats['false_pos'] == 0)
check('estimated offset matches the fall time',
      abs(stats['offset'] - dk.fall_time(0.086)) < 3e-3,
      f"{stats['offset']*1000:.1f} ms vs {dk.fall_time(0.086)*1000:.1f} ms")

# --- 4. the failure mode Phase 1 exists to catch ---------------------------
print('\ndetection when some drops entrain no bubble')
silent = {5, 17, 18, 31}
wav2 = dk.synthesise(os.path.join(tmp, 'gappy.wav'), Ts, Ms, fs=FS, silent=silent)
ch2, _ = dk.read_wav(wav2)
ac2 = dk.detect(ch2[0], fs)
gt2 = dk.detect(ch2[1], fs, f_lo=200, f_hi=3000, k=5.0)
s2 = dk.match(ac2, gt2)

check('silent drops are missing from the acoustic channel',
      len(ac2) == len(Ts) - len(silent), f'{len(ac2)}, expected {len(Ts)-len(silent)}')
check('validation COUNTS the missed drops', s2['missed'] == len(silent),
      f"reported {s2['missed']}, planted {len(silent)}")
check('validation reports the degraded rate',
      abs(s2['detection_rate'] - (len(Ts)-len(silent))/len(Ts)) < 0.02,
      f"{s2['detection_rate']*100:.1f}%")

bad = dk.intervals(ac2)
check('and the corrupted intervals are visibly wrong',
      max(bad) > 1.6*max(dk.intervals(acoustic)),
      f'merged interval {max(bad)*1000:.1f} ms vs clean max {max(dk.intervals(acoustic))*1000:.1f} ms')

print('\n' + ('ALL CHECKS PASS' if not fails else f'{len(fails)} FAILED: {fails}'))
raise SystemExit(1 if fails else 0)
