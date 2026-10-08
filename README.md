# Hydrodynamics of the chaotic dripping faucet using acoustic signals

B.Tech project, 2026–27.

Using the sound a drop makes as the measurement instrument for the dripping faucet's route to chaos.

**[Read the site →](https://rtb-12.github.io/Hydrodynamics-of-the-chaotic-dripping-faucet-using-acoustic-signals/)**

## The gap this fills

The literature splits cleanly into two bodies of work that have never been joined.

Every dripping-faucet chaos study detects drops **optically** and contains no acoustics.
Every drop-impact acoustics study examines **isolated single drops** and contains no nonlinear dynamics.
Nobody has used the plink as the event clock for a period-doubling cascade.

The quantitative argument is timing resolution.
Sekatchev reports a 0.025 s floor on his phototransistor and states that it caused missed drops across 1.17 million events.
A microphone at 96 kHz timestamps a drop to about 10 µs, roughly three orders of magnitude sharper.

## Repository layout

```
site/        Static site. Opens by double-clicking site/index.html. No build step.
analysis/    Drip analysis toolkit (standard library only) and the recording pipeline (runs through uv).
papers/      Source PDFs, with citations and DOIs in papers/README.md.
tools/       Repository checks that CI runs.
```

## Running things

```bash
python3 analysis/test_dripkit.py    # 19 checks on the analysis toolkit
python3 tools/validate_site.py      # structural checks on the site
```

Both run on a stock Python 3 with nothing installed.

### Recordings

`site/recording.html` plays a drip video beside its audio, locked to one clock, with the return map of the detected drops.
Each recording is prepared once by the pipeline, which needs only [uv](https://docs.astral.sh/uv/):

```bash
uv run analysis/recording.py exp/1.MOV --audio "exp/WhatsApp Audio.mp4" --name exp1
uv run analysis/recording.py slowmo.mov --audio audio.mp4 --capture-fps 120 --slow 10 179    # phone slow motion
uv run analysis/recording.py --selftest    # synthetic clips with known drops, run by CI
```

It finds the drop column and the water line, levels and crops the video (tone-mapping iPhone HDR), aligns the separately recorded audio by cross-correlation, cleans it, and times every drop.
Drops are counted from the video alone: a pixel band on the drop's path is compared with its own background colour, after the lamp flicker each pixel follows has been fitted and removed.
The count is then set against the sound onsets in the audio, which is the check on the microphone as a drop counter.
Everything it found is drawn on `site/recordings/<name>/check.jpg`; `--impact`, `--tilt` and `--gate` override it when a scene fools it, and `--detect-only` skips the slow video encode while placing them.

A phone slow-motion clip is stored at 30 fps with both ends at normal speed.
`--capture-fps` is the rate it was shot at and `--slow` the clip seconds between which it plays slowed; only that stretch is used, timed in real seconds.

## The site

Twelve pages, about 300 KB before any recordings, zero external requests, no libraries.

| Page | What it holds |
| --- | --- |
| Overview | The thesis, the gap, the failure mode that could sink it |
| Explainers | Three animated mechanisms: the plink, why the rhythm splits, why 4.669 matters |
| Interactive lab | Live faucet simulator with audio, bifurcation diagram, plink synthesiser, nozzle designer |
| Recordings | Real drip videos beside their audio on one clock, zoomable to a single impact, with the return map |
| Choosing sensors | Which microphone and hydrophone to buy, and why |
| Build guide | Shopping list, overflow cross-sections, the optical gate circuit, day-one checks |
| Data pipeline | Sensor roles, signal chain, six analysis stages |
| Experimental setups | Every rig in the literature drawn in one visual language |
| Proposed rig | The design, bill of materials, derived hydrophone specification |
| Theory | Four model families with equations, dimensionless groups |
| Papers, Concepts | Reference indexes |

The faucet simulator integrates D'Innocenzo and Renna's variable-mass oscillator live.
It reproduces their published result: period-1 to period-2 between R = 0.605 and 0.610 against their reported 0.61, with intervals spanning 0.026 to 0.076 s against their 0.025 to 0.078 s.

## Design decisions worth knowing

**Both tanks hold a constant level by overflow.**
Flow rate is the bifurcation parameter, so a draining reservoir smears the measurement across the diagram instead of sampling a point on it.
The impact tank needs one too: at a 12 inch footprint and 30 ml/min the level climbs about 2 cm per hour, which is a 20-odd percent drift in impact velocity against an 86 mm fall.

**All sensors share one converter clock.**
A light gate's output is just a voltage, so it goes on the fourth channel of the same audio interface as the microphone and hydrophone.
No drift, no timestamp alignment, no sync protocol.

**The optical gate is a calibration instrument, not the measurement.**
Bubble entrainment fails outside roughly 1 to 5 mm drops, so some impacts are acoustically silent.
A missed drop merges two intervals and fabricates a point in the return map.
Proving where the acoustic clock is complete is the first result.
