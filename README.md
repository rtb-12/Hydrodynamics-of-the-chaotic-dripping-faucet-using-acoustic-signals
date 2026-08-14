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
analysis/    Drip analysis toolkit. Standard library only, no install needed.
papers/      Source PDFs, with citations and DOIs in papers/README.md.
tools/       Repository checks that CI runs.
```

## Running things

```bash
python3 analysis/test_dripkit.py    # 19 checks on the analysis toolkit
python3 tools/validate_site.py      # structural checks on the site
```

Both run on a stock Python 3 with nothing installed.

## The site

Ten pages, about 290 KB, zero external requests, no libraries.

| Page | What it holds |
| --- | --- |
| Overview | The thesis, the gap, the failure mode that could sink it |
| Explainers | Three animated mechanisms: the plink, why the rhythm splits, why 4.669 matters |
| Interactive lab | Live faucet simulator with audio, bifurcation diagram, plink synthesiser, nozzle designer |
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
