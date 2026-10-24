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

Audio with only a few loud moments can match the video's sound track in more than one place.
`--audio-offset` gives the rough audio time of the clip's start, and the alignment is then searched within a second of it.

### Drop height

`site/heights.html` sets the sound of one drop beside the same drip falling from other heights: waveform and spectrogram per recording, all of them overlaid, and how each measure of the sound moves with height.
Each recording is a phone audio file of the same faucet at the same flow rate, one per height:

```bash
uv run analysis/heights.py --clip 15 "audios /water drop 20 cm.mp4" --clip 35 "audios /water drop 40 cm .mp4"
uv run analysis/heights.py --selftest    # a synthetic recording with known drops, run by CI
```

The first set's files are named 5 cm higher than the drops really fell (20, 25 and 40 cm for 15, 20 and 35 cm).
A second set goes in its own folder with `--set NAME`, shown on the same page through its picker.
When the tap was left alone between recordings, `--same-tap` finds one drip period from the loud drops of all of them, which a slow drip with few drops per recording needs, and counts ringing against every drip that fell.
Each recording is cut to its steadiest minute (`--span`), the stretch with the most evenly spaced drops and no missed ones, so every height is judged on its cleanest drops.
Background music below 1 kHz is filtered out and the steady hiss turned down for listening and for finding faint drops.
Drops are placed one per drip period by dynamic programming, so a run of faint drops cannot lose the beat, and drops inside talking or handling noise are left out.
Every measure is read from the raw 2 to 9 kHz band, since the phone's codec invents the sound above 9 kHz and the cleaning's gain opens around each drop.

A slow-motion video of a height's drip goes beside its recording with `--video HEIGHT VIDEO X Y`, where X is the pixel column the drops fall down and Y the water line.
The clip's slowed stretch is found from its own sound, which is slowed into a low murmur, and its frames are taken as shot at 240 fps; the drip rate the video then shows must match the sound's.
Each drop is followed down the column frame by frame and timed where its fall meets the water line.
The video and the sound were recorded together on two phones and are lined up by the drops themselves: the one shift that puts the most video drops within 30 ms of a heard drop, each of which is then pinned to it.
With fewer than three drops matched, or another shift fitting as many, the video is only placed in the drip's rhythm and the page says so.

```bash
uv run analysis/heights.py --set sweep --same-tap --span 40 --clip 35 "heights/@35cm/WhatsApp Audio ....mpeg" --video 35 heights/@35cm/IMG_5388.mov 1000 500
```

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
