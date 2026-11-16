# Measured results

All numbers come from `com.classcounter.tools.ParamTuner` on **SCUT-HEAD Part B** (train split).
The raw console output of both runs is next to this file (`tuner-run1-threshold.log`, `tuner-run2-shape-hough.log`).

## Setup

| Item | Value |
|---|---|
| Tuning images | 120 random images from the Part B train list (seed 1) |
| Held-out images | 60 other images the search never saw |
| Image size | downscaled to at most 1280 px wide; ground-truth boxes scaled the same way |
| Hit rule | a detection is correct if its IoU with an unused ground-truth head is at least 0.5 |
| Ground truth used | heads of at least 400 px² (`--min-gt-area 400`); smaller heads are ignored |
| Objective | `mix` = F1 x (1 - mean count error) |

## Results (held-out images, the numbers to quote)

| Parameter set | F1 | Precision | Recall | Mean abs. count error |
|---|---|---|---|---|
| Hand-picked starting values | 0.000 | 0.000 | 0.000 | 73.5 % |
| Run 1 best (threshold method) | 0.070 | 0.078 | 0.064 | 56.8 % |
| **Run 2 best (current defaults)** | **0.074** | **0.083** | **0.067** | **51.5 %** |

On the 120 tuning images the run 2 best scored F1 0.108, precision 0.135, recall 0.090, count error 48.9 %.
The drop to the held-out numbers is the normal amount of overfitting to the tuning set, so use the held-out line.

## Settings found (now the defaults in `HeadDetector.Params`)

Method `THRESHOLD`, global threshold 35 on the grayscale image (dark pixels = foreground), no Gaussian blur,
7x7 elliptical opening and closing, area 223 to 5755 px², aspect ratio 0.42 to 1.84, IoU overlap limit 0.22,
eccentricity at most 9.7.

## What the numbers mean

* The pipeline finds roughly **7 % of the labelled heads**, and about 8 % of its boxes are on a real head.
  It counts dark, roughly round blobs of plausible size. It does not recognise heads.
* It works better on sparse scenes than on crowded ones. In the eight debug images of run 2 the per-image counts
  were 8/10, 4/10, 16/9, 2/5, 5/6, 5/4, 6/13 and 4/15 (detected/true).
* Hough circles, solidity and eccentricity filters were available to the search in run 2 and did not beat the plain
  dark-pixel threshold (F1 0.074 against 0.070, within noise).
* The scale-space `BLOB` method was added afterwards. **Its tuned score has not been measured yet**, so no
  claim is made about it here. Run `ParamTuner` with `--method blob` and add the result to this table.

## Limits of these results

* Only SCUT-HEAD Part B was used; other classrooms may behave differently.
* One random split and one seed, so differences of a few thousandths of F1 are noise.
* The scoring ignores heads under 400 px², so the numbers describe the larger, easier heads.
* The offline parameter search fits settings to labelled data. The detector itself is a fixed pipeline of
  image-processing operations.
