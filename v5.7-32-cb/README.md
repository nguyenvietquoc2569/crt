# v5.7 — All 31 Acoustic Codebooks (CB1-CB31)

## Problem

v5.6 predicts only CB1-CB18 (18 of 31 acoustic codebooks). CB19-CB31 encode
fine acoustic texture — breathiness, micro-prosody, lip noise, room tone.
Without them, the decoded audio sounds "clean but dead" / robotic.

## Solution

Change `--n_learn 18` to `--n_learn 31`. The model now predicts ALL 31 acoustic
codebooks. The extra 13 output heads add ~34M parameters (heads only).

## Changes from v5.6

| Parameter | v5.6 | v5.7 |
|-----------|------|------|
| n_learn | 18 | **31** |
| batch_size | 128 | **64** (more VRAM needed) |
| Output heads | 18 | **31** |
| Total params | ~706M | ~740M |

Everything else identical: d_model=1280, n_layers=16, curriculum corruption,
SECS loss, merged dataset (3242 speakers), 120k steps.

## Expected Impact

- CB19-CB31 are lower-information codebooks (fine acoustic detail)
- They should be easier to predict than CB1-CB6 (higher accuracy)
- Decoded audio should have richer texture and less robotic quality
