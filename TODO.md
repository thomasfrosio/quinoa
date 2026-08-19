### `CTF`
- `Check`: tile grid edges - one tile per dim
- `Check`: background with bad spectrum, make sure it doesn’t fail
- `Check`: check why defocus error increases with recovery. Is it 48 vs 64 lines (decrease ZNCC)? Maybe the spectra are better resolved and the thickness region negatively affects the fit. I think it increases the frequency window and exposes the flipped oscillations making it worse, but in principle the autotunning should prevent that. Similarly, why is the recovery for 10164 so high res?


- `Fix`: On some shells, the glob expansion breaks. Investigate...
- `Fix`: noa's `ctf.fftfreq_at` with Cs=0. At the moment, Metadata clamps it to 1e-5.
- `Fix`: Defocus handedness. Polish code
- `Fix`: Baseline fit. Exposure the smoothing parameters and fix the end cutoff. If passed 4A, try double baseline subtraction? Maybe just after recovery?


- `Feature`: Thickness and initial frequency cutoff. When an initial thickness estimate is specified, allow to extend the initial cutoff of the patches to include at least one full thickness node? Same for recovery that has a thickness estimate, expand to at least include the full node.

- `Feature`: Thickness and autotuning. look at the high resolution recovery and the 90% threshold; does it work if the thickness node is close to the end? When estimating the fitting range, should we look at “nodes” and restart from there, making sure that if before the first node the peaks stop early we still account for the flipped peak in the first node?

- `Feature`: add frame support


### `Tilt alignment`

- `Feature`: For faster common-line alignment, rotational binning (or spectrum2polar) and sum power spectra to get the best line. This ignores the FOV changes, which is fine for coarse search.
