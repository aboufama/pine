# Research pass: cheap improvements for the Pine motion model (2025-2026 literature)

Starting point: `runs/pi_pass/` found that `chunk16` works best (16-frame residual chunks, execute 8, then re-plan).
Free-running, it scores 14.7 deg mean joint error over 2 s, against 17.5 deg for holding the last pose. That pass
already covered residual targets, pi0 chunking and FAST/DCT targets, so they are not repeated here. Effort
estimates are for this codebase (NumPy reference model, 2-core CPU): S = under a day, M = a few days, L = a week
or more.

## Ranked ideas

1. **Absolute targets for the channels the suit does not measure (any-subset sensor conditioning).**
   *Towards Real-World Wearable Motion Reconstruction* (WHIP, 2026, https://arxiv.org/abs/2607.09780); background:
   *MaskedMimic* (2024, https://arxiv.org/abs/2409.14393).
   - What it is: one generative model reconstructs the full body from any subset of wearable sensors and stays
     robust when some are missing. MaskedMimic does the same for any masked subset of motion targets.
   - Why it matters for Pine: our check below shows that pi_pass's residual targets make the suit fill-in
     useless. With EXO-1 masks, waist and arm error 200 ms ahead is 21.2 deg, while always predicting the mean
     pose scores 21.3 deg. A residual needs the current value of the joint, and the suit never has it. The fix
     is a residual target for observed channels and an absolute target for masked ones, chosen per channel by
     the mask.
   - Effort: S (about 10 lines in the objective). This is the first thing to test.
2. **Noise-augmented history (tested below).**
   *Diffusion Models Are Real-Time Game Engines* (GameNGen, ICLR 2025, https://arxiv.org/abs/2408.14837);
   *History-Guided Video Diffusion* (ICML 2025, https://arxiv.org/abs/2502.06764).
   - What it is: train on context frames that have been corrupted on purpose, so the model learns to recover
     from its own drift. GameNGen also gives the model the noise level as an input.
   - Why it matters for Pine: a 20 Hz suit-to-robot stream re-plans every frame. That is the mode where our
     model drifted, and this fix costs no extra compute.
   - Effort: S. The test cut every-frame re-planning error by 1.7 deg. Next step: add the noise level as an
     input, which should remove the small one-step cost.
3. **Scale wearer diversity, not hours.**
   *Data Scaling Laws in Imitation Learning for Robotic Manipulation* (ICLR 2025,
   https://arxiv.org/abs/2410.18647); *Scaling Laws of Motion Forecasting and Planning* (Waymo, 2025,
   https://arxiv.org/abs/2506.08228).
   - What they found: success follows a power law in the number of environments and objects, and more
     demonstrations per environment barely help. Waymo finds that optimal model size grows about as C^0.63 and
     data as C^0.44. Waymo also finds that logs of other drivers transfer zero-shot to the self-driving car, at
     roughly 2-3 demonstrated miles per 10 observed miles.
   - Why it matters for Pine: held-out people score 3x worse than held-out takes, and model size saturates at
     12 h. Both point to data that is limited by diversity. Re-running the S data sweep with the number of CMU
     subjects (102 in train) as the axis would tell Pine whether to recruit more wearers or record longer.
     Waymo's "observed agents" result is the closest published analogue to the exo-to-humanoid bet.
   - Effort: S (about 25 min of CPU).
4. **Retargeting artifact metrics, then fixes.**
   *Retargeting Matters: General Motion Retargeting for Humanoid Motion Tracking* (GMR, 2025,
   https://arxiv.org/abs/2510.02252); *PHUMA: Physically Reliable Humanoid Locomotion Dataset* (2025,
   https://arxiv.org/abs/2510.26236); *Make Tracking Easy: Neural Motion Retargeting* (2026,
   https://arxiv.org/abs/2603.22201).
   - What they found: foot sliding, ground or self penetration and joint jumps in retargeted references lower
     tracking success. On G1, NMR's references gave 31/33 short clips tracked vs 25/33 for GMR's IK references.
   - Why it matters for Pine: the data engine's product is robot-space data. Pine's QA checks keypoint error,
     but not foot skate, penetration or joint-velocity spikes, so a buyer would find them first.
   - Effort: S for the metrics, using the existing FK and the stored `err`. M for a contact-aware IK pass.
5. **Scheduled sampling from the model's own rollouts (tested below).**
   *MotionStreamer* (ICCV 2025, https://arxiv.org/abs/2503.15451); *DART* (ICLR 2025,
   https://arxiv.org/abs/2410.05260); *Self Forcing* (NeurIPS 2025, https://arxiv.org/abs/2506.08009).
   - What it is: part of the history is replaced by the model's own predictions during training (two forward
     passes, with the gradient taken only on the second). Self Forcing goes further and trains on full
     rollouts with a sequence-level loss.
   - Why it matters for Pine: it is the principled exposure-bias fix. In DART, removing it raised segment FID
     from 3.8 to 8.1.
   - Effort: S, about +7% CPU. In the test it was about a third as effective as noise for every-frame
     re-planning, but it had no one-step cost.
6. **Probabilistic outputs for under-determined joints, plus foot-pressure insoles.**
   *Probabilistic Inertial Poser (ProbIP)* (ICCV 2025, https://openaccess.thecvf.com/content/ICCV2025/html/Kim_Probabilistic_Inertial_Poser_ProbIP_Uncertainty-aware_Human_Motion_Modeling_from_Sparse_ICCV_2025_paper.html);
   *UMotion* (CVPR 2025, https://arxiv.org/abs/2505.09393); *Ground Reaction Inertial Poser (GRIP)* (2026,
   https://arxiv.org/abs/2603.16233).
   - What they do: predict a distribution, not a point, for poses that sparse sensors cannot pin down. GRIP adds
     insole pressure, with 2 wrist IMUs and 2 insoles, and reports 182 mm global MPJPE vs 249 mm for 6-IMU PIP.
   - Why it matters for Pine: arm angles cannot really be read off a leg suit (about 20 deg error vs 21 deg for
     the mean pose). A variance head would let downstream training weight or drop guessed channels. Insoles
     are a cheap sensor that could join EXO-1.
   - Effort: M.
7. **Frequency-domain loss.**
   *FreDF: Learning to Forecast in the Frequency Domain* (ICLR 2025, https://arxiv.org/abs/2402.02399).
   - What it is: add an L1 loss on the FFT of the forecast along the horizon to the time-domain loss, weighted
     by alpha.
   - Why it matters for Pine: per-step MSE on autocorrelated futures is a biased objective, and 16-frame chunks
     are exactly that case. In NumPy it costs one matmul with a DFT matrix.
   - Effort: S. Untested; expect a small effect, since dct16 already matched chunk16.
8. **B-spline (or wavelet) chunk targets.**
   *BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences* (NeurIPS 2025,
   https://arxiv.org/abs/2506.06072); *OmniSAT* (2025, https://arxiv.org/abs/2510.09667); *Wavelet Policy*
   (2025, https://arxiv.org/abs/2504.04991).
   - What it is: fit each channel's chunk with a few B-spline control points and pin the first one to the last
     executed frame, so chunks join continuously by construction. Wavelet Policy splits action chunks into
     scales instead.
   - Why it matters for Pine: smooth chunk boundaries for robot playback. On our metric, dct16 already showed
     that compressing the chunk neither helps nor hurts.
   - Effort: S, but expect little gain here.
9. **Learned residual-VQ or ordered action tokenizers.**
   *FASTer* (2025, https://arxiv.org/abs/2512.04952); *Ordered Action Tokens (OAT)* (2026,
   https://arxiv.org/abs/2607.21670); *NAC: Neural Action Codec* (2026, https://arxiv.org/abs/2606.21372).
   - What they do: learned coarse-to-fine discrete tokens. FASTer reports 97.9% vs pi0-FAST's 94.2% on LIBERO,
     and OAT shows that any prefix of its tokens decodes to a full chunk.
   - Why it matters for Pine: only if the motion model turns into discrete tokens for co-training with a VLA.
     It is not needed for the current regression model.
   - Effort: L (GPU).

Context for the exo-as-data-source bet (not model changes):

- *HumanoidExo* (2025, https://arxiv.org/abs/2510.03022): 5 teleop + 195 exoskeleton demos matched 200 teleop
  demos (80%) on a G1 laundry task.
- *AirExo-2* (CoRL 2025, https://arxiv.org/abs/2503.03081): a policy trained on exoskeleton data alone reached
  85% average success vs 93.75% with teleop.
- *SONIC* (2025, https://arxiv.org/abs/2511.07820, 700 h of mocap, up to 42M params) and *Humanoid-GPT* (2026,
  https://arxiv.org/abs/2606.03985, about 2B frames): humanoid tracking keeps improving with data and model
  size, and transformers scale where MLPs plateau.

## Tests run here (ideas 2 and 5)

**Setup.** Both ideas are added to pi_pass `chunk16`, and everything else is identical: size S, 900 steps,
seed 0, the same batches and suit masks, and the same checkpoint rule.
- `research/history_corruption.py --check` confirms that with no corruption the loop reproduces
  `pi_pass.train` bit for bit.
- Scoring calls `pi_pass.score` unchanged: free-running joint-angle RMSE on the same 200 held-out test windows
  (184 CMU held-out people, 16 Bandai held-out takes) vs hold-last-pose, plus 200 ms error from real frames.
- A chunk16 control at seed 1 measures run-to-run noise.
- Costs per run: about 335 s wall when the CPU was free; CPU time 518 s (control), 516 s (noisyhist) and
  552 s (selfhist). The selfhist wall time is longer because the watchdog paused it while other sessions' jobs
  ran.

**What each variant does.**
- `noisyhist`: on half the training windows, a random suffix of the context (4-32 frames) gets smooth,
  integrated random-walk drift (up to 1 std) on the observed channels. Targets are the clean future, written as
  offsets from the corrupted frame.
- `selfhist`: from step 150, a share of windows (ramped to 25%) gets its last n context frames (n = 1-16)
  replaced by the model's own open-loop chunk, predicted without gradient.

Mean joint-angle RMSE in degrees on 200 windows (lower is better; hold-last-pose = 17.50 mean, 19.87 at 2 s):

| model | re-plan every frame, 2 s mean | at 2 s | execute 8, 2 s mean | at 2 s | 200 ms from real frames |
|---|---|---|---|---|---|
| chunk16 (pi_pass, seed 0) | 17.14 | 20.10 | 14.69 | 16.60 | 8.79 |
| chunk16 control (seed 1) | 17.31 | 20.61 | 14.75 | 16.88 | 8.77 |
| + noisyhist (seed 0) | **15.57** | **17.84** | 14.61 | 16.49 | 8.93 |
| + selfhist (seed 0) | 16.79 | 19.48 | 14.65 | 16.53 | 8.80 |

**Paired differences vs the seed-1 control** (same windows, 95% bootstrap interval over windows):

| | re-plan every frame | execute 8 | 200 ms from real |
|---|---|---|---|
| noisyhist | -1.74 (-2.26 to -1.28) | -0.14 (-0.26 to -0.04) | +0.16 (+0.05 to +0.28) |
| selfhist | -0.51 (-0.95 to -0.13) | -0.11 (-0.20 to -0.02) | +0.03 (-0.05 to +0.13) |

The two chunk16 seeds differ by 0.16 (every frame), 0.06 (execute 8, mean), 0.28 (execute 8, at 2 s) and 0.02
(200 ms). These intervals cover window sampling only, not training-seed noise.

**Verdict.**
- Noise-augmented history is a real, free gain when the model re-plans every frame. The 2 s error drops by
  1.6-1.7 deg against either seed, about 10x the seed spread, and 20.6 at 2 s becomes 17.8, so it now beats
  holding the pose at every horizon. The cost is about +0.15 deg (2%) on the 200 ms one-step forecast, and the
  suit-mode leg forecast at 200 ms rises from 8.58 to 8.87.
- In the best deployment mode (chunk16, execute 8), neither idea gains beyond seed noise: 0.1 deg out of 14.7.
  Chunked execution with re-planning had already removed most of the drift that "past hold by 1 s" described.
- Scheduled sampling is cheaper on one-step accuracy but about a third as effective. On this model and budget
  it is not worth its extra code over noise.
- Recommendation: adopt noise augmentation for any per-frame streaming use. Add a noise-level input, set to 0
  at inference, to try to recover the 200 ms cost. Do not expect it to move the execute-8 headline number.

**Side finding (light check, no training: `research/exo_check.py`, `exo_check.json`).** This one matters more
than either test. pi_pass evaluates with all channels visible. With EXO-1 masks, the residual chunk models
predict waist and arm joints 200 ms ahead at:
- 9.7 deg if handed the true current hidden angles, which is just holding them (9.9 deg), and which a real suit
  never has;
- 21.2 deg without those angles, the same as always predicting the mean pose (21.3 deg).

The older direct-target L model reaches 19.9 deg on the same check, and the S sweep run 20.4 deg vs 22.8 deg
for the mean pose. Moving the whole model to residual targets would therefore silently remove the suit
fill-in. Use mask-dependent targets (idea 1) before adopting chunk16 for suit data.

Files are in `runs/research/`:
- `noisyhist.json`, `selfhist.json`, `none_s1.json`: full scores with per-window errors.
- `compare.json`: paired bootstrap.
- `exo_check.json`.
- `weights_*.npz` and logs.

`selfhist_aborted.log` is a first selfhist start that loaded a half-edited script. It was stopped at step 150
and re-run, and none of its numbers are used. Code is in `research/` (`history_corruption.py`, `compare.py`,
`exo_check.py`, `watchdog.py`, `run_one.sh`).
