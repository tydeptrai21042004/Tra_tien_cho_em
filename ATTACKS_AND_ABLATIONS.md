# Experimental design notes

## Why the ablation is stronger

The ablation is organized by **mechanism**, so each row answers a specific methodological question.

| Variant | Removed/replaced mechanism | Question tested |
|---|---|---|
| full | nothing | Reference proposal |
| q_only | R branch + adaptive selection | Is Q alone sufficient? |
| r_only | Q branch + adaptive selection | Is R alone sufficient? |
| distortion_only | robustness-aware score | Does robustness information improve branch choice? |
| robustness_only | distortion normalization | Is the distortion term necessary for imperceptibility/selection balance? |
| raw_score | perturbation ensemble | Does local perturbation testing add robustness? |
| no_safe_margin | explicit Q/R margins | Do safety margins protect decoding? |
| hard_vote | soft reliability magnitudes | Does soft evidence improve repeated decoding? |
| raw_decoder | denoised candidates and confidence selection | Does confidence-guided decoder adaptation help? |
| fixed_mild_decoder | confidence selection | Is selecting among candidates better than always denoising? |
| repeat_1 | repeated embedding | What robustness is contributed by repetition? |

The generated paired-delta CSV is particularly useful for a paper because every ablated result is compared with `full` on the same host, watermark, and attack.

## Attack profiles

`paper` preserves the attack settings currently described in the manuscript. `representative` is deliberately compact for ablation. `extended` broadens attack *types*. `stress` additionally broadens *severity levels*.

Do not mix results from different profiles into one average without reporting which attacks were included.
