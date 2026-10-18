# Hardware delivery project

This repository is the independent hardware project zen5x5/ebv-eye_snn_hardware.
Read README.md, docs/design.md, docs/results.md and results/README.md first.

- Current implementation: hardware/rtl/pursuit_state_depth_timing_top.sv with STATELESS_DEPTH=0/1/2/3.
- Formal model: seed2021 Pursuit user10, 400 events, aligned cache, T=2 repeated complete input.
- Four independently trained variants: full_state, l1_stateless, l1_l2_stateless, l1_l2_l3_stateless.
- All pass complete routed 160 MHz setup/hold/pulse timing. Cycles: 105305/90525/78817/72101.
- LUT: 29894/29544/29546/29287; FF: 17142/17130/17113/17097; DSP:308; BRAM36eq:170.5/132.5/122/120.
- Results, DCPs, 64 activity traces and power reports are retained. Total energy:402.257/331.583/285.527/258.212 uJ.
- Power is a routed-functional-activity tool estimate at 25 C, typical process, nominal voltages, user10 and 16 matched inputs. No SDF; host loading and board peripherals excluded.
- L1-L2 24-user fixed-point result already exists:0.8254806991646753 deg vs float0.8239446895575812. Other three variants currently have user10 fixed-point metrics; their other 23 users would total69 models, not92.
- Sparse L1/L2 design is documentation and metrics only:docs/sparse_comparison.md. Sparse code/artifacts and failed attempts are removed.
- No dependency on the neighboring software directory. hardware/quantization/ contains its own model and metric definitions.
- Preserve raw result bytes and numeric provenance. Compression is lossless with original hashes. Do not regenerate results simply to make documentation or tables.
- python scripts/verify.py --full validates evidence; scripts/report.py exports paper tables without simulation.
- New runs must use a new explicit runs/ path. FPGA board deployment is outside this paper scope.
- Explain in plain Chinese; formulas plain text; no defensive programming or defensive replies.

