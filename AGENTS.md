# Hardware delivery project

This repository is the independent hardware project zen5x5/ebv-eye_snn_hardware.
Read README.md, docs/design.md, docs/results.md and results/README.md first.

- Current implementation: hardware/rtl/pursuit_state_depth_timing_top.sv with STATELESS_DEPTH=0/1/2/3/4.
- Formal model: seed2021 Pursuit user10, 400 events, aligned cache, T=2 repeated complete input.
- Five independently trained Pursuit variants: full_state, l1_stateless, l1_l2_stateless, l1_l2_l3_stateless, l1_l2_l3_l4_stateless.
- All pass complete routed 160 MHz setup/hold/pulse timing. Cycles: 105305/90525/78817/72101/68293. BRAM36eq:170.5/132.5/122/120/120. Five-depth latency/BRAM table:results/latency_bram.json and .csv.
- Original four depths: LUT29894/29544/29546/29287; FF17142/17130/17113/17097; DSP308; BRAM36eq170.5/132.5/122/120.
- Original four-depth results, DCPs,64 activity traces and power reports are retained. Total energy:402.257/331.583/285.527/258.212 uJ.
- The depth-4 extension has 29659 LUT,17112 FF,308 DSP,120 BRAM36eq; setup/hold/pulse slack +0.091/+0.011/+2.583 ns. Sixteen RTL samples check2255616 values without mismatch. L5 current replay uses12x8x24 bits; first four layers emit once. Depth4 has a separate default-activity power report; no depth4 sample-activity power run has been performed.
- Original four-depth RTL bytes remain in results/provenance/measured_sources.zip and four_depth_sources.zip. Original DCP/report/source hashes are unchanged; current parameterized RTL adds depth4. Verification checks old evidence against frozen sources and the new result against current sources.
- Table II continuous_xy Pursuit/Saccade use identical fixed scheduling and runtime-loaded parameter layouts at each matched depth. All192 original model configurations match;64 actual Saccade RTL replays match Pursuit layer cycles/state accesses and have zero numerical mismatch. results/task_equivalence/verification.json records the proof for depths0..3. The newer grid-spike Saccade head is outside this identical-model comparison. Continuous-head Saccade depth4 training is now complete for24 users.
- Power is a routed-functional-activity tool estimate at 25 C, typical process, nominal voltages, user10 and 16 matched inputs. No SDF; host loading and board peripherals excluded.
- Additional five-depth default-activity estimates are in results/power_vectorless/ at160MHz,25C,typical process,nominal voltages. No SAIF or manual activity overrides. Pursuit/Saccade continuous_xy share the same routed circuit at matched depth, so five reports cover both tasks. All reports retain Vivado Low confidence and internal-control activity warnings. Keep these estimates separate from the original64 sample-activity power reports. Reproduction: scripts/power_vectorless.py; independent verification: scripts/verify_vectorless_power.py.
- All four depths now have complete 24-user test quantization for both Pursuit and continuous-xy Saccade:192 models,51 verified retained results and141 new CPU evaluations on carbonite. Equal-subject fixed-point means: Pursuit0.810896/0.816982/0.825481/0.823448 deg; Saccade1.322249/1.314573/1.326037/1.326348 deg. Evidence:results/quantization/state_depth/; verify with scripts/verify_state_depth_quantization.py. FPGA/power evidence remains the existing Pursuit user10 experiment.
- Pursuit depth4 now also has complete 24-user test quantization:24 newly evaluated CPU models,25716 test inputs, equal-subject floating/fixed means0.8341314497331875/0.8327973161956191 deg. Each model passes61 independent integer probe arrays; saved predictions,means and original input hashes independently verified. All1194 user10 predictions match the previous saved predictions exactly. Evidence:results/quantization/l1_l4/. No paper,training,FPGA or power changes in this quantization batch.
- Saccade continuous_xy depth4 also has complete 24-user test quantization:24 newly evaluated CPU models,28389 test inputs, equal-subject floating/fixed means1.3500876128673553/1.342359105399257 deg, pixel errors26.59480881690979/26.44682172841815. All24 models pass61 independent integer probe arrays each; predictions,means and original model/cache hashes independently verified. Evidence:results/quantization/saccade_l1_l4/. All five depths now have complete24-user accuracy for both tasks. This batch changes no paper,trained model,FPGA or power result.
- Sparse L1/L2 design is documentation and metrics only:docs/sparse_comparison.md. Sparse code/artifacts and failed attempts are removed.
- No dependency on the neighboring software directory. hardware/quantization/ contains its own model and metric definitions.
- Preserve raw result bytes and numeric provenance. Compression is lossless with original hashes. Do not regenerate results simply to make documentation or tables.
- python scripts/verify.py --full validates evidence; scripts/report.py exports paper tables without simulation.
- New runs must use a new explicit runs/ path. FPGA board deployment is outside this paper scope.
- Explain in plain Chinese; formulas plain text; no defensive programming or defensive replies.

