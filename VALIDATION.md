# Export validation

Date: 2026-10-13

The standalone repository was validated from its staging root, using only the code and evidence
that are included in the repository.

| Check | Result |
|---|---|
| Python syntax, `python -m compileall -q code scripts` | passed |
| Portable evidence hashes, `python scripts/verify_evidence.py` | 116 artifacts checked, passed |
| Final whole-network Icarus RTL replay | 159,408 values, 0 mismatches, 0 range errors, 67,929 cycles |
| Verilator 5.052 `--Wall` lint of the final top | passed |
| Input accepted by the final top | 400 events |

The replay used Icarus Verilog 13.0 and the portable copies under
`evidence/golden_vectors_seed2022/` and `evidence/hardware_image_stateless_seed2022/`.
Yosys was deliberately skipped during this export smoke test because the archived mapping result
is already provided in `evidence/rtl_three_route/final_layer_adjacent/`; removing
`--skip-mapping` from the command in `README.md` reruns it.

