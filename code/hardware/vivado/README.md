# Vivado OOC implementation

These scripts target the RealDigital AUP-ZU3 device
`xczu3eg-sfvc784-2-e` and constrain the compute clock to 200 MHz. They run
protocol-neutral accelerator tops out of context, so no board pins, PS block
design, or physical board are required.

There are two entry points. `run_ooc.sh` retains the dense-raster correctness
and state-memory baseline. `run_sparse_ooc.sh` implements the two current
L1/L2 event-driven candidates.

## Sparse candidates

Run the sparse front end with the old time-major deep schedule:

```bash
code/hardware/vivado/run_sparse_ooc.sh \
  /absolute/output/sparse_tm sparse_tm
```

Run the selected layer-adjacent schedule, called plan A in the design report:

```bash
code/hardware/vivado/run_sparse_ooc.sh \
  /absolute/output/sparse_lm_a sparse_lm_a
```

Both tops consume sorted unit-event packets in L1, use a 72-lane LUT add-only
scatter, use 48 DSP count multipliers in L2, and share a 288-DSP core for
L3--L8. Sixteen DSPs implement H1/H2 and two implement readout, giving a
total of 354 DSP48E2 in both complete Yosys mappings, versus 360 on the
XCZU3EG. This 98.3% allocation is a placement and routing risk; only the
routed report can establish whether the 5 ns constraint is practical. The
Yosys physical-LUT bounds also straddle the device capacity for both routes,
so open-source mapping does not establish that either top fits.

`sparse_tm` retains the full-layer L3 current cache and top-level on-chip
deep-state RAM. `sparse_lm_a` loads each L4--L8 weight tile once for adjacent `t0/t1`
scans, keeps the intermediate membrane local, and uses a 4.5 KiB tile-local
L3 current buffer. It is one sequential shared core, not a cross-layer or
inter-sample wavefront.

## Dense baseline

Run the final L1/L2-stateless dense-raster baseline with:

```bash
code/hardware/vivado/run_ooc.sh /absolute/output/dense_stateless 0
```

Run the matched full-state FS-TM comparison with:

```bash
code/hardware/vivado/run_ooc.sh /absolute/output/dense_full_state 1
```

The `FULL_STATE` generic changes only the convolution state-store depth.
Dense build `0` supports SL-TM and SL-HYB. Build `1` adds the L1/L2 state
capacity required by FS-TM. Both retain the same 288-lane convolution,
16-lane dense, and 2-lane readout datapaths. They do not skip zero spikes and
remain correctness, state-memory, and temporal-reuse baselines.

## Reports and evidence boundary

Each run refuses a nonempty output directory, then writes post-synthesis and
post-route checkpoints plus hierarchical utilization, timing, DRC,
methodology, clock, critical-warning, and vectorless power reports. Its
manifest records the Vivado version, top, part, 5 ns constraint, and SHA-256
for the Tcl, XDC, and route-specific RTL sources. The batch fails on negative
routed setup or hold slack, `no_clock`, absent or partial I/O-delay coverage,
or any unconstrained internal endpoint; critical warnings are retained as a
separate review artifact. Optional third and fourth arguments override the
part and job count:

```text
run_sparse_ooc.sh OUTPUT_DIR {sparse_tm|sparse_lm_a} [PART] [JOBS]
```

The server where these files were prepared has no Vivado. The shell entry
points and Tcl source/control flow were syntax-checked, while device
elaboration, inferred-memory shape, placed-and-routed utilization, WNS, and
Fmax remain unmeasured. The 200 MHz latency values in the hardware report are
therefore conditional on nonnegative routed WNS.

The generated vectorless power report is not an acceptance criterion and can
only be labeled a preliminary estimate. A paper power number needs a validation replay captured as SAIF/VCD
or a defined board-level measurement. Accelerator-kernel latency also excludes
event arrival, ROI filtering, sorting, PS/AXI transfer, and sensor time.
