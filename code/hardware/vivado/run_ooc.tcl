# Usage:
# vivado -mode batch -source run_ooc.tcl -tclargs OUTPUT_DIR FULL_STATE PART JOBS
# FULL_STATE is 0 for the final L1/L2-stateless deployment or 1 for FS-TM.

if {$argc < 1 || $argc > 4} {
    error "usage: run_ooc.tcl OUTPUT_DIR ?FULL_STATE=0? ?PART=xczu3eg-sfvc784-2-e? ?JOBS=8?"
}
set output_directory [file normalize [lindex $argv 0]]
set full_state 0
set target_part "xczu3eg-sfvc784-2-e"
set jobs 8
if {$argc >= 2} {
    set full_state [lindex $argv 1]
}
if {$argc >= 3} {
    set target_part [lindex $argv 2]
}
if {$argc >= 4} {
    set jobs [lindex $argv 3]
}
if {$full_state != 0 && $full_state != 1} {
    error "FULL_STATE must be 0 or 1"
}
if {$jobs < 1} {
    error "JOBS must be positive"
}
file mkdir $output_directory
set report_directory [file join $output_directory reports]
file mkdir $report_directory

source [file join [file dirname [file normalize [info script]]] sources.tcl]
set_param general.maxThreads $jobs

synth_design \
    -top pursuit_snn_accelerator_top \
    -part $target_part \
    -mode out_of_context \
    -flatten_hierarchy rebuilt \
    -generic FULL_STATE_CAPABLE=$full_state
write_checkpoint -force [file join $output_directory post_synth.dcp]
report_utilization -hierarchical -file \
    [file join $report_directory post_synth_utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 -report_unconstrained \
    -file [file join $report_directory post_synth_timing.rpt]
report_methodology -file [file join $report_directory post_synth_methodology.rpt]

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $output_directory post_route.dcp]
report_utilization -hierarchical -file \
    [file join $report_directory post_route_utilization.rpt]
report_timing_summary -delay_type min_max -max_paths 50 -report_unconstrained \
    -file [file join $report_directory post_route_timing.rpt]
report_clock_utilization -file \
    [file join $report_directory post_route_clock_utilization.rpt]
report_drc -file [file join $report_directory post_route_drc.rpt]
report_methodology -file [file join $report_directory post_route_methodology.rpt]
report_power -file [file join $report_directory post_route_power_vectorless.rpt]

set timing_paths [get_timing_paths -delay_type max -max_paths 1]
set worst_slack [get_property SLACK $timing_paths]
set summary_file [file join $output_directory run_summary.txt]
set summary [open $summary_file w]
puts $summary "part=$target_part"
puts $summary "full_state_capable=$full_state"
puts $summary "target_clock_period_ns=5.000"
puts $summary "worst_setup_slack_ns=$worst_slack"
puts $summary "timing_met=[expr {$worst_slack >= 0.0}]"
puts $summary "power_scope=vectorless_estimate_only"
close $summary

if {$worst_slack < 0.0} {
    error "routed design misses the 200 MHz OOC constraint: WNS=$worst_slack ns"
}
