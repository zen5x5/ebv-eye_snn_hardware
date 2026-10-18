# Implement the frozen, already verified synthesis checkpoint. No board bitstream.
if {$argc != 2} { error "usage: implement.tcl SYNTHESIS_DIRECTORY OUTPUT_DIRECTORY" }
set input_directory [file normalize [lindex $argv 0]]
set output_directory [file normalize [lindex $argv 1]]
file mkdir $output_directory
set reports [file join $output_directory reports]
set snapshot [file join $output_directory source_snapshot]
file mkdir $reports
file copy [file join $input_directory source_snapshot] $snapshot
file copy [info script] $snapshot
file copy [file join $input_directory source_sha256.txt] [file join $output_directory synthesis_source_sha256.txt]
set checkpoint [file join $input_directory post_synth.dcp]
set manifest [open [file join $output_directory input_checkpoint_sha256.txt] w]
puts $manifest [exec sha256sum -- $checkpoint]
close $manifest
set_param general.maxThreads 4
open_checkpoint $checkpoint
# Constraints, including the 6.250 ns clock, are already stored in this DCP.
opt_design
place_design
phys_opt_design
route_design
write_checkpoint [file join $output_directory post_route.dcp]
write_xdc [file join $output_directory implemented_constraints.xdc]
report_route_status -file [file join $reports post_route_status.rpt]
report_utilization -file [file join $reports post_route_utilization.rpt]
report_utilization -hierarchical -file [file join $reports post_route_utilization_hierarchical.rpt]
report_timing_summary -delay_type min_max -max_paths 30 -report_unconstrained -file [file join $reports post_route_timing.rpt]
report_timing -delay_type min -max_paths 20 -input_pins -file [file join $reports post_route_hold_paths.rpt]
report_timing -delay_type max -max_paths 20 -input_pins -file [file join $reports post_route_setup_paths.rpt]
report_clock_utilization -file [file join $reports post_route_clock_utilization.rpt]
report_drc -file [file join $reports post_route_drc.rpt]
report_methodology -file [file join $reports post_route_methodology.rpt]
check_timing -verbose -file [file join $reports post_route_check_timing.rpt]
report_io -file [file join $reports post_route_io.rpt]
# Functional gate-level simulation provides activity names identical to this netlist.
write_verilog -mode funcsim [file join $output_directory post_route_funcsim.v]
set setup_slack [get_property SLACK [get_timing_paths -setup -max_paths 1]]
set hold_slack [get_property SLACK [get_timing_paths -hold -max_paths 1]]
set summary [open [file join $output_directory run_summary.txt] w]
puts $summary "stage=post_route"
puts $summary "vivado_version=[version -short]"
puts $summary "top=pursuit_state_depth_timing_top"
puts $summary "part=[get_property PART [current_design]]"
puts $summary "target_clock_period_ns=[get_property PERIOD [get_clocks compute_clock]]"
puts $summary "worst_setup_slack_ns=$setup_slack"
puts $summary "worst_hold_slack_ns=$hold_slack"
puts $summary "critical_warning_count=[get_msg_config -severity {CRITICAL WARNING} -count]"
close $summary
if {$setup_slack < 0 || $hold_slack < 0} { error "Physical timing failed: setup=$setup_slack hold=$hold_slack ns" }
exit
