# Matched full-top synthesis for separately trained state-depth variants.
if {$argc != 2} { error "usage: synthesize.tcl OUTPUT_DIRECTORY STATELESS_DEPTH" }
set output_directory [file normalize [lindex $argv 0]]
set depth [lindex $argv 1]
set script_path [file normalize [info script]]
set scripts [file dirname $script_path]
set rtl [file normalize [file join $scripts .. rtl]]
set xdc [file join $scripts constraints.xdc]
file mkdir $output_directory
set snapshot [file join $output_directory source_snapshot]
set reports [file join $output_directory reports]
file mkdir $snapshot $reports
set names {
    programmable_padded_line_buffer_3x3.sv programmable_maxpool2x2_stream.sv
    pipelined_count_mac_array.sv lif_multispike.sv paired_linear_lif_layer_engine.sv readout_engine.sv
    dense_state_depth_conv_engine.sv
    pursuit_state_depth_feature_store.sv
    pursuit_state_depth_accelerator_top.sv
    pursuit_state_depth_timing_top.sv
}
set manifest [open [file join $output_directory source_sha256.txt] w]
set frozen {}
foreach name $names {
    set path [file join $rtl $name]
    file copy $path $snapshot
    puts $manifest [exec sha256sum -- $path]
    lappend frozen [file join $snapshot [file tail $name]]
}
foreach path [list $script_path $xdc] {
    file copy $path $snapshot
    puts $manifest [exec sha256sum -- $path]
}
close $manifest
set_param general.maxThreads 4
create_project -in_memory -part xczu3eg-sfvc784-2-e
add_files $frozen
add_files -fileset constrs_1 [file join $snapshot constraints.xdc]
set_property top pursuit_state_depth_timing_top [current_fileset]
synth_design -top pursuit_state_depth_timing_top -part xczu3eg-sfvc784-2-e -generic STATELESS_DEPTH=$depth -flatten_hierarchy rebuilt
write_checkpoint [file join $output_directory post_synth.dcp]
report_utilization -file [file join $reports post_synth_utilization.rpt]
report_utilization -hierarchical -file [file join $reports post_synth_utilization_hierarchical.rpt]
report_timing_summary -delay_type min_max -report_unconstrained -file [file join $reports post_synth_timing.rpt]
report_drc -file [file join $reports post_synth_drc.rpt]
set record [open [file join $output_directory run_summary.txt] w]
puts $record "stateless_depth=$depth"
puts $record "part=xczu3eg-sfvc784-2-e"
puts $record "top=pursuit_state_depth_timing_top"
puts $record "target_clock_period_ns=6.250"
puts $record "stage=synthesis"
puts $record "critical_warning_count=[get_msg_config -severity {CRITICAL WARNING} -count]"
close $record
exit
