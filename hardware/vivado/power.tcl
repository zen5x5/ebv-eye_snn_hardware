if {$argc != 3} { error "usage: power.tcl POST_ROUTE_DCP SAIF_DIRECTORY OUTPUT_DIRECTORY" }
set checkpoint [file normalize [lindex $argv 0]]
set activity_directory [file normalize [lindex $argv 1]]
set output_directory [file normalize [lindex $argv 2]]
file mkdir $output_directory
set_param general.maxThreads 4
open_checkpoint $checkpoint
# A fixed 25 C junction temperature isolates architecture differences from
# assumed cooling. Nominal device voltages and typical process are used.
set_operating_conditions -process typical -junction_temp 25
report_operating_conditions -all -file [file join $output_directory operating_conditions.rpt]
set activity_files [lsort [glob [file join $activity_directory sample_* inference.saif]]]
foreach activity $activity_files {
    set name [file tail [file dirname $activity]]
    set reports [file join $output_directory $name]
    file mkdir $reports
    reset_switching_activity -default
    read_saif -strip_path tb_state_depth_power/dut -out_file [file join $reports saif_annotation.rpt] $activity
    report_power -hier all -hierarchical_depth 0 -advisory -l 20 -file [file join $reports power.rpt]
    report_power -hier all -hierarchical_depth 0 -format xml -file [file join $reports power.xml]
    report_switching_activity -signal_rate -static_probability -file [file join $reports input_activity.rpt] [get_ports *]
    report_switching_activity -signal_rate -static_probability -type bram_enable -all -file [file join $reports bram_activity.rpt]
}
exit
