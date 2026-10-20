if {$argc != 2} { error "usage: power_vectorless.tcl POST_ROUTE_DCP OUTPUT_DIRECTORY" }
set checkpoint [file normalize [lindex $argv 0]]
set output_directory [file normalize [lindex $argv 1]]
file mkdir $output_directory
set_param general.maxThreads 4
open_checkpoint $checkpoint
set period [get_property PERIOD [get_clocks compute_clock]]
if {$period != 6.25} { error "Expected a 6.25 ns compute clock; found $period" }

# Remove annotations and restore Vivado's default activity assumptions.
# No simulation activity file or user-specified activity is applied.
reset_switching_activity -all
reset_switching_activity -default
set_operating_conditions -process typical -junction_temp 25
report_operating_conditions -all -file [file join $output_directory operating_conditions.rpt]
report_power -hier all -hierarchical_depth 0 -advisory -l 20 -file [file join $output_directory power.rpt]
report_power -hier all -hierarchical_depth 0 -format xml -file [file join $output_directory power.xml]
report_switching_activity -signal_rate -static_probability -file [file join $output_directory input_activity.rpt] [get_ports *]
report_switching_activity -signal_rate -static_probability -type bram_enable -all -file [file join $output_directory bram_activity.rpt]
set audit [open [file join $output_directory settings.txt] w]
puts $audit "vivado_version=[version -short]"
puts $audit "part=[get_property PART [current_design]]"
puts $audit "clock_period_ns=$period"
puts $audit "method=vectorless_default"
puts $audit "simulation_activity_file=none"
puts $audit "process=typical"
puts $audit "junction_temperature_c=25"
puts $audit "voltage=nominal"
close $audit
exit
