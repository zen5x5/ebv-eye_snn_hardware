if {$argc != 1} { error "usage: audit.tcl RUN_DIRECTORY" }
set run [file normalize [lindex $argv 0]]
set_param general.maxThreads 4
open_checkpoint [file join $run post_route.dcp]
set registers [all_registers]
set core_registers [filter $registers {NAME =~ core/*}]
set record [open [file join $run physical_audit.txt] w]
puts $record "vivado_version=[version -short]"
puts $record "part=[get_property PART [current_design]]"
puts $record "clock_period_ns=[get_property PERIOD [get_clocks compute_clock]]"
set clock_net [get_nets -of_objects [get_pins compute_clock_buffer/O]]
puts $record "clock_net=$clock_net"
puts $record "clock_route_status=[get_property ROUTE_STATUS $clock_net]"
puts $record "clock_buffer_site=[get_property LOC [get_cells compute_clock_buffer]]"
puts $record "clock_package_pin=[get_property PACKAGE_PIN [get_ports clk]]"
puts $record "core_register_count=[llength $core_registers]"
puts $record "critical_warning_count=[get_msg_config -severity {CRITICAL WARNING} -count]"
foreach {name from to} [list core_to_core $core_registers $core_registers \
    input_to_registers [all_inputs] $registers registers_to_output $registers [all_outputs]] {
    foreach analysis {setup hold} {
        set path [get_timing_paths -$analysis -from $from -to $to -max_paths 1]
        puts $record "$name -$analysis slack=[get_property SLACK $path] start=[get_property STARTPOINT_PIN $path] end=[get_property ENDPOINT_PIN $path]"
    }
}
close $record
set scopes [open [file join $run netlist_signal_scopes.txt] w]
puts $scopes "."
foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE == false}]] {
    puts $scopes [get_property NAME $cell]
}
close $scopes
exit
