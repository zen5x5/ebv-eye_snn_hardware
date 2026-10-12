# OOC timing contract for the pursuit accelerator core.
# Board clocks and PS/AXI pin constraints belong in the eventual board wrapper.
create_clock -name compute_clock -period 5.000 [get_ports clk]

set non_clock_inputs [remove_from_collection [all_inputs] [get_ports clk]]
if {[llength $non_clock_inputs] > 0} {
    set_input_delay -clock compute_clock 1.000 $non_clock_inputs
}
set all_core_outputs [all_outputs]
if {[llength $all_core_outputs] > 0} {
    set_output_delay -clock compute_clock 1.000 $all_core_outputs
}
