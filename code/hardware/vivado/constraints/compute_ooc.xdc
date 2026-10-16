# OOC timing contract for the pursuit accelerator core.
# Board clocks and PS/AXI pin constraints belong in the eventual board wrapper.
create_clock -name compute_clock -period 5.000 [get_ports clk]

set_input_delay -clock compute_clock 1.000 \
    [get_ports -filter {DIRECTION == IN && NAME != clk}]
set_output_delay -clock compute_clock 1.000 [all_outputs]
