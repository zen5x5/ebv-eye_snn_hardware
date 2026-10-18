set scope_file [open netlist_signal_scopes.txt]
set netlist_scopes [split [string trim [read $scope_file]] "\n"]
close $scope_file
add_condition -name power_start {/tb_state_depth_power/power_active == 1} {
    open_saif inference.saif
    set signal_count 0
    foreach name $netlist_scopes {
        if {$name eq "."} {
            current_scope /tb_state_depth_power/dut
        } else {
            current_scope /tb_state_depth_power/dut/$name
        }
        set signals [get_objects]
        log_saif $signals
        incr signal_count [llength $signals]
    }
    current_scope /tb_state_depth_power
    puts "SAIF_NETLIST_OBJECTS $signal_count"
    puts "SAIF_STARTED [current_time]"
}
add_condition -name power_end {/tb_state_depth_power/power_active == 0} {
    if {[current_time] > 0} {
        close_saif
        puts "SAIF_CLOSED [current_time]"
    }
}
run all
quit
