set vivado_script_directory [file dirname [file normalize [info script]]]
set rtl_directory [file normalize [file join $vivado_script_directory .. rtl]]

set rtl_sources [list \
    [file join $rtl_directory programmable_padded_line_buffer_3x3.sv] \
    [file join $rtl_directory programmable_maxpool2x2_stream.sv] \
    [file join $rtl_directory pipelined_count_mac_array.sv] \
    [file join $rtl_directory lif_multispike.sv] \
    [file join $rtl_directory conv3x3_layer_engine.sv] \
    [file join $rtl_directory linear_lif_layer_engine.sv] \
    [file join $rtl_directory readout_engine.sv] \
    [file join $rtl_directory cached_current_lif_replay.sv] \
    [file join $rtl_directory pursuit_activation_store.sv] \
    [file join $rtl_directory pursuit_conv_state_store.sv] \
    [file join $rtl_directory pursuit_l3_current_cache.sv] \
    [file join $rtl_directory pursuit_snn_accelerator_top.sv] \
]

foreach source $rtl_sources {
    if {![file exists $source]} {
        error "missing RTL source: $source"
    }
}
read_verilog -sv $rtl_sources
read_xdc [file join $vivado_script_directory constraints compute_ooc.xdc]
