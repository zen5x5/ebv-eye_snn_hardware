set vivado_script_directory [file dirname [file normalize [info script]]]
set rtl_directory [file normalize [file join $vivado_script_directory .. rtl]]

set common_sources [list \
    [file join $rtl_directory sparse_conv3x3_pool_lif_core.sv] \
    [file join $rtl_directory pursuit_sparse_l1_event_layer.sv] \
    [file join $rtl_directory pursuit_sparse_l2_packet_layer.sv] \
    [file join $rtl_directory pursuit_sparse_frontend.sv] \
    [file join $rtl_directory programmable_padded_line_buffer_3x3.sv] \
    [file join $rtl_directory programmable_maxpool2x2_stream.sv] \
    [file join $rtl_directory pipelined_count_mac_array.sv] \
    [file join $rtl_directory lif_multispike.sv] \
    [file join $rtl_directory readout_engine.sv] \
]

set time_major_sources [list \
    [file join $rtl_directory conv3x3_layer_engine.sv] \
    [file join $rtl_directory cached_current_lif_replay.sv] \
    [file join $rtl_directory pursuit_l3_current_cache.sv] \
    [file join $rtl_directory pursuit_activation_store.sv] \
    [file join $rtl_directory pursuit_conv_state_store.sv] \
    [file join $rtl_directory linear_lif_layer_engine.sv] \
    [file join $rtl_directory pursuit_sparse_time_major_accelerator_top.sv] \
]

set layer_major_sources [list \
    [file join $rtl_directory paired_conv3x3_layer_engine.sv] \
    [file join $rtl_directory pursuit_paired_feature_store.sv] \
    [file join $rtl_directory paired_linear_lif_layer_engine.sv] \
    [file join $rtl_directory pursuit_sparse_layer_major_accelerator_top.sv] \
]

if {$sparse_variant eq "sparse_tm"} {
    set rtl_sources [concat $common_sources $time_major_sources]
} elseif {$sparse_variant eq "sparse_lm_a"} {
    set rtl_sources [concat $common_sources $layer_major_sources]
} else {
    error "unknown sparse variant: $sparse_variant"
}

foreach source $rtl_sources {
    if {![file exists $source]} {
        error "missing RTL source: $source"
    }
}
read_verilog -sv $rtl_sources
read_xdc [file join $vivado_script_directory constraints compute_ooc.xdc]
