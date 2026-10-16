`timescale 1ns/1ps
module full_walkthrough_monitor;
    integer states [0:15];
    integer i;
    integer cycle = 0;
    integer last_input = -1;
    integer first_l1 = -1;
    integer first_packet = -1;
    integer last_packet = -1;
    integer waiting_for_l2 = 0;
    initial begin
        for (i=0; i<16; i=i+1) states[i]=0;
    end
    always @(posedge tb_pursuit_sparse_layer_major_accelerator_top.clk) begin
        if (tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1.core.busy) begin
            states[tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1.core.control_state] =
                states[tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1.core.control_state] + 1;
        end
        if (tb_pursuit_sparse_layer_major_accelerator_top.busy) begin
            cycle=cycle+1;
            if (tb_pursuit_sparse_layer_major_accelerator_top.event_valid
                    && tb_pursuit_sparse_layer_major_accelerator_top.event_ready)
                last_input=cycle;
            if (tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1_trace_valid
                    && first_l1==-1) first_l1=cycle;
            if (tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1_packet_valid
                    && tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1_packet_ready) begin
                if (first_packet==-1) first_packet=cycle;
                last_packet=cycle;
            end
            if (tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1_packet_valid
                    && !tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1_packet_ready)
                waiting_for_l2=waiting_for_l2+1;
        end
    end
    final begin
        $display("WALKTHROUGH_FULL l1_wrapper_cycles=%0d l1_core_cycles=%0d l2_cycles=%0d last_input=%0d first_l1=%0d first_packet=%0d last_packet=%0d packet_wait_for_l2=%0d",
            tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1_cycle_count,
            tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l1_core_cycle_count,
            tb_pursuit_sparse_layer_major_accelerator_top.dut.sparse_frontend.l2_cycle_count,
            last_input,first_l1,first_packet,last_packet,waiting_for_l2);
        for (i=0;i<16;i=i+1)
            $display("WALKTHROUGH_FULL state=%0d cycles=%0d",i,states[i]);
    end
endmodule
