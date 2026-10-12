# Usage:
# vivado -mode batch -source run_sparse_ooc.tcl \
#   -tclargs OUTPUT_DIR VARIANT PART JOBS
# VARIANT is sparse_tm or sparse_lm_a.

if {$argc < 2 || $argc > 4} {
    error "usage: run_sparse_ooc.tcl OUTPUT_DIR VARIANT ?PART=xczu3eg-sfvc784-2-e? ?JOBS=8?"
}
set output_directory [file normalize [lindex $argv 0]]
set sparse_variant [lindex $argv 1]
set target_part "xczu3eg-sfvc784-2-e"
set jobs 8
if {$argc >= 3} {
    set target_part [lindex $argv 2]
}
if {$argc >= 4} {
    set jobs [lindex $argv 3]
}
if {$sparse_variant eq "sparse_tm"} {
    set top_module pursuit_sparse_time_major_accelerator_top
} elseif {$sparse_variant eq "sparse_lm_a"} {
    set top_module pursuit_sparse_layer_major_accelerator_top
} else {
    error "VARIANT must be sparse_tm or sparse_lm_a"
}
if {![string is integer -strict $jobs] || $jobs < 1} {
    error "JOBS must be a positive integer"
}

if {[file exists $output_directory]} {
    if {![file isdirectory $output_directory]} {
        error "OUTPUT_DIR exists and is not a directory: $output_directory"
    }
    set output_entries {}
    foreach entry [concat \
            [glob -nocomplain -directory $output_directory -- *] \
            [glob -nocomplain -directory $output_directory -- .*]] {
        if {[file tail $entry] ni {. ..}} {
            lappend output_entries $entry
        }
    }
    if {[llength $output_entries] != 0} {
        error "refusing to reuse nonempty OUTPUT_DIR: $output_directory"
    }
}
file mkdir $output_directory
set report_directory [file join $output_directory reports]
file mkdir $report_directory

set run_script_path [file normalize [info script]]
set vivado_script_directory [file dirname $run_script_path]
set shell_script_path [file join $vivado_script_directory run_sparse_ooc.sh]
set source_script_path [file join $vivado_script_directory sources_sparse.tcl]
set constraint_path [file join $vivado_script_directory constraints compute_ooc.xdc]
source $source_script_path

proc sha256_file {path} {
    if {![file isfile $path]} {
        error "provenance input is not a regular file: $path"
    }
    if {[catch {exec sha256sum -- $path} digest_line]} {
        error "sha256sum failed for $path: $digest_line"
    }
    set digest [lindex $digest_line 0]
    if {![regexp {^[0-9a-fA-F]{64}$} $digest]} {
        error "could not parse SHA-256 for $path: $digest_line"
    }
    return [string tolower $digest]
}

set provenance_sources [concat \
    [list $run_script_path $shell_script_path $source_script_path $constraint_path] \
    $rtl_sources]
set initial_source_hashes [dict create]
foreach path $provenance_sources {
    set normalized_path [file normalize $path]
    if {[dict exists $initial_source_hashes $normalized_path]} {
        error "duplicate provenance source: $normalized_path"
    }
    dict set initial_source_hashes $normalized_path [sha256_file $normalized_path]
}

set source_hash_file [file join $output_directory source_sha256.txt]
set source_hash_channel [open $source_hash_file w]
foreach path [lsort [dict keys $initial_source_hashes]] {
    puts $source_hash_channel "[dict get $initial_source_hashes $path]  $path"
}
close $source_hash_channel

set provenance_file [file join $output_directory run_provenance.txt]
set provenance_channel [open $provenance_file w]
puts $provenance_channel "vivado_version=[version -short]"
puts $provenance_channel "variant=$sparse_variant"
puts $provenance_channel "top=$top_module"
puts $provenance_channel "part=$target_part"
puts $provenance_channel "target_clock_period_ns=5.000"
puts $provenance_channel "jobs=$jobs"
puts $provenance_channel "source_sha256_manifest=$source_hash_file"
puts $provenance_channel "power_scope=vectorless_estimate_only_not_acceptance_evidence"
close $provenance_channel

set_param general.maxThreads $jobs

synth_design \
    -top $top_module \
    -part $target_part \
    -mode out_of_context \
    -flatten_hierarchy rebuilt
write_checkpoint -force [file join $output_directory post_synth.dcp]
report_utilization -hierarchical -file \
    [file join $report_directory post_synth_utilization.rpt]
report_timing_summary -delay_type max -max_paths 20 -report_unconstrained \
    -file [file join $report_directory post_synth_timing.rpt]
report_methodology -file [file join $report_directory post_synth_methodology.rpt]

opt_design
place_design
phys_opt_design
route_design
write_checkpoint -force [file join $output_directory post_route.dcp]
report_utilization -hierarchical -file \
    [file join $report_directory post_route_utilization.rpt]
report_timing_summary -delay_type min_max -max_paths 50 -report_unconstrained \
    -file [file join $report_directory post_route_timing.rpt]
report_clock_utilization -file \
    [file join $report_directory post_route_clock_utilization.rpt]
report_drc -file [file join $report_directory post_route_drc.rpt]
report_methodology -file [file join $report_directory post_route_methodology.rpt]
report_power -file [file join $report_directory post_route_power_vectorless.rpt]

proc require_finite_worst_slack {analysis_switch} {
    set timing_paths [get_timing_paths $analysis_switch -max_paths 1 -nworst 1 \
        -no_report_unconstrained]
    if {[llength $timing_paths] != 1} {
        error "expected one constrained $analysis_switch timing path, got [llength $timing_paths]"
    }
    set slack [get_property SLACK [lindex $timing_paths 0]]
    if {![string is double -strict $slack] || [regexp -nocase {inf|nan} $slack]} {
        error "invalid $analysis_switch slack: $slack"
    }
    return $slack
}

proc timing_check_issue_count {check_name report_channel} {
    set report [check_timing -verbose -override_defaults [list $check_name] \
        -return_string]
    puts $report_channel $report
    set matched_summary 0
    set issue_count 0
    foreach line [split $report "\n"] {
        if {[regexp -nocase \
                {There[[:space:]]+(is|are)[[:space:]]+([0-9][0-9,]*)} \
                $line unused verb count_text]} {
            incr matched_summary
            set count_text [string map [list "," ""] $count_text]
            incr issue_count $count_text
        }
    }
    if {$matched_summary == 0} {
        error "could not parse check_timing result for $check_name"
    }
    return $issue_count
}

set worst_setup_slack [require_finite_worst_slack -setup]
set worst_hold_slack [require_finite_worst_slack -hold]

set check_timing_file [file join $report_directory post_route_check_timing.rpt]
set check_timing_channel [open $check_timing_file w]
set unconstrained_check_names [list \
    no_clock \
    no_input_delay \
    no_output_delay \
    partial_input_delay \
    partial_output_delay \
    unconstrained_internal_endpoints]
set unconstrained_issue_count 0
foreach check_name $unconstrained_check_names {
    incr unconstrained_issue_count \
        [timing_check_issue_count $check_name $check_timing_channel]
}
close $check_timing_channel

set critical_warning_file \
    [file join $report_directory post_route_critical_warnings.rpt]
write_messages -force -severity {CRITICAL WARNING} $critical_warning_file
set critical_warning_count \
    [get_msg_config -severity {CRITICAL WARNING} -count]

foreach path [dict keys $initial_source_hashes] {
    set final_digest [sha256_file $path]
    set initial_digest [dict get $initial_source_hashes $path]
    if {$final_digest ne $initial_digest} {
        error "provenance source changed during the run: $path"
    }
}

set summary_file [file join $output_directory run_summary.txt]
set summary [open $summary_file w]
puts $summary "variant=$sparse_variant"
puts $summary "top=$top_module"
puts $summary "part=$target_part"
puts $summary "target_clock_period_ns=5.000"
puts $summary "worst_setup_slack_ns=$worst_setup_slack"
puts $summary "worst_hold_slack_ns=$worst_hold_slack"
puts $summary "unconstrained_timing_issue_count=$unconstrained_issue_count"
puts $summary "critical_warning_count=$critical_warning_count"
puts $summary "critical_warnings_report=$critical_warning_file"
puts $summary "source_hashes_verified_unchanged=1"
puts $summary "timing_met=[expr {\
    $worst_setup_slack >= 0.0 && \
    $worst_hold_slack >= 0.0 && \
    $unconstrained_issue_count == 0}]"
puts $summary "power_scope=vectorless_estimate_only_not_acceptance_evidence"
close $summary

set acceptance_failures {}
if {$worst_setup_slack < 0.0} {
    lappend acceptance_failures \
        "negative routed setup slack: $worst_setup_slack ns"
}
if {$worst_hold_slack < 0.0} {
    lappend acceptance_failures \
        "negative routed hold slack: $worst_hold_slack ns"
}
if {$unconstrained_issue_count != 0} {
    lappend acceptance_failures \
        "$unconstrained_issue_count unconstrained/incompletely constrained timing issues"
}
if {[llength $acceptance_failures] != 0} {
    error "routed OOC acceptance failed: [join $acceptance_failures {; }]"
}
