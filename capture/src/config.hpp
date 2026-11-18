// Configuration: a key=value file (serial numbers and defaults) plus command-line options that override it.
#pragma once

#include <string>

namespace cap {

struct Config {
    std::string config_path = "capture.conf";
    std::string serial[2];                  // A, B
    std::string sink = "shm";               // shm | stats
    std::string shm_name = "/d435_pairs";
    double duration_s = 0;                  // 0: run until SIGINT/SIGTERM
    double status_interval_s = 10;
    std::string pair_csv;                   // empty: no CSV
    std::string hist_out = "latency_histograms.csv";
    std::string final_json;                 // empty: no JSON
    std::string delay_hist_out = "output_delay_histograms.csv";   // gap 3: output delay histograms at shutdown
    // CPU affinity (R8, off by default): -1 = not set
    int cpu_matcher = -1, cpu_supervisor = -1, cpu_csv = -1, cpu_callback[2] = {-1, -1};
    // test option (watchdog test): camera index, start (s after stream start), length (s); cam -1 = off
    int test_ignore_cam = -1;
    double test_ignore_at_s = 0, test_ignore_for_s = 0;
    // test option (gap 2): hardware resets, one camera letter per reset (e.g. AAAAAAAAAABBBBBBBBBB); the first at
    // test_hw_first_s after start, each next one test_hw_interval_s after the previous camera delivers pairs again
    std::string test_hw_seq;
    double test_hw_first_s = 0, test_hw_interval_s = 0;
};

// Parses argv (and the config file named by --config, default capture.conf). Returns false with err set on error;
// sets help=true for --help.
bool load_config(int argc, char** argv, Config& cfg, std::string& err, bool& help);
const char* usage_text();

}  // namespace cap
