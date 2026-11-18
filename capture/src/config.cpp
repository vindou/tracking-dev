#include "config.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <vector>

namespace cap {

const char* usage_text() {
    return "usage: d435_capture [options]\n"
           "  --config FILE          key=value file (default capture.conf): A=<serial>, B=<serial> (or serial_a/serial_b),\n"
           "                         optional: sink, shm_name, status_interval, pair_csv, hist_out\n"
           "  --sink shm|stats       pair output (default shm)\n"
           "  --shm-name NAME        POSIX shared-memory name (default /d435_pairs)\n"
           "  --duration S           stop after S seconds (default: run until SIGINT/SIGTERM)\n"
           "  --status-interval S    status line period (default 10)\n"
           "  --pair-csv FILE        write one CSV line per pair (separate writer thread)\n"
           "  --hist-out FILE        latency histograms at shutdown (default latency_histograms.csv)\n"
           "  --final-json FILE      final counters as JSON\n"
           "  --affinity LIST        CPU affinity, e.g. matcher=2,supervisor=3,csv=3,callback_a=4,callback_b=5\n"
           "  --delay-hist-out FILE  output delay histograms at shutdown (default output_delay_histograms.csv)\n"
           "  --test-ignore CAM:AT:FOR   test only: callback of CAM (A|B) ignores frames from AT s after start for FOR s\n"
           "  --test-hw-reset SEQ:FIRST:INTERVAL  test only: hardware reset of the cameras in SEQ (letters A|B, e.g.\n"
           "                         AAAAABBBBB), the first FIRST s after start, each next one INTERVAL s after the\n"
           "                         previous camera delivers pairs again\n";
}

namespace {

std::string trim(const std::string& s) {
    const auto b = s.find_first_not_of(" \t\r\n");
    if (b == std::string::npos) return "";
    const auto e = s.find_last_not_of(" \t\r\n");
    return s.substr(b, e - b + 1);
}

bool to_double(const std::string& s, double& v) {
    char* end = nullptr;
    v = std::strtod(s.c_str(), &end);
    return end && *end == '\0' && !s.empty();
}

bool to_int(const std::string& s, int& v) {
    char* end = nullptr;
    const long l = std::strtol(s.c_str(), &end, 10);
    v = int(l);
    return end && *end == '\0' && !s.empty();
}

bool apply(Config& c, const std::string& key, const std::string& val, std::string& err) {
    double d = 0;
    if (key == "A" || key == "serial_a") c.serial[0] = val;
    else if (key == "B" || key == "serial_b") c.serial[1] = val;
    else if (key == "sink") c.sink = val;
    else if (key == "shm_name") c.shm_name = val;
    else if (key == "pair_csv") c.pair_csv = val;
    else if (key == "hist_out") c.hist_out = val;
    else if (key == "delay_hist_out") c.delay_hist_out = val;
    else if (key == "status_interval") {
        if (!to_double(val, d) || d <= 0) { err = "bad status_interval: " + val; return false; }
        c.status_interval_s = d;
    } else {
        err = "unknown config key: " + key;
        return false;
    }
    return true;
}

bool parse_affinity(Config& c, const std::string& list, std::string& err) {
    std::stringstream ss(list);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto eq = item.find('=');
        int cpu = -1;
        if (eq == std::string::npos || !to_int(item.substr(eq + 1), cpu) || cpu < 0) {
            err = "bad --affinity item: " + item;
            return false;
        }
        const std::string k = item.substr(0, eq);
        if (k == "matcher") c.cpu_matcher = cpu;
        else if (k == "supervisor") c.cpu_supervisor = cpu;
        else if (k == "csv") c.cpu_csv = cpu;
        else if (k == "callback_a") c.cpu_callback[0] = cpu;
        else if (k == "callback_b") c.cpu_callback[1] = cpu;
        else { err = "bad --affinity key: " + k; return false; }
    }
    return true;
}

}  // namespace

bool load_config(int argc, char** argv, Config& cfg, std::string& err, bool& help) {
    help = false;
    std::vector<std::string> args(argv + 1, argv + argc);
    // pass 1: --config
    for (std::size_t i = 0; i + 1 < args.size(); i++)
        if (args[i] == "--config") cfg.config_path = args[i + 1];
    std::ifstream f(cfg.config_path);
    if (!f) {
        err = "cannot read config file " + cfg.config_path;
        return false;
    }
    std::string line;
    int ln = 0;
    while (std::getline(f, line)) {
        ln++;
        const auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) {
            err = cfg.config_path + ":" + std::to_string(ln) + ": expected key=value";
            return false;
        }
        if (!apply(cfg, trim(line.substr(0, eq)), trim(line.substr(eq + 1)), err)) {
            err = cfg.config_path + ":" + std::to_string(ln) + ": " + err;
            return false;
        }
    }
    // pass 2: options override the file
    for (std::size_t i = 0; i < args.size(); i++) {
        const std::string& a = args[i];
        auto val = [&](std::string& out) -> bool {
            if (i + 1 >= args.size()) { err = "missing value for " + a; return false; }
            out = args[++i];
            return true;
        };
        std::string v;
        double d = 0;
        if (a == "--help" || a == "-h") { help = true; return true; }
        else if (a == "--config") { if (!val(v)) return false; }
        else if (a == "--sink") { if (!val(cfg.sink)) return false; }
        else if (a == "--shm-name") { if (!val(cfg.shm_name)) return false; }
        else if (a == "--pair-csv") { if (!val(cfg.pair_csv)) return false; }
        else if (a == "--hist-out") { if (!val(cfg.hist_out)) return false; }
        else if (a == "--final-json") { if (!val(cfg.final_json)) return false; }
        else if (a == "--delay-hist-out") { if (!val(cfg.delay_hist_out)) return false; }
        else if (a == "--test-hw-reset") {
            if (!val(v)) return false;
            std::stringstream ss(v);
            std::string seq, first, iv;
            if (!std::getline(ss, seq, ':') || !std::getline(ss, first, ':') || !std::getline(ss, iv) || seq.empty() ||
                seq.find_first_not_of("AB") != std::string::npos || !to_double(first, cfg.test_hw_first_s) ||
                !to_double(iv, cfg.test_hw_interval_s) || cfg.test_hw_first_s < 0 || cfg.test_hw_interval_s < 0) {
                err = "bad --test-hw-reset (expected AAAAABBBBB:20:20)";
                return false;
            }
            cfg.test_hw_seq = seq;
        }
        else if (a == "--duration" || a == "--status-interval") {
            if (!val(v) || !to_double(v, d) || d < 0) { err = "bad value for " + a; return false; }
            (a == "--duration" ? cfg.duration_s : cfg.status_interval_s) = d;
        } else if (a == "--affinity") {
            if (!val(v) || !parse_affinity(cfg, v, err)) return false;
        } else if (a == "--test-ignore") {
            if (!val(v)) return false;
            std::stringstream ss(v);
            std::string cam, at, len;
            if (!std::getline(ss, cam, ':') || !std::getline(ss, at, ':') || !std::getline(ss, len) ||
                (cam != "A" && cam != "B") || !to_double(at, cfg.test_ignore_at_s) || !to_double(len, cfg.test_ignore_for_s)) {
                err = "bad --test-ignore (expected A:20:1)";
                return false;
            }
            cfg.test_ignore_cam = cam == "A" ? 0 : 1;
        } else {
            err = "unknown option " + a;
            return false;
        }
    }
    if (cfg.serial[0].empty() || cfg.serial[1].empty()) { err = "serial numbers for A and B are required in " + cfg.config_path; return false; }
    if (cfg.serial[0] == cfg.serial[1]) { err = "A and B have the same serial number"; return false; }
    if (cfg.sink != "shm" && cfg.sink != "stats") { err = "sink must be shm or stats"; return false; }
    if (cfg.status_interval_s <= 0) { err = "status interval must be > 0"; return false; }
    return true;
}

}  // namespace cap
