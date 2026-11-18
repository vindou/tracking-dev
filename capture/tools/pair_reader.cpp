// Example reader for the d435_capture shared memory (layout version 2).
// Attaches, reads every pair (futex notification, no polling while pairs arrive), and follows writer restarts:
//  - writer not alive (heartbeat older than the liveness timeout: killed or hung) or replaced by a new writer:
//    the reader attaches again by name as soon as a new writer instance exists (retry every 100 ms while there is
//    none; it continues with the old writer if that one becomes alive again);
//  - writer stopped cleanly: the reader exits, or with --follow waits for a new writer like above.
// Reports the number of attaches and, for each writer instance, its id, pairs received and missing.
//
//   d435_pair_reader [--shm-name /d435_pairs] [--duration S] [--slow-ms N] [--wait-attach S] [--reattach-wait S]
//                    [--follow] [--json FILE] [--quiet]
//
// Exits when the writer stops (without --follow), when no new writer comes within --reattach-wait seconds, after
// --duration, or on SIGINT/SIGTERM. Exit code 0 if no pair was missing, 1 otherwise, 3 if the first attach failed.
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "shm_sink.hpp"

using namespace cap;

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop.store(true); }

// Statistics for one writer instance (one attach).
struct Instance {
    uint64_t id = 0;
    int pid = 0;
    int64_t attach_real_ns = 0, writer_start_real_ns = 0;
    uint64_t start_seq = 0, received = 0, first_seq = UINT64_MAX, last_seq = 0, seq_gaps = 0;
    uint64_t missing0 = 0, discarded0 = 0, missing = 0, discarded = 0;   // reader totals at attach; deltas
    int64_t first_recv_real_ns = 0, first_publish_real_ns = 0;
    std::string end = "running";   // writer_stopped | writer_not_alive | writer_replaced | duration | signal | ...
    int64_t end_real_ns = 0;
    double end_heartbeat_age_ms = 0;
};

void print_attach(const ShmReader& r, const std::string& name, const Instance& in) {
    const ShmHeader& h = r.header();
    std::printf("attach %" PRIu64 ": %s layout v%u, %ux%u, %u slots, writer instance %016" PRIx64
                " pid %d, start at pair %" PRIu64 "\n",
                r.attaches(), name.c_str(), h.layout_version, h.image_width, h.image_height, h.slot_count, in.id, in.pid,
                in.start_seq);
    for (int c = 0; c < 2; c++)
        std::printf("  camera %s: serial %s depth_scale %.6g fx %.3f fy %.3f ppx %.3f ppy %.3f\n", c ? "B" : "A",
                    h.cam[c].serial, h.cam[c].depth_scale, h.cam[c].fx, h.cam[c].fy, h.cam[c].ppx, h.cam[c].ppy);
    std::fflush(stdout);
}

Instance new_instance(const ShmReader& r) {
    Instance in;
    in.id = r.instance_id();
    in.pid = r.header().writer_pid;
    in.attach_real_ns = real_ns();
    in.writer_start_real_ns = r.header().writer_start_real_ns;
    in.start_seq = r.start_seq();
    in.missing0 = r.missing();
    in.discarded0 = r.discarded_reads();
    return in;
}

void close_instance(Instance& in, const ShmReader& r, const char* end) {
    in.missing = r.missing() - in.missing0;
    in.discarded = r.discarded_reads() - in.discarded0;
    in.end = end;
    in.end_real_ns = real_ns();
    in.end_heartbeat_age_ms = double(r.heartbeat_age_ns()) / 1e6;
}
}  // namespace

int main(int argc, char** argv) {
    std::string name = "/d435_pairs", json;
    double duration = 0, wait_attach = 10, reattach_wait = 30;
    int slow_ms = 0;
    bool quiet = false, follow = false;
    for (int i = 1; i < argc; i++) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "missing value for %s\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--shm-name") name = next();
        else if (a == "--duration") duration = std::atof(next());
        else if (a == "--slow-ms") slow_ms = std::atoi(next());
        else if (a == "--wait-attach") wait_attach = std::atof(next());
        else if (a == "--reattach-wait") reattach_wait = std::atof(next());
        else if (a == "--follow") follow = true;
        else if (a == "--json") json = next();
        else if (a == "--quiet") quiet = true;
        else {
            std::fprintf(stderr,
                         "usage: %s [--shm-name NAME] [--duration S] [--slow-ms N] [--wait-attach S] "
                         "[--reattach-wait S] [--follow] [--json FILE] [--quiet]\n",
                         argv[0]);
            return 2;
        }
    }
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    const auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    auto duration_over = [&] { return duration > 0 && elapsed() >= duration; };

    // First attach: the writer may not exist yet, retry every 100 ms for --wait-attach seconds.
    std::unique_ptr<ShmReader> r;
    std::string err;
    while (!g_stop.load()) {
        r = ShmReader::attach(name, err);
        if (r) break;
        if (elapsed() > wait_attach) {
            std::fprintf(stderr, "cannot attach to %s: %s\n", name.c_str(), err.c_str());
            return 3;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!r) return 3;
    std::vector<Instance> inst;
    inst.push_back(new_instance(*r));
    print_attach(*r, name, inst.back());

    std::vector<uint16_t> a(r->image_bytes() / 2), b(r->image_bytes() / 2);
    auto t_print = std::chrono::steady_clock::now();
    const char* end_reason = "signal";
    while (!g_stop.load()) {
        if (duration_over()) {
            end_reason = "duration";
            break;
        }
        PairRecord rec;
        const auto res = r->next(rec, a.data(), b.data(), 1000000000LL);
        if (res == ShmReader::Result::Ok) {
            Instance& in = inst.back();
            if (in.received == 0) {
                in.first_seq = rec.pair_seq;
                in.first_recv_real_ns = real_ns();
                in.first_publish_real_ns = rec.publish_real_ns;
            } else if (rec.pair_seq != in.last_seq + 1) {
                in.seq_gaps += rec.pair_seq - in.last_seq - 1;
            }
            in.last_seq = rec.pair_seq;
            in.received++;
            if (slow_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(slow_ms));
            const auto now = std::chrono::steady_clock::now();
            if (!quiet && now - t_print >= std::chrono::seconds(10)) {
                t_print = now;
                std::printf("reader: instance %016" PRIx64 " received=%" PRIu64 " last_seq=%" PRIu64 " missing=%" PRIu64
                            " discarded_reads=%" PRIu64 " dt=%.3f ms delay=%.2f ms\n",
                            in.id, in.received, in.last_seq, r->missing() - in.missing0,
                            r->discarded_reads() - in.discarded0, rec.dt_ms,
                            double(real_ns() - rec.publish_real_ns) / 1e6);
                std::fflush(stdout);
            }
            continue;
        }
        if (res == ShmReader::Result::Timeout) continue;
        // Writer stopped, not alive, or replaced.
        const char* what = ShmReader::result_name(res);
        std::printf("reader: writer instance %016" PRIx64 ": %s (heartbeat age %.0f ms) at real %.3f s\n",
                    inst.back().id, what, double(r->heartbeat_age_ns()) / 1e6, double(real_ns()) / 1e9);
        std::fflush(stdout);
        close_instance(inst.back(), *r, what);
        if (res == ShmReader::Result::WriterStopped && !follow) {
            end_reason = what;
            break;
        }
        // Wait for a new writer instance (retry every 100 ms; this is the only polling and only while no writer
        // is present). A writer that was only "not alive" (e.g. stopped by a debugger) may become alive again.
        const auto t_lost = std::chrono::steady_clock::now();
        bool again = false, resumed = false;
        while (!g_stop.load() && !duration_over()) {
            if (r->reattach(err)) {
                again = true;
                break;
            }
            if (res == ShmReader::Result::WriterNotAlive && r->writer_alive()) {
                resumed = true;
                break;
            }
            if (std::chrono::duration<double>(std::chrono::steady_clock::now() - t_lost).count() > reattach_wait) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (resumed) {
            std::printf("reader: writer instance %016" PRIx64 " alive again\n", inst.back().id);
            inst.back().end = "running";
            continue;
        }
        if (!again) {
            end_reason = g_stop.load() ? "signal" : duration_over() ? "duration" : "no_new_writer";
            break;
        }
        if (r->image_bytes() / 2 != a.size()) {
            a.assign(r->image_bytes() / 2, 0);
            b.assign(r->image_bytes() / 2, 0);
        }
        inst.push_back(new_instance(*r));
        print_attach(*r, name, inst.back());
    }
    if (inst.back().end == "running") close_instance(inst.back(), *r, end_reason);

    // Missing = numbers the reader never got after its first pair of an instance: counted by the reader protocol
    // (overwritten / torn) and cross-checked with gaps in the received sequence numbers.
    uint64_t received = 0, missing = 0, gaps = 0;
    for (const Instance& in : inst) {
        received += in.received;
        missing += in.missing;
        gaps += in.seq_gaps;
        std::printf("reader instance %016" PRIx64 " pid %d: received=%" PRIu64 " first_seq=%" PRIu64 " last_seq=%" PRIu64
                    " missing=%" PRIu64 " seq_gaps=%" PRIu64 " discarded_reads=%" PRIu64
                    " first_pair_delay_ms=%.3f end=%s\n",
                    in.id, in.pid, in.received, in.received ? in.first_seq : 0, in.last_seq, in.missing, in.seq_gaps,
                    in.discarded, in.received ? double(in.first_recv_real_ns - in.first_publish_real_ns) / 1e6 : -1.0,
                    in.end.c_str());
    }
    std::printf("reader final: attaches=%" PRIu64 " received=%" PRIu64 " first_seq=%" PRIu64 " last_seq=%" PRIu64
                " missing=%" PRIu64 " seq_gaps=%" PRIu64 " discarded_reads=%" PRIu64 " writer_stopped=%d end=%s\n",
                r->attaches(), received, inst.front().received ? inst.front().first_seq : 0, inst.back().last_seq,
                missing, gaps, r->discarded_reads(), inst.back().end == "writer_stopped" ? 1 : 0, end_reason);
    if (!json.empty()) {
        if (FILE* f = std::fopen(json.c_str(), "w")) {
            std::fprintf(f,
                         "{\"attaches\": %" PRIu64 ", \"received\": %" PRIu64 ", \"start_seq\": %" PRIu64
                         ", \"first_seq\": %" PRIu64 ", \"last_seq\": %" PRIu64 ", \"missing\": %" PRIu64
                         ", \"seq_gaps\": %" PRIu64 ", \"discarded_reads\": %" PRIu64
                         ", \"writer_stopped\": %s, \"end\": \"%s\", \"instances\": [",
                         r->attaches(), received, inst.front().start_seq,
                         inst.front().received ? inst.front().first_seq : 0, inst.back().last_seq, missing, gaps,
                         r->discarded_reads(), inst.back().end == "writer_stopped" ? "true" : "false", end_reason);
            for (std::size_t i = 0; i < inst.size(); i++) {
                const Instance& in = inst[i];
                std::fprintf(f,
                             "%s\n  {\"instance_id\": \"%016" PRIx64 "\", \"writer_pid\": %d, \"writer_start_real_ns\": %" PRId64
                             ", \"attach_real_ns\": %" PRId64 ", \"start_seq\": %" PRIu64 ", \"received\": %" PRIu64
                             ", \"first_seq\": %" PRId64 ", \"last_seq\": %" PRIu64 ", \"missing\": %" PRIu64
                             ", \"seq_gaps\": %" PRIu64 ", \"discarded_reads\": %" PRIu64
                             ", \"first_recv_real_ns\": %" PRId64 ", \"first_publish_real_ns\": %" PRId64
                             ", \"end\": \"%s\", \"end_real_ns\": %" PRId64 ", \"end_heartbeat_age_ms\": %.1f}",
                             i ? "," : "", in.id, in.pid, in.writer_start_real_ns, in.attach_real_ns, in.start_seq,
                             in.received, in.received ? int64_t(in.first_seq) : int64_t(-1), in.last_seq, in.missing,
                             in.seq_gaps, in.discarded, in.first_recv_real_ns, in.first_publish_real_ns, in.end.c_str(),
                             in.end_real_ns, in.end_heartbeat_age_ms);
            }
            std::fprintf(f, "\n]}\n");
            std::fclose(f);
        }
    }
    return (missing == 0 && gaps == 0) ? 0 : 1;
}
