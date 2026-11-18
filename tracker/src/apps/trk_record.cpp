// T3 recorder: pairs from the pair buffer into a pair file (record/replay format).
//   trk_record --out FILE [--pairs N] [--shm-name /d435_pairs] [--wait S]
// The pairs are kept in memory while recording (the file system cannot take 145 MB/s) and written at the end.
// Every pair is read in full (record and both images) with the capture library reader; missing pairs are reported.
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "common/clock.hpp"
#include "records/pairfile.hpp"
#include "shm_sink.hpp"

using namespace trk;

int main(int argc, char** argv) {
    std::string out, name = "/d435_pairs";
    long pairs = 300;
    double wait_s = 20;
    for (int i = 1; i + 1 < argc; i += 2) {
        const std::string a = argv[i];
        if (a == "--out") out = argv[i + 1];
        else if (a == "--pairs") pairs = std::atol(argv[i + 1]);
        else if (a == "--shm-name") name = argv[i + 1];
        else if (a == "--wait") wait_s = std::atof(argv[i + 1]);
        else { std::fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (out.empty() || pairs <= 0) { std::fprintf(stderr, "usage: trk_record --out FILE [--pairs N] [--shm-name NAME] [--wait S]\n"); return 2; }
    std::string err;
    std::unique_ptr<cap::ShmReader> r;
    const int64_t t0 = mono_ns();
    while (!(r = cap::ShmReader::attach(name, err))) {
        if (double(mono_ns() - t0) / 1e9 > wait_s) { std::fprintf(stderr, "cannot attach %s: %s\n", name.c_str(), err.c_str()); return 3; }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    const std::size_t n = r->image_bytes() / 2;
    std::vector<cap::PairRecord> recs{std::size_t(pairs)};
    std::vector<uint16_t> imgs(std::size_t(pairs) * 2 * n, 0);
    long got = 0;
    while (got < pairs) {
        const auto res = r->next(recs[std::size_t(got)], &imgs[std::size_t(got) * 2 * n], &imgs[(std::size_t(got) * 2 + 1) * n], 5000000000LL);
        if (res == cap::ShmReader::Result::Ok) { got++; continue; }
        if (res != cap::ShmReader::Result::Timeout) { std::fprintf(stderr, "writer: %s\n", cap::ShmReader::result_name(res)); break; }
    }
    const cap::ShmHeader& h = r->header();
    cap::CameraInfo ci[2] = {h.cam[0], h.cam[1]};
    PairFileWriter w;
    if (!w.open(out, ci, 0, h.frame_period_ms, h.tolerance_ms, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 3; }
    for (long k = 0; k < got; k++)
        w.append(recs[std::size_t(k)], &imgs[std::size_t(k) * 2 * n], &imgs[(std::size_t(k) * 2 + 1) * n]);
    w.close();
    std::printf("%s: %ld pairs (seq %" PRIu64 "..%" PRIu64 "), missing %" PRIu64 ", %.1f MB, capture instance %016" PRIx64 "\n",
                out.c_str(), got, got ? recs[0].pair_seq : 0, got ? recs[std::size_t(got - 1)].pair_seq : 0, r->missing(),
                double(w.bytes()) / 1e6, r->instance_id());
    return r->missing() == 0 && got == pairs ? 0 : 1;
}
