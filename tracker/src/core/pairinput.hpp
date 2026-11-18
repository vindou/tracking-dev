// Pair input for the setup tools: the live pair buffer (capture or replay) or a pair file.
#pragma once
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "../common/clock.hpp"
#include "../records/pairfile.hpp"
#include "shm_sink.hpp"

namespace trk {

class PairInput {
public:
    // source: a pair file path, or "shm:NAME" for a pair buffer (waits up to wait_s for a writer).
    bool open(const std::string& source, double wait_s, std::string& err) {
        if (source.rfind("shm:", 0) == 0) {
            const std::string name = source.substr(4);
            const int64_t t0 = mono_ns();
            while (!(r_ = cap::ShmReader::attach(name, err))) {
                if (double(mono_ns() - t0) / 1e9 > wait_s) return false;
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            cams_[0] = r_->header().cam[0];
            cams_[1] = r_->header().cam[1];
            n_ = r_->image_bytes() / 2;
            return true;
        }
        if (!f_.open(source, err)) return false;
        cams_[0] = f_.header().cam[0];
        cams_[1] = f_.header().cam[1];
        n_ = std::size_t(f_.header().width) * f_.header().height;
        return true;
    }
    const cap::CameraInfo* cameras() const { return cams_; }
    std::size_t pixels() const { return n_; }
    // Next pair (full images). false at the end of a file or when the writer stopped / died.
    bool next(cap::PairRecord& rec, uint16_t* a, uint16_t* b, double* truth16 = nullptr) {
        if (r_) {
            for (;;) {
                const auto res = r_->next(rec, a, b, 2000000000LL);
                if (res == cap::ShmReader::Result::Ok) return true;
                if (res != cap::ShmReader::Result::Timeout) return false;
            }
        }
        PairChunkHeader ch;
        std::string err;
        if (!f_.next(ch, a, b, err)) return false;
        rec = ch.rec;
        if (truth16 && (ch.flags & 1)) for (int i = 0; i < 16; i++) truth16[i] = ch.truth[i];
        return true;
    }
    bool live() const { return r_ != nullptr; }
private:
    std::unique_ptr<cap::ShmReader> r_;
    PairFileReader f_;
    cap::CameraInfo cams_[2] = {};
    std::size_t n_ = 0;
};

}  // namespace trk
