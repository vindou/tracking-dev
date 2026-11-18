#include "shm_sink.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <random>
#include <thread>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "event_signal.hpp"
#include "thread_util.hpp"

#if defined(__SANITIZE_THREAD__)
#define CAP_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define CAP_TSAN 1
#endif
#endif

namespace cap {

namespace {
std::string errstr(const char* what) { return std::string(what) + ": " + std::strerror(errno); }

uint64_t new_instance_id() {
    std::random_device rd;
    uint64_t v = (uint64_t(rd()) << 32) ^ uint64_t(rd());
    v ^= uint64_t(real_ns()) * 0x9E3779B97F4A7C15ULL;
    v ^= uint64_t(getpid()) << 17;
    return v ? v : 1;
}

std::atomic<uint64_t>& magic_ref(ShmHeader* h) { return *reinterpret_cast<std::atomic<uint64_t>*>(&h->magic); }
const std::atomic<uint64_t>& magic_ref(const ShmHeader* h) {
    return *reinterpret_cast<const std::atomic<uint64_t>*>(&h->magic);
}

// An object of the same name exists: tell its readers that it is replaced (version 2) or stopped (version 1 readers
// know only "stopped"), and wake them. Returns a description for the log.
std::string mark_old_object(const std::string& name, uint64_t new_instance) {
    const int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) return "old object could not be opened (" + std::string(std::strerror(errno)) + ")";
    struct stat st;
    std::string what;
    if (fstat(fd, &st) == 0 && std::size_t(st.st_size) >= kShmHeaderSize) {
        void* p = mmap(nullptr, kShmHeaderSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p != MAP_FAILED) {
            auto* h = static_cast<ShmHeader*>(p);
            if (magic_ref(h).load(std::memory_order_acquire) == kShmMagic) {
                char buf[160];
                if (h->layout_version == kShmLayoutVersion) {
                    std::snprintf(buf, sizeof buf, "instance %016llx, writer pid %d, state %u",
                                  (unsigned long long)h->writer_instance_id, h->writer_pid,
                                  h->writer_state.load(std::memory_order_acquire));
                    h->replaced_by_instance.store(new_instance, std::memory_order_relaxed);
                    h->writer_state.store(kWriterReplaced, std::memory_order_release);
                } else {
                    std::snprintf(buf, sizeof buf, "layout version %u, writer pid %d", h->layout_version, h->writer_pid);
                    h->writer_state.store(kWriterStopped, std::memory_order_release);   // offset 68 in version 1 too
                }
                h->notify_word.fetch_add(1, std::memory_order_release);
                shared_word_wake_all(&h->notify_word);
                what = buf;
            } else {
                what = "header not initialised";
            }
            munmap(p, kShmHeaderSize);
        }
    }
    ::close(fd);
    return what;
}
}  // namespace

#if defined(CAP_TSAN)
// Not instrumented by ThreadSanitizer; volatile so that the compiler does not turn the loop into a memcpy call (which
// TSan would intercept). Only for the sanitizer build; the normal build uses memcpy.
__attribute__((no_sanitize("thread"))) void seqlock_copy(void* dst, const void* src, std::size_t n) {
    auto* d = static_cast<unsigned char*>(dst);
    auto* s = static_cast<const unsigned char*>(src);
    std::size_t i = 0;
    if ((reinterpret_cast<uintptr_t>(d) % 8) == 0 && (reinterpret_cast<uintptr_t>(s) % 8) == 0) {
        auto* d8 = reinterpret_cast<volatile uint64_t*>(d);
        auto* s8 = reinterpret_cast<const volatile uint64_t*>(s);
        for (; i + 8 <= n; i += 8) d8[i / 8] = s8[i / 8];
    }
    for (; i < n; i++) static_cast<volatile unsigned char*>(d)[i] = static_cast<const volatile unsigned char*>(s)[i];
}
#else
void seqlock_copy(void* dst, const void* src, std::size_t n) { std::memcpy(dst, src, n); }
#endif

// ------------------------------------------------------------------------------------------------ writer
std::unique_ptr<ShmSink> ShmSink::create(const std::string& name, const CameraInfo cams[2], std::size_t image_bytes,
                                         std::string& err, std::string* note) {
    std::unique_ptr<ShmSink> s(new ShmSink());
    s->name_ = name;
    s->image_bytes_ = image_bytes;
    s->size_ = shm_total_size(image_bytes, kShmSlots);
    s->instance_ = new_instance_id();

    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0660);
    if (fd < 0 && errno == EEXIST) {
        // Old object (a writer that crashed, or one that still runs): mark it replaced and wake its readers first,
        // then remove the name. Readers still attached to it get WriterReplaced and attach again by name.
        const std::string what = mark_old_object(name, s->instance_);
        if (shm_unlink(name.c_str()) != 0 && errno != ENOENT) {
            err = errstr("shm_unlink of old object");
            return nullptr;
        }
        if (note) *note = "replaced old shared-memory object " + name + " (" + what + ")";
        fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0660);
    }
    if (fd < 0) {
        err = errstr("shm_open");
        return nullptr;
    }
    if (ftruncate(fd, off_t(s->size_)) != 0) {
        err = errstr("ftruncate");
        ::close(fd);
        shm_unlink(name.c_str());
        return nullptr;
    }
    void* p = mmap(nullptr, s->size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        err = errstr("mmap");
        shm_unlink(name.c_str());
        return nullptr;
    }
    s->base_ = static_cast<uint8_t*>(p);
    std::memset(s->base_, 0, s->size_);  // also touches every page now, not in the matcher later

    ShmHeader* h = reinterpret_cast<ShmHeader*>(s->base_);
    h->layout_version = kShmLayoutVersion;
    h->header_size = uint32_t(kShmHeaderSize);
    h->slot_count = uint32_t(kShmSlots);
    h->image_width = kWidth;
    h->image_height = kHeight;
    h->bytes_per_pixel = 2;
    h->image_bytes = image_bytes;
    h->slot_stride = shm_slot_stride(image_bytes);
    h->slots_offset = kShmHeaderSize;
    h->total_size = s->size_;
    h->writer_pid = int32_t(getpid());
    h->record_offset = uint32_t(kShmRecordOffset);
    h->image_offset = uint32_t(kShmImageOffset);
    h->record_size = uint32_t(sizeof(PairRecord));
    h->frame_period_ms = 0;
    h->tolerance_ms = 0;
    for (int c = 0; c < 2; c++) h->cam[c] = cams[c];
    h->writer_instance_id = s->instance_;
    h->heartbeat_interval_ms = kHeartbeatIntervalMs;
    h->liveness_timeout_ms = kLivenessTimeoutMs;
    h->writer_start_real_ns = real_ns();
    h->replaced_by_instance.store(0, std::memory_order_relaxed);
    h->heartbeat_count.store(0, std::memory_order_relaxed);
    h->heartbeat_mono_ns.store(uint64_t(mono_ns()), std::memory_order_relaxed);
    h->pairs_published.store(0, std::memory_order_relaxed);
    h->notify_word.store(0, std::memory_order_relaxed);
    h->writer_state.store(kWriterRunning, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    // magic last: a reader that sees the magic (acquire) sees a complete header
    magic_ref(h).store(kShmMagic, std::memory_order_release);
    s->hdr_ = h;
    ShmSink* raw = s.get();
    s->hb_th_ = std::thread([raw] { raw->heartbeat_loop(); });
    return s;
}

ShmSink::~ShmSink() { close(); }

// Heartbeat thread (gap 1): not the callback, not the matcher. Every kHeartbeatIntervalMs.
void ShmSink::heartbeat_loop() {
    set_thread_name("shm-heartbeat");
    for (;;) {
        const uint32_t seen = hb_sig_.value();
        if (hb_stop_.load(std::memory_order_acquire)) break;
        hdr_->heartbeat_mono_ns.store(uint64_t(mono_ns()), std::memory_order_release);
        hdr_->heartbeat_count.fetch_add(1, std::memory_order_relaxed);
        hb_sig_.wait(seen, int64_t(kHeartbeatIntervalMs) * 1000000LL);
    }
}

uint64_t ShmSink::heartbeats() const { return hdr_ ? hdr_->heartbeat_count.load(std::memory_order_relaxed) : 0; }

void ShmSink::on_period(double period_ms, double tolerance_ms) {
    hdr_->frame_period_ms = period_ms;
    hdr_->tolerance_ms = tolerance_ms;
}

void ShmSink::on_pair(const PairRecord& rec, const uint16_t* depth_a, const uint16_t* depth_b) {
    const uint64_t n = rec.pair_seq;
    uint8_t* slot = base_ + kShmHeaderSize + (n % kShmSlots) * shm_slot_stride(image_bytes_);
    auto* sh = reinterpret_cast<ShmSlotHead*>(slot);
    sh->seq.store(2 * n + 1, std::memory_order_relaxed);   // odd: being written
    std::atomic_thread_fence(std::memory_order_release);
    seqlock_copy(slot + kShmRecordOffset, &rec, sizeof(PairRecord));
    seqlock_copy(slot + kShmImageOffset, depth_a, image_bytes_);
    seqlock_copy(slot + kShmImageOffset + image_bytes_, depth_b, image_bytes_);
    sh->seq.store(2 * n + 2, std::memory_order_release);   // complete
    hdr_->pairs_published.store(n + 1, std::memory_order_release);
    hdr_->notify_word.fetch_add(1, std::memory_order_release);
    shared_word_wake_all(&hdr_->notify_word);              // never blocks
}

void ShmSink::close() {
    if (closed_) return;
    closed_ = true;
    if (hb_th_.joinable()) {
        hb_stop_.store(true, std::memory_order_release);
        hb_sig_.notify();
        hb_th_.join();
    }
    bool replaced = false;
    if (hdr_) {
        uint32_t st = hdr_->writer_state.load(std::memory_order_acquire);
        replaced = st == kWriterReplaced;
        if (!replaced) hdr_->writer_state.store(kWriterStopped, std::memory_order_release);
        hdr_->notify_word.fetch_add(1, std::memory_order_release);
        shared_word_wake_all(&hdr_->notify_word);
    }
    if (base_) munmap(base_, size_);
    base_ = nullptr;
    hdr_ = nullptr;
    if (!replaced) shm_unlink(name_.c_str());   // replaced: the name belongs to the new writer
}

// ------------------------------------------------------------------------------------------------ reader
const char* ShmReader::result_name(Result r) {
    switch (r) {
        case Result::Ok: return "ok";
        case Result::Timeout: return "timeout";
        case Result::WriterStopped: return "writer_stopped";
        case Result::WriterNotAlive: return "writer_not_alive";
        case Result::WriterReplaced: return "writer_replaced";
    }
    return "?";
}

std::unique_ptr<ShmReader> ShmReader::from_mapping(const uint8_t* base, std::size_t size, bool owns, std::string& err) {
    std::unique_ptr<ShmReader> r(new ShmReader());
    r->base_ = base;
    r->size_ = size;
    r->owns_ = owns;
    r->hdr_ = reinterpret_cast<const ShmHeader*>(base);
    const uint64_t magic = magic_ref(r->hdr_).load(std::memory_order_acquire);
    if (magic != kShmMagic) {
        err = "wrong magic (writer not ready or not a d435 pair object)";
        return nullptr;
    }
    const uint32_t v = r->hdr_->layout_version;
    if (v != kShmLayoutVersion) {
        err = "shared-memory layout version " + std::to_string(v) + ", but this reader supports only version " +
              std::to_string(kShmLayoutVersion) + (v < kShmLayoutVersion ? " (writer is an older d435_capture)"
                                                                         : " (reader is older than the writer)");
        return nullptr;
    }
    // The object can be larger than total_size (macOS rounds shared memory up to whole pages).
    if (r->hdr_->header_size != kShmHeaderSize || r->hdr_->total_size > r->size_ || r->hdr_->slot_count == 0 ||
        r->hdr_->record_size < sizeof(PairRecord) ||
        r->hdr_->slots_offset + r->hdr_->slot_count * r->hdr_->slot_stride > r->size_) {
        err = "inconsistent header sizes";
        return nullptr;
    }
    r->instance_ = r->hdr_->writer_instance_id;
    r->attaches_ = 1;
    r->next_ = r->start_seq_ = r->hdr_->pairs_published.load(std::memory_order_acquire);
    return r;
}

std::unique_ptr<ShmReader> ShmReader::attach(const std::string& name, std::string& err) {
    int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
        err = errstr("shm_open");
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || std::size_t(st.st_size) < kShmHeaderSize) {
        err = "shared-memory object too small or not ready";
        ::close(fd);
        return nullptr;
    }
    void* p = mmap(nullptr, std::size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        err = errstr("mmap");
        return nullptr;
    }
    auto r = from_mapping(static_cast<const uint8_t*>(p), std::size_t(st.st_size), true, err);
    if (!r) {
        munmap(p, std::size_t(st.st_size));
        return nullptr;
    }
    r->name_ = name;
    return r;
}

std::unique_ptr<ShmReader> ShmReader::attach_in_process(const ShmSink& sink, std::string& err) {
    auto r = from_mapping(sink.mapping(), sink.mapping_size(), false, err);
    if (r) r->name_ = sink.shm_name();
    return r;
}

void ShmReader::release() {
    if (base_ && owns_) munmap(const_cast<uint8_t*>(base_), size_);
    base_ = nullptr;
    hdr_ = nullptr;
}

ShmReader::~ShmReader() { release(); }

bool ShmReader::reattach(std::string& err) {
    auto n = attach(name_, err);
    if (!n) return false;
    if (n->instance_ == instance_) {
        err = "still the same writer instance (no new writer yet)";
        return false;
    }
    release();
    base_ = n->base_;
    size_ = n->size_;
    owns_ = n->owns_;
    hdr_ = n->hdr_;
    instance_ = n->instance_;
    next_ = start_seq_ = n->next_;
    attaches_++;
    n->base_ = nullptr;   // the mapping now belongs to this reader
    n->hdr_ = nullptr;
    return true;
}

int64_t ShmReader::heartbeat_age_ns() const {
    return mono_ns() - int64_t(hdr_->heartbeat_mono_ns.load(std::memory_order_acquire));
}

bool ShmReader::writer_alive() const {
    if (hdr_->writer_state.load(std::memory_order_acquire) != kWriterRunning) return false;
    const uint32_t lt = hdr_->liveness_timeout_ms ? hdr_->liveness_timeout_ms : kLivenessTimeoutMs;
    return heartbeat_age_ns() <= int64_t(lt) * 1000000LL;
}

ShmReader::Result ShmReader::next(PairRecord& rec, uint16_t* depth_a, uint16_t* depth_b, int64_t timeout_ns,
                                  int64_t copy_delay_ns) {
    auto* notify = const_cast<std::atomic<uint32_t>*>(&hdr_->notify_word);
    const uint64_t slots = hdr_->slot_count;
    const std::size_t img = std::size_t(hdr_->image_bytes);
    const int64_t deadline = timeout_ns < 0 ? INT64_MAX : mono_ns() + timeout_ns;
    constexpr int64_t kCheckNs = 200000000;   // liveness check at least every 200 ms while waiting
    for (;;) {
        // Load the notify word before pairs_published: a pair published after this load also changes the word.
        const uint32_t w = notify->load(std::memory_order_acquire);
        const uint64_t pub = hdr_->pairs_published.load(std::memory_order_acquire);
        // Replaced has priority over unread pairs: a replaced writer may still run and publish, and the reader must
        // move to the new writer at once (otherwise a reader slower than the old writer would never leave it).
        if (hdr_->writer_state.load(std::memory_order_acquire) == kWriterReplaced) {
            if (next_ < pub) missing_ += pub - next_;
            next_ = pub;
            return Result::WriterReplaced;
        }
        if (next_ < pub) {
            if (pub - next_ > slots) {           // the oldest ones are already overwritten
                missing_ += pub - slots - next_;
                next_ = pub - slots;
            }
            const uint8_t* slot = base_ + hdr_->slots_offset + (next_ % slots) * hdr_->slot_stride;
            auto* sh = reinterpret_cast<const ShmSlotHead*>(slot);
            const uint64_t want = 2 * next_ + 2;
            const uint64_t s1 = sh->seq.load(std::memory_order_acquire);
            if (s1 != want) {                    // overwritten by a newer pair (or being overwritten)
                missing_++;
                next_++;
                continue;
            }
            seqlock_copy(&rec, slot + hdr_->record_offset, sizeof(PairRecord));
            if (copy_delay_ns > 0) {   // test hook: only the first read attempt of this call is slowed down
                std::this_thread::sleep_for(std::chrono::nanoseconds(copy_delay_ns));
                copy_delay_ns = 0;
            }
            if (depth_a) seqlock_copy(depth_a, slot + hdr_->image_offset, img);
            if (depth_b) seqlock_copy(depth_b, slot + hdr_->image_offset + img, img);
            std::atomic_thread_fence(std::memory_order_acquire);
            const uint64_t s2 = sh->seq.load(std::memory_order_relaxed);
            if (s2 != s1) {                      // the writer changed the slot during the read
                discarded_++;
                missing_++;
                next_++;
                continue;
            }
            next_++;
            received_++;
            return Result::Ok;
        }
        // Nothing new: writer state, then liveness.
        const uint32_t st = hdr_->writer_state.load(std::memory_order_acquire);
        if (st == kWriterStopped) return Result::WriterStopped;
        if (st == kWriterReplaced) return Result::WriterReplaced;
        if (!writer_alive()) return Result::WriterNotAlive;
        const int64_t now = mono_ns();
        if (now >= deadline) return Result::Timeout;
        const int64_t left = deadline - now;
        shared_word_wait(notify, w, left < kCheckNs ? left : kCheckNs);
    }
}

}  // namespace cap
