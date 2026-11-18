#include "shm_ring.hpp"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "../common/clock.hpp"
#include "../common/futex.hpp"

#if defined(__SANITIZE_THREAD__)
#define TRK_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define TRK_TSAN 1
#endif
#endif

namespace trk {

namespace {
std::string errstr(const char* what) { return std::string(what) + ": " + std::strerror(errno); }

uint64_t new_instance_id() {
    std::random_device rd;
    uint64_t v = (uint64_t(rd()) << 32) ^ uint64_t(rd());
    v ^= uint64_t(real_ns()) * 0x9E3779B97F4A7C15ULL;
    v ^= uint64_t(getpid()) << 17;
    return v ? v : 1;
}

std::atomic<uint64_t>& magic_ref(RingHeader* h) { return *reinterpret_cast<std::atomic<uint64_t>*>(&h->magic); }
const std::atomic<uint64_t>& magic_ref(const RingHeader* h) {
    return *reinterpret_cast<const std::atomic<uint64_t>*>(&h->magic);
}

// An object of this name exists: mark it replaced and wake its readers.
std::string mark_old(const std::string& name, uint64_t new_instance) {
    const int fd = shm_open(name.c_str(), O_RDWR, 0);
    if (fd < 0) return std::string("old object not opened: ") + std::strerror(errno);
    struct stat st;
    std::string what = "too small";
    if (fstat(fd, &st) == 0 && std::size_t(st.st_size) >= kRingHeaderSize) {
        void* p = mmap(nullptr, kRingHeaderSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p != MAP_FAILED) {
            auto* h = static_cast<RingHeader*>(p);
            if (magic_ref(h).load(std::memory_order_acquire) == kRingMagic) {
                char buf[128];
                std::snprintf(buf, sizeof buf, "instance %016llx, pid %d, state %u",
                              (unsigned long long)h->writer_instance_id, h->writer_pid,
                              h->writer_state.load(std::memory_order_acquire));
                what = buf;
                h->replaced_by_instance.store(new_instance, std::memory_order_relaxed);
                h->writer_state.store(kWriterReplaced, std::memory_order_release);
                h->notify_word.fetch_add(1, std::memory_order_release);
                futex_wake_all(&h->notify_word);
            } else {
                what = "not a ring object";
            }
            munmap(p, kRingHeaderSize);
        }
    }
    ::close(fd);
    return what;
}
}  // namespace

#if defined(TRK_TSAN)
__attribute__((no_sanitize("thread"))) void ring_seqlock_copy(void* dst, const void* src, std::size_t n) {
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
void ring_seqlock_copy(void* dst, const void* src, std::size_t n) { std::memcpy(dst, src, n); }
#endif

// ------------------------------------------------------------------------------------------------ writer
std::unique_ptr<RingWriter> RingWriter::create(const std::string& name, const char* record_type,
                                               uint32_t record_version, uint32_t record_size, uint32_t slot_count,
                                               std::string& err, std::string* note) {
    if (slot_count == 0 || record_size == 0 || std::strlen(record_type) >= 16) {
        err = "bad ring parameters";
        return nullptr;
    }
    std::unique_ptr<RingWriter> w(new RingWriter());
    w->name_ = name;
    w->instance_ = new_instance_id();
    const uint64_t stride = ring_slot_stride(record_size);
    w->size_ = std::size_t(kRingHeaderSize + stride * slot_count);
    int fd = shm_open(name.c_str(), O_CREAT | O_EXCL | O_RDWR, 0660);
    if (fd < 0 && errno == EEXIST) {
        const std::string what = mark_old(name, w->instance_);
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
    if (ftruncate(fd, off_t(w->size_)) != 0) {
        err = errstr("ftruncate");
        ::close(fd);
        shm_unlink(name.c_str());
        return nullptr;
    }
    void* p = mmap(nullptr, w->size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        err = errstr("mmap");
        shm_unlink(name.c_str());
        return nullptr;
    }
    w->base_ = static_cast<uint8_t*>(p);
    std::memset(w->base_, 0, w->size_);   // touches every page now
    RingHeader* h = reinterpret_cast<RingHeader*>(w->base_);
    h->layout_version = kRingLayoutVersion;
    h->header_size = kRingHeaderSize;
    h->slot_count = slot_count;
    h->record_size = record_size;
    h->slot_stride = stride;
    h->slots_offset = kRingHeaderSize;
    h->total_size = w->size_;
    std::snprintf(h->record_type, sizeof h->record_type, "%s", record_type);
    h->record_version = record_version;
    h->writer_pid = int32_t(getpid());
    h->writer_instance_id = w->instance_;
    h->heartbeat_interval_ms = kRingHeartbeatMs;
    h->liveness_timeout_ms = kRingLivenessMs;
    h->writer_start_real_ns = real_ns();
    h->heartbeat_mono_ns.store(uint64_t(mono_ns()), std::memory_order_relaxed);
    h->writer_state.store(kWriterRunning, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    magic_ref(h).store(kRingMagic, std::memory_order_release);   // last: header complete
    w->hdr_ = h;
    RingWriter* raw = w.get();
    w->hb_th_ = std::thread([raw] { raw->heartbeat_loop(); });
    return w;
}

RingWriter::~RingWriter() { close(); }

void RingWriter::heartbeat_loop() {
    for (;;) {
        const uint32_t seen = hb_word_.load(std::memory_order_acquire);
        if (hb_stop_.load(std::memory_order_acquire)) break;
        hdr_->heartbeat_mono_ns.store(uint64_t(mono_ns()), std::memory_order_release);
        hdr_->heartbeat_count.fetch_add(1, std::memory_order_relaxed);
        futex_wait(&hb_word_, seen, int64_t(kRingHeartbeatMs) * 1000000LL);
    }
}

void* RingWriter::begin() {
    const uint64_t n = next_;
    open_slot_ = base_ + kRingHeaderSize + (n % hdr_->slot_count) * hdr_->slot_stride;
    reinterpret_cast<std::atomic<uint64_t>*>(open_slot_)->store(2 * n + 1, std::memory_order_relaxed);
    std::atomic_thread_fence(std::memory_order_release);
    return open_slot_ + kRingRecordOffset;
}

uint64_t RingWriter::commit() {
    const uint64_t n = next_++;
    reinterpret_cast<std::atomic<uint64_t>*>(open_slot_)->store(2 * n + 2, std::memory_order_release);
    open_slot_ = nullptr;
    hdr_->records_published.store(n + 1, std::memory_order_release);
    hdr_->notify_word.fetch_add(1, std::memory_order_release);
    futex_wake_all(&hdr_->notify_word);
    return n;
}

uint64_t RingWriter::publish(const void* record) {
    void* dst = begin();
    ring_seqlock_copy(dst, record, hdr_->record_size);
    return commit();
}

void RingWriter::close() {
    if (closed_) return;
    closed_ = true;
    if (hb_th_.joinable()) {
        hb_stop_.store(true, std::memory_order_release);
        hb_word_.fetch_add(1, std::memory_order_release);
        futex_wake_all(&hb_word_);
        hb_th_.join();
    }
    bool replaced = false;
    if (hdr_) {
        replaced = hdr_->writer_state.load(std::memory_order_acquire) == kWriterReplaced;
        if (!replaced) hdr_->writer_state.store(kWriterStopped, std::memory_order_release);
        hdr_->notify_word.fetch_add(1, std::memory_order_release);
        futex_wake_all(&hdr_->notify_word);
    }
    if (base_) munmap(base_, size_);
    base_ = nullptr;
    hdr_ = nullptr;
    if (!replaced) shm_unlink(name_.c_str());
}

// ------------------------------------------------------------------------------------------------ reader
const char* RingReader::result_name(Result r) {
    switch (r) {
        case Result::Ok: return "ok";
        case Result::Timeout: return "timeout";
        case Result::WriterStopped: return "writer_stopped";
        case Result::WriterNotAlive: return "writer_not_alive";
        case Result::WriterReplaced: return "writer_replaced";
    }
    return "?";
}

std::unique_ptr<RingReader> RingReader::from_mapping(const uint8_t* base, std::size_t size, bool owns,
                                                     const char* type, uint32_t version, uint32_t rsize,
                                                     std::string& err) {
    std::unique_ptr<RingReader> r(new RingReader());
    r->base_ = base;
    r->size_ = size;
    r->owns_ = owns;
    r->type_ = type;
    r->version_ = version;
    r->rsize_ = rsize;
    r->hdr_ = reinterpret_cast<const RingHeader*>(base);
    if (magic_ref(r->hdr_).load(std::memory_order_acquire) != kRingMagic) {
        err = "wrong magic (writer not ready or not a tracker ring)";
        r->owns_ = false;
        return nullptr;
    }
    const RingHeader& h = *r->hdr_;
    char t[17];
    std::memcpy(t, h.record_type, 16);
    t[16] = 0;
    if (h.layout_version != kRingLayoutVersion) {
        err = "ring layout version " + std::to_string(h.layout_version) + ", reader supports only " +
              std::to_string(kRingLayoutVersion);
    } else if (std::strcmp(t, type) != 0 || h.record_version != version || h.record_size != rsize) {
        err = std::string("record type/version/size ") + t + "/" + std::to_string(h.record_version) + "/" +
              std::to_string(h.record_size) + ", expected " + type + "/" + std::to_string(version) + "/" +
              std::to_string(rsize);
    } else if (h.header_size != kRingHeaderSize || h.slot_count == 0 || h.slot_stride < kRingRecordOffset + rsize ||
               h.slots_offset + h.slot_count * h.slot_stride > size) {
        err = "inconsistent ring header";
    } else {
        r->instance_ = h.writer_instance_id;
        r->attaches_ = 1;
        r->next_ = h.records_published.load(std::memory_order_acquire);
        return r;
    }
    r->owns_ = false;
    return nullptr;
}

std::unique_ptr<RingReader> RingReader::attach(const std::string& name, const char* type, uint32_t version,
                                               uint32_t rsize, std::string& err) {
    const int fd = shm_open(name.c_str(), O_RDONLY, 0);
    if (fd < 0) {
        err = errstr("shm_open");
        return nullptr;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || std::size_t(st.st_size) < kRingHeaderSize) {
        err = "object too small or not ready";
        ::close(fd);
        return nullptr;
    }
    void* p = mmap(nullptr, std::size_t(st.st_size), PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        err = errstr("mmap");
        return nullptr;
    }
    auto r = from_mapping(static_cast<const uint8_t*>(p), std::size_t(st.st_size), true, type, version, rsize, err);
    if (!r) {
        munmap(p, std::size_t(st.st_size));
        return nullptr;
    }
    r->name_ = name;
    return r;
}

std::unique_ptr<RingReader> RingReader::attach_in_process(const RingWriter& w, const char* type, uint32_t version,
                                                          uint32_t rsize, std::string& err) {
    auto r = from_mapping(w.mapping(), w.mapping_size(), false, type, version, rsize, err);
    if (r) r->name_ = w.name();
    return r;
}

void RingReader::release() {
    if (base_ && owns_) munmap(const_cast<uint8_t*>(base_), size_);
    base_ = nullptr;
    hdr_ = nullptr;
}

RingReader::~RingReader() { release(); }

bool RingReader::reattach(std::string& err) {
    auto n = attach(name_, type_.c_str(), version_, rsize_, err);
    if (!n) return false;
    if (n->instance_ == instance_) {
        err = "still the same writer instance";
        return false;
    }
    release();
    base_ = n->base_;
    size_ = n->size_;
    owns_ = n->owns_;
    hdr_ = n->hdr_;
    instance_ = n->instance_;
    next_ = n->next_;
    attaches_++;
    n->base_ = nullptr;
    n->hdr_ = nullptr;
    return true;
}

int64_t RingReader::heartbeat_age_ns() const {
    return mono_ns() - int64_t(hdr_->heartbeat_mono_ns.load(std::memory_order_acquire));
}

bool RingReader::writer_alive() const {
    if (hdr_->writer_state.load(std::memory_order_acquire) != kWriterRunning) return false;
    const uint32_t lt = hdr_->liveness_timeout_ms ? hdr_->liveness_timeout_ms : kRingLivenessMs;
    return heartbeat_age_ns() <= int64_t(lt) * 1000000LL;
}

// Reads record n into rec. false: overwritten or changed during the read.
bool RingReader::read_slot(uint64_t n, void* rec, int64_t copy_delay_ns) {
    const uint8_t* slot = base_ + hdr_->slots_offset + (n % hdr_->slot_count) * hdr_->slot_stride;
    auto* seq = reinterpret_cast<const std::atomic<uint64_t>*>(slot);
    const uint64_t s1 = seq->load(std::memory_order_acquire);
    if (s1 != 2 * n + 2) return false;
    if (copy_delay_ns > 0) {   // test hook: copy the first half, sleep, copy the rest
        ring_seqlock_copy(rec, slot + kRingRecordOffset, rsize_ / 2);
        std::this_thread::sleep_for(std::chrono::nanoseconds(copy_delay_ns));
        ring_seqlock_copy(static_cast<uint8_t*>(rec) + rsize_ / 2, slot + kRingRecordOffset + rsize_ / 2,
                          rsize_ - rsize_ / 2);
    } else {
        ring_seqlock_copy(rec, slot + kRingRecordOffset, rsize_);
    }
    std::atomic_thread_fence(std::memory_order_acquire);
    return seq->load(std::memory_order_relaxed) == s1;
}

RingReader::Result RingReader::next(void* rec, int64_t timeout_ns, int64_t copy_delay_ns) {
    auto* notify = const_cast<std::atomic<uint32_t>*>(&hdr_->notify_word);
    const uint64_t slots = hdr_->slot_count;
    const int64_t deadline = timeout_ns < 0 ? INT64_MAX : mono_ns() + timeout_ns;
    constexpr int64_t kCheckNs = 200000000;
    for (;;) {
        const uint32_t w = notify->load(std::memory_order_acquire);
        const uint64_t pub = hdr_->records_published.load(std::memory_order_acquire);
        if (hdr_->writer_state.load(std::memory_order_acquire) == kWriterReplaced) {
            if (next_ < pub) missing_ += pub - next_;
            next_ = pub;
            return Result::WriterReplaced;
        }
        if (next_ < pub) {
            if (pub - next_ > slots) {
                missing_ += pub - slots - next_;
                next_ = pub - slots;
            }
            const uint8_t* slot = base_ + hdr_->slots_offset + (next_ % slots) * hdr_->slot_stride;
            const uint64_t s1 = reinterpret_cast<const std::atomic<uint64_t>*>(slot)->load(std::memory_order_acquire);
            if (s1 != 2 * next_ + 2) {
                missing_++;
                next_++;
                continue;
            }
            const bool ok = read_slot(next_, rec, copy_delay_ns);
            copy_delay_ns = 0;
            next_++;
            if (!ok) {
                discarded_++;
                missing_++;
                continue;
            }
            received_++;
            return Result::Ok;
        }
        const uint32_t st = hdr_->writer_state.load(std::memory_order_acquire);
        if (st == kWriterStopped) return Result::WriterStopped;
        if (!writer_alive()) return Result::WriterNotAlive;
        const int64_t now = mono_ns();
        if (now >= deadline) return Result::Timeout;
        const int64_t left = deadline - now;
        futex_wait(notify, w, left < kCheckNs ? left : kCheckNs);
    }
}

RingReader::Result RingReader::latest(void* rec) {
    for (int attempt = 0; attempt < 4; attempt++) {
        const uint64_t pub = hdr_->records_published.load(std::memory_order_acquire);
        const uint32_t st = hdr_->writer_state.load(std::memory_order_acquire);
        if (st == kWriterReplaced) return Result::WriterReplaced;
        if (pub == 0 || pub <= next_) {
            if (st == kWriterStopped) return Result::WriterStopped;
            if (!writer_alive()) return Result::WriterNotAlive;
            return Result::Timeout;
        }
        if (read_slot(pub - 1, rec, 0)) {
            next_ = pub;
            received_++;
            return Result::Ok;
        }
        discarded_++;
    }
    return Result::Timeout;
}

}  // namespace trk
