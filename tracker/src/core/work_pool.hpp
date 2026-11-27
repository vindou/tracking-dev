// A small fixed pool for the hint trials (DESIGN.md section 13): the threads are created once, run() hands out job
// indices 0 .. n-1 to the pool threads and to the caller (worker 0) and returns when all are done. Nothing is
// allocated in run(). The order in which jobs are taken is not defined; callers must not depend on it.
#pragma once
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#ifdef __linux__
#include <pthread.h>
#endif

namespace trk {

class WorkPool {
public:
    using Fn = void (*)(void* ctx, int job, int worker);
    ~WorkPool() { stop(); }
    // threads: total number of workers including the caller of run(); 1 or less: no thread is created.
    void start(int threads) {
        stop();
        stop_ = false;
        for (int w = 1; w < threads; w++)
            th_.emplace_back([this, w] {
#ifdef __linux__
                pthread_setname_np(pthread_self(), "trk-hint");
#endif
                uint64_t seen = 0;
                for (;;) {
                    {
                        std::unique_lock<std::mutex> lk(m_);
                        cv_.wait(lk, [&] { return stop_ || gen_ != seen; });
                        if (stop_) return;
                        seen = gen_;
                    }
                    work(w);
                    {
                        std::lock_guard<std::mutex> lk(m_);
                        if (--pending_ == 0) done_.notify_one();
                    }
                }
            });
    }
    void stop() {
        {
            std::lock_guard<std::mutex> lk(m_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto& t : th_) t.join();
        th_.clear();
    }
    int workers() const { return int(th_.size()) + 1; }
    void run(int njobs, Fn fn, void* ctx) {
        {
            std::lock_guard<std::mutex> lk(m_);
            fn_ = fn;
            ctx_ = ctx;
            njobs_ = njobs;
            next_.store(0, std::memory_order_relaxed);
            pending_ = int(th_.size());
            gen_++;
        }
        cv_.notify_all();
        work(0);
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [&] { return pending_ == 0; });
    }

private:
    void work(int w) {
        for (;;) {
            const int j = next_.fetch_add(1, std::memory_order_relaxed);
            if (j >= njobs_) break;
            fn_(ctx_, j, w);
        }
    }
    std::vector<std::thread> th_;
    std::mutex m_;
    std::condition_variable cv_, done_;
    std::atomic<int> next_{0};
    Fn fn_ = nullptr;
    void* ctx_ = nullptr;
    int njobs_ = 0, pending_ = 0;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

}  // namespace trk
