// Background load for the revised exp 4: N threads, each copies an 8 MB buffer in a loop
// (memory-bandwidth + CPU load, a pessimistic stand-in for a second camera pipeline + matcher).
// Usage: cpuload N SECONDS   -> prints GB/s copied per thread at the end.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>
#include <atomic>
int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : 1; double secs = argc > 2 ? atof(argv[2]) : 10;
    std::atomic<bool> run{true}; std::vector<double> gb(n); std::vector<std::thread> th;
    for (int i = 0; i < n; i++) th.emplace_back([&, i] {
        std::vector<char> a(8 << 20, 1), b(8 << 20); double bytes = 0;
        while (run.load(std::memory_order_relaxed)) { memcpy(b.data(), a.data(), a.size()); a[0] = b[1]; bytes += a.size(); }
        gb[i] = bytes / 1e9; });
    std::this_thread::sleep_for(std::chrono::duration<double>(secs)); run = false;
    for (auto& t : th) t.join();
    for (int i = 0; i < n; i++) printf("thread %d: %.2f GB/s\n", i, gb[i] / secs);
}
