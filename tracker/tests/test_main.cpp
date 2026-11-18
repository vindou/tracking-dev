#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>

#include "test_framework.hpp"

namespace tf {
std::string& self_path() { static std::string p; return p; }
}  // namespace tf

int helper_ring_writer(int argc, char** argv);   // test_shm_ring.cpp

int main(int argc, char** argv) {
    tf::self_path() = argv[0];
    if (argc > 1 && std::strcmp(argv[1], "--helper-ring-writer") == 0) return helper_ring_writer(argc, argv);
    const char* filter = argc > 1 ? argv[1] : nullptr;
    int run = 0, failed = 0;
    for (auto& t : tf::registry()) {
        if (filter && !std::strstr(t.name, filter)) continue;
        const int before = tf::failures();
        const auto t0 = std::chrono::steady_clock::now();
        std::printf("[ RUN  ] %s\n", t.name);
        std::fflush(stdout);
        t.fn();
        const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        const bool ok = tf::failures() == before;
        std::printf("[ %s ] %s (%.2f s)\n", ok ? "PASS" : "FAIL", t.name, s);
        std::fflush(stdout);
        run++;
        if (!ok) failed++;
    }
    std::printf("%d tests, %d failed\n", run, failed);
    return failed;
}
