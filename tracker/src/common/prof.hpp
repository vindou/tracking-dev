// Section timers for profiling builds (cmake -DTRK_PROFILE=ON). Without TRK_PROFILE the macro expands to nothing.
// Times accumulate per context (what the cycle is doing: search step, hint trial, other) and section; one thread
// only (the profile is taken with the hint ICPs run sequentially). report() prints the table.
#pragma once
#ifdef TRK_PROFILE
#include <cstdint>
#include <cstdio>

#include "clock.hpp"

namespace trk {
namespace prof {
enum Context { kOther = 0, kSearch = 1, kHint = 2, kContexts = 3 };
enum Section {
    kBuild, kFilter, kCenters, kSelect, kTransform, kCorres, kAccumulate, kSolve, kFitCheck, kVisibility, kSections
};
struct State {
    int ctx = kOther;
    int64_t ns[kContexts][kSections] = {};
    int64_t calls[kContexts][kSections] = {};
    int64_t cycles[kContexts] = {};
};
inline State& state() { static State s; return s; }
struct Scope {
    int sec;
    int64_t t0;
    explicit Scope(int s) : sec(s), t0(mono_ns()) {}
    ~Scope() {
        State& s = state();
        s.ns[s.ctx][sec] += mono_ns() - t0;
        s.calls[s.ctx][sec]++;
    }
};
struct Ctx {   // sets the context for a scope and counts one cycle of that kind
    int prev;
    explicit Ctx(int c) : prev(state().ctx) { state().ctx = c; state().cycles[c]++; }
    ~Ctx() { state().ctx = prev; }
};
inline void report(FILE* f) {
    static const char* sn[kSections] = {"cloud building", "scene filter", "search start positions", "point selection", "point transform",
                                        "KD-tree queries", "normal equations", "6x6 solve", "fit check (eigenvalues, covariance)", "free-space check"};
    static const char* cn[kContexts] = {"outside search step and hint trial", "search step", "hint trial"};
    const State& s = state();
    for (int c = 0; c < kContexts; c++) {
        int64_t tot = 0;
        for (int k = 0; k < kSections; k++) tot += s.ns[c][k];
        if (!tot) continue;
        std::fprintf(f, "%s: %lld cycles, %.3f ms per cycle in the timed sections\n", cn[c], (long long)s.cycles[c],
                     s.cycles[c] ? double(tot) / 1e6 / double(s.cycles[c]) : 0.0);
        for (int k = 0; k < kSections; k++)
            if (s.calls[c][k])
                std::fprintf(f, "  %-38s %5.1f %%  %9.3f ms/cycle  %8.2f us/call  %lld calls\n", sn[k], 100.0 * double(s.ns[c][k]) / double(tot),
                             s.cycles[c] ? double(s.ns[c][k]) / 1e6 / double(s.cycles[c]) : 0.0, double(s.ns[c][k]) / 1e3 / double(s.calls[c][k]),
                             (long long)s.calls[c][k]);
    }
}
}  // namespace prof
}  // namespace trk
#define TRK_PROF_CAT2(a, b) a##b
#define TRK_PROF_CAT(a, b) TRK_PROF_CAT2(a, b)
#define TRK_PROF(section) ::trk::prof::Scope TRK_PROF_CAT(trk_prof_scope_, __LINE__)(::trk::prof::section)
#define TRK_PROF_CTX(context) ::trk::prof::Ctx TRK_PROF_CAT(trk_prof_ctx_, __LINE__)(::trk::prof::context)
#else
#define TRK_PROF(section)
#define TRK_PROF_CTX(context)
#endif
