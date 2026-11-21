// Turns off transparent huge pages for this process (Linux prctl, no privileges needed). The Jetson runs THP in
// "always" mode with khugepaged max_ptes_none = 511: khugepaged collapses any 2 MB range with one resident page into a
// huge page at a random time during its scan pass (about every 10 min), which raises the resident memory by up to
// 2 MB in one step and stalls the process while it copies. The tracker and the bridge allocate and touch all their
// memory at start; with THP off, their resident memory stays at that level.
#pragma once
#if defined(__linux__)
#include <sys/prctl.h>
#endif
#include <cstdio>

namespace trk {
inline void disable_transparent_huge_pages() {
#if defined(__linux__) && defined(PR_SET_THP_DISABLE)
    if (prctl(PR_SET_THP_DISABLE, 1, 0, 0, 0) != 0) std::perror("prctl(PR_SET_THP_DISABLE)");
#endif
}
}  // namespace trk
