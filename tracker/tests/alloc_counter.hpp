// Counts heap allocations (operator new) while counting is on. Replaces the global operator new in the test binary,
// except in sanitizer builds (the sanitizers provide their own); there alloc_counting_available() is false.
#pragma once
#include <atomic>
#include <cstdint>
bool alloc_counting_available();
void alloc_count_start();
uint64_t alloc_count_stop();   // allocations since start
