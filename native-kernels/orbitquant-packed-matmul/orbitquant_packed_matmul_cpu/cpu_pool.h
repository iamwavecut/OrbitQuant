#pragma once

#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

namespace orbitquant::cpu {

// Executes fn over each [start, end) range using a lazily created persistent
// worker pool plus the calling thread. Ranges are disjoint, so results stay
// deterministic regardless of which thread runs which range. A single range
// (or an empty pool) runs inline on the caller.
void run_ranges(
    std::vector<std::pair<std::int64_t, std::int64_t>> const &ranges,
    std::function<void(std::int64_t, std::int64_t)> const &fn);

}  // namespace orbitquant::cpu
