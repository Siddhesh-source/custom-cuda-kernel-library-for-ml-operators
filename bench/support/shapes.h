#ifndef CCML_BENCH_SUPPORT_SHAPES_H
#define CCML_BENCH_SUPPORT_SHAPES_H

#include <string>
#include <vector>

namespace ccml_bench {

/*
 * bench/shapes.json is the single source of truth for every benchmark shape,
 * shared by the C++ benches and scripts/bench_torch.py (reference.md §3
 * "shape list", §5 "target shapes plus odd shapes").
 *
 *   { "matmul":    [ {"tag": "...", "M": 1, "N": 2304, "K": 768}, ... ],
 *     "layernorm": [ {"tag": "...", "rows": 128, "cols": 768}, ... ],
 *     "softmax":   [ {"tag": "...", "rows": 1536, "cols": 128}, ... ] }
 *
 * Fields that do not apply to an operator are 0.
 */
struct ShapeEntry {
  std::string tag;
  long long   M = 0, N = 0, K = 0;
  long long   rows = 0, cols = 0;
};

/* Throws std::runtime_error with the offending position on malformed input. */
std::vector<ShapeEntry> load_shapes(const std::string& path, const std::string& op);

/* Parses JSON text instead of a file. Exposed for tests. */
std::vector<ShapeEntry> parse_shapes(const std::string& json, const std::string& op);

}  // namespace ccml_bench

#endif /* CCML_BENCH_SUPPORT_SHAPES_H */
