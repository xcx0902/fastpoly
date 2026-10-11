// Independent transforms for partial outputs: unspecified scratch is never
// consumed as a later input. Only guaranteed output windows are checksummed.
// Arguments: prefix-quarter|prefix-half|highhalf N loops reps warmup.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>
#include "fastpoly/fastpoly.hpp"

int main(int argc, char** argv) {
  if (argc != 6) return 2;
  const std::string_view op = argv[1];
  if (op != "prefix-quarter" && op != "prefix-half" && op != "highhalf") return 2;
  const uint32_t n = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  const size_t loops = std::strtoull(argv[3], nullptr, 10);
  const int reps = std::atoi(argv[4]), warmup = std::atoi(argv[5]);
  if (n < 4 || (n & (n - 1)) || loops == 0 || loops > SIZE_MAX / n || reps < 1 || warmup < 0)
    return 2;
  using M = fpx::mod998244353;
  const auto& plan = fpx::NttPlan<M>::get_ref(n);
  std::vector<uint32_t> input(n * loops), a(input.size());
  for (size_t i = 0; i < input.size(); ++i)
    input[i] = M::from_int((uint64_t(i + 1) * 0x9e3779b9u) ^ 0x12345678u).raw_val();
  const uint32_t first = op == "highhalf" ? n / 2 : 0;
  const uint32_t last = op == "highhalf" ? n : op == "prefix-half" ? n / 2 : n / 4;
  std::vector<double> times;
  uint64_t checksum = 0;
  for (int r = -warmup; r < reps; ++r) {
    std::copy(input.begin(), input.end(), a.begin());
    const auto start = std::chrono::steady_clock::now();
    if (op == "highhalf") {
      for (size_t i = 0; i < a.size(); i += n) plan.inverse_highhalf(a.data() + i);
    } else {
      for (size_t i = 0; i < a.size(); i += n) plan.inverse_prefix(a.data() + i, last);
    }
    const auto stop = std::chrono::steady_clock::now();
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < a.size(); i += n)
      for (uint32_t j = first; j < last; ++j) {
        hash ^= a[i + j]; hash *= 1099511628211ull;
      }
    if (r != -warmup && hash != checksum) return 3;
    checksum = hash;
    if (r >= 0)
      times.push_back(std::chrono::duration<double, std::nano>(stop - start).count() / loops);
  }
  std::sort(times.begin(), times.end());
  const double median = (times[(times.size() - 1) / 2] + times[times.size() / 2]) / 2;
  std::printf("{\"operation\":\"%s\",\"size\":%u,\"loops\":%zu,\"reps\":%d,"
              "\"warmup\":%d,\"median_ns\":%.9f,\"best_ns\":%.9f,\"checksum\":\"%016llx\"}\n",
              argv[1], n, loops, reps, warmup, median, times.front(),
              static_cast<unsigned long long>(checksum));
}
