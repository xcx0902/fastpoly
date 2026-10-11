// Batch tiny transforms so clock resolution does not dominate their timings.
// Compile each header version separately with -std=c++20 -O2 -DNDEBUG -pthread
// and -include benchmark-qos.hpp. Arguments: forward|inverse N loops reps warmup.
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
  if (op != "forward" && op != "inverse") return 2;
  const uint32_t n = static_cast<uint32_t>(std::strtoul(argv[2], nullptr, 10));
  const size_t loops = std::strtoull(argv[3], nullptr, 10);
  const int reps = std::atoi(argv[4]), warmup = std::atoi(argv[5]);
  if (loops == 0 || reps < 1 || warmup < 0) return 2;
  using M = fpx::mod998244353;
  const auto& plan = fpx::NttPlan<M>::get_ref(n);
  std::vector<uint32_t> input(n), a(n);
  for (uint32_t i = 0; i < n; ++i)
    input[i] = M::from_int((uint64_t(i + 1) * 0x9e3779b9u) ^ 0x12345678u).raw_val();
  std::vector<double> times;
  uint64_t checksum = 0;
  for (int r = -warmup; r < reps; ++r) {
    a = input;
    const auto start = std::chrono::steady_clock::now();
    if (op == "forward") {
      for (size_t i = 0; i < loops; ++i) plan.forward(a.data());
    } else {
      for (size_t i = 0; i < loops; ++i) plan.inverse(a.data());
    }
    const auto stop = std::chrono::steady_clock::now();
    uint64_t hash = 1469598103934665603ull;
    for (uint32_t x : a) { hash ^= x; hash *= 1099511628211ull; }
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
