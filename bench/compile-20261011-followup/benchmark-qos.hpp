// Benchmark harness only: never included by the library.
#pragma once
#if defined(__APPLE__)
#include <pthread.h>
#include <pthread/qos.h>
#include <cstdio>
#include <cstdlib>
namespace fastpoly_benchmark_qos {
inline const bool configured = [] {
  if (pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0) != 0) {
    std::fputs("cannot set benchmark QoS\n", stderr);
    std::abort();
  }
  return true;
}();
}
#endif
