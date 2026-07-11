#include "packed_matmul_cpu.h"

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#pragma intrinsic(_xgetbv)
#endif
#endif

namespace orbitquant::cpu {
namespace {

bool runtime_has_avx2_fma_f16c() {
#if defined(_MSC_VER) && defined(_M_X64)
  int registers[4]{};
  __cpuid(registers, 1);
  const bool osxsave = (registers[2] & (1 << 27)) != 0;
  const bool avx = (registers[2] & (1 << 28)) != 0;
  const bool fma = (registers[2] & (1 << 12)) != 0;
  const bool f16c = (registers[2] & (1 << 29)) != 0;
  if (!osxsave || !avx || !fma || !f16c || (_xgetbv(0) & 0x6) != 0x6) {
    return false;
  }
  __cpuidex(registers, 7, 0);
  return (registers[1] & (1 << 5)) != 0;
#elif defined(__x86_64__)
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") &&
      __builtin_cpu_supports("f16c");
#else
  return false;
#endif
}

}  // namespace

bool packed_matmul_x86_avx2_available() {
  static const bool available = runtime_has_avx2_fma_f16c();
  return available;
}

}  // namespace orbitquant::cpu
