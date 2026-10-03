# fastpoly

`fastpoly` 是一个 **header-only** 的 C++20 多项式模板库：模数 `mod` 取 32 位 NTT 友好素数
（`mod - 1` 含大 2 幂因子），所有无限长结果都要求显式给出截断长度 `n`，在 `mod x^n` 下计算。

- 多项式基本运算：加、减、取负、数乘、平移 `x^k`、求值、微分、积分、`trim`
- 卷积（完整 / 截断 `mod x^n`）
- 求逆 `inv`、对数 `log`、指数 `exp`、开方 `sqrt`、幂 `pow`、带余除法 `divmod`（`div`/`mod`）
- NTT 核心：**无位反转**的 radix-4 DIF/DIT 对，SIMD 化蝴蝶
- SIMD 后端：**AVX2**（x86-64）、**AVX-512F/VL**（可选）、**NEON**（arm64）、标量兜底

```
include/fastpoly/modint.hpp   Montgomery 模数层
include/fastpoly/simd.hpp     AVX2 / AVX-512 / NEON / 标量 的向量原语
include/fastpoly/ntt.hpp      NTT plan（预计算 twiddle、按 n 缓存共享）
include/fastpoly/poly.hpp     多项式模板与全部算术
include/fastpoly/fastpoly.hpp 总入口
tests/  bench/  scripts/run-tests.sh
```

## 快速开始

```cpp
#include "fastpoly/fastpoly.hpp"

using ctx = fpx::mod998244353;     // 其他模数见 modint.hpp 末尾的类型别名
using Poly = fpx::Poly<ctx>;

int main() {
  Poly a = ctx::from_ints({1, 2, 3});      // 1 + 2x + 3x^2
  Poly b{1, 1};                            // 1 + x

  Poly c = a * b;                          // 卷积
  Poly q = a.inv(10);                      // 1/a  mod x^10
  Poly l = b.log(10);                      // log(b) mod x^10
  Poly e = l.exp(10);                      // exp(log(b)) == b mod x^10
  Poly s = a.sqrt(10);                     // 开方，非二次剩余会抛 domain_error
  Poly p = a.pow(1000003, 64);             // a^k mod x^64（k 可为任意 64 位整数）
  auto [quot, rem] = a.divmod(b);          // a = quot*b + rem, deg(rem) < deg(b)
}
```

`Poly<M>` 继承 `std::vector<M>`，也可以用自由函数风格：`fpx::poly::conv(a, b, n)`、
`fpx::poly::inv(a, n)`、`fpx::poly::exp(a, n)` …

编译：单头文件、无外部依赖，C++20。

```bash
# x86-64：AVX2 不是 -O2 的默认目标，需要显式打开
clang++ -std=c++20 -O2 -mavx2 main.cpp
# arm64：NEON 属于 AArch64 基线，无需额外开关
clang++ -std=c++20 -O2 main.cpp
```

CMake（可选）：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./scripts/run-tests.sh          # 一次跑遍所有可用后端
```

## 算法与优化

**Montgomery 表示。** 所有元素以 `a·2^32 mod p` 存储，`modint.hpp` 里没有一次 64 位除法：
`mul` 退化成 `mul + madd + shift + 条件减`，这正是 SIMD 需要的形态。`R²` 常量在编译期由
Newton 迭代求 `-p^{-1} mod 2^32` 得到。

**SIMD 蝴蝶（`simd.hpp`）。** `mulmod` 是向量化的 Montgomery 约简：

| 后端 | 通道 | 实现要点 |
| --- | --- | --- |
| AVX2 | 8×u32 | `_mm256_mul_epu32` 取偶/奇双通道 64 位积，`_mm256_srli_epi64` 取 Montgomery 商，再 `or + slli` 拼回 8 通道 |
| AVX-512F/VL | 16×u32 | 同上，加宽到 512 位，条件减用掩码 `_mm512_mask_sub_epi32` |
| NEON | 4×u32 | `vmull_u32` 半宽扩展乘 + `vmovn_u64`/`vshrn_n_u64` 取低/高半 |
| 标量 | 1 | 参照实现，也是小规模的兜底路径 |

加/减用 `min_epu32(r, r-mod)` 与算术右移掩码完成，全程无分支。

**无位反转的 radix-4 NTT（`ntt.hpp`）。** 正向走 DIF、逆向走配对的 DIT，两者复合为
`n·I`，因此

* 完全不需要位反转置换（省掉一趟 O(n) 随机访存）；
* 点值乘法在同一个 base-4 位反序下进行，顺序对卷积没有影响。

radix-4 把两级 radix-2 合并进一个蝴蝶：乘法次数与 radix-2 相同，但**访存趟数和 twiddle
流量减半**。推导中有一个省事的结论：蝴蝶里那个固定的 `w^m` 恒等于 `root_n^(n/4)`，与阶段
无关，全阶段共用同一个 4 次单位根常量（逆向用它的逆，即取负）。

twiddle 按阶段预计算成**连续数组** `(w^k, w^2k, w^3k)` 及其逆，所以 SIMD 内核是纯向量加载；
`log2(n)` 为奇数时最后补一级 len=2 的 radix-2（该蝴蝶是自转置的，正逆共用同一份代码）。
plan 按 `n` 缓存并共享，尺寸上限由 `v2(mod-1)` 决定，越界直接抛 `ntt_size_error`。

**Newton 迭代 + middle product。** `inv` 在 `m → m2` 步只用到 `a·b` 的**高半部分**，而高半
在变换长度 `2m` 下不受循环卷积混叠影响（混叠只污染下标 `< m` 的低半）——于是变换长度可以
留在 `2m` 而不是 `4m`，每次迭代的变换规模减半，整体约 2 倍加速。`log`/`exp`/`sqrt` 都建立在
这一 `inv` 上，`exp`/`sqrt` 的每一步分别是 `b(1 + a - log b)` 与 `(b + a/b)/2`。
`pow` 对小指数走截断快速幂，对大指数走 `exp(k·log a)`：对 `j < n ≤ 2^23 ≪ mod`，二项式系数
`binom(k, j)` 只依赖 `k mod mod`，所以 `k` 可以任意大。小规模卷积自动退回 O(nm) 朴素实现
（阈值 40）。

## 验证

`./scripts/run-tests.sh` 会按本机可用性依次构建并运行：原生架构、`-march=native`、
ASan+UBSan，以及在 Apple Silicon 上通过 **Rosetta 交叉运行 x86-64 的 AVX2 版本**
（macOS SDK 是 universal 的，clang 能直接产出 x86-64 二进制）。

测试不依赖库自身的自洽性，而是对照**独立定义**：

* 模数层：与 `__uint128_t` 参考实现逐位比对（含 `mod-1`、`0`、`mod-2` 等边界）；
* NTT：往返恒等；单项式 `x^j` 变换后必须是 `{w^{jk}}` 的多重集（这把变换钉死为**真正的
  DFT**，而不只是"自洽的正逆对"）；与 O(n²) 卷积逐系数比对；
* 多项式：`a·inv(a) ≡ 1`；`log(exp(b)) == b` 且 `exp(log(a)) == a`；`exp' = exp·a'`；
  `sqrt(a)² == a`；`pow` 与朴素连乘/`a^k·a == a^{k+1}` 一致；`divmod` 还原 `a = q·b + r`
  且 `deg r < deg b`。另有一个 O(n²) 级数 `exp` 递推作为独立参照。

当前状态：6 个 NTT 模数 × {NEON, AVX2, 标量} 全绿，ASan/UBSan 无告警。

## 性能

Apple M2 Pro（arm64 / NEON，4 通道），`mod = 998244353`，单线程：

```
  NTT n=1024      forward    0.004 ms   inverse    0.004 ms   0.38 ns/(elem*log n)
  NTT n=16384     forward    0.073 ms   inverse    0.077 ms   0.32 ns/(elem*log n)
  NTT n=262144    forward    1.425 ms   inverse    1.506 ms   0.30 ns/(elem*log n)

  conv  n=16384        0.502 ms      conv  n=262144     11.818 ms

  series n=262144   inv  14.9  log  25.7  exp  67.6  sqrt  47.9  pow  93.7 ms
```

x86-64 的 AVX2 路径只验证了**正确性**（经 Rosetta），未测真实性能；在原生 x86-64 上
8 通道的 AVX2 预期比 NEON 快约 1.5–2 倍，有 AVX-512 的机器再翻倍。

## 边界与已知限制

* `mod < 2^31`（32 位有符号通道内比较所需）；SIMD 内核按通道宽度做无符号/有符号比较，
  已在 `1224736769` 这类接近 `2^30` 的模数上验证。
* NTT 长度上限 `2^{v2(mod-1)}`：`998244353` 为 `2^23`，`469762049` 为 `2^26`。
* `integral`/`log` 需要 `n < mod`（`1/i` 的线性递推前提），在 NTT 长度范围内自动满足。
* `log` 要求常数项为 1、`exp` 要求常数项为 0，否则抛 `domain_error`；`sqrt` 会在常数项为
  非二次剩余或 x-adic 赋值（valuation）为奇数时抛错。
