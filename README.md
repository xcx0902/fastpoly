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
include/fastpoly/memory.hpp   64 字节对齐存储、按线程复用的有界临时内存池
include/fastpoly/ntt.hpp      NTT plan（预计算 twiddle、按 n 缓存共享）
include/fastpoly/poly.hpp     多项式模板与全部算术
include/fastpoly/fastpoly.hpp 总入口
fastpoly.hpp                自动生成的单文件发行版
tests/  bench/  scripts/run-tests.sh  scripts/amalgamate.py
```

## 快速开始

```cpp
#include "fastpoly/fastpoly.hpp"

using ctx = fpx::mod998244353;     // 其他模数见 modint.hpp 末尾的类型别名
using Poly = fpx::Poly<ctx>;

int main() {
  Poly a = Poly::from_ints({1, 2, 3});     // 1 + 2x + 3x^2
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
| AVX-512F/VL | 16×u32 | 同上，加宽到 512 位 |
| NEON | 4×u32 | 用四通道 `vmulq_u32` 独立计算 Montgomery 数位，两条 widening multiply-add 链合并积与约简，`vuzp2q_u32` 提取高半；减少窄化重排和依赖 |
| 标量 | 1 | 参照实现，也是小规模的兜底路径 |

三处与"少做一次归约"有关的细节：

* **`add`/`sub` 各 3 条指令。** `add(a,b,R) = min(a+b, a+b-R)`（`a+b < 2R` 时无符号
  `min` 即条件减）；`sub(a,b,R) = min(a-b, a-b+R)`——`a<b` 时 `a-b` 回绕成巨大的无符号
  数、`a-b+R` 落在 `[0,R)`，`min` 恰好选后者。**不需要比较掩码、不需要算术右移**，比
  "移位取借位 + and + add" 少一条指令。`mulmod` 末尾的条件减同样用 `min`。
* **lazy reduction（`mod < 2^30`）。** 见下节。
* **未归约加法 `add_wide`。** 蝴蝶里唯一"只喂给乘以 twiddle 的乘法"的和可以完全不归约：
  两个 `< 2*mod` 的值相加 `< 4*mod`，而 `4*mod² < mod·2^32`（即 `mod < 2^30`）时这样的
  乘积仍落在 REDC 的合法域内。

**Lazy reduction。** 当 `mod < 2^30` 时（本库全部常用素数除 `1224736769` 外都满足），
中间量从 `[0, mod)` 放宽到 `[0, 2·mod)`，取 `R = 2·mod` 作为"约简模数"：

* `R` 是 `mod` 的倍数，所以按 `R` 归约得到的结果仍是合法的剩余类；
* `4·mod² < mod·2^32` 成立，于是**两个 `< 2·mod` 的值做 Montgomery 乘，结果自动落在
  `[0, 2·mod)`，末尾的条件减可以整条删掉**——蝴蝶的 4 次乘法每次都省 2 条指令。

`forward` 的最后一个 stage 与 `inverse` 的最后一级蝶形负责把数据还原成 `[0, mod)` 的标准
代表。内部卷积用 `forward_lazy` 保留 `[0, 2·mod)`，直接交给点值乘法和逆变换，省去中途规范化。

"lazy lazy reduction"（再放宽一级）在本库不划算：若要把不变式放到 `[0, 4·mod)`，乘法需要
`16·mod² < mod·2^32`，即 `mod < 2^28`；`998244353 ≈ 2^29.9`、`469762049 ≈ 2^28.8` 都不满足，
而放宽一级对 `add`/`sub` 的指令数没有任何好处。真正能省下的是"只喂给 twiddle 乘法的那一个
加"——已由 `add_wide` 覆盖。

**无位反转的 radix-4 NTT（`ntt.hpp`）。** 正向走 DIF、逆向走配对的 DIT，两者复合为
`n·I`，因此

* 完全不需要位反转置换（省掉一趟 O(n) 随机访存）；
* 点值乘法在同一个 base-4 位反序下进行，顺序对卷积没有影响。

radix-4 把两级 radix-2 合并进一个蝴蝶：乘法次数与 radix-2 相同，但**访存趟数和 twiddle
流量减半**。推导中有一个省事的结论：蝴蝶里那个固定的 `w^m` 恒等于 `root_n^(n/4)`，与阶段
无关，全阶段共用同一个 4 次单位根常量（逆向用它的逆，即取负）。

twiddle 按阶段预计算成**连续数组** `(w^k, w^2k, w^3k)` 及其逆，所以 SIMD 内核是纯向量加载；
较大阶段的六组 twiddle 同时用独立 SIMD 乘法链生成，以 `w^lane` 为步长连续写入，隐藏乘法依赖；
每次更新仍归约到 `[0, mod)`。小阶段、剩余元素和标量后端沿用标量生成。
`log2(n)` 为奇数时最后补一级 len=2 的 radix-2（该蝴蝶是自转置的，正逆共用同一份代码，且用
`swap_pairs + pick_odd` 向量化成"成对互换 + 选择"三件套，不再是标量循环）。
plan 按 `log2(n)` 缓存并共享：每个尺寸用独立的 `call_once` 构建，不同尺寸可并发预计算；
线程缓存使热查找不经过互斥锁。内部用 `get_ref(n)` 借用缓存持有的 plan，避免热路径上的
`shared_ptr` 原子引用计数；原有 `get(n)` 接口保留。阶段长度和偏移用固定数组保存，twiddle
使用 64 字节对齐的未初始化存储，直接生成有效数据；各级单位根由上一层平方两次得到。
尺寸上限由 `v2(mod-1)` 决定，越界直接抛 `ntt_size_error`。

小阶段按最多 **4096 个连续元素** 分块，在一个缓存块内完成所有剩余层，再处理下一块。
末层 `len=4` 的三个 twiddle 都是 1，直接删除相应乘法。逆变换把 `1/n` 预乘进最后一级的
三个 twiddle，只额外缩放该蝶形的第一个输入；归一化乘法从 `n` 次降到 `n/4` 次，并省去单独遍历。

**小 `m` 阶段的 chunked 内核（访存连续性的关键）。** 通用蝴蝶要求子块宽度 `m = len/4` 至少
等于一个向量的通道数，否则整级都会掉进标量兜底。而 DIF 的**最后几级恰好是 `m = 4, 2, 1`**：
实测在 NEON 上 `len=4` 一级就占正向变换的 **36%**，`len=8` 一级占 **31%**——因为这几级对同一
个 k 只用一组标量 twiddle，标量蝴蝶的依赖链根本喂不饱流水线。

解决办法是把 `lane/m` 个连续 block 塞进一个向量：`simd::load4m<M>` 把窗口内每块的第
`r` 个 `M` 元子块搬到 `xj`，蝴蝶照常做，`store4m<M>` 精确还原。twiddle 向量按 lane 模式
复制（`tw[i] = A[i % M]`）——**同一行各 lane 共享同一个 k，所以 block 在 lane 内的顺序可以
任意，只要 store 用的排列一致**。各后端的 gather 恰好都是便宜操作：

| 后端 | M=1 | M=2 | M=4 |
| --- | --- | --- | --- |
| NEON | `ld4/st4` 一条指令 | 4 次 `vcombine`（64 位半重组） | — |
| AVX2 | 4×4 转置网络（8 条 unpack + 4 条 permute2x128） | 4 permute2x128 + 4 unpack64 | 4 条 permute2x128 |
| AVX-512 | 上面 AVX2 版本各做两个 256 位半宽再 `inserti64x4` 拼起来 | 同左 | 同左；M=8 用连续加载与 256 位半块交换 |

AVX-512 复用 AVX2 的 256 位网络（在 Rosetta 下可测），只是把两个半宽结果拼成 512 位。
`n < 4·lane` 时退回逐 block 标量，保证小尺寸正确。

**Newton 迭代 + middle product。** `inv` 在 `m → m2` 步只用到 `a·b` 的**高半部分**，而高半
在变换长度 `2m` 下不受循环卷积混叠影响（混叠只污染下标 `< m` 的低半）——于是变换长度可以
留在 `2m` 而不是 `4m`。求逆先用 32 项递推启动，后续复用两个变换缓冲区，把高半误差搬回
原缓冲区，不再分配中间的 `h` 和 `hv`。

求逆先检查有效输入前缀，忽略输出范围之外的系数和末尾零。线性输入的逆元是等比数列，
用四组独立 SIMD 链生成；次数不超过 4 的输入直接递推，`n >= 4096` 时放宽至次数 8。
这些阈值由实测选择，较高次数继续使用 Newton/NTT。

`exp` 和 `sqrt` 保存已有逆元前缀，随精度增长继续扩展。`exp` 从 `b'=a'b` 的高半残差出发，
除以 `b`、积分，再乘以缓存的 `b` 频谱，只补新增系数；`sqrt` 用 `(a-b²)/(2b)` 的高半修正。
两者的主变换长度均为 `2m`，消除了每轮从头求逆及完整 `log`/卷积。最终只剩至多 8 项时，
直接递推追加，避免刚越过二次幂就再跑一整组变换。短输入的 `exp` 用系数递推；常数开方直接返回。
线性 `exp` 单独生成相邻系数，先计算与前一项无关的乘法因子，避免通用求和循环。

`pow` 对小指数走截断快速幂，对大指数走 `exp(k·log a)`：对 `j < n ≤ 2^23 ≪ mod`，二项式系数
`binom(k, j)` 只依赖 `k mod mod`，所以 `k` 可以任意大。小规模卷积自动退回 O(nm) 朴素实现
（短边阈值 40），广播短边系数并连续 SIMD 扫描长边和结果。同一对象的平方只做一次正向变换。

**点值乘法与数乘向量化。** `conv`/`inv` 里的逐点 Montgomery 乘法是一整趟 O(n) 遍历，
标量实现下 `conv` 有约 15% 的时间花在这一趟上（`1.96 → 1.70 ns/(elem*log n)`）；换成
`simd::mulmod` 后只剩一趟向量遍历，`mul_scalar`、微分、积分和指数修正同理向量化。

**临时内存池与对齐。** 卷积的第二个频谱，以及 `inv`/`exp`/`sqrt` 的临时频谱和工作区，
使用内部 `ScratchBuffer`：连续存放原始 32 位 Montgomery 数位，首地址按 64 字节对齐。
按二次幂尺寸分桶，每个线程独立复用，不需要共享分配锁；同时存活的缓冲区拥有独立存储。
每次只复制有效系数、清零填充尾部，增长时不复制旧频谱；返回结果仍是 `std::vector<M>`，
卷积直接返回自己的结果缓冲区。Newton 输出预留最终容量，避免逐次扩容和搬运。
临时工作区也一次预留本次运算实际需要的最大 NTT 容量，倍增期间复用同一个地址；最后至多
8 项直接递推时，预留量保持在前一个二次幂，不申请额外的双倍工作区。

默认每线程最多缓存 **64 MiB 空闲存储**（包含缓存块头），每个尺寸最多保留四块；活跃缓冲区
和长期存活的 NTT plan 不计入该上限。线程退出时释放缓存，也可主动释放当前线程的空闲块：

```cpp
auto stats = fpx::scratch_memory_stats(); // 当前线程：缓存字节数、块数、系统申请数、命中数
fpx::release_scratch_memory();           // 不影响正在使用的缓冲区或 NTT plan
```

编译前定义 `FASTPOLY_SCRATCH_CACHE_BYTES` 可调整上限，设为 `0` 则不保留空闲块；所有翻译单元
必须使用一致的值。公开 `Poly` / `poly::vec` 继续兼容 `std::vector`。小指数幂保留中间乘积的
实际长度，到最终输出才补零；大指数幂原地数乘。常数幂直接生成，线性输入且新增系数下标小于
模数时用二项式递推，并在指数模 `mod` 后的有效次数结束。除法只反转商所需的高位系数，余数只计算低于
除数次数的部分，常数除数直接执行一次 SIMD 数乘。

## 验证

`./scripts/run-tests.sh` 会按本机可用性依次构建并运行：原生架构、`-march=native`、
ASan+UBSan，以及在 Apple Silicon 上通过 **Rosetta 交叉运行 x86-64 的 AVX2 版本**
（macOS SDK 是 universal 的，clang 能直接产出 x86-64 二进制）。

测试不依赖库自身的自洽性，而是对照**独立定义**：

* 模数层：与 `__uint128_t` 参考实现逐位比对（含 `mod-1`、`0`、`mod-2` 等边界）；
* NTT：往返恒等；单项式 `x^j` 变换后必须是 `{w^{jk}}` 的多重集（这把变换钉死为**真正的
  DFT**，而不只是"自洽的正逆对"）——该检查是 O(n log n) 的，因此单独有一条 `2^17` 规模的
  用例，专门覆盖"chunked 小 `m` 内核接管之后"的大尺寸、以及 `log2(n)` 奇偶两种 `m` 序列；
  与 O(n²) 卷积逐系数比对；
* 多项式：`a·inv(a) ≡ 1`；`log(exp(b)) == b` 且 `exp(log(a)) == a`；`exp' = exp·a'`；
  `sqrt(a)² == a`；`pow` 与朴素连乘/`a^k·a == a^{k+1}` 一致；`divmod` 还原 `a = q·b + r`
  且 `deg r < deg b`。另有一个 O(n²) 级数 `exp` 递推作为独立参照。

另有惰性表示范围、未对齐缓冲区、并发冷 plan 构建、窄卷积、平方别名、非整幂截断及带赋值开方
的边界回归。单文件发行版由维护头文件生成，用同一套测试验证：

`test_memory` 还覆盖缓存污染后的六模数独立递推对照、分配复用计数、缓存上限、嵌套所有权、
跨线程移动、异常安全和线程/静态析构顺序；可用 `-DFASTPOLY_SCRATCH_CACHE_BYTES=0` 验证禁用缓存。

```bash
python3 scripts/amalgamate.py
python3 scripts/amalgamate.py --check
FASTPOLY_TEST_SINGLE_HEADER=1 ./scripts/run-tests.sh
```

## 性能

### 2026-10-04：对齐内存池与运算路径优化

基线是本轮修改前的 `57b9540`。本机 Apple Silicon / NEON，Clang 21、
`-std=c++20 -O2 -DNDEBUG`、单线程。两版使用同一份基准源码和确定性输入，交替运行三轮，
每轮每项预热 2 次、测量 7 次，取各轮中位数的中位数。64 至 `2^20` 的 225 组输出 checksum
全部一致，完整数据见 `bench/performance-20261004.csv`。

| 操作（n = 2^20） | 本轮优化前 ms | 本轮优化后 ms | 加速 |
| --- | ---: | ---: | ---: |
| NTT 正向 | 2.801 | 2.804 | 1.00× |
| NTT 逆向 | 3.164 | 3.068 | 1.03× |
| 卷积 | 24.048 | 20.881 | 1.15× |
| 求逆 | 32.788 | 28.753 | 1.14× |
| log | 58.659 | 54.428 | 1.08× |
| exp | 70.611 | 63.497 | 1.11× |
| sqrt | 47.859 | 43.403 | 1.10× |
| pow（指数 1000003） | 130.840 | 119.133 | 1.10× |
| 线性求逆 `1/(1-x)` | 32.138 | 0.588 | 54.68× |
| 线性 exp `exp(x)` | 12.697 | 6.090 | 2.09× |
| 短输入幂 `(1+x)^64` | 75.297 | 0.307 | 245.07× |
| 线性幂 `(1+x)^1000003` | 108.414 | 9.947 | 10.90× |
| 除法（除数次数 40） | 69.397 | 51.197 | 1.36× |

主扫描中出现微秒级回退的四项额外用 9 轮、每轮 1001 次、预热 50 次复测：128 项卷积
1.04×、128 项平方 1.00×、256 项 NTT 正向 1.00×、512 项求逆 1.05×；所有 checksum 一致。
这些小尺寸测量见 `bench/performance-20261004-small.csv`，主扫描原始结果仍完整保留。

冷 plan 另用 13 组交替的新进程测量，`2^20` 构建中位数 **1.699 → 1.179 ms（1.44×）**，
数据见 `bench/performance-20261004-plan.csv`。热查表使用借用接口 `get_ref` 后，4M 次查询
平均每次 **5.683 → 2.208 ns**（9 组交替测量，六个尺寸循环），记录在
`bench/performance-20261004-lookup.csv`。

临时池统计独立于计时，见 `bench/performance-20261004-memory.csv`：`n=2^20` 时首次
`inv`/`exp`/`sqrt` 分别只申请 **2 / 3 / 2** 个临时块；预热后的调用申请数均为 **0**，
分别保留约 **8 / 12 / 8 MiB** 空闲块。公开结果的 vector 分配和 NTT plan 存储不计入这些计数。

复现成对测量（使用当前基准源码分别编译旧、新头文件）：

```bash
mkdir -p /tmp/fastpoly-before
git archive 57b9540 include | tar -x -C /tmp/fastpoly-before
clang++ -std=c++20 -O2 -DNDEBUG -I/tmp/fastpoly-before/include bench/bench.cpp -o /tmp/fastpoly-before/bench
clang++ -std=c++20 -O2 -DNDEBUG -Iinclude bench/bench.cpp -o /tmp/fastpoly-after
python3 scripts/compare-bench.py --before /tmp/fastpoly-before/bench --after /tmp/fastpoly-after \
  --min-size 64 --max-size 1048576 --runs 3 --reps 7 --warmup 2 --output /tmp/fastpoly-comparison.csv
./build/fastpoly_bench --size 1048576 --op exp --reps 3 --warmup 2 --memory
./build/fastpoly_bench --size 1048576 --op exp --reps 3 --warmup 2 --memory --cold-scratch
```

### 2026-10-03：此前的 NTT 与级数优化

2026-10-03，本机 Apple Silicon（arm64 / NEON，4 通道），`mod = 998244353`，单线程。
Clang 21，`-std=c++20 -O2`；同一基准源码、同一确定性输入，每项预热 2 次、测量 7 次，
下表取**中位数**，plan 已预热。优化前为 `6e25c22`，优化后为 2026-10-03 的实现。

| 操作（n = 2^20） | 优化前 ms | 优化后 ms | 加速 |
| --- | ---: | ---: | ---: |
| NTT 正向 | 3.911 | 2.892 | 1.35× |
| NTT 逆向 | 4.181 | 3.231 | 1.29× |
| 卷积 | 27.479 | 20.319 | 1.35× |
| 求逆 | 39.475 | 31.009 | 1.27× |
| 对数 | 71.513 | 53.716 | 1.33× |
| 指数 | 185.451 | 66.031 | 2.81× |
| 开方 | 157.631 | 47.999 | 3.28× |
| 幂 | 256.031 | 130.155 | 1.97× |

`1024..1048576` 的每个二次幂均有覆盖，含奇数 `log2(n)`：88 组配对结果的输出校验和全部一致，
各组中位耗时均有改善。完整结果含每次测量组的最小值、中位数及输出校验和，保存在
[bench/performance-20261003.csv](bench/performance-20261003.csv)。不同运行之间仍可能有约 ±15% 波动。

专用路径（同一测量方式）：

| 场景 | 优化前 ms | 优化后 ms | 加速 |
| --- | ---: | ---: | ---: |
| 平方，n = 2^20 | 28.075 | 14.181 | 1.98× |
| 窄卷积，40 × 2^20 | 27.527 | 11.969 | 2.30× |
| exp(x)，n = 2^20 | 182.684 | 11.931 | 15.31× |
| 指数，n = 2^18 + 1 | 71.289 | 14.703 | 4.85× |
| 开方，n = 2^18 + 1 | 59.420 | 10.233 | 5.81× |

首次 `get(2^20)` 单独计时，在 9 组交替启动的新进程中取中位数：
**2.295 → 1.218 ms（1.89×）**。
此项包含 twiddle 分配与预计算，后续正向变换仅用于校验生成结果，计时范围只有 `get()`。
复现冷启动须每次启动新进程，禁止预热和在同一尺寸上重复测量：

```bash
/tmp/fastpoly_bench --size 1048576 --op plan --reps 1 --warmup 0 --csv
```

复现优化后的测量：

```bash
./build/fastpoly_bench 1048576 --op all --reps 7 --warmup 2 --csv
./build/fastpoly_bench --size 1048576 --op square --reps 7 --warmup 2 --csv
./build/fastpoly_bench --size 1048576 --op skinny --reps 7 --warmup 2 --csv
./build/fastpoly_bench --size 1048576 --op exp-linear --reps 7 --warmup 2 --csv
./build/fastpoly_bench --size 262145 --op series --reps 7 --warmup 2 --csv
```

测量表使用 `-O2`；要逐项对照上述数值，可直接构建：
`clang++ -std=c++20 -O2 -Iinclude bench/bench.cpp -o /tmp/fastpoly_bench`。
基准的输入复原、输出校验和计算及输出销毁不计入耗时。运算基准排除冷 plan 构建和线程启动；冷 plan 数据单独记录。
AVX2 的正确性通过 Rosetta 验证；AVX-512 通过编译检查，本机未做原生 x86 性能测量。

## 边界与已知限制

* `mod < 2^31`（32 位有符号通道内比较所需）；SIMD 内核按通道宽度做无符号/有符号比较，
  已在 `1224736769` 这类接近 `2^30` 的模数上验证。
* lazy reduction（中间量放宽到 `[0, 2·mod)`）要求 `mod < 2^30`；`1224736769` 不满足，
  自动退回全程 `[0, mod)` 的路径，结果一致、只是每次乘法多一条条件减。
* `inv_series` 用线性递推实现。`mod/i` 与 `mod%i` 共用一次 **32 位变除数整数除法**；普通
  整数商直接乘已有 Montgomery 原始值，再按常量模数归约，省去转入 Montgomery 表示及一次 REDC。
* NTT 长度上限 `2^{v2(mod-1)}`：`998244353` 为 `2^23`，`469762049` 为 `2^26`。
* `integral`/`log` 需要 `n < mod`（`1/i` 的线性递推前提），在 NTT 长度范围内自动满足。
* `log` 要求常数项为 1、`exp` 要求常数项为 0，否则抛 `domain_error`；`sqrt` 会在常数项为
  非二次剩余或 x-adic 赋值（valuation）为奇数时抛错。
