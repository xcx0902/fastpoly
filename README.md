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

**NEON 的固定乘数路径（`mod < 2^30`）。** NTT 的 twiddle、数乘和窄卷积里的短边系数
都是固定乘数，可以用预计算商代替一般 Montgomery 乘法。正输入的四通道乘法只需要
`sqdmulh + mul + mls` 三条指令；蝶形的差直接保留为有符号数，用 `sqrdmulh` 估商，
结果加一次 `mod` 即回到 `[0, 2·mod)`，省去乘法前的条件约减。

twiddle **只保存一个 32 位 Q31 商**，加载后用一次 `sqrdmulh` 精确恢复普通乘数，表的大小
仍和原来一样。若 `t` 是规范的 Montgomery 数位，`m = low32(t·NINV)`，则普通乘数
`w = (t+m·mod)/2^32` 满足 `w·2^31/mod = m/2 + t/(2·mod)`。因此正向需要的最近整数商
就是 `ceil(m/2)`，逆向需要的向下取整商就是 `m>>1`；生成时直接编码，不增加除法、额外表或
转换遍历。plan 的六条递推链同样使用固定乘数路径，每步仍还原成规范 Montgomery 数位。
使用固定乘数近似商和冗余表示的基础方法可参见
[David Harvey 的 NTT 算术讲义](https://web.maths.unsw.edu.au/~davidharvey/talks/fastntt-talk.pdf)。

**AVX2 / AVX-512 的固定乘数路径。** 固定乘数保存普通值 `w` 和 Q32 商
`q = floor(w·2^32/mod)`，用两次偶/奇通道 `mul_epu32` 估商，再用两次全通道
`mullo_epi32` 计算余数，替代通用 Montgomery 的六次宽乘。估商与低位乘积可并行；
`mod < 2^30` 时结果保留在 `[0, 2·mod)`，加法输入可直接放宽到 `< 4·mod`。
更大模数继续规范化。对于规范 Montgomery 数位 `t`，上述 `q` 恰好等于
`low32(t·NINV)`，普通值由 `(t+q·mod)>>32` 恢复，初始化不需要除法。

NTT 只在 `len <= 2048` 的阶段保存普通乘数与商两组连续平面；较大阶段仍用单字
Montgomery 表，限制带宽和内存增长。正逆两张表合计额外存储始终小于 **16 KiB**，
不增加分配次数。建表先用六条独立 Montgomery 链连续写入，再只转换这些缓存内的小平面，
避免把转换临时值带入大尺寸生成循环。AVX2 的双字蝶形循环单向量展开，减少 16 个寄存器
的压力；AVX-512 保留双向量展开。数乘、窄卷积和线性求逆也使用 Q32 固定乘法。

三处与"少做一次归约"有关的细节：

* **`add`/`sub` 各 3 条指令。** `add(a,b,R) = min(a+b, a+b-R)`（`a+b < 2R` 时无符号
  `min` 即条件减）；`sub(a,b,R) = min(a-b, a-b+R)`——`a<b` 时 `a-b` 回绕成巨大的无符号
  数、`a-b+R` 落在 `[0,R)`，`min` 恰好选后者。**不需要比较掩码、不需要算术右移**，比
  "移位取借位 + and + add" 少一条指令。`mulmod` 末尾的条件减同样用 `min`。
* **lazy reduction（`mod < 2^30`）。** 见下节。
* **未归约加法 `add_wide`。** Montgomery 蝴蝶里唯一"只喂给乘以 twiddle 的乘法"的和可以完全不归约：
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
加"——Montgomery 和 x86 Q32 路径由 `add_wide` 覆盖，NEON 固定乘数路径直接把和减去 `2·mod` 后估商。

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
`log2(n)` 为奇数时，未融合的后端最后补一级 len=2 的 radix-2（该蝴蝶是自转置的，正逆共用同一份代码，且用
`swap_pairs + pick_odd` 向量化成"成对互换 + 选择"三件套，不再是标量循环）。
plan 按 `log2(n)` 缓存并共享：每个尺寸用独立的 `call_once` 构建，不同尺寸可并发预计算；
线程缓存使热查找不经过互斥锁。内部用 `get_ref(n)` 借用缓存持有的 plan，避免热路径上的
`shared_ptr` 原子引用计数；原有 `get(n)` 接口保留。阶段长度和偏移用固定数组保存，twiddle
使用 64 字节对齐的未初始化存储，直接生成有效数据；各级单位根由上一层平方两次得到。
尺寸上限由 `v2(mod-1)` 决定，越界直接抛 `ntt_size_error`。

小阶段按最多 **4096 个连续元素** 分块（x86 为 **2048**），在一个缓存块内完成所有剩余层，再处理下一块。
NEON 的惰性路径和 x86 路径还把终端 `len=8` 的 radix-4 与 `len=2` 的 radix-2 合并，在向量寄存器
内完成，不再单独加载、写回末级；逆向按相反顺序在同一次加载中处理两级。
一次处理四个 8 点块时进一步按 `k=0/1` 分列：`k=0` 的三个 twiddle 全为 1，删除对应乘法；
配对的 radix-2 各用一整向量保存和、差，避免在相邻通道里重复计算。
x86 让偶通道的单位 twiddle 直接通过，只计算奇通道的三个宽乘，并预计算
`twiddle·NINV`，缩短乘法依赖；四次单位根在编译期生成，省去循环内常量加载。
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
| AVX2 | 4 次 256 位连续加载 + 8 条通道内 unpack；写回用逆网络 | 4 permute2x128 + 4 unpack64 | 4 条 permute2x128 |
| AVX-512 | 4 次 512 位连续加载 + 8 条通道内 unpack；写回用逆网络 | 4 shuffle_i64x2 + 4 unpack64；写回用 4 unpack64 + 4 permutex2var | 8 条 shuffle_i64x2；M=8 用连续加载与 256 位半块交换 |

AVX-512 的小阶段全部使用原生 512 位网络，避免半宽结果的提取和拼接。
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
NEON 在 `n >= 256` 时直接处理 `exp(c·x)[i]=c^i/i!`：十六条独立连乘链计算最后一个阶乘，
随后从高位向低位生成系数。每块用四组 SIMD 前缀乘积和一次端点扫描同时产生十六项，
只在块与块之间传递一个端点；不申请逆元表，临时空间固定为十六个数位和少量寄存器。

逆元表 `1/i` 在 `i > 65536` 后按相同的 `floor(mod/i)` 分组，每组只计算一次商和区间末端，
组内余数递减且全部引用已经完成的前缀。NEON 的小模数路径把四个独立乘法合并；其他后端
保留标量乘法并省去逐项除法。线性求逆遇到常数等比数列直接填充，短周期数列重复规范种子。

`pow` 对小指数走截断快速幂，对大指数走 `exp(k·log a)`：对 `j < n ≤ 2^23 ≪ mod`，二项式系数
`binom(k, j)` 只依赖 `k mod mod`，所以 `k` 可以任意大。小规模卷积自动退回 O(nm) 朴素实现
（短边阈值 40），广播短边系数并连续 SIMD 扫描长边和结果。同一对象的平方只做一次正向变换。
窄卷积在每个 4096 项输出块内完成全部累加，重复写入保持在缓存中；除第一块外，每个短边系数
从相同的输出地址开始写入。单项式系数乘法直接执行数乘，零系数不扫描长边。

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
的边界回归。单文件发行版由维护头文件生成，用同一套测试验证。

固定乘数测试对照独立整数定义，检查 Q31 编码、乘数恢复、和/差乘法及输出范围；额外覆盖
接近 `2^30` 的 `1073741789`、`1073479681`，以及接近 `2^31` 的 `2013265921`。
x86 另检查 Q32 编码和单位偶通道乘法；转置测试按独立下标公式验证每列及列间运算，
覆盖 `M=1/2/4/8/16`、各后端支持的宽度和多个未对齐偏移。
逆元表覆盖 65536 分组边界，线性指数逐项检查 `i·b[i] = c·b[i-1]`，窄卷积覆盖 4096 分块边界。

`test_memory` 还覆盖缓存污染后的六模数独立递推对照、分配复用计数、缓存上限、嵌套所有权、
跨线程移动、异常安全和线程/静态析构顺序；可用 `-DFASTPOLY_SCRATCH_CACHE_BYTES=0` 验证禁用缓存。

```bash
python3 scripts/amalgamate.py
python3 scripts/amalgamate.py --check
python3 tests/test_amalgamate.py
FASTPOLY_TEST_SINGLE_HEADER=1 ./scripts/run-tests.sh
```

无 AVX-512 硬件时，可以用 [SIMDe](https://github.com/simd-everywhere/simde) 的可移植语义执行
AVX-512 分支的全部四套测试（已验证 v0.8.2，无需给库增加依赖）：

```bash
FASTPOLY_SIMDE_DIR=/path/to/simde ./scripts/run-tests-simde.sh
FASTPOLY_SIMDE_DIR=/path/to/simde FASTPOLY_TEST_SINGLE_HEADER=1 ./scripts/run-tests-simde.sh
```

软件模型检查运算结果和范围；原生 AVX-512 的指令支持另用 `-mavx512f -mavx512vl`
编译检查，性能仍需在实际硬件上测量。

合并脚本按 C++ token 压缩空白，并把重复序列提取为临时宏；宏展开后的公共名称、
类型和实现保持不变。临时宏使用 Clang / GCC / MSVC 支持的 `push_macro` / `pop_macro`
保存和恢复调用方定义，且全部在头文件末尾清理。系统头文件在定义临时宏前包含。
生成结果确定，可用 `--check` 检查；Python 回归测试还对四种 SIMD 分支的预处理 token
与维护头文件逐一比对，覆盖配置覆盖、重复包含和临时宏泄漏。

## 性能

### 2026-10-09：AVX2 / AVX-512 固定乘法与全宽转置

基线为上一小节所述 NEON 优化完成后的工作区版本，本节报告 x86 路径的追加收益。
环境、参数及两版维护头文件的 SHA-256 见 [测量记录](bench/performance-20261009-x86.json)。
Apple Silicon 上通过 **Rosetta 执行 AVX2**，Clang 21，`-std=c++20 -O2 -DNDEBUG -mavx2`，
单线程；两版使用同一份基准源码、交替运行三轮。64 至 `2^18` 的主扫描每轮预热 5 次、
测量 9 次，共 234 组成对结果；`2^20` 每轮预热 3 次、测量 5 次，另测 12 组。
全部 checksum 一致，原始结果见 [主扫描](bench/performance-20261009-x86-avx2.csv) 和
[大尺寸数据](bench/performance-20261009-x86-large.csv)。

| 操作（n = 2^20） | 优化前 ms | 优化后 ms | 加速 |
| --- | ---: | ---: | ---: |
| NTT 正向 | 20.990 | 19.562 | 1.07× |
| NTT 逆向 | 24.467 | 21.163 | 1.16× |
| 卷积 | 142.362 | 128.885 | 1.10× |
| 平方 | 99.641 | 87.164 | 1.14× |
| 求逆 | 214.105 | 187.139 | 1.14× |
| log | 362.169 | 320.961 | 1.13× |
| exp | 445.390 | 391.233 | 1.14× |
| sqrt | 317.780 | 279.211 | 1.14× |
| pow（指数 1000003） | 808.154 | 722.229 | 1.12× |
| 窄卷积（40 × N） | 69.134 | 60.494 | 1.14× |
| 数乘（系数 7） | 1.473 | 1.205 | 1.22× |
| 线性求逆 `1/(1-7x)` | 1.157 | 0.907 | 1.28× |

主扫描的 NTT 正向、逆向几何平均加速分别为 **1.16×、1.21×**；通用级数约为
**1.16–1.17×**，线性求逆为 **1.29×**。没有耗时至少 5 微秒且回退超过 7% 的项目。
256 项常数等比求逆的纳秒级差异用 9 轮、每轮 1001 次、预热 50 次复测，得到
**0.167 → 0.167 微秒**，见 [小尺寸复测](bench/performance-20261009-x86-small.csv)。

另外三个模数在 `1024..131072` 上各测量 48 组，每轮预热 5 次、测量 7 次、三轮交替，
checksum 全部一致。`1073479681` 接近 `2^30`，`1224736769 > 2^30` 覆盖全归约路径；
下表为 `n = 2^17` 的加速，详见 [多模数数据](bench/performance-20261009-x86-moduli.csv)。

| mod | NTT 正向 | NTT 逆向 | 卷积 | 数乘 | 线性求逆 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 469762049 | 1.13× | 1.17× | 1.16× | 1.05× | 1.35× |
| 1073479681 | 1.13× | 1.20× | 1.12× | 1.26× | 1.37× |
| 1224736769 | 1.14× | 1.19× | 1.12× | 1.25× | 1.11× |

原生 x86 汇编用 [codegen.cpp](bench/codegen.cpp) 单独检查连续加载、写回和固定乘法，
指令计数见 [汇编统计](bench/performance-20261009-x86-codegen.csv)。Clang 21 的
`M=1` 列转置函数，AVX2 从 **36 → 21** 条指令，AVX-512 从 **69 → 21** 条；
AVX-512 对应写回从 **68 → 21**，`M=2` 写回从 **37 → 24**，`M=4` 写回从 **29 → 19**。
数乘 7 的宽乘从六次减少到两次，另有一次全通道低位乘；乘 7 被编译器改写为移位和减法。
指令数包含函数进出开销；实际吞吐取决于 CPU。

每个 plan 的正逆表额外商存储小于 **16 KiB**，`n = 2^20` 为 **8184 字节**，
仍只有两次 64 字节对齐表分配。冷建表用 13 组交替的新进程测量，见
[建表数据](bench/performance-20261009-x86-plan.csv)：`2^20` 为 **3.685 → 3.668 ms**，
1024 和 65536 项的首次构建分别增加约 **6.3、6.7 微秒**；缓存内商平面的转换总量有固定上限。
运算基准预热了 plan，此成本由尺寸缓存只付一次。
[临时池统计](bench/performance-20261009-x86-memory.csv) 确认 `conv`/`inv`/`exp`/`sqrt`
首次申请仍为 **1 / 2 / 3 / 2** 块，预热后均为 **0** 次系统申请；保留空间约
**8 / 8 / 12 / 8 MiB**，与基线一致。plan 和输出 vector 的分配不计入池内统计。

NEON 另做 63 组成对控制测量，几何平均为 **0.98–1.01×**，没有耗时至少 5 微秒且回退
超过 7% 的项目，见 [NEON 对照](bench/performance-20261009-x86-neon-control.csv)。
尝试后舍弃了全阶段双字表、较大缓存块和 x86 线性指数的向量前缀扫描；它们增加带宽或
没有稳定收益。六链建表保留单字 Montgomery 常量，商转换单独限制在小平面内，消除大尺寸建表回退。

维护头文件和单文件版均通过 NEON、`-march=native`、ASan+UBSan、Rosetta AVX2、
标量及 AVX-512 软件模型的全部四套测试；另通过原生 AVX-512F/VL 编译检查和
9 项单文件生成器测试。**本机没有原生 x86 的计时结果，AVX-512 未做硬件运行计时。**

用保存的对照版维护头文件复现主扫描（macOS 交叉编译命令；原生 x86 平台省略 `-arch x86_64`）：

```bash
clang++ -std=c++20 -O2 -DNDEBUG -arch x86_64 -mavx2 -I/path/to/before/include bench/bench.cpp -o /tmp/fastpoly-before-avx2
clang++ -std=c++20 -O2 -DNDEBUG -arch x86_64 -mavx2 -Iinclude bench/bench.cpp -o /tmp/fastpoly-after-avx2
python3 scripts/compare-bench.py --before /tmp/fastpoly-before-avx2 --after /tmp/fastpoly-after-avx2 \
  --min-size 64 --max-size 262144 --runs 3 --reps 9 --warmup 5 \
  --ops all,square,skinny,exp-linear,inv-short,pow-short,pow-linear,div-small,inv-series,scale,inv-linear \
  --output /tmp/fastpoly-x86-comparison.csv
clang++ -std=c++20 -O2 -DNDEBUG -arch x86_64 -mavx512f -mavx512vl -Iinclude bench/bench.cpp -o /tmp/fastpoly-after-avx512
clang++ -std=c++20 -O2 -arch x86_64 -mavx512f -mavx512vl -Iinclude -S bench/codegen.cpp -o /tmp/fastpoly-codegen-avx512.s
```

对照目录也可用 `git archive 3ab9f13 include` 提取，此时测得的是两轮优化的累计收益。

### 2026-10-09：小模数固定乘法与连续分块

基线为修改前的 `3ab9f13`。本机 Apple Silicon / NEON，`mod = 998244353`，Clang 21，
`-std=c++20 -O2 -DNDEBUG`，单线程。两版使用当前同一份基准源码，交替运行三轮，
每轮预热 5 次、测量 9 次，取各轮中位数的中位数。64 至 `2^20` 的 270 组成对结果
checksum 全部一致，完整记录见 [NEON 数据](bench/performance-20261009-neon.csv)。

| 操作（n = 2^20） | 优化前 ms | 优化后 ms | 加速 |
| --- | ---: | ---: | ---: |
| NTT 正向 | 2.763 | 2.088 | 1.32× |
| NTT 逆向 | 3.063 | 2.242 | 1.37× |
| 卷积 | 19.200 | 13.843 | 1.39× |
| 平方 | 13.348 | 9.679 | 1.38× |
| 求逆 | 28.231 | 21.031 | 1.34× |
| log | 49.576 | 37.013 | 1.34× |
| exp | 60.206 | 45.039 | 1.34× |
| sqrt | 42.181 | 31.337 | 1.35× |
| pow（指数 1000003） | 112.399 | 82.340 | 1.37× |
| 窄卷积（40 × N） | 12.886 | 6.819 | 1.89× |
| 线性 exp `exp(x)` | 5.161 | 1.304 | 3.96× |
| 逆元表 `1/i` | 1.439 | 0.723 | 1.99× |
| 线性求逆 `1/(1-7x)` | 0.333 | 0.168 | 1.99× |
| 常数等比求逆 `1/(1-x)` | 0.342 | 0.039 | 8.82× |
| 数乘（系数 7） | 0.184 | 0.134 | 1.38× |
| 除法（除数次数 40） | 47.667 | 35.809 | 1.33× |

主扫描没有耗时至少 5 微秒且回退超过 7% 的项目。64 项 `pow` 的微秒级回退用 9 轮、
每轮 1001 次、预热 50 次复测，得到 **4.292 → 4.042 微秒（1.06×）**；原始扫描和
[复测数据](bench/performance-20261009-small.csv) 均保留。

另外六个模数在 `1024..131072` 上各测量 56 组，三轮交替、每轮预热 5 次、测量 7 次，
checksum 全部一致，见 [多模数数据](bench/performance-20261009-moduli.csv)。下表是
`n = 2^17` 的加速；`1224736769 > 2^30` 用于验证原有全归约路径。

| mod | NTT 正向 | NTT 逆向 | 卷积 | exp(x) |
| ---: | ---: | ---: | ---: | ---: |
| 1004535809 | 1.41× | 1.42× | 1.35× | 3.87× |
| 469762049 | 1.51× | 1.44× | 1.34× | 3.85× |
| 167772161 | 1.36× | 1.44× | 1.37× | 3.82× |
| 754974721 | 1.36× | 1.49× | 1.31× | 3.92× |
| 1073479681 | 1.41× | 1.43× | 1.38× | 3.79× |
| 1224736769 | 1.01× | 1.01× | 0.99× | 3.06× |

上述 NEON 优化阶段的 AVX2 控制测量通过 Rosetta 成对运行，通用 NTT/级数运算基本持平；`n = 2^20` 的逆元表
**1.69×**、`exp(x)` **1.16×**、`1/(1-x)` **16.45×**。数据分别见
[AVX2 扫描](bench/performance-20261009-avx2.csv) 和
[AVX2 级数扫描](bench/performance-20261009-avx2-series.csv)。这些数字来自翻译执行，
本机没有原生 x86 的性能数据。

冷 plan 用 13 组交替的新进程测量，见 [建表数据](bench/performance-20261009-plan.csv)。
NEON `2^20` 首次构建 **1.043 → 0.857 ms（1.22×）**；65536 项 **63.792 → 60.542 微秒**。
1024 项首次构建 **4.958 → 6.791 微秒**，小尺寸初始化多约 1.8 微秒；此成本由尺寸缓存只付一次。
运算计时排除冷建表，输入复原、输出校验和销毁也在计时区域之外。

twiddle 保持每项一个 32 位字和 64 字节对齐，没有增加表存储或转换遍历。
[内存统计](bench/performance-20261009-memory.csv) 确认 `n = 2^20` 的首次
`conv`/`inv`/`exp`/`sqrt` 分别申请 **1 / 2 / 3 / 2** 个池内临时块，预热后系统申请均为 **0**，
保留约 **8 / 8 / 12 / 8 MiB**，与基线一致。NEON 线性 `exp` 删除 `n` 项逆元表，
此尺寸节省约 **4 MiB** 临时空间；vector 和 NTT plan 的分配不计入池内统计。

尝试后没有保留的方案包括全阶段双字 twiddle 表（访存增加）、其他后端的分组逆元表向量化
（实测不如标量常商区间）、更大的 NTT 缓存块（没有稳定收益）和显式 blend
（Clang 已生成相同指令）。最终维护头文件和单文件版均通过 NEON、`-march=native`、
ASan+UBSan、Rosetta AVX2 和标量测试；另通过 GCC 16 的 NTT/多项式测试、AVX-512 编译检查
及 9 项单文件生成器测试。

复现主扫描：

```bash
mkdir -p /tmp/fastpoly-before-20261009
git archive 3ab9f13 include | tar -x -C /tmp/fastpoly-before-20261009
clang++ -std=c++20 -O2 -DNDEBUG -I/tmp/fastpoly-before-20261009/include bench/bench.cpp -o /tmp/fastpoly-before-20261009/bench
clang++ -std=c++20 -O2 -DNDEBUG -Iinclude bench/bench.cpp -o /tmp/fastpoly-after-20261009
python3 scripts/compare-bench.py --before /tmp/fastpoly-before-20261009/bench --after /tmp/fastpoly-after-20261009 \
  --min-size 64 --max-size 1048576 --runs 3 --reps 9 --warmup 5 \
  --ops all,square,skinny,exp-linear,inv-short,pow-short,pow-linear,div-small,inv-series,scale,inv-linear \
  --output /tmp/fastpoly-comparison-20261009.csv
```

其他模数用 `-DFASTPOLY_BENCH_MOD=... -DFASTPOLY_BENCH_ROOT=...` 编译两版；上表中的
`754974721`、`1073479681` 取原根 11，其余取 3。冷 plan 使用 `--op plan --size N --warmup 0 --reps 1`
并逐次启动新进程；临时池统计使用 `--memory`，首次调用设 `--warmup 0 --reps 1`。

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
AVX2 的正确性通过 Rosetta 验证；AVX-512 通过原生编译检查和软件模型测试，本机未做原生 x86 性能测量。

## 边界与已知限制

* `mod < 2^31`（32 位有符号通道内比较所需）；SIMD 内核按通道宽度做无符号/有符号比较，
  已在 `1224736769` 这类接近 `2^30` 的模数上验证。
* lazy reduction（中间量放宽到 `[0, 2·mod)`）要求 `mod < 2^30`；`1224736769` 不满足，
  自动退回全程 `[0, mod)` 的路径，结果一致、只是每次乘法多一条条件减。
* `inv_series` 用线性递推实现。前 65536 项的 `mod/i` 与 `mod%i` 共用一次 **32 位变除数整数除法**；
  之后按常商区间推进，省去逐项除法。标量普通整数商直接乘已有 Montgomery 原始值，
  再按常量模数归约；NEON 小模数区间用固定乘数 SIMD 处理四项。
* NTT 长度上限 `2^{v2(mod-1)}`：`998244353` 为 `2^23`，`469762049` 为 `2^26`。
* `integral`/`log` 需要 `n < mod`（`1/i` 的线性递推前提），在 NTT 长度范围内自动满足。
* `log` 要求常数项为 1、`exp` 要求常数项为 0，否则抛 `domain_error`；`sqrt` 会在常数项为
  非二次剩余或 x-adic 赋值（valuation）为奇数时抛错。
