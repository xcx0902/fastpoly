# O2 编译优化的增量测量

日期：2026-10-11，时区 Asia/Shanghai。这里对照的是本轮开始时已经优化过的工作树：
`9f0dad9b0e52dfcb0499e57cb44a12f4b7032466` 加上 `baseline.patch`。
维护头文件树的 SHA-256 指纹为
`2633696ac5bca85cf0cea7528f00870109486dee75608e4001c952c2def0830f`，
与上一轮 `../compile-20261011-neon.json` 的优化后指纹相同。

本轮编译器为 Apple Clang 21（`clang-2100.3.34.2`）和 Homebrew GCC 16.1.0。
三套主要测试与基准的编译耗时合计，Clang / NEON 减少 10.5%、Clang / AVX2 减少 12.7%、
GCC / NEON 减少 13.2%；默认模数单文件消费示例减少 6.8%–8.9%，特殊模数示例最高减少 48.0%。
1,264 组成对运行计时（含复测）的输出全部一致，复测未发现稳定的运行性能回退。

## 数据含义

- `compile-*.csv`：每次重新编译目标文件，前后交替三轮，取中位数，包含解析、模板实例化、
  优化与代码生成，不包含链接。同名 JSON 保存每轮墙钟时间、子进程 CPU 时间、命令和源码指纹。
  `cpu_speedup` 与 `speedup` 都是优化前耗时除以优化后耗时。
- `runtime-*.csv`：使用同一份 `bench/bench.cpp`，交替运行两个独立二进制。
  同名 JSON 保存各轮返回的原始行，包含该轮重复测量的中位数、最小值与完整输出 checksum。
  `runtime-recheck-*` 是对主扫描异常点增加重复次数的复测，主扫描数据完整保留。
- `batch-*.csv`：小尺寸完整 NTT 的批量计时，每个批次反复变换同一缓冲区，计时后校验所有输出。
  截断逆变换的 `window-*.csv` 使用独立输入块，每块只变换一次，只校验接口承诺的输出范围。
  两类方法都把计时器开销分摊到大量变换上；同名 JSON 保存每个进程的样本。
- `codegen-*.csv`：`bench/codegen.cpp` 中底层 SIMD 原语的完整汇编比较，模数为 998244353。
  这不表示整个 NTT 或多项式程序的汇编相同。
- `benchmark-builds.json`、`batch-builds.json`、`window-builds.json` 保存二进制构建命令及指纹；
  `compile-commands.json`、`runtime-commands.json` 保存测量命令。
- `validation.json` 与 `validation/` 保存测试配置、通过数量和日志。
  `summary.json` 汇总结果、源码指纹及测量限制。

编译计时和运行计时均串行执行，期间不并行运行本项目的构建或测试。
运行基准通过 `-include benchmark-qos.hpp` 将自身线程设为 macOS 的
`QOS_CLASS_USER_INITIATED`，减少后台进程调度造成的波动；设置失败会终止基准。
该辅助文件只用于测量，不包含在库或单文件发行版中。编译计时不使用这个辅助文件。
AVX2 在 Apple Silicon 上经 Rosetta 执行；AVX-512 只有编译与 SIMDe 正确性验证，
没有原生 x86 硬件性能数据。

## 复现

从仓库根目录重建增量基线：

```bash
project="$PWD"
work="$(mktemp -d)"
git archive 9f0dad9b0e52dfcb0499e57cb44a12f4b7032466 include scripts/amalgamate.py | tar -x -C "$work"
(cd "$work" && git apply "$project/bench/compile-20261011-followup/baseline.patch")
python3 "$work/scripts/amalgamate.py"
python3 "$work/scripts/amalgamate.py" --paths avx2 avx512 --output "$work/fastpoly-x86.hpp"
python3 scripts/compare-compile.py --before "$work/include" --runs 3 --output "$work/compile-neon.csv"
python3 scripts/compare-compile.py --before "$work/include" --runs 3 \
  --flag=-arch --flag=x86_64 --flag=-mavx2 --output "$work/compile-avx2.csv"
```

原生 x86 编译省略 `--flag=-arch --flag=x86_64`。可用 `--cxx` 更换编译器；
普通质数消费示例使用 `--modulus 1000000007 --cases conv,series`。
单文件消费示例使用 `--before "$work" --after . --header fastpoly.hpp --cases include,conv,series`。

复现运行对照的一个完整扫描：

```bash
clang++ -std=c++20 -O2 -DNDEBUG -pthread -I"$work/include" \
  -include bench/compile-20261011-followup/benchmark-qos.hpp bench/bench.cpp -o "$work/before"
clang++ -std=c++20 -O2 -DNDEBUG -pthread -Iinclude \
  -include bench/compile-20261011-followup/benchmark-qos.hpp bench/bench.cpp -o "$work/after"
python3 scripts/compare-bench.py --before "$work/before" --after "$work/after" \
  --min-size 64 --max-size 262144 --runs 5 --reps 21 --warmup 10 \
  --ops all,square,skinny,exp-linear,inv-short,pow-short,pow-linear,div-small,inv-series,scale,inv-linear,conv-trunc,conv-tail,div,mod,divmod,add-assign,eval \
  --output "$work/runtime-neon.csv"
```

其他模数、后端、批量小变换与异常点复测的参数见对应 JSON。
