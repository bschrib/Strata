# Independent GP100 VMAD candidate

Baseline: Strata v0.1.39, `6f32ec070f23ced9f50e704d854d775da52591ab`.
Branch: `experiment/p100-vmad-dp4a`. No deployment pin or llama.cpp source changes.

Only `__CUDA_ARCH__ == 600` selects four signed-byte VMAD instructions. Other architectures keep the original source path. The mechanism removes explicit byte extraction from the native DP4A emulation. It independently implements [patch 01's idea](https://github.com/shinbunbun/llama-cpp-p100-patches/blob/939dd95f9f926c87769bd0442fbcf8a40aff2597/patches/01-vmad-dp4a-sm60.patch), with a scoped PTX accumulator. The output is written after all inputs have been consumed, so input/output register sharing is safe. No memory side effects or floating-point changes occur.

[CUDA 12.9 PTX VMAD semantics](https://docs.nvidia.com/cuda/archive/12.9.1/parallel-thread-execution/index.html#scalar-video-instructions-vmad) specify signed byte selectors and support from sm_20. Without saturation or scaling, four operations produce the low 32 bits of the signed-byte dot plus accumulator. The [inline PTX guide](https://docs.nvidia.com/cuda/archive/12.9.1/inline-ptx-assembly/index.html) documents scoped temporary registers and input/output constraints.

The original scalar expression has undefined behavior if **any intermediate signed addition** overflows, even if a later term cancels it. Real quant block sums are bounded far below int32 limits. Device parity checks the intended wrapping integer contract near the limits against a widened oracle. CUDA compilation, SASS, device parity, and performance remain unverified.

## Safe local CPU verification

```sh
uv run --no-project python tests/p100/verify_offline.py
```

This tests all signed-byte pairs in each selector, mixed bytes, accumulator edges, random cases, and aliases against an independent int64 oracle. UBSan checks host arithmetic. Preprocessing compares untouched architecture paths with the pristine header. It does **not** execute or compile production CUDA assembly.

## Deferred compile-only inspection on a CUDA 12.9 Linux preparation host

These commands compile without launching GPU work.

```sh
mkdir -p build-p100-vmad
/usr/local/cuda-12.9/bin/nvcc -std=c++17 -O3 -arch=sm_60 -Iinclude \
  --ptx tests/p100/dp4a_compile_probe.cu -o build-p100-vmad/probe.ptx
/usr/local/cuda-12.9/bin/nvcc -std=c++17 -O3 -arch=sm_60 -Iinclude \
  --cubin -Xptxas=-v tests/p100/dp4a_compile_probe.cu -o build-p100-vmad/probe.cubin \
  2>build-p100-vmad/ptxas.txt
/usr/local/cuda-12.9/bin/cuobjdump --dump-sass build-p100-vmad/probe.cubin \
  >build-p100-vmad/probe.sass
/usr/local/cuda-12.9/bin/nvcc -std=c++17 -O3 -arch=sm_60 -Iinclude \
  -Xptxas=-v tests/p100/dp4a_cuda.cu -o build-p100-vmad/dp4a_cuda \
  2>build-p100-vmad/benchmark-ptxas.txt
```

Inspect both probe kernels and actual IQ/native kernels for instruction lowering, register count, spills, and resource changes. A standalone probe cannot establish real kernel instruction counts. Compile/disassemble corresponding native kernels on sm_61 and a supported modern target in pristine and candidate trees, checking that the lower source path stays unchanged.

## Deferred execution after the user frees the node

First archive pristine pinned single/dual end-to-end baselines. Never run these while acceptance tests continue.

```sh
CUDA_VISIBLE_DEVICES=0 build-p100-vmad/dp4a_cuda --run-on-authorized-p100 \
  >build-p100-vmad/raw-gpu0.csv 2>build-p100-vmad/gpu0.log
CUDA_VISIBLE_DEVICES=1 build-p100-vmad/dp4a_cuda --run-on-authorized-p100 \
  >build-p100-vmad/raw-gpu1.csv 2>build-p100-vmad/gpu1.log
```

The same binary compares pristine scalar and production VMAD in ABBA/BAAB order, checks device outputs, and retains all event timings. Latency and four-chain throughput use synthetic operands and bounded accumulators. Operand preparation and stores add common overhead: this filters candidates, not full-kernel speedups. Run compute-sanitizer memcheck, IQ fixture and multi-column parity, MMVQ parity, grouped/native expert parity, and model logits/token correctness. Then repeat identical IQ2_XS single/dual benchmarks with one candidate only, fixed context/KV/MTP/sampler, 200 W policy, topology, thermals, expert residency, and raw telemetry. Compare archived pristine and contemporaneous control. Keep/revert based on measured results.
