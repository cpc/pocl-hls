# PoCL AlmaIF HLS: OpenCL-to-FPGA Synthesis via MLIR

This document describes how to build and use PoCL's MLIR-based HLS compilation
flow, which automatically synthesizes FPGA accelerators from standard OpenCL C
programs. The toolchain compiles OpenCL kernels through MLIR intermediate
representations and generates FPGA bitstreams using vendor back-end tools
(Vitis HLS / Vivado for AMD, Intel AOC for Altera).

For the background and evaluation results, see:

> T. Leppänen, L. Leppänen, Z. Jamil, J. Solanti, J. Multanen, and P. Jääskeläinen,
> "Composable Open-Source Toolchain for Synthesizing Hardware Accelerators from
> OpenCL Command Buffers," *ACM Trans. Reconfig. Technol. Syst.*, 2026.
> https://doi.org/10.1145/3786204

## Overview

The compilation flow is:

```
OpenCL C  -->  Polygeist/ClangIR  -->  MLIR (upstream dialects)
         -->  PoCL middle-end passes (workgroup generation, barrier elimination)
         -->  ScaleHLS / Hida optimization passes
         -->  scalehls-translate (emit HLS C++)
         -->  Vitis HLS (C++ to RTL)
         -->  Vivado (RTL wrapping as AlmaIF accelerator)
         -->  v++ (bitstream generation)
```

The resulting bitstream is loaded at runtime by the AlmaIF device driver via
XRT (AMD) or OPAE (Altera). Compilation is JIT: the FPGA bitstream is
generated the first time a kernel (or command buffer) is enqueued, then cached.

## Prerequisites

### Required software

| Component | Version | Purpose |
|-----------|---------|---------|
| ClangIR | [github/ClangIR](https://github.com/llvm/clangir/tree/d4ebb05f347d8d9d62968676d5b2bbc1338de499)| Base MLIR infrastructure, and one of the front-ends |
| Polygeist (optional front-end) | [github/cpc/Polygeist](https://github.com/cpc/polygeist)                | OpenCL C to MLIR front-end                          |
| ScaleHLS            | [github/cpc/Hida](https://github.com/cpc/hida)                                     | HLS optimization passes and C++ emission            |
| OpenASIP            | [github/cpc/OpenASIP](https://github.com/cpc/tce)                                  | Soft processor acting as a controller               |
| ISL                 | (tested with:) 0.26-3build1.1 (from apt)                                           | Used by affine passes ported from Enzyme-JAX        |
| Vitis               | 2022.1                                                                             | C++ to bitstream (xclbin)                           |
| XRT                 | 2023.2                                                                             | Runtime for AMD FPGAs                               |
| FPGA Platform shell | `xilinx_u280_gen3x16_xdma_base_1`                                                  | Specific to Alveo U280                              |

### Hardware

Tested with the following hardware:

- AMD: Alveo U280 (`xcu280-fsvh2892-2L-e`)
- Altera: BittWare IA-420f (Intel Agilex 7)

## Building PoCL with HLS Support

```bash
mkdir build && cd build

source /path/to/xrt/setup.sh

cmake .. \
  -DENABLE_CLANGIR=ON \
  -DENABLE_ALMAIF_DEVICE=ON \
  -DSCALEHLS_DIR=/path/to/scalehls/install \
  -DOPENASIP_LLVM_DIR=/path/to/openasip-llvm/install \
  -DWITH_LLVM_CONFIG=/path/to/llvm-config

make -j$(nproc)
```

### Key CMake variables

| Variable | Description |
|----------|-------------|
| `ENABLE_CLANGIR=ON` | Enables the MLIR compiler path (required) |
| `ENABLE_ALMAIF_DEVICE=ON` | Builds the AlmaIF accelerator device driver |
| `POLYGEIST_BINDIR` | Path to directory containing the `cgeist` binary (optional) |
| `SCALEHLS_DIR` | Root of ScaleHLS installation |
| `OPENASIP_LLVM_DIR` | LLVM installation root used by OpenASIP (as it can differ from the main ClangIR LLVM) |

## End-to-End Compilation Flow

### Single kernel compilation

When an OpenCL program calls `clBuildProgram`, PoCL:

1. **Front-end**: ClangIR (or Polygeist) converts OpenCL C to MLIR using upstream
   dialects (scf, affine, arith, memref, func, gpu).

2. **Middle-end**: PoCL MLIR passes generate the workgroup function:
   - Links OpenCL built-in functions (implemented in MLIR)
   - Wraps the SPMD kernel in `affine.parallel` (local size bounds)
   - Eliminates barriers (Polygeist's barrier elimination pass)
   - Allocates local memory
   - Runs affine optimization passes (loop fusion, coalescing, LICM, CSE, mem2reg)

3. **HLS back-end** (at `clEnqueue` or `clFinalizeCommandBufferKHR`):
   - ScaleHLS optimization passes (dataflow, pipelining, array partitioning)
   - `scalehls-translate --scalehls-emit-hlscpp` emits Vitis-compatible C++
   - Vitis HLS synthesizes RTL from the C++
   - Vivado wraps the RTL in an AlmaIF accelerator block design with an OpenASIP
     command processor
   - `v++` generates the final `.xclbin` bitstream

### Command buffer compilation

When using `cl_khr_command_buffer`, at `clFinalizeCommandBufferKHR`:

1. Each kernel in the command buffer is compiled to a workgroup function
2. A *command buffer function* is generated that calls all kernels sequentially,
   with all arguments and launch parameters specialized as constants
3. The fused function is compiled through the HLS back-end as a single accelerator
4. The bitstream contains one combined accelerator for the entire command buffer

This enables cross-kernel optimizations: constant propagation of arguments,
known loop bounds, and potential inter-kernel dataflow optimizations.

## Execution Modes

Three setup scripts in `tools/scripts/` configure the environment for different
execution modes. Source one before running a benchmark:

- **Software emulation** (`source tools/scripts/setup_emu.sh`): Runs on the host
  CPU (via LLVM host target) without any FPGA hardware or Vitis simulation.
  Uses the AlmaIF emulation device (`POCL_ALMAIF0_PARAMETERS=0xE,...`).
  Useful for testing the compiler front- and middle-ends without synthesizing hardware.

- **Hardware emulation** (`source tools/scripts/setup_hw_emu.sh`): Runs in the
  Vitis `hw_emu` simulator. Simulates the actual RTL but takes up to 20 minutes
  to generate the xclbin. Use small dataset sizes.

- **Real hardware** (`source tools/scripts/setup_hw.sh`): Runs on physical FPGA
  hardware. Bitstream generation takes ~2 hours. To replicate the paper results,
  disable the small dataset size.

## Running Benchmarks (PolybenchGPU)

The PolybenchGPU benchmark suite is available with OpenCL command buffer
support at: [github/cpc/polybench](https://github.com/cpc/polybench)

### Building

```bash
cd build
cmake .. -DENABLE_TESTSUITES=polybenchGPU  # add to your existing cmake args
make prepare_examples
```

This clones and builds the polybench suite as an external project.

### Running

For XRT-based execution, make sure you have sourced the XRT `setup.sh` and
the Vivado `settings.sh`:
Then, source one of the setup scripts above, and run the benchmarks:


```bash
cd build/examples/polybenchGPU/src/polybenchGPU-build

# Standard OpenCL version:
./OpenCL/GEMM/gemm

# Command buffer version:
./OpenCL-command-buffer/GEMM/gemm_cmd_buffer
```

For hw_emu-mode, you should also set:

```bash
export XRT_INI_PATH=/path/to/pocl/lib/CL/devices/almaif/mlir/xrt.ini # speeds up hw_emu
export EMCONFIG_PATH=/path/to/emconfig.json
```

### Running the full test suite

```bash
cd build
ctest -L polybenchGPU
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `POCL_DEVICES` | Set to `almaif` to select the AlmaIF device driver |
| `POCL_ALMAIF0_PARAMETERS` | Device parameters: (0xE for emulation; 0xA for hw_emu or hw), Initial xclbin (or 'none') path with 0xA, Kernel id (65535 represents HLS-generated kernels) |
| `POCL_ALMAIF_EXTERNALREGION` | External (DDR/HBM) memory region base address and size |
| `POCL_CACHE_DIR` | Directory for caching compiled kernels and bitstreams |
| `XCL_EMULATION_MODE` | Set to `hw_emu` for Vitis RTL simulation, *unset* for real FPGA |

## Inspecting Intermediate Files

PoCL caches intermediate compilation artifacts. Set `POCL_CACHE_DIR=/where/you/want/to/cache`.
The following intermediate files are generated:

- `parallel.mlir` -- workgroup function after middle-end passes
- `parallel_hls.mlir` -- after ScaleHLS HLS optimization passes
- `parallel_hls.cpp` -- emitted HLS C++ for Vitis HLS
- `parallel.xo` -- Vivado-packaged XO file
- `parallel.xclbin` -- final FPGA bitstream
- `firmware.img` -- OpenASIP command processor firmware

## Available PoCL MLIR Passes

| Pass | Description |
|------|-------------|
| `pocl-workgroup` | Generates the workgroup function from an SPMD kernel |
| `pocl-distribute-barriers` | Barrier elimination using min-cut distribution (ported from Polygeist) |
| `pocl-mem2reg` | Memory-to-register promotion (ported from Polygeist) |
| `pocl-affine-cfg` | Raises `scf`/`memref`-ops to affine equivalents (ported from Polygeist/Enzyme-JAX) |
| `pocl-detect-reduction` | Detects and marks reduction patterns for HLS (ported from intel/llvm/mlir)|
| `pocl-affine-parallel-to-for` | Converts `affine.parallel` to `affine.for` loops |
| `pocl-convert-memref-to-llvm-kernel-args` | Converts memref kernel arguments for the arg-buffer launcher (LLVM lowering) |
| `pocl-strip-mem-spaces` | Removes memory space attributes before LLVM lowering |

## Runtime Architecture

The system is structured around the AlmaIF accelerator interface, which
provides a vendor-portable memory-mapped protocol for controlling accelerators:

The OpenASIP command processor reads work packets from the host, configures the
HLS-generated accelerator IP with kernel arguments and launch parameters, and
signals completion. This wrapper is generated by the Vivado TCL scripts
(`generate_xo.tcl`) and uses RTL generated by OpenASIP (`generateprocessor`).

## Troubleshooting

**ScaleHLS pass failures**: The HLS pass pipeline has a fallback mechanism. If
the full pipeline (with affine raising and ScaleHLS dataflow passes) fails, it
automatically retries without affine raising, and then without the more fragile
ScaleHLS passes. Check `POCL_DEBUG=almaif` output for retry messages.

**Affine verification errors**: If you see "is not a valid symbol" errors from
the MLIR verifier, this is likely due to `arith.index_cast` results being used
in affine map positions. The `pocl-raise-to-affine` pass handles most cases,
but complex control flow may trigger this. See the fallback mechanism above.

# Original PoCL README
# Portable Computing Language (PoCL)

PoCL is a conformant implementation (for [CPU](https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_450)
and [Level Zero GPU](https://www.khronos.org/conformance/adopters/conformant-products/opencl#submission_453) targets)
of the OpenCL 3.0 standard which can be easily adapted for new targets.

[Official web page](http://portablecl.org)

[Full documentation](http://portablecl.org/docs/html/)

[![OpenSSF Best Practices](https://www.bestpractices.dev/projects/9499/badge)](https://www.bestpractices.dev/projects/9499)
[![Coverity Scan Build Status](https://scan.coverity.com/projects/30739/badge.svg)](https://scan.coverity.com/projects/pocl-pocl)

## Building

This section contains instructions for building PoCL in its default
configuration and a subset of driver backends. You can find the full build
instructions including a list of available options
in the [install guide](http://portablecl.org/docs/html/install.html).

### Requirements

In order to build PoCL, you need the following support libraries and
tools:

  * Latest released version of LLVM & Clang
  * development files for LLVM & Clang + their transitive dependencies
    (e.g. `libclang-dev`, `libclang-cpp-dev`, `libllvm-dev`, `zlib1g-dev`,
    `libtinfo-dev`...)
  * CMake 3.15 or newer
  * GNU make or ninja
  * Optional: pkg-config
  * Optional: hwloc v1.0 or newer (e.g. `libhwloc-dev`)
  * Optional (but enabled by default): python3 (for support of LLVM bitcode with SPIR target)
  * Optional: llvm-spirv (version-compatible with LLVM) and spirv-tools
    (required for SPIR-V support in CPU / CUDA; Vulkan driver supports SPIR-V through clspv)

For more details, consult the [install guide](http://portablecl.org/docs/html/install.html).

Building PoCL follows the usual CMake build steps. Note however, that PoCL
can be used from the build directory (without installing it system-wide).

## Supported environments

### CI status:

![x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux_gh.yml/badge.svg?event=push&branch=main)
![x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux.yml/badge.svg?event=push&branch=main)
![ARM64](https://github.com/pocl/pocl/actions/workflows/build_arm64.yml/badge.svg?event=push&branch=main)
![CUDA](https://github.com/pocl/pocl/actions/workflows/build_cuda.yml/badge.svg?event=push&branch=main)
![Level Zero](https://github.com/pocl/pocl/actions/workflows/build_level0.yml/badge.svg?event=push&branch=main)
![OpenASIP+Vulkan](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml/badge.svg?event=push&branch=main)
![Remote](https://github.com/pocl/pocl/actions/workflows/build_remote.yml/badge.svg?event=push&branch=main)
![Apple Silicon](https://github.com/pocl/pocl/actions/workflows/build_macos.yml/badge.svg?event=push&branch=main)
![Windows](https://github.com/pocl/pocl/actions/workflows/build_msvc.yml/badge.svg?event=push&branch=main)

### Support Matrix legend:

:large_blue_diamond: Achieved status of OpenCL conformant implementation

:large_orange_diamond: Tested in CI extensively, including OpenCL-CTS tests

:green_circle: : Tested in CI

:yellow_circle: : Should work, but is untested

:x: : Unsupported

### Linux

| CPU device  |     LLVM 17    |     LLVM 18     |      LLVM 19     |     LLVM 20     |     LLVM 21     |     LLVM 22     |
|:------------|:--------------:|:---------------:|:----------------:|:---------------:|:---------------:|:---------------:|
| [x86-64](https://github.com/pocl/pocl/actions/workflows/build_linux_gh.yml) | :green_circle: | :green_circle: :large_blue_diamond: | :green_circle: |  :large_orange_diamond: | :large_orange_diamond: | :green_circle: |
| [ARM64](https://github.com/pocl/pocl/actions/workflows/build_arm64.yml)     | :yellow_circle: | :yellow_circle: | :yellow_circle: |  :yellow_circle: | :green_circle: |:yellow_circle: |
| i686    | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| ARM32   | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| RISC-V  | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |
| PowerPC | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: |

| GPU device  |     LLVM 17    |     LLVM 18     |      LLVM 19     |     LLVM 20     |     LLVM 21     |
|:------------|:--------------:|:---------------:|:----------------:|:---------------:|:---------------:|
| [CUDA SM5.0](https://github.com/pocl/pocl/actions/workflows/build_cuda.yml) | :yellow_circle: | :green_circle: | :yellow_circle: | :green_circle: | :x: |
| CUDA SM other than 5.0                                                      | :yellow_circle: | :yellow_circle: | :yellow_circle: | :yellow_circle: | :x: |
| [Level Zero](https://github.com/pocl/pocl/actions/workflows/build_level0.yml) | :yellow_circle: | :yellow_circle: | :green_circle: | :green_circle: | :large_orange_diamond: | 
| [Vulkan](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml) | :green_circle: | :x: | :x: | :x: | :x: |

Note: CUDA with LLVM 21 is broken due to a bug in Clang (https://github.com/llvm/llvm-project/issues/154772).

| Special device |    LLVM 17    |     LLVM 18     |      LLVM 19     |     LLVM 20     |     LLVM 21     |
|:---------------|:-------------:|:---------------:|:----------------:|:---------------:|:---------------:|
| [OpenASIP](https://github.com/pocl/pocl/actions/workflows/build_openasip_vulkan.yml) | :green_circle: | :x: | :x: | :x: |  :x: |
| [Remote](https://github.com/pocl/pocl/actions/workflows/build_remote.yml) | :green_circle: | :green_circle:  | :green_circle: | :green_circle: | :yellow_circle: |


### Mac OS X

| CPU device  |     LLVM 17    |     LLVM 18     |      LLVM 19     |     LLVM 20     |      LLVM 21     |
|:------------|:--------------:|:---------------:|:----------------:|:---------------:|:----------------:|
| [Apple Silicon](https://github.com/pocl/pocl/actions/workflows/build_macos.yml) | :yellow_circle: | :yellow_circle: | :yellow_circle: | :green_circle: | :green_circle: |
| [Intel CPU](https://github.com/pocl/pocl/actions/workflows/build_macos.yml)     | :yellow_circle: | :yellow_circle: | :x: | :x: | :x: |

### Windows

| CPU device  |     LLVM 18    |  LLVM 19        |     LLVM 20     |     LLVM 21     |
|:------------|:--------------:|:---------------:|:---------------:|:---------------:|
| [MinGW](https://github.com/pocl/pocl/actions/workflows/build_mingw.yml) / x86-64  | :yellow_circle: | :green_circle: | :yellow_circle: | :yellow_circle: |
| [MSVC](https://github.com/pocl/pocl/actions/workflows/build_msvc.yml) / x86-64    | :yellow_circle: | :green_circle: | :green_circle:  | :yellow_circle: |


## Binary packages

### Linux distros

PoCL with CPU device support can be found on many linux distribution managers.
See [![latest packaged version(s)](https://repology.org/badge/latest-versions/pocl.svg)](https://repology.org/project/pocl/versions)

### PoCL with CUDA driver

PoCL with CUDA driver support for Linux `x86_64`, `aarch64` and `ppc64le`
can be found on conda-forge distribution and can be installed with

    wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh   # install mambaforge

To install pocl with cuda driver

    mamba install pocl-cuda

To install all drivers

    mamba install pocl

### macOS

#### Homebrew

PoCL with CPU driver support Intel and Apple Silicon chips can be
found on homebrew and can be installed with

    brew install pocl

Note that this installs an ICD loader from KhronoGroup and the builtin
OpenCL implementation will be invisible when your application is linked
to this loader.

#### Conda

PoCL with CPU driver support Intel and Apple Silicon chips
can be found on conda-forge distribution and can be installed with

    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
    bash Mambaforge-$(uname)-$(uname -m).sh

To install the CPU driver

    mamba install pocl

Note that this installs an ICD loader from KhronosGroup and the builtin
OpenCL implementation will be invisible when your application is linked
to this loader. To make both pocl and the builtin OpenCL implementaiton
visible, do

    mamba install pocl ocl_icd_wrapper_apple

## License

PoCL is distributed under the terms of the MIT license. Contributions are expected
to be made with the same terms.
