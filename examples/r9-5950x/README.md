# Config for a Ryzen 9 5950X + RTX 3090 alone (Linux)

The config measured on a 16-core 5950X with 64 GB of RAM and one RTX 3090 (300 W) in a Strata server:

- `ud-iq4_xs-3090.json`: unsloth's UD-IQ4_XS on the RTX 3090 alone — **99 tok/s** of decode (six-run average) and
  2,268 tok/s of prefill at 18K, which is what the PCIe 4.0 x8 slot gives. Long context holds: 32K/64K/120K prompts
  at 2,596/2,628/2,582 tok/s (120K in 46.5 s) and 72/90/73 tok/s of decode after. The runs:
  [bench/results/2026-10-03-5950x-opt](../../bench/results/2026-10-03-5950x-opt/README.md).

What the 5950X needs is two numbers: `--pool-workers 14` (one worker per thread but two, so the host thread and the
GPU drain keep a core) and `--adapt-swaps 192` (a single GPU swaps experts in bursts): **68.8 -> 99.3 tok/s**. It
also needed the adaptive-tier fix (a batch a pump retired while the last request ended was never admitted, which
froze the VRAM cache from 6 workers up); without it more workers make things slower.

With 64 GB of RAM there is no `--ple-io ram`: the 28.8 GB PLE table would not fit beside the experts. And
UD-Q4_K_XL does not belong on this machine: its 71.7 GiB of experts do not fit in RAM, and by mmap it decodes at
~30 tok/s.

The paths are relative to the Strata folder and follow setup's data layout (`../Strata-data`). To build and run it:

1. Build the engine from this branch for the RTX 30 card:

       cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
         "-DCMAKE_CUDA_ARCHITECTURES=86"
       cmake --build build --target strata

2. Download the model files into `../Strata-data/models/UD-IQ4_XS` (three `Qwen3.8-Flash-Next-UD-IQ4_XS-0000N-of-00003.gguf`)
   and make its pack, as in [examples/r7-5700x](../r7-5700x/README.md).

3. Start it with Niko1221/Strata's server, from the Strata folder:

       git clone https://github.com/Niko1221/Strata ../Strata-niko-server
       .venv/bin/python ../Strata-niko-server/serve/server.py --engine strata \
         --config examples/r9-5950x/ud-iq4_xs-3090.json --port 8092
