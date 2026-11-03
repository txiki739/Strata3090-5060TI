# Troubleshooting

The engine's log is the config's `log`, `lamina-<model>.log` in the clone's folder; the server writes to its window.
`setup --check` says what the PC has and what a model needs.

## Setup

**"missing for the build: ...".** Setup installs no system software: install what it names (the CUDA toolkit 13.0 or
newer; on Windows Visual Studio 2022's Build Tools with the workload "Desktop development with C++"; on Linux a C++
compiler, `build-essential` on Debian and Ubuntu), then run setup again. `LAMINA_NVCC=<path to nvcc>` and
`LAMINA_VCVARS=<path to vcvars64.bat>` pick a toolkit and a Visual Studio when there are several.

**"no NVIDIA GPU of compute capability 8.0 or newer ... was found by nvidia-smi".** The NVIDIA driver is missing, or
`nvidia-smi` is not on the PATH, or the GPUs are older than the RTX 30 series, which Lamina does not run on.

**Setup warns about CUDA 13.2.0 or 13.2.1.** Their compiler builds some of the engine's kernels wrong for RTX 50 GPUs
(sm_120): the answers are garbage. Use CUDA 13.0, 13.1 or 13.2.2 and newer; `LAMINA_NVCC` picks one installed beside
it.

**Linux: the build stops at `unsupported GNU version`.** The GCC is newer than the CUDA toolkit accepts. Install an
older one beside it (g++-14) and name it: `CXX=g++-14 ./setup.sh ...` (`CC`, `CXX` and `CUDAHOSTCXX` name the build's
compilers).

**A download broke off.** Run setup again: it goes on from where it stopped.

**"... has the right size but not the pinned file's SHA-256", or "... bytes, not the pinned file's".** The file in the
data folder is an older or damaged copy. Move it away and run setup again to download it.

## Starting

**The PC is slow or stops responding for a minute or two while the model loads.** The engine reads the experts into
RAM and locks them there for the GPUs (the server's window says when). Wait; the next starts are faster while the
model file stays in the OS file cache.

**The start fails, or the PC pages heavily, with too little free RAM.** The experts stay locked in RAM while the model
runs (the README's table). Close other programs, or use IQ3_S, which takes ~28 GB less than UD-Q4_K_XL.

**"port N is already in use - is Lamina (or another server) already running?".** Another server listens on the
config's port: close it, or start this one with `--port`.

**"Windows blocked ... Smart App Control".** Smart App Control on Windows 11 refuses programs that are not signed, and
the engine and the image encoder are built on this PC. Turn it off (Windows Security > App & browser control > Smart
App Control settings), or, if only the image encoder is blocked, run setup with `--vision off`.

**"ExpertCache: cudaMalloc(...) failed: out of memory" although the GPU has VRAM free.** On Windows, every allocation
on a GPU also counts against Windows' commit limit (RAM and the page file), and the locked experts already take much of
it. Give the page file a fixed size, the same initial and maximum size (e.g. 65,536 MB: System > About > Advanced system
settings > Performance > Advanced > Virtual memory), and restart Windows: a page file that grows on demand may not grow
in time.

**"MiB of VRAM free with everything loaded - LOW: requests may stall".** The expert cache and the K/V left the GPU
almost no VRAM; Windows then pages the GPU's memory, and a request can stall. Raise `--vram-reserve-mib` in the config's
`args` (default 700), or lower `--max-context`, and close what else uses that GPU.

## While it answers

**Generation is far slower than expected, a few tokens a second.** Something else uses the main GPU: a display
plugged into it, a browser or a game drawing on it, another CUDA program. The engine's GPU waits on the host by
spinning, and a GPU shared in time slices stalls it. Plug the displays into another GPU (setup picks a GPU that drives
no display as the main one) and check `nvidia-smi` for other processes on it.

**It is slower than [COMPARISON.md](COMPARISON.md).** Those are the development PC's numbers: three GPUs holding most
experts, a 3090's VRAM bandwidth, 8 cores and DDR5-5600 for the rest. Fewer or smaller GPUs leave more experts to the
CPU, and the RAM's bandwidth sets its speed: run the RAM at its rated speed (EXPO or XMP), in both channels.
`BENCH.bat` (`./bench.sh`) measures an install the way it is used, through the API, and writes a report without
personal details, to compare after a change or to attach to an issue.

**Answers turn into nonsense, or the PC crashes under load.** Check the RAM first: an overclock that passes everyday
use can flip bits when the CPU and the GPUs' copies read it at once, which on the development PC (DDR5-6000 XMP with
four DIMMs) gave crashes and changing answers. Test at the RAM's stock speed. A display driver reset followed by a
blue screen in `nvlddmkm` (0x116) points at the NVIDIA driver: try another version.

**"the engine stopped unexpectedly (exit code ...)".** The engine ended mid-request; the next request starts it again.
The end of its log says why. On Linux the usual cause is the RAM running out: `sudo dmesg | grep -i -E 'killed
process|out of memory'`.

**"request exceeds the context window".** The conversation and the answer's room are longer than the context. Start a
new conversation, or run setup again with a larger `--context` (with `--kv-grow`, the configs' setting, a larger
context takes VRAM only as a conversation reaches it).

**"the engine said nothing for ... s during the request".** The engine and the server lost step, or the engine froze
(a driver stall): the server ends the engine and the next request starts it again. A prompt that takes longer than
that on a slow PC: raise `engine_silence_s` in the config (default 300).

**"this server was started without the vision encoder".** The model was set up without pictures: run setup again with
`--vision on`.

**Pictures take seconds.** The image encoder runs on the CPU: about 9 s for a picture of ~1,000 image tokens on an
8-core CPU. The config's `vision.max_tokens` caps a picture's tokens (1,024 in the configs). With a GPU that has
~1.3 GB to spare, setup's `--vision gpu` runs the encoder there (about 2 s for such a picture).

**Two runs of one greedy request differ.** Where an expert is computed changes its rounding a little, and the GPUs'
expert caches change as their copies land, so runs can part at a near-tie
([HOW_IT_WORKS.md](HOW_IT_WORKS.md#exactness)).

## Reporting a problem

Open an [issue](https://github.com/eddoursul/Lamina/issues) with the engine's log and the server's output, the output
of `setup --check`, the commit (`git rev-parse --short HEAD`) and what the request was.
