#!/usr/bin/env python3
"""Lamina's setup for Windows and Linux PCs with NVIDIA GPUs (RTX 30 series or newer). It builds the engine from this
checkout, prepares a model, and writes the model's config and launcher. setup.bat and setup.sh make the Python
environment (.venv) and run this file in it. It asks nothing: the flags say what to do.

    setup.bat --model iq3_s               Qwen3.8-Flash-Next, ISTA-DASLab's GSQ-RCO IQ3_S (84 GB of files)
    ./setup.sh --model ud-q4_k_xl         Qwen3.8-Flash-Next, unsloth's UD-Q4_K_XL (112 GB)
    setup.bat --model qwen36-ud-q4_k_xl   Qwen3.6-35B-A3B, unsloth's UD-Q4_K_XL (23 GB)
    setup.bat --model qwen38-27b-ud-q6_k_m    Qwen3.8-27B, unsloth's UD-Q6_K_M (23 GB; its layers split over two GPUs)
    setup.bat                             the engine only
    setup.bat --check [--model M]         what this PC has and what the model needs; nothing is changed

The steps:
1. llama.cpp at the pinned commit, into third_party/llama.cpp: ggml for the engine, mtmd for pictures, gguf-py for
   the image projector's conversion, and the server's libraries (cpp-httplib, nlohmann/json, Jinja).
2. lamina and lamina-vision, built with CMake, Ninja, the CUDA toolkit, and MSVC or GCC. Missing tools are named, not
   installed.
3. The model's files from Hugging Face at pinned revisions. Downloads resume, and each file's size and SHA-256 are
   checked.
4. Their preparation: for Flash-Next the draft layer and the Q8_0 image projector. The engine reads the rest from
   the GGUF files.
5. lamina-<model>.json and run-<model>.bat (or .sh) beside this file, from examples/<model>.json.
A step whose result is already there is skipped. Files already in place are checked and kept: setup moves or deletes
nothing.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import http.client
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
WIN = os.name == "nt"
EXE = ".exe" if WIN else ""
LLAMA = ROOT / "third_party" / "llama.cpp"
LLAMA_COMMIT = "3cf03257f219afbe7334045ff7c6a06ac68c627d"   # the engine's CMakeLists.txt checks its mmq.cuh
CUDA_MIN = (13, 0)   # RTX 50 engines built with CUDA 12.8 crashed in long prompts (upstream #220, #224)

# Each file comes from a fixed commit of its Hugging Face repository, with a fixed size and SHA-256
REPOS = {
    "ista": ("ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "ed59f92082b1e93c0e96d60a8b11aab089b52f09"),
    "unsloth": ("unsloth/Qwen3.8-Flash-Next-GGUF", "38bb39ee97821de2c9009abb7e93950eec396e66"),
    "qwen36": ("unsloth/Qwen3.6-35B-A3B-MTP-GGUF", "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d"),
    "qwen38_27b": ("unsloth/Qwen3.8-27B-GGUF", "4ca720788d1e01f1bff70c033e0d0028fd02e502"),
    "draft": ("eddoursul/Lamina-Qwen3.8-Flash-Next-MTP", "362952648189e083ee413b3cdcac53eb5f0d183e"),
}


@dataclass(frozen=True)
class File:
    repo: str      # a key of REPOS
    path: str      # in the repository
    dest: str      # under the data folder
    size: int
    sha256: str


IQ3_S = [
    File("ista", f"IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{i}-of-00002.gguf",
         f"models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{i}-of-00002.gguf", size, sha)
    for i, size, sha in (
        (1, 54817524224, "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
        (2, 28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"))
]
UD_Q4_K_XL = [
    File("unsloth", f"UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf",
         f"models/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-0000{i}-of-00004.gguf", size, sha)
    for i, size, sha in (
        (1, 10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
        (2, 49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
        (3, 49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
        (4, 12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"))
]
QWEN36 = [
    File("qwen36", "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf", "models/qwen36-UD-Q4_K_XL/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
         22853663008, "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1"),
]
QWEN36_MMPROJ = File("qwen36", "mmproj-F16.gguf", "models/qwen36-UD-Q4_K_XL/mmproj-F16.gguf",
                     899283584, "71f3cbc1f7cc0f30d09d41cfa924c0060827ebc33bf15ace7e86661e856f0160")
QWEN38_27B = [
    File("qwen38_27b", "Qwen3.8-27B-UD-Q6_K_M.gguf", "models/qwen38-27b-UD-Q6_K_M/Qwen3.8-27B-UD-Q6_K_M.gguf",
         23088409504, "493301830a596b8ad56dc1329f80bbcb578c8e910da395feafdc9cd8263430bb"),
]
QWEN38_27B_MMPROJ = File("qwen38_27b", "mmproj-F16.gguf", "models/qwen38-27b-UD-Q6_K_M/mmproj-F16.gguf",
                         927607488, "cbb841a9ee0636b2ec172f5bb8df2ea8dfeb01e90fe7c6126581d662a0b4e43e")
# Qwen3.8-Flash-Next's image projector (the same for every quant). The CPU encoder reads it as Q8_0 (tools/mmproj_q8.py).
FLASH_MMPROJ = File("ista", "mmproj-Qwen3.8-Flash-Next-BF16.gguf", "models/mmproj/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
                    907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0")
FLASH_MMPROJ_Q8 = "models/mmproj/mmproj-Qwen3.8-Flash-Next-Q8_0.gguf"
# The MTP draft layer of both Flash-Next quants (the repository's README describes it)
DRAFT = [
    File("draft", name, f"mtp/rt-distilled/{name}", size, sha)
    for name, size, sha in (
        ("experts.bin", 707788800, "09398406be61f1f54c93861f449e48b8df0bfccbc9ec9b2b7636775a6ea9244f"),
        ("dense.bin", 116099072, "a46561c5d3e76e78eebb9b54bbf4ae87ba1f99a62cad85562cbb516141cd5e47"),
        ("dense.txt", 1880, "8773c81ebb0986e37fe94a8a9933e87be48b1fabc6889a0106bcdab179d1c2ac"),
        ("draft_vocab.bin", 162100, "369151522226a5edaa5f12cfd1e2ae7db8f4fbdbd222f3dcf327dced9597fb25"))
]


@dataclass(frozen=True)
class Model:
    title: str
    files: list
    flash_next: bool      # Qwen3.8-Flash-Next: the draft layer too; else the Qwen3.6 path
    port: int
    ram_gb: int           # the RAM it takes while running (its experts pinned)
    mmproj: File | None = None   # the Qwen3.6 path's image projector (Flash-Next's: FLASH_MMPROJ)
    vram_gb: int = 0      # the VRAM of every GPU together it needs (its layers split over them), 0: one GPU's


MODELS = {
    "iq3_s": Model("Qwen3.8-Flash-Next IQ3_S", IQ3_S, True, 8081, 53),
    "ud-q4_k_xl": Model("Qwen3.8-Flash-Next UD-Q4_K_XL", UD_Q4_K_XL, True, 8082, 81),
    "qwen36-ud-q4_k_xl": Model("Qwen3.6-35B-A3B UD-Q4_K_XL", QWEN36, False, 8083, 22, QWEN36_MMPROJ),
    "qwen38-27b-ud-q6_k_m": Model("Qwen3.8-27B UD-Q6_K_M", QWEN38_27B, False, 8084, 2, QWEN38_27B_MMPROJ, 36),
}


def say(msg=""):
    print(msg, flush=True)


def ok(msg):
    say(f"  ok: {msg}")


def warn(msg):
    say(f"  note: {msg}")


def fail(msg, hint=None):
    say(f"\nsetup stopped: {msg}")
    if hint:
        say(f"  {hint}")
    sys.exit(1)


def out(cmd, **kw) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120, **kw).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def run(cmd, env=None):
    cmd = [str(c) for c in cmd]
    say("  $ " + " ".join(cmd))
    try:
        code = subprocess.run(cmd, env=env).returncode
    except OSError as e:
        fail(f"{cmd[0]}: {e}")
    if code != 0:
        fail(f"{Path(cmd[0]).name} failed (above)")


def clean_env(env=None) -> dict:
    """The environment for setup's child processes, without a PYTHONPATH or PIP_TARGET meant for another Python."""
    env = dict(os.environ if env is None else env)
    for k in ("PYTHONPATH", "PIP_TARGET"):
        env.pop(k, None)
    return env


# ---- the PC

@dataclass
class Gpu:
    index: int            # nvidia-smi's index (the bus order, which the engine uses too)
    name: str
    uuid: str
    mem_mib: int
    cc: int               # compute capability x 10
    display: bool


def gpus() -> list:
    text = out(["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,compute_cap,display_active",
                "--format=csv,noheader,nounits"])
    res = []
    for line in text.strip().splitlines():
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 6 and f[0].isdigit():
            try:
                cc = round(float(f[4]) * 10)
            except ValueError:
                cc = 0
            res.append(Gpu(int(f[0]), f[1], f[2], int(f[3]) if f[3].isdigit() else 0, cc, f[5].lower() == "enabled"))
    return res


def driver_version() -> str:
    lines = out(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]).split()
    return lines[0] if lines else "?"


def ram_gb() -> float:
    if WIN:
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + \
                       [(n, ctypes.c_ulonglong) for n in ("total", "avail", "pt", "pa", "vt", "va", "ve")]
        m = MemoryStatus()
        m.length = ctypes.sizeof(m)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
        return m.total / 1e9
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) * 1024 / 1e9
    return 0.0


def cpu_avx2() -> bool:
    if WIN:
        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(40))   # PF_AVX2_INSTRUCTIONS_AVAILABLE
    flags = next((ln for ln in Path("/proc/cpuinfo").read_text().splitlines() if ln.startswith("flags")), "")
    return "avx2" in flags.split()


def free_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def gpu_spec(g: Gpu, all_gpus: list) -> str:
    """How the config names a GPU: by its name when no other GPU has it, else by its UUID."""
    return g.name if sum(1 for x in all_gpus if x.name == g.name) == 1 else g.uuid


def pick_main(usable: list) -> Gpu:
    """The GPU the model runs on: the one with the most VRAM among those not driving a display, else among all."""
    pool = [g for g in usable if not g.display] or usable
    return max(pool, key=lambda g: (g.mem_mib, -g.index))


def pick_vision_gpu(spec: str | None, usable: list, main: Gpu | None) -> Gpu | None:
    """The GPU of --vision gpu[:GPU]: the one named (its name, UUID or nvidia-smi index), else, besides the model's
    main GPU, the one with the least VRAM (where the encoder's ~1.3 GB of experts matter least) among those driving no
    display, as for the main GPU (the engine leaves a desktop's card a larger reserve), else among all of them, else
    the main GPU. None when the name matches no GPU."""
    if spec:
        return next((g for g in usable if spec in (g.name, g.uuid, str(g.index))), None) or \
            next((g for g in usable if spec.lower() in g.name.lower()), None)
    others = [g for g in usable if main is None or g.index != main.index] or usable
    pool = [g for g in others if not g.display] or others
    return min(pool, key=lambda g: (g.mem_mib, -g.index)) if pool else None


# ---- the tools

def find_tool(name):
    """A tool in this Python's environment (CMake and Ninja from requirements-build.txt), else on the PATH."""
    for d in (Path(sys.executable).parent, Path(sys.executable).parent / "Scripts"):
        if (d / (name + EXE)).exists():
            return str(d / (name + EXE))
    return shutil.which(name)


def find_nvcc():
    """The newest CUDA toolkit's nvcc and its (major, minor); LAMINA_NVCC=<path to nvcc> picks one."""
    pick = os.environ.get("LAMINA_NVCC")
    cands = [pick] if pick else [shutil.which("nvcc")]
    if not pick:
        cands += [str(Path(os.environ[v]) / "bin" / ("nvcc" + EXE)) for v in ("CUDA_PATH", "CUDA_HOME") if os.environ.get(v)]
        if WIN:
            base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
            cands += [str(p / "bin" / "nvcc.exe") for p in base.glob("v*")]
        else:
            cands += [str(p / "bin" / "nvcc") for p in [*Path("/usr/local").glob("cuda*"), *Path("/opt").glob("cuda*")]]
    best, best_v = None, None
    for c in dict.fromkeys(x for x in cands if x):
        if Path(c).exists():
            v = re.search(r"release (\d+)\.(\d+)", out([c, "--version"]))
            ver = (int(v.group(1)), int(v.group(2))) if v else None
            if ver and (best_v is None or ver > best_v):
                best, best_v = c, ver
    return best, best_v


def nvcc_build(nvcc) -> int | None:
    """An nvcc's build number ("V13.2.86" -> 86)."""
    m = re.search(r"\bV\d+\.\d+\.(\d+)", out([nvcc, "--version"]) if nvcc else "")
    return int(m.group(1)) if m else None


def find_vcvars(cuda_v):
    """Visual Studio's vcvars64.bat with the C++ tools: 2019 or 2022, or 2026 with CUDA 13.3 or newer (the versions
    CUDA accepts). LAMINA_VCVARS=<path to vcvars64.bat> picks one."""
    if os.environ.get("LAMINA_VCVARS"):
        v = Path(os.environ["LAMINA_VCVARS"])
        return v if v.exists() else None
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / \
        "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if not vswhere.exists():
        return None
    upper = "19.0" if cuda_v and cuda_v >= (13, 3) else "18.0"
    p = out([str(vswhere), "-latest", "-products", "*", "-version", f"[16.0,{upper})", "-requires",
             "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath"]).strip()
    v = Path(p) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat" if p else None
    return v if v and v.exists() else None


def vs_environment(vcvars: Path) -> dict:
    """The environment vcvars64.bat makes, for the build."""
    env = clean_env()
    env.pop("NoDefaultCurrentDirectoryInExePath", None)   # vcvars runs vswhere.exe from its own folder
    r = subprocess.run(f'"{vcvars}" >nul && set', capture_output=True, text=True, shell=True, env=env)
    res = {}
    for line in r.stdout.splitlines():
        k, sep, v = line.partition("=")
        if sep and k:
            res[k] = v
    if "INCLUDE" not in res:
        fail(f"{vcvars} gave no compiler environment")
    return clean_env(res)


@dataclass
class Toolchain:
    cmake: str | None
    ninja: str | None
    nvcc: str | None
    cuda: tuple | None
    vcvars: Path | None    # Windows
    cxx: str | None        # Linux

    def problems(self) -> list:
        """What the build needs and setup does not install (CMake and Ninja come from requirements-build.txt)."""
        p = []
        if not self.nvcc:
            p.append(f"the CUDA toolkit {CUDA_MIN[0]}.{CUDA_MIN[1]} or newer (https://developer.nvidia.com/cuda-downloads;"
                     " LAMINA_NVCC=<path to nvcc> picks one)")
        elif self.cuda < CUDA_MIN:
            p.append(f"the CUDA toolkit {CUDA_MIN[0]}.{CUDA_MIN[1]} or newer: {self.nvcc} is {self.cuda[0]}.{self.cuda[1]}")
        if WIN and not self.vcvars:
            p.append("Visual Studio 2022's C++ tools (the free Build Tools, workload 'Desktop development with C++';"
                     " LAMINA_VCVARS=<path to vcvars64.bat> picks one)")
        if not WIN and not self.cxx:
            p.append("a C++ compiler (Debian and Ubuntu: build-essential)")
        return p


def toolchain() -> Toolchain:
    nvcc, cuda = find_nvcc()
    return Toolchain(find_tool("cmake"), find_tool("ninja"), nvcc, cuda,
                     find_vcvars(cuda) if WIN else None, None if WIN else shutil.which("g++"))


def install_requirements(build: bool):
    """The server's Python packages, and CMake and Ninja for a build, into this environment (a venv only)."""
    if sys.prefix == sys.base_prefix:
        fail(f"{sys.executable} is not a virtual environment", "run setup.bat or setup.sh: they make .venv")
    reqs = ["-r", ROOT / "requirements.txt"] + (["-r", ROOT / "requirements-build.txt"] if build else [])
    run([sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "-q", *reqs], env=clean_env())


# ---- llama.cpp and the builds

def fetch_llama():
    """llama.cpp at the pinned commit, from GitHub's zip of it (no git needed). llama.cpp's web UI (tools/ui) is left
    out: its deep paths can pass Windows' 260-character limit."""
    mark = LLAMA / ".lamina-commit"
    if mark.exists() and mark.read_text().strip() == LLAMA_COMMIT:
        ok(f"llama.cpp {LLAMA_COMMIT[:8]}")
        return
    if LLAMA.exists():
        fail(f"{LLAMA} holds a llama.cpp that setup did not put there",
             f"the engine is built against {LLAMA_COMMIT[:8]}: move that folder away and run setup again")
    z = LLAMA.parent / f"llama.cpp-{LLAMA_COMMIT[:8]}.zip"
    say(f"  llama.cpp {LLAMA_COMMIT[:8]} into {LLAMA.relative_to(ROOT)} ...")
    get(f"https://github.com/ggml-org/llama.cpp/archive/{LLAMA_COMMIT}.zip", z, "llama.cpp")
    tmp = LLAMA.parent / "llama.cpp.unpack"
    shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(z) as f:
        if f.comment.decode(errors="replace") != LLAMA_COMMIT:       # git archive notes the commit there
            z.unlink()
            fail(f"the llama.cpp download is not commit {LLAMA_COMMIT}", "run setup again")
        f.extractall(tmp, [m for m in f.namelist() if "/tools/ui/" not in m])
    for attempt in range(5):     # an antivirus scanner can still hold a file of the fresh unpack (upstream #63)
        try:
            os.replace(next(tmp.iterdir()), LLAMA)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(2)
    tmp.rmdir()
    z.unlink()
    mark.write_text(LLAMA_COMMIT + "\n")
    ok(f"llama.cpp {LLAMA_COMMIT[:8]}")


def cuda_archs(usable: list) -> str:
    return ";".join(str(c) for c in sorted({g.cc for g in usable}))


def cuda_defs(nvcc: str) -> list:
    """The CUDA compiler, the toolkit its libraries come from (CMake can otherwise link another toolkit's cudart,
    e.g. the distribution's CUDA 12 beside /opt/cuda-13.0), and the host compilers CC, CXX and CUDAHOSTCXX name, as
    definitions, so that a build folder made before takes them too."""
    root = Path(nvcc).resolve().parent.parent
    cc, cxx = os.environ.get("CC", "").strip(), os.environ.get("CXX", "").strip()
    host = os.environ.get("CUDAHOSTCXX", "").strip() or cxx
    return [f"-DCMAKE_CUDA_COMPILER={Path(nvcc).as_posix()}"] + \
        ([] if root == Path("/usr") else [f"-DCUDAToolkit_ROOT={root.as_posix()}"]) + \
        ([f"-DCMAKE_C_COMPILER={cc}"] if cc else []) + ([f"-DCMAKE_CXX_COMPILER={cxx}"] if cxx else []) + \
        ([f"-DCMAKE_CUDA_HOST_COMPILER={host}"] if host else [])


def cmake_build(tc: Toolchain, src: Path, b: Path, target: str, defs: list, env: dict):
    run([tc.cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={tc.ninja}", "-S", src, "-B", b, "-DCMAKE_BUILD_TYPE=Release",
         "-DGGML_CCACHE=OFF", *defs], env=env)
    # half the CPU's threads (nvcc takes GBs of RAM per file) unless CMAKE_BUILD_PARALLEL_LEVEL says. A failed build is
    # tried once more: CUDA 13.0's ptxas now and then fails to read a PTX file it just wrote, and the same command then
    # gets past it (upstream #45).
    jobs = os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL") or str(max(2, (os.cpu_count() or 4) // 2))
    build = [tc.cmake, "--build", b, "--target", *target.split(), "-j", jobs]
    say("  $ " + " ".join(map(str, build)))
    if subprocess.run([str(c) for c in build], env=env).returncode != 0:
        say("  (the build stopped: trying it once more)")
        run(build, env=env)


def build_engine(tc: Toolchain, archs: str, portable: bool, env: dict):
    b = ROOT / "build"
    cmake_build(tc, ROOT, b, "lamina lamina-server", ["-DLAMINA_ENABLE_CUDA=ON", "-DLAMINA_BUILD_TESTS=OFF",
                                         f"-DCMAKE_CUDA_ARCHITECTURES={archs}", f"-DLAMINA_GGML_DIR={LLAMA.as_posix()}",
                                         f"-DLAMINA_PORTABLE={'ON' if portable else 'OFF'}", *cuda_defs(tc.nvcc)], env)
    if WIN:   # the CUDA runtime beside the engine
        for dll in cuda_dlls(tc.nvcc):
            shutil.copy2(dll, b / dll.name)
    ok(f"the engine and the server: build/lamina{EXE}, build/lamina-server{EXE} (CUDA architectures {archs})")


def cuda_dlls(nvcc: str) -> list:
    """Windows: the CUDA runtime and cuBLAS DLLs of nvcc's toolkit (CUDA 13 keeps them in bin\\x64, 12 in bin)."""
    tk = Path(nvcc).parent
    return [dll for pat in ("cudart64_*.dll", "cublas64_*.dll", "cublasLt64_*.dll")
            for dll in [*(tk / "x64").glob(pat), *tk.glob(pat)]]


def build_vision(tc: Toolchain, env: dict, cuda_archs: str | None = None):
    """The image encoder runs on the CPU, built for any AVX2 CPU: a native build on an AVX-512 CPU runs without flash
    attention, about half as fast. With --vision gpu (`cuda_archs`) it is built with CUDA, into build-vision-cuda."""
    if not cuda_archs:
        cmake_build(tc, ROOT / "tools" / "vision", ROOT / "build-vision", "lamina-vision",
                    [f"-DLLAMA_DIR={LLAMA.as_posix()}", "-DLAMINA_VISION_CUDA=OFF", "-DLAMINA_PORTABLE=ON"], env)
        ok(f"the image encoder: build-vision/bin/lamina-vision{EXE}")
        return
    b = ROOT / "build-vision-cuda"
    cmake_build(tc, ROOT / "tools" / "vision", b, "lamina-vision",
                [f"-DLLAMA_DIR={LLAMA.as_posix()}", "-DLAMINA_VISION_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
                 "-DLAMINA_PORTABLE=ON", *cuda_defs(tc.nvcc)], env)
    if WIN:   # the CUDA runtime beside the encoder
        for dll in cuda_dlls(tc.nvcc):
            shutil.copy2(dll, b / "bin" / dll.name)
    ok(f"the image encoder: build-vision-cuda/bin/lamina-vision{EXE} (CUDA architectures {cuda_archs})")


def cuda_lib_dir(nvcc: str | None):
    """Linux: the toolkit's library folder, for the config's lib_dirs (the server puts them on LD_LIBRARY_PATH)."""
    if not nvcc:
        return None
    tk = Path(nvcc).resolve().parent.parent
    for d in (tk / "lib64", tk / "targets" / "x86_64-linux" / "lib"):
        if list(d.glob("libcudart.so*")):
            return str(d)
    return None


# ---- the files

def get(url: str, dst: Path, label: str, size: int | None = None, h=None):
    """url into dst through dst.part, resumed after a break. h: a hash fed with every byte of the file."""
    part = dst.with_name(dst.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    if have and h is not None:
        hash_file(part, h)                             # the bytes already there
    for _ in range(20):
        if size is not None and have >= size:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "lamina-setup", **({"Range": f"bytes={have}-"} if have else {})})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as fh:
                if have and r.status != 206:           # the server sent the whole file
                    fh.seek(0)
                    fh.truncate()
                    have = 0
                    h = hashlib.sha256() if h is not None else None
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    fh.write(b)
                    if h is not None:
                        h.update(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        total = f" of {size / 1e9:.2f}" if size else ""
                        print(f"\r  {label}: {have / 1e9:.2f}{total} GB   ", end="", flush=True)
            print()
            if size is None:
                break
        except (OSError, http.client.HTTPException) as e:
            print()
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                fail(f"{url}: HTTP {e.code}")
            warn(f"{label}: {e}; trying again in 10 s")
            time.sleep(10)
    else:
        fail(f"{label}: the download kept breaking", "run setup again: it goes on from where it stopped")
    os.replace(part, dst)
    return h


def hash_file(p: Path, h=None, label: str = ""):
    h = h or hashlib.sha256()
    size, done, last = p.stat().st_size, 0, 0.0
    with open(p, "rb") as fh:
        while b := fh.read(16 << 20):
            h.update(b)
            done += len(b)
            if label and time.time() - last > 2:
                last = time.time()
                print(f"\r  checking {label}: {100 * done / max(size, 1):.0f}%   ", end="", flush=True)
    if label and last:
        print("\r" + " " * 70 + "\r", end="", flush=True)
    return h


def checked(dst: Path, size: int, sha256: str) -> bool:
    """Whether dst is the pinned file: its size, and its SHA-256 once (<name>.sha256 beside it remembers the match)."""
    mark = dst.with_name(dst.name + ".sha256")
    if not dst.exists() or dst.stat().st_size != size:
        return False
    if mark.exists() and mark.read_text().split()[:1] == [sha256]:
        return True
    if hash_file(dst, label=dst.name).hexdigest() != sha256:
        fail(f"{dst} has the right size but not the pinned file's SHA-256",
             "an older or damaged copy: move it away and run setup again to download it")
    mark.write_text(f"{sha256}  {dst.name}\n")
    return True


def hf_endpoint() -> str:
    return (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")


def download(url: str, dst: Path, size: int, sha256: str):
    """The pinned file at url into dst, unless dst is it already."""
    if checked(dst, size, sha256):
        ok(f"{dst.name} (checked)")
        return
    if dst.exists():
        fail(f"{dst} is {dst.stat().st_size:,} bytes, not the pinned file's {size:,}",
             "an older or damaged copy: move it away and run setup again to download it")
    dst.parent.mkdir(parents=True, exist_ok=True)
    h = get(url, dst, dst.name, size, hashlib.sha256())
    if dst.stat().st_size != size or h.hexdigest() != sha256:
        dst.unlink()
        fail(f"{dst.name}: the download is not the pinned file (its size or SHA-256 differs)",
             "run setup again to download it anew")
    dst.with_name(dst.name + ".sha256").write_text(f"{sha256}  {dst.name}\n")
    ok(f"{dst.name} downloaded and checked")


def fetch(f: File, data: Path):
    repo, rev = REPOS[f.repo]
    download(f"{hf_endpoint()}/{repo}/resolve/{rev}/{f.path}", data / f.dest, f.size, f.sha256)


# ---- the model's preparation

def tool(name: str, *args):
    run([sys.executable, ROOT / "tools" / name, *args], env=clean_env({**os.environ, "LAMINA_GGUF_PY": str(LLAMA / "gguf-py")}))


def prepare(key: str, m: Model, data: Path, vision: bool):
    """What the GGUF files do not hold: Flash-Next's draft layer and, with pictures, the image projector (Flash-Next's
    in Q8_0). The engine computes its float tensors' forms from the GGUF at load, and the server reads the tokenizer
    and the chat template from it."""
    if m.flash_next:
        for f in DRAFT:
            fetch(f, data)
        if vision:
            fetch(FLASH_MMPROJ, data)
            if (data / FLASH_MMPROJ_Q8).exists():
                ok(f"the image projector: {data / FLASH_MMPROJ_Q8}")
            else:
                tool("mmproj_q8.py", data / FLASH_MMPROJ.dest, data / FLASH_MMPROJ_Q8)
    elif vision and m.mmproj:
        fetch(m.mmproj, data)


# ---- the config and the launcher

EXAMPLE_DATA = "../Lamina-data"   # the data folder of examples/*.json


def vision_section(v: dict, gpu: Gpu | None) -> dict:
    """The config's `vision` section from the example's: the CPU encoder on this PC's logical cores, or the CUDA build
    on `gpu` (`cuda_device`: the server runs the encoder on that card alone, #408)."""
    v = dict(v)
    if gpu is None:
        v["threads"] = os.cpu_count() or 8
        return v
    v["exe"] = v["exe"].replace("/build-vision/", "/build-vision-cuda/", 1)
    v["gpu"] = True
    v["cuda_device"] = gpu.uuid
    v.pop("threads", None)
    return v


def write_config(key: str, data: Path, main_gpu: str, port: int, vision: bool, context, nvcc,
                 vision_gpu: Gpu | None = None) -> Path:
    """lamina-<key>.json from examples/<key>.json. The example's keys are written anew; the other keys of a config
    written before (an API key, the "images" of tools/image/install.py) are kept."""
    cfg = json.loads((ROOT / "examples" / f"{key}.json").read_text(encoding="utf-8"))
    path = ROOT / f"lamina-{key}.json"
    old = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else {}
    keep = {k: v for k, v in old.items() if k not in cfg}
    try:
        ref = Path(os.path.relpath(data, ROOT)).as_posix()   # relative to this checkout where it can be
    except ValueError:                                     # another drive
        ref = data.as_posix()

    def fix(v):
        if isinstance(v, str):
            if v.startswith(EXAMPLE_DATA + "/"):
                v = ref + v[len(EXAMPLE_DATA):]
            if not WIN and v.startswith("./build") and v.endswith(".exe"):
                v = v[:-4]
        return v

    args = [fix(a) for a in cfg["args"]]
    if "--main-gpu" in args:
        args[args.index("--main-gpu") + 1] = main_gpu
    if context:
        args[args.index("--max-context") + 1] = str(context)
    if vision:
        cfg["vision"] = vision_section({k: fix(v) for k, v in cfg["vision"].items()}, vision_gpu)
    else:
        args = [a for a in args if a != "--vision"]
        cfg.pop("vision", None)
    cfg["args"] = args
    if "log" in cfg:
        cfg["log"] = fix(cfg["log"])
    if "expert_profile_save" in cfg:                    # what the runs learn, kept with the model's data
        cfg["expert_profile_save"] = f"{ref}/{key}-learned.bin"
    cfg["port"] = port
    if not WIN:
        lib = cuda_lib_dir(nvcc)
        cfg["lib_dirs"] = [lib] if lib else []
    path.write_text(json.dumps({**cfg, **keep}, indent=2) + "\n", encoding="utf-8", newline="\n")
    return path


def write_launcher(key: str, m: Model, cfg: Path, port: int) -> Path:
    if WIN:
        p = ROOT / f"run-{key}.bat"
        lines = ["@echo off", f"title Lamina {m.title}",
                 f"rem {m.title} on http://127.0.0.1:{port}/v1 ({cfg.name}, written by setup.py).",
                 'cd /d "%~dp0"', f'build\\lamina-server.exe --config {cfg.name} %*', "pause"]
        p.write_bytes(("\r\n".join(lines) + "\r\n").encode())
    else:
        p = ROOT / f"run-{key}.sh"
        lines = ["#!/bin/sh", f"# {m.title} on http://127.0.0.1:{port}/v1 ({cfg.name}, written by setup.py).",
                 'cd "$(dirname "$0")" || exit 1', f'exec build/lamina-server --config {cfg.name} "$@"']
        p.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
        p.chmod(0o755)
    return p


# ---- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog=__doc__.split("\n\n", 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), help="the model to prepare (without it: the engine only)")
    ap.add_argument("--data-dir", default=str(ROOT.parent / "Lamina-data"),
                    help="the models, the draft layer and the image projector (default: Lamina-data beside this checkout)")
    ap.add_argument("--vision", default="on", metavar="on|off|gpu[:GPU]",
                    help="pictures: on (the encoder runs on the CPU), off, or gpu (built with CUDA and run on a GPU: its "
                         "name, UUID or nvidia-smi index; default the one with the least VRAM besides the main GPU, "
                         "of those that drive no display if any)")
    ap.add_argument("--main-gpu", help="the GPU the model runs on: its name, UUID or nvidia-smi index (default: the one "
                                       "with the most VRAM that drives no display)")
    ap.add_argument("--context", type=int, help="the context in tokens (default: the example config's)")
    ap.add_argument("--port", type=int, help="the server's port (default: 8081 to 8084 by model)")
    ap.add_argument("--check", action="store_true", help="say what this PC has and what the model needs; change nothing")
    ap.add_argument("--update", action="store_true", help="git pull this checkout first")
    ap.add_argument("--no-build", action="store_true", help="use the binaries already built")
    ap.add_argument("--cuda-archs", help="the CUDA architectures to build for, e.g. 86;120 (default: this PC's GPUs)")
    ap.add_argument("--portable", action="store_true",
                    help="build the engine for other PCs (any AVX2 CPU), as the Docker image does")
    a = ap.parse_args()
    if a.vision not in ("on", "off", "gpu") and not a.vision.startswith("gpu:"):
        ap.error(f"--vision {a.vision}: on, off, gpu or gpu:GPU")

    say(f"Lamina setup in {ROOT}")
    if platform.machine().lower() not in ("amd64", "x86_64"):
        fail(f"{platform.machine()}: Lamina runs on x86-64 PCs")
    gl = gpus()
    usable = [g for g in gl if g.cc >= 80]
    for g in gl:
        say(f"  GPU {g.index}: {g.name}, {g.mem_mib / 1024:.1f} GiB, compute capability {g.cc / 10:.1f}"
            f"{', drives a display' if g.display else ''}{'' if g.cc >= 80 else ' (too old: left out)'}")
    if gl:
        say(f"  NVIDIA driver {driver_version()}")
    say(f"  RAM {ram_gb():.0f} GB, CPU {'with' if cpu_avx2() else 'WITHOUT'} AVX2, Python {platform.python_version()}")
    tc = toolchain()
    say(f"  CMake and Ninja {'ok' if tc.cmake and tc.ninja else 'to install'}, CUDA "
        f"{'.'.join(map(str, tc.cuda)) if tc.cuda else 'missing'}, "
        + (f"MSVC {'ok' if tc.vcvars else 'missing'}" if WIN else f"g++ {'ok' if tc.cxx else 'missing'}"))
    if not cpu_avx2():
        fail("the CPU has no AVX2: the engine's CPU kernels need it")
    if not usable and (a.model or not a.cuda_archs):
        fail("no NVIDIA GPU of compute capability 8.0 or newer (RTX 30 series or newer) was found by nvidia-smi")

    m = MODELS.get(a.model) if a.model else None
    data = Path(a.data_dir).resolve()
    main_gpu = None
    if m:
        if a.main_gpu:
            main_gpu = next((g for g in usable if a.main_gpu in (g.name, g.uuid, str(g.index))), None) or \
                next((g for g in usable if a.main_gpu.lower() in g.name.lower()), None)
            if main_gpu is None:
                fail(f"--main-gpu {a.main_gpu}: no such GPU of compute capability 8.0 or newer")
        else:
            main_gpu = pick_main(usable)
        example = json.loads((ROOT / "examples" / f"{a.model}.json").read_text(encoding="utf-8"))
        others = len(usable) - 1 if "--second-gpu" in example["args"] else 0
        say(f"\n{m.title} runs on GPU {main_gpu.index}, {main_gpu.name}"
            + (f", with {others} more as expert tiers" if others and m.flash_next else
               f", with up to {others} more for a layer split (the fewest that hold it)" if others else ""))
        if m.vram_gb and sum(g.mem_mib for g in usable) < m.vram_gb * 1024:
            warn(f"its layers need ~{m.vram_gb} GB of VRAM over every GPU, and these have "
                 f"{sum(g.mem_mib for g in usable) / 1024:.0f} GB: a shorter --context, or another GPU")
        files = m.files + (DRAFT if m.flash_next else [])
        need = sum(f.size for f in files)
        missing = sum(f.size for f in files if not (data / f.dest).exists())
        say(f"  {need / 1e9:.0f} GB of files in {data}: {missing / 1e9:.0f} GB to download, {free_gb(data):.0f} GB free")
        if ram_gb() < m.ram_gb + 8:
            warn(f"it takes ~{m.ram_gb} GB of RAM while running, and this PC has {ram_gb():.0f} GB")
        if free_gb(data) < missing / 1e9 + 5:
            warn("the data folder's drive is short of room for the downloads")
    build = not (a.no_build and (ROOT / "build" / f"lamina-server{EXE}").exists())
    archs = a.cuda_archs or cuda_archs(usable)
    vision_gpu = None
    if a.vision.startswith("gpu"):
        vision_gpu = pick_vision_gpu(a.vision[4:] or None, usable, main_gpu)
        if vision_gpu is None:
            fail(f"--vision {a.vision}: no such GPU of compute capability 8.0 or newer")
        say(f"  pictures: the image encoder on GPU {vision_gpu.index}, {vision_gpu.name} (~1.3 GB of its VRAM)")
    problems = tc.problems() if build else []
    if problems:
        say("\nmissing for the build:")
        for p in problems:
            say(f"  - {p}")
    elif build and tc.cuda == (13, 2) and max(map(int, re.findall(r"\d+", archs)), default=0) >= 120 and \
            (nvcc_build(tc.nvcc) or 0) < 86:
        warn("CUDA 13.2.0 and 13.2.1's nvcc made garbage answers on RTX 50 GPUs (sm_120; upstream #892, #968), where "
             "13.0, 13.1 and 13.2.2 or newer are right: LAMINA_NVCC=<path to nvcc> picks another toolkit")
    if a.check:
        return 0
    if problems:
        fail("install what is missing (above), then run setup again")

    if a.update:
        say("\nupdating this checkout")
        run(["git", "-C", ROOT, "pull", "--ff-only"])
        # the rest by the setup.py just pulled
        return subprocess.call([sys.executable, __file__, *(x for x in sys.argv[1:] if x != "--update")])
    say("\n1. Python packages and llama.cpp")
    install_requirements(build)
    if build:
        tc = toolchain()                               # CMake and Ninja from requirements-build.txt
        if not tc.cmake or not tc.ninja:
            fail("CMake or Ninja is not in this environment after pip installed requirements-build.txt")
    fetch_llama()
    vision = a.vision != "off"
    if build:
        say("\n2. the engine and the image encoder")
        env = vs_environment(tc.vcvars) if WIN else clean_env()
        build_engine(tc, archs, a.portable, env)
        build_vision(tc, env, archs if vision_gpu else None)
    else:
        say("\n2. the binaries as they are (--no-build)")
        if vision_gpu and not (ROOT / "build-vision-cuda" / "bin" / f"lamina-vision{EXE}").exists():
            fail("--vision gpu: build-vision-cuda/bin/lamina-vision is not built: run setup without --no-build")
    if not m:
        say("\nDone: the engine is built. Prepare a model with --model (" + ", ".join(MODELS) + ").")
        return 0
    say(f"\n3. {m.title}'s files")
    for f in m.files:
        fetch(f, data)
    say("\n4. preparing them")
    prepare(a.model, m, data, vision)
    say("\n5. the config and the launcher")
    port = a.port or m.port
    cfg = write_config(a.model, data, gpu_spec(main_gpu, gl), port, vision, a.context, tc.nvcc, vision_gpu)
    launcher = write_launcher(a.model, m, cfg, port)
    ok(f"{cfg.name}, {launcher.name}")
    say(f"\nDone. Start it with {launcher.name}: the API on http://127.0.0.1:{port}/v1 (OpenAI and Anthropic), "
        f"a chat page on http://127.0.0.1:{port}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
