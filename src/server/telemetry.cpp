// src/server/telemetry.cpp - hardware readings for the web app's Monitor tab (serve/telemetry.py).
//
// The GPU readings come from NVML, loaded at run time with its few entry points declared here (no NVML headers or
// import library). CPU, RAM and disk are read as psutil reads them (the Python server's psutil path, so the same
// figures): on Windows GetSystemTimes with the per-CPU interrupt and DPC times, GetPerformanceInfo, and
// IOCTL_DISK_PERFORMANCE on each PhysicalDriveN; on Linux /proc/stat, /proc/meminfo, /proc/diskstats (the whole disks
// /sys/block lists) and the CPU topology in /sys. Python's arithmetic is kept where it shows in the figures: round()
// (correctly rounded), sum() of floats (Python 3.12's compensated sum), time.time().
#include "telemetry.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

// The Linux readers use standard C++ only (the files under /proc and /sys), so the check driver can also build them
// on Windows (LAMINA_TELEMETRY_LINUX) and point them at a made-up /proc and /sys.
#if !defined(_WIN32) || defined(LAMINA_TELEMETRY_LINUX)
#define LAMINA_TELEMETRY_PROCFS 1
#include <filesystem>
#include <fstream>
#include <iterator>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#include <winioctl.h>
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

namespace lamina {
namespace {

constexpr size_t kHistory = 60;
// the series the sparklines draw
constexpr const char* kSeries[] = {"gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu",
                                   "ram_used", "disk_read_mb", "tok_s", "prefill_tok_s_mean"};

// ---------------------------------------------------------------------------------------------- Python's numbers

/// sum() of floats as Python 3.12 adds them: Neumaier's compensated sum
struct FloatSum {
    double f, c = 0.0;
    explicit FloatSum(double first) : f(first) {}
    void add(double x) {
        const double t = f + x;
        c += std::fabs(f) >= std::fabs(x) ? (f - t) + x : (x - t) + f;
        f = t;
    }
    double result() const { return c != 0.0 && std::isfinite(c) ? f + c : f; }
};

double py_fsum(const std::vector<double>& v) {
    if (v.empty()) return 0.0;
    FloatSum s(0.0 + v[0]);
    for (size_t i = 1; i < v.size(); ++i) s.add(v[i]);
    return s.result();
}

/// sum() of JSON numbers: whole numbers exactly, floats from the first float on
json py_sum(const std::vector<json>& v) {
    int64_t whole = 0;
    size_t i = 0;
    for (; i < v.size() && !v[i].is_number_float(); ++i) whole += v[i].get<int64_t>();
    if (i == v.size()) return whole;
    FloatSum s((double) whole + v[i].get<double>());
    for (++i; i < v.size(); ++i) {
        if (v[i].is_number_float()) s.add(v[i].get<double>());
        else s.f += (double) v[i].get<int64_t>();
    }
    return s.result();
}

/// max(): the first of the largest
json py_max(const std::vector<json>& v) {
    size_t best = 0;
    for (size_t i = 1; i < v.size(); ++i)
        if (v[i].get<double>() > v[best].get<double>()) best = i;
    return v[best];
}

/// time.time()
double time_time() {
    const int64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                           std::chrono::system_clock::now().time_since_epoch())
                           .count();
    return ns % 1000000000 == 0 ? (double) (ns / 1000000000) : (double) ns / 1e9;
}

/// bytes.decode("utf-8", errors="replace"): each maximal invalid subpart becomes one U+FFFD
std::string utf8_replace(std::string_view b) {
    std::string out;
    out.reserve(b.size());
    for (size_t i = 0; i < b.size();) {
        const auto c = (unsigned char) b[i];
        if (c < 0x80) {
            out += (char) c;
            ++i;
            continue;
        }
        int need = 0;
        unsigned char lo = 0x80, hi = 0xBF;   // the second byte's range
        if (c >= 0xC2 && c <= 0xDF) {
            need = 1;
        } else if (c >= 0xE0 && c <= 0xEF) {
            need = 2;
            if (c == 0xE0) lo = 0xA0;
            if (c == 0xED) hi = 0x9F;
        } else if (c >= 0xF0 && c <= 0xF4) {
            need = 3;
            if (c == 0xF0) lo = 0x90;
            if (c == 0xF4) hi = 0x8F;
        }
        size_t j = i + 1;
        int got = 0;
        for (; got < need && j < b.size(); ++got, ++j) {
            const auto d = (unsigned char) b[j];
            if (d < (got ? 0x80 : lo) || d > (got ? 0xBF : hi)) break;
        }
        if (need && got == need) out.append(b.substr(i, j - i));
        else out += "\xEF\xBF\xBD";
        i = need ? j : i + 1;
    }
    return out;
}

/// a disk's byte counters (psutil.disk_io_counters' read_bytes and write_bytes)
struct DiskBytes {
    std::string name;
    int64_t read = 0, written = 0;
};

#ifdef _WIN32
std::optional<std::wstring> env_w(const wchar_t* name) {
    const DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (!n) return std::nullopt;
    std::wstring v(n, L'\0');
    const DWORD got = GetEnvironmentVariableW(name, v.data(), n);
    if (got >= n) return std::nullopt;
    v.resize(got);
    return v;
}

std::string utf8_of(std::wstring_view w) {
    if (w.empty()) return {};
    const int n = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    if (n <= 0) return {};
    std::string out((size_t) n, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), out.data(), n, nullptr, nullptr);
    return out;
}

/// os.path's full path (GetFullPathNameW)
std::wstring full_path(const std::wstring& p) {
    const DWORD n = GetFullPathNameW(p.c_str(), 0, nullptr, nullptr);
    if (!n) return p;
    std::wstring out(n, L'\0');
    const DWORD got = GetFullPathNameW(p.c_str(), n, out.data(), nullptr);
    if (!got || got >= n) return p;
    out.resize(got);
    return out;
}
#endif

// ---------------------------------------------------------------------------------------------- NVML
struct NvmlUtilization {
    unsigned gpu, memory;
};
struct NvmlMemory {
    unsigned long long total, free, used;
};
using NvmlInitFn = int (*)();
using NvmlHandleFn = int (*)(unsigned, void**);
using NvmlNameFn = int (*)(void*, char*, unsigned);
using NvmlUtilizationFn = int (*)(void*, NvmlUtilization*);
using NvmlMemoryFn = int (*)(void*, NvmlMemory*);
using NvmlUintFn = int (*)(void*, unsigned*);
using NvmlUintOfFn = int (*)(void*, unsigned, unsigned*);   // of a sensor or a counter

struct Nvml {
    void* lib = nullptr;
    bool ready = false;   // nvmlInit succeeded
    NvmlInitFn init = nullptr;
    NvmlHandleFn handle = nullptr;
    NvmlNameFn name = nullptr;
    NvmlUtilizationFn utilization = nullptr;
    NvmlMemoryFn memory = nullptr;
    NvmlUintOfFn temperature = nullptr, pcie_throughput = nullptr;
    NvmlUintFn power = nullptr, power_limit = nullptr, pcie_gen = nullptr, pcie_gen_max = nullptr,
               pcie_width = nullptr;
};

void* nvml_open() {
#ifdef _WIN32
    // ctypes.CDLL's search for a bare name (the program's folder, System32 and AddDllDirectory's, not PATH), then the
    // driver's NVSMI folder by its full path
    if (HMODULE h = LoadLibraryExW(L"nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) return h;
    std::wstring dir = env_w(L"ProgramFiles").value_or(L"C:\\Program Files");
    if (!dir.empty() && dir.back() != L'\\' && dir.back() != L'/') dir += L'\\';
    const std::wstring path = full_path(dir + L"NVIDIA Corporation\\NVSMI\\nvml.dll");
    return LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR);
#else
    for (const char* n : {"libnvidia-ml.so.1", "libnvidia-ml.so"})
        if (void* h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
    return nullptr;
#endif
}

template <class F>
F nvml_fn(void* lib, const char* name, const char* fallback = nullptr) {
#ifdef _WIN32
    auto find = [lib](const char* n) { return reinterpret_cast<void*>(GetProcAddress(static_cast<HMODULE>(lib), n)); };
#else
    auto find = [lib](const char* n) { return dlsym(lib, n); };
#endif
    void* p = find(name);
    if (!p && fallback) p = find(fallback);
    return reinterpret_cast<F>(p);
}

std::mutex& nvml_mu() {
    static std::mutex m;
    return m;
}

Nvml& nvml_state() {
    static Nvml n;
    return n;
}

/// NVML loaded and initialized. Each Python reader loads the library and calls nvmlInit: a failure is tried again by
/// the next reader; once it succeeds NVML stays initialized for the life of the process.
const Nvml& nvml() {
    std::lock_guard<std::mutex> lk(nvml_mu());
    Nvml& n = nvml_state();
    if (n.ready) return n;
    if (!n.lib) {
        n.lib = nvml_open();
        if (n.lib) {
            n.init = nvml_fn<NvmlInitFn>(n.lib, "nvmlInit_v2", "nvmlInit");
            n.handle = nvml_fn<NvmlHandleFn>(n.lib, "nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetHandleByIndex");
            n.name = nvml_fn<NvmlNameFn>(n.lib, "nvmlDeviceGetName");
            n.utilization = nvml_fn<NvmlUtilizationFn>(n.lib, "nvmlDeviceGetUtilizationRates");
            n.memory = nvml_fn<NvmlMemoryFn>(n.lib, "nvmlDeviceGetMemoryInfo");
            n.temperature = nvml_fn<NvmlUintOfFn>(n.lib, "nvmlDeviceGetTemperature");
            n.power = nvml_fn<NvmlUintFn>(n.lib, "nvmlDeviceGetPowerUsage");
            n.power_limit = nvml_fn<NvmlUintFn>(n.lib, "nvmlDeviceGetEnforcedPowerLimit");
            n.pcie_gen = nvml_fn<NvmlUintFn>(n.lib, "nvmlDeviceGetCurrPcieLinkGeneration");
            n.pcie_gen_max = nvml_fn<NvmlUintFn>(n.lib, "nvmlDeviceGetMaxPcieLinkGeneration");
            n.pcie_width = nvml_fn<NvmlUintFn>(n.lib, "nvmlDeviceGetCurrPcieLinkWidth");
            n.pcie_throughput = nvml_fn<NvmlUintOfFn>(n.lib, "nvmlDeviceGetPcieThroughput");
        }
    }
    n.ready = n.lib && n.init && n.init() == 0;
    return n;
}

/// one card's readings (serve/telemetry.py's _Nvml)
class GpuReader {
public:
    explicit GpuReader(int index) {
        const Nvml& n = nvml();
        ok_ = n.ready && n.handle && n.handle((unsigned) index, &dev_) == 0;
    }

    bool ok() const { return ok_; }

    json name() const {
        const Nvml& n = nvml_state();
        char buf[96] = {};
        if (!ok_ || !n.name || n.name(dev_, buf, (unsigned) sizeof buf) != 0) return nullptr;
        return utf8_replace(std::string_view(buf, (size_t) (std::find(buf, buf + sizeof buf, '\0') - buf)));
    }

    std::optional<NvmlMemory> memory() const {
        const Nvml& n = nvml_state();
        NvmlMemory m{};
        if (ok_ && n.memory && n.memory(dev_, &m) == 0) return m;
        return std::nullopt;
    }

    /// util, mem_used and mem_total when they could be read; temp, power and the PCIe link and traffic (null when not)
    json read() const {
        const Nvml& n = nvml_state();
        json out = json::object();
        NvmlUtilization u{};
        if (ok_ && n.utilization && n.utilization(dev_, &u) == 0) out["util"] = (int64_t) u.gpu;
        if (const auto m = memory()) {
            out["mem_used"] = (int64_t) m->used;
            out["mem_total"] = (int64_t) m->total;
        }
        auto value = [&](NvmlUintFn f) {
            unsigned v = 0;
            return ok_ && f && f(dev_, &v) == 0 ? std::optional<unsigned>(v) : std::nullopt;
        };
        auto value_of = [&](NvmlUintOfFn f, unsigned what) {
            unsigned v = 0;
            return ok_ && f && f(dev_, what, &v) == 0 ? std::optional<unsigned>(v) : std::nullopt;
        };
        auto whole = [](std::optional<unsigned> v) { return v ? json((int64_t) *v) : json(nullptr); };
        auto scaled = [](std::optional<unsigned> v, double unit) {
            return v ? json((double) *v / unit) : json(nullptr);
        };
        out["temp"] = whole(value_of(n.temperature, 0));            // NVML_TEMPERATURE_GPU
        out["power"] = scaled(value(n.power), 1000.0);              // mW
        out["power_limit"] = scaled(value(n.power_limit), 1000.0);
        out["pcie_gen"] = whole(value(n.pcie_gen));                 // drops at idle (power saving)
        out["pcie_gen_max"] = whole(value(n.pcie_gen_max));
        out["pcie_width"] = whole(value(n.pcie_width));
        const auto rx = value_of(n.pcie_throughput, 1);             // NVML_PCIE_UTIL_RX_BYTES, KB/s
        const auto tx = value_of(n.pcie_throughput, 0);
        out["pcie_rx_mb"] = scaled(rx, 1024.0);
        out["pcie_tx_mb"] = scaled(tx, 1024.0);
        return out;
    }

private:
    bool ok_ = false;
    void* dev_ = nullptr;
};

#ifdef _WIN32
// ---------------------------------------------------------------------------------------------- Windows
namespace win {

/// psutil's FILETIME / LARGE_INTEGER in seconds (HI_T * high + LO_T * low)
double seconds(double high, double low) { return 429.4967296 * high + 1e-7 * low; }

struct ProcessorTimes {   // SYSTEM_PROCESSOR_PERFORMANCE_INFORMATION
    LARGE_INTEGER idle, kernel, user, dpc, interrupt;
    ULONG interrupts;
};
using NtQuerySystemInformationFn = LONG(WINAPI*)(ULONG, void*, ULONG, ULONG*);

/// psutil.cpu_times(): user, system (kernel time less idle), idle, and the per-CPU interrupt and DPC times summed
std::optional<std::vector<double>> cpu_times() {
    FILETIME idle, kernel, user;
    if (!GetSystemTimes(&idle, &kernel, &user)) return std::nullopt;
    auto ft = [](const FILETIME& t) { return seconds((double) t.dwHighDateTime, (double) t.dwLowDateTime); };
    auto li = [](const LARGE_INTEGER& t) { return seconds((double) t.HighPart, (double) t.LowPart); };
    const double i = ft(idle), k = ft(kernel), u = ft(user);
    static const auto query = reinterpret_cast<NtQuerySystemInformationFn>(
        reinterpret_cast<void*>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtQuerySystemInformation")));
    const DWORD ncpus = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    if (!query || !ncpus) return std::nullopt;
    std::vector<ProcessorTimes> per(ncpus);
    ULONG got = 0;
    constexpr ULONG kProcessorPerformance = 8;   // SystemProcessorPerformanceInformation
    if (query(kProcessorPerformance, per.data(), (ULONG) (per.size() * sizeof(ProcessorTimes)), &got) < 0)
        return std::nullopt;
    per.resize(std::min<size_t>(per.size(), got / sizeof(ProcessorTimes)));
    std::vector<double> interrupt, dpc;
    for (const ProcessorTimes& c : per) {
        interrupt.push_back(li(c.interrupt));
        dpc.push_back(li(c.dpc));
    }
    return std::vector<double>{u, k - i, i, py_fsum(interrupt), py_fsum(dpc)};
}

/// psutil.virtual_memory()'s (total, available), from GetPerformanceInfo's pages
std::optional<std::pair<int64_t, int64_t>> virtual_memory() {
    PERFORMANCE_INFORMATION p{};
    if (!K32GetPerformanceInfo(&p, (DWORD) sizeof p)) return std::nullopt;
    return std::pair<int64_t, int64_t>((int64_t) (p.PhysicalTotal * p.PageSize),
                                       (int64_t) (p.PhysicalAvailable * p.PageSize));
}

/// psutil's disk counters: each PhysicalDriveN that answers IOCTL_DISK_PERFORMANCE; nullopt where psutil raises
std::optional<std::vector<DiskBytes>> disk_counters() {
    std::vector<DiskBytes> out;
    for (int n = 0; n <= 32; ++n) {   // psutil's guess: more than there are drive letters
        const std::wstring dev = L"\\\\.\\PhysicalDrive" + std::to_wstring(n);
        const HANDLE h =
            CreateFileW(dev.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h == INVALID_HANDLE_VALUE) continue;
        std::vector<unsigned char> buf(sizeof(DISK_PERFORMANCE));
        DWORD got = 0, err = 0;
        for (int tries = 0; tries < 8; ++tries) {
            err = DeviceIoControl(h, IOCTL_DISK_PERFORMANCE, nullptr, 0, buf.data(), (DWORD) buf.size(), &got, nullptr)
                      ? 0
                      : GetLastError();
            if (err != ERROR_INSUFFICIENT_BUFFER) break;
            buf.resize(buf.size() * 2);
        }
        CloseHandle(h);
        if (err == ERROR_INVALID_FUNCTION || err == ERROR_NOT_SUPPORTED) continue;   // an exotic disk: psutil skips it
        if (err) return std::nullopt;
        DISK_PERFORMANCE dp;
        std::memcpy(&dp, buf.data(), sizeof dp);
        out.push_back({"PhysicalDrive" + std::to_string(n), dp.BytesRead.QuadPart, dp.BytesWritten.QuadPart});
    }
    return out;
}

/// the records GetLogicalProcessorInformationEx gives for `relation` (nullopt when it fails)
std::optional<std::vector<unsigned char>> processor_records(LOGICAL_PROCESSOR_RELATIONSHIP relation, DWORD& length) {
    std::vector<unsigned char> buf;
    length = 0;
    while (!GetLogicalProcessorInformationEx(
        relation, buf.empty() ? nullptr : reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buf.data()),
        &length)) {
        if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || !length) return std::nullopt;
        buf.assign(length, 0);
    }
    return buf;
}

/// how many of the records are `relation`'s
int64_t count_records(const std::vector<unsigned char>& buf, DWORD length, LOGICAL_PROCESSOR_RELATIONSHIP relation) {
    int64_t n = 0;
    for (size_t off = 0; off + 8 <= length && off + 8 <= buf.size();) {
        uint32_t rel, size;
        std::memcpy(&rel, buf.data() + off, 4);
        std::memcpy(&size, buf.data() + off + 4, 4);
        if (!size) break;
        if (rel == (uint32_t) relation) ++n;
        off += size;
    }
    return n;
}

/// _cpu_topology(): (CPU packages, NUMA nodes)
std::pair<json, json> topology() {
    DWORD length = 0;
    json sockets = nullptr;
    if (const auto buf = processor_records(RelationProcessorPackage, length))
        if (const int64_t n = count_records(*buf, length, RelationProcessorPackage)) sockets = n;
    ULONG high = 0;
    return {sockets, GetNumaHighestNodeNumber(&high) ? json((int64_t) high + 1) : json(nullptr)};
}

/// psutil.cpu_count(logical=False)
json cpu_cores() {
    DWORD length = 0;
    const auto buf = processor_records(RelationAll, length);
    const int64_t n = buf ? count_records(*buf, length, RelationProcessorCore) : 0;
    return n ? json(n) : json(nullptr);
}

/// os.cpu_count()
json cpu_threads() {
    const DWORD n = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
    return n ? json((int64_t) n) : json(nullptr);
}

/// _cpu_name(): the registry's ProcessorNameString, else platform.processor()
json cpu_name() {
    const wchar_t* key = L"HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0";
    DWORD size = 0;
    if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"ProcessorNameString", RRF_RT_REG_SZ, nullptr, nullptr, &size) ==
        ERROR_SUCCESS) {
        std::wstring v(size / sizeof(wchar_t) + 1, L'\0');
        size = (DWORD) (v.size() * sizeof(wchar_t));
        if (RegGetValueW(HKEY_LOCAL_MACHINE, key, L"ProcessorNameString", RRF_RT_REG_SZ, nullptr, v.data(), &size) ==
            ERROR_SUCCESS) {
            v.resize(wcsnlen(v.c_str(), v.size()));
            return std::string(strip(utf8_of(v)));
        }
    }
    // platform.processor(): the CPU's description (Python asks WMI for it first, whose text is this variable's)
    std::optional<std::wstring> p = env_w(L"PROCESSOR_IDENTIFIER");
    if (!p) {
        p = env_w(L"PROCESSOR_ARCHITEW6432");
        if (!p || p->empty()) p = env_w(L"PROCESSOR_ARCHITECTURE");
    }
    const std::string s = p ? utf8_of(*p) : "";
    return s.empty() || s == "unknown" ? json(nullptr) : json(s);
}

}  // namespace win
#endif

#ifdef LAMINA_TELEMETRY_PROCFS
// ---------------------------------------------------------------------------------------------- Linux
namespace procfs {

namespace fs = std::filesystem;

bool read_file(const std::string& path, std::string& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    out.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
    return !f.bad();
}

/// os.access(path, F_OK): it exists (a link: its target)
bool exists(const std::string& path) {
    std::error_code ec;
    return fs::exists(fs::status(path, ec));
}

/// os.path.lexists: what glob lists
bool lexists(const std::string& path) {
    std::error_code ec;
    return fs::exists(fs::symlink_status(path, ec));
}

bool space(char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f'; }

/// bytes.strip()
std::string_view strip_bytes(std::string_view s) {
    while (!s.empty() && space(s.front())) s.remove_prefix(1);
    while (!s.empty() && space(s.back())) s.remove_suffix(1);
    return s;
}

/// bytes.split()
std::vector<std::string_view> words(std::string_view s) {
    std::vector<std::string_view> out;
    for (size_t i = 0;;) {
        while (i < s.size() && space(s[i])) ++i;
        if (i >= s.size()) break;
        size_t j = i;
        while (j < s.size() && !space(s[j])) ++j;
        out.push_back(s.substr(i, j - i));
        i = j;
    }
    return out;
}

/// a file's lines as iterating it yields them (without their "\n")
std::vector<std::string_view> lines(std::string_view s) {
    std::vector<std::string_view> out;
    for (size_t i = 0; i < s.size();) {
        size_t j = s.find('\n', i);
        if (j == std::string_view::npos) j = s.size();
        out.push_back(s.substr(i, j - i));
        i = j + 1;
    }
    return out;
}

/// int() of a decimal number (spaces around, a sign)
std::optional<int64_t> to_int(std::string_view s) {
    s = strip_bytes(s);
    const bool neg = !s.empty() && s[0] == '-';
    if (!s.empty() && (s[0] == '-' || s[0] == '+')) s.remove_prefix(1);
    if (s.empty()) return std::nullopt;
    uint64_t v = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc() || r.ptr != s.data() + s.size()) return std::nullopt;
    return neg ? -(int64_t) v : (int64_t) v;
}

/// float() of a number
std::optional<double> to_float(std::string_view s) {
    s = strip_bytes(s);
    if (!s.empty() && s[0] == '+') s.remove_prefix(1);
    if (s.empty()) return std::nullopt;
    double v = 0;
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v);
    if (r.ec != std::errc() || r.ptr != s.data() + s.size()) return std::nullopt;
    return v;
}

/// the names in `dir` that glob's prefix[0-9]* matches
std::vector<std::string> numbered(const std::string& dir, std::string_view prefix) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end; it.increment(ec)) {
        const std::string n = it->path().filename().string();
        if (n.size() > prefix.size() && n.compare(0, prefix.size(), prefix) == 0 && n[prefix.size()] >= '0' &&
            n[prefix.size()] <= '9')
            out.push_back(n);
    }
    return out;
}

/// psutil.cpu_times(): /proc/stat's first line in seconds (user nice system idle iowait irq softirq [steal [guest
/// [guest_nice]]])
std::optional<std::vector<double>> cpu_times(const std::string& proc, double clk_tck) {
    std::string text;
    if (!read_file(proc + "/stat", text)) return std::nullopt;
    const auto w = words(std::string_view(text).substr(0, text.find('\n')));
    if (w.size() < 8) return std::nullopt;   // "cpu" and user .. softirq at least
    std::vector<double> out;
    for (size_t i = 1; i < w.size() && i <= 10; ++i) {
        const auto v = to_float(w[i]);
        if (!v) return std::nullopt;
        out.push_back(*v / clk_tck);
    }
    return out;
}

/// calculate_avail_vmem(): the kernel's MemAvailable estimate where /proc/meminfo lacks it (kernels before 3.14) or
/// gives 0 (psutil #1915)
int64_t avail_estimate(const std::map<std::string, int64_t>& mems, const std::string& proc, int64_t page_size) {
    auto get = [&](const char* k) {
        const auto it = mems.find(k);
        return it == mems.end() ? std::nullopt : std::optional<int64_t>(it->second);
    };
    const int64_t mem_free = get("MemFree:").value_or(0);
    const int64_t fallback = mem_free + get("Cached:").value_or(0);
    const auto active = get("Active(file):"), inactive = get("Inactive(file):"), reclaimable = get("SReclaimable:");
    std::string zone;
    if (!active || !inactive || !reclaimable || !read_file(proc + "/zoneinfo", zone)) return fallback;
    int64_t low = 0;
    for (const auto line : lines(zone)) {
        const auto l = strip_bytes(line);
        if (l.substr(0, 3) != "low") continue;
        const auto w = words(l);
        if (w.size() < 2) continue;
        if (const auto v = to_int(w[1])) low += *v;
    }
    low *= page_size;
    const double wl = (double) low;
    double avail = (double) (mem_free - low);
    double pagecache = (double) (*active + *inactive);
    pagecache -= std::min(pagecache / 2, wl);
    avail += pagecache;
    avail += (double) *reclaimable - std::min((double) *reclaimable / 2.0, wl);
    return (int64_t) avail;
}

/// psutil.virtual_memory()'s (total, available)
std::optional<std::pair<int64_t, int64_t>> virtual_memory(const std::string& proc, int64_t page_size) {
    std::string text;
    if (!read_file(proc + "/meminfo", text)) return std::nullopt;
    std::map<std::string, int64_t> mems;
    for (const auto line : lines(text)) {
        const auto w = words(line);
        if (w.size() < 2) continue;
        if (const auto v = to_int(w[1])) mems[std::string(w[0])] = *v * 1024;
    }
    const auto total = mems.find("MemTotal:"), mem_free = mems.find("MemFree:");
    if (total == mems.end() || mem_free == mems.end()) return std::nullopt;
    const auto given = mems.find("MemAvailable:");
    int64_t avail = given != mems.end() && given->second != 0 ? given->second : avail_estimate(mems, proc, page_size);
    if (avail < 0) avail = 0;
    else if (avail > total->second) avail = mem_free->second;   // a container's figures (LXC): procps does the same
    return std::pair<int64_t, int64_t>(total->second, avail);
}

/// psutil's disk counters: /proc/diskstats' lines of the whole disks (the names /sys/block has; partitions are in
/// their disk's figures); nullopt where psutil raises
std::optional<std::vector<DiskBytes>> disk_counters(const std::string& proc, const std::string& sys) {
    std::string text;
    if (!read_file(proc + "/diskstats", text)) return std::nullopt;
    std::vector<DiskBytes> out;
    for (const auto line : lines(text)) {
        const auto f = words(line);
        // the line's layouts: Linux 2.4 (15 fields), a disk (14, 4.18+: 18, 5.5+: 20), a 2.6 partition (7)
        size_t name, first, last, rd, wr;
        if (f.size() == 15) name = 3, first = 2, last = 14, rd = 5, wr = 9;
        else if (f.size() == 14 || f.size() >= 18) name = 2, first = 3, last = 14, rd = 5, wr = 9;
        else if (f.size() == 7) name = 2, first = 3, last = 7, rd = 4, wr = 6;
        else return std::nullopt;
        std::vector<int64_t> v(last);
        for (size_t i = first; i < last; ++i) {
            if (i == name) continue;
            const auto x = to_int(f[i]);
            if (!x) return std::nullopt;
            v[i] = *x;
        }
        std::string n(f[name]), dev = n;
        std::replace(dev.begin(), dev.end(), '/', '!');
        if (!exists(sys + "/block/" + dev)) continue;
        const DiskBytes d{n, v[rd] * 512, v[wr] * 512};
        const auto same = std::find_if(out.begin(), out.end(), [&](const DiskBytes& o) { return o.name == n; });
        if (same != out.end()) *same = d;
        else out.push_back(d);
    }
    return out;
}

/// psutil.cpu_count(logical=False): the distinct core sibling lists in /sys, else /proc/cpuinfo's "cpu cores" of
/// each "physical id"
json cpu_cores(const std::string& proc, const std::string& sys) {
    const std::string dir = sys + "/devices/system/cpu";
    const auto cpus = numbered(dir, "cpu");
    std::set<std::string> lists;
    for (const char* file : {"core_cpus_list", "thread_siblings_list"}) {   // the newer name, the older one
        bool any = false;
        for (const auto& c : cpus) {
            const std::string path = dir + "/" + c + "/topology/" + file;
            std::string text;
            if (!lexists(path)) continue;
            any = true;
            if (read_file(path, text)) lists.insert(std::string(strip_bytes(text)));
        }
        if (any) break;
    }
    if (!lists.empty()) return (int64_t) lists.size();
    std::string text;
    if (!read_file(proc + "/cpuinfo", text)) return nullptr;
    std::map<int64_t, int64_t> cores;          // physical id -> cpu cores
    std::map<std::string, int64_t> section;
    for (const auto line : lines(text)) {
        const std::string l = lower(strip_bytes(line));
        if (l.empty()) {
            const auto id = section.find("physical id"), n = section.find("cpu cores");
            if (id != section.end() && n != section.end()) cores[id->second] = n->second;
            section.clear();
        } else if (starts_with(l, "physical id") || starts_with(l, "cpu cores")) {
            const size_t at = l.find("\t:");
            if (at == std::string::npos) continue;
            if (const auto v = to_int(std::string_view(l).substr(at + 2))) section[l.substr(0, at)] = *v;
        }
    }
    int64_t sum = 0;
    for (const auto& kv : cores) sum += kv.second;
    return sum >= 1 ? json(sum) : json(nullptr);
}

/// _cpu_topology(): (CPU packages, NUMA nodes)
std::pair<json, json> topology(const std::string& sys) {
    const std::string root = sys + "/devices/system";
    std::set<std::string> packages;
    for (const auto& c : numbered(root + "/cpu", "cpu")) {
        std::string text;
        if (read_file(root + "/cpu/" + c + "/topology/physical_package_id", text))
            packages.insert(std::string(strip(text)));
    }
    const size_t nodes = numbered(root + "/node", "node").size();
    return {packages.empty() ? json(nullptr) : json((int64_t) packages.size()),
            nodes ? json((int64_t) nodes) : json(nullptr)};
}

/// _cpu_name(): /proc/cpuinfo's first "model name" (null: none)
json cpu_name(const std::string& proc) {
    std::string text;
    if (!read_file(proc + "/cpuinfo", text)) return nullptr;
    const std::string decoded = utf8_replace(text);
    for (const auto line : lines(decoded)) {
        if (!starts_with(line, "model name")) continue;
        const size_t colon = line.find(':');
        if (colon != std::string_view::npos) return std::string(strip(line.substr(colon + 1)));
    }
    return nullptr;
}

}  // namespace procfs
#endif

#ifndef _WIN32
/// platform.processor() on Linux: `uname -p` (none when it says "unknown")
json uname_p() {
    FILE* p = popen("uname -p 2>/dev/null", "r");
    if (!p) return nullptr;
    std::string out;
    char buf[256];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, p)) > 0;) out.append(buf, n);
    if (pclose(p) != 0) return nullptr;
    const std::string s(strip(out));
    return s.empty() || s == "unknown" ? json(nullptr) : json(s);
}
#endif

// ---------------------------------------------------------------------------------------------- this computer
#ifdef _WIN32
constexpr bool kProcStat = false;   // the CPU times' fields: Windows' (user system idle interrupt dpc)
#else
constexpr bool kProcStat = true;    // /proc/stat's
#endif

std::optional<std::vector<double>> os_cpu_times() {
#ifdef _WIN32
    return win::cpu_times();
#else
    return procfs::cpu_times("/proc", (double) sysconf(_SC_CLK_TCK));
#endif
}

std::optional<std::pair<int64_t, int64_t>> os_virtual_memory() {
#ifdef _WIN32
    return win::virtual_memory();
#else
    return procfs::virtual_memory("/proc", (int64_t) sysconf(_SC_PAGESIZE));
#endif
}

std::optional<std::vector<DiskBytes>> os_disk_counters() {
#ifdef _WIN32
    return win::disk_counters();
#else
    return procfs::disk_counters("/proc", "/sys");
#endif
}

// ---------------------------------------------------------------------------------------------- the readings
/// psutil.cpu_percent()'s figure from two cpu_times(): the deltas (clamped at 0: CPU times can step back), their
/// total less Linux's guest times (already in user and nice), busy = total - idle (- Linux's iowait)
double cpu_busy_percent(const std::vector<double>& t1, const std::vector<double>& t2, bool proc_stat) {
    std::vector<double> d(t2.size());
    for (size_t i = 0; i < d.size(); ++i) d[i] = std::max(0.0, t2[i] - (i < t1.size() ? t1[i] : 0.0));
    double total = py_fsum(d);
    if (proc_stat && d.size() > 8) total -= d[8];   // guest
    if (proc_stat && d.size() > 9) total -= d[9];   // guest_nice
    const size_t idle = proc_stat ? 3 : 2;
    double busy = total - (idle < d.size() ? d[idle] : 0.0);
    if (proc_stat && d.size() > 4) busy -= d[4];    // iowait
    if (total == 0.0) return 0.0;
    return py_round(busy / total * 100, 1);
}

/// psutil.cpu_percent(interval=None): the CPU's busy share since the last call (the first call compares two readings
/// taken back to back: psutil's "meaningless 0.0")
class CpuPercent {
public:
    std::optional<double> percent() {
        std::optional<std::vector<double>> t1 = last_;
        if (!t1) t1 = os_cpu_times();
        if (!t1) return std::nullopt;
        const auto t2 = os_cpu_times();
        if (!t2 || t2->size() != t1->size()) return std::nullopt;
        last_ = t2;
        return cpu_busy_percent(*t1, *t2, kProcStat);
    }

private:
    std::optional<std::vector<double>> last_;
};

/// psutil's nowrap: a disk counter that went back (its counters reset, a 32-bit counter wrapped) goes on from the
/// sum of what it had counted
class WrapNumbers {
public:
    std::vector<DiskBytes> run(const std::vector<DiskBytes>& in) {
        if (!last_) {
            last_ = in;
            return in;
        }
        for (const DiskBytes& was : *last_) {   // a disk that is gone takes its reminders with it
            if (find(in, was.name)) continue;
            for (const int i : reminder_keys_[was.name]) reminders_.erase({was.name, i});
            reminder_keys_.erase(was.name);
        }
        std::vector<DiskBytes> out;
        for (const DiskBytes& d : in) {
            const DiskBytes* was = find(*last_, d.name);
            if (!was) {
                out.push_back(d);
                continue;
            }
            DiskBytes x = d;
            int64_t* const now[2] = {&x.read, &x.written};
            const int64_t before[2] = {was->read, was->written};
            for (int i = 0; i < 2; ++i) {
                if (*now[i] < before[i]) {
                    reminders_[{d.name, i}] += before[i];
                    reminder_keys_[d.name].insert(i);
                }
                const auto r = reminders_.find({d.name, i});
                if (r != reminders_.end()) *now[i] += r->second;
            }
            out.push_back(x);
        }
        last_ = in;
        return out;
    }

private:
    static const DiskBytes* find(const std::vector<DiskBytes>& v, const std::string& name) {
        for (const DiskBytes& d : v)
            if (d.name == name) return &d;
        return nullptr;
    }

    std::optional<std::vector<DiskBytes>> last_;
    std::map<std::pair<std::string, int>, int64_t> reminders_;
    std::map<std::string, std::set<int>> reminder_keys_;
};

/// Telemetry._disk: the disks' read and write rates in MiB/s since the last reading; null the first time, and when
/// the counters cannot be read or there is no disk (a gVisor container; Windows with its disk counters off)
class DiskRate {
public:
    std::pair<json, json> rates(const std::optional<std::vector<DiskBytes>>& raw, double t) {
        if (!raw || raw->empty()) return {nullptr, nullptr};
        Reading now{t, {}};
        for (const DiskBytes& d : wrap_.run(*raw)) now.disks[d.name] = {d.read, d.written};
        const std::optional<Reading> prev = std::move(prev_);
        prev_ = std::move(now);
        if (!prev || t <= prev->t) return {nullptr, nullptr};
        // each disk against its own last reading: one that came or went since moves no sum (it made the rate jump or
        // go negative)
        int64_t rd = 0, wr = 0;
        for (const auto& [name, v] : prev_->disks)
            if (const auto p = prev->disks.find(name); p != prev->disks.end()) {
                rd += v.first - p->second.first;
                wr += v.second - p->second.second;
            }
        const double dt = t - prev->t;
        return {(double) rd / dt / 1048576.0, (double) wr / dt / 1048576.0};
    }

private:
    struct Reading {
        double t;
        std::map<std::string, std::pair<int64_t, int64_t>> disks;   // name -> (read, written)
    };
    WrapNumbers wrap_;
    std::optional<Reading> prev_;
};

/// Telemetry.sample's figures from its readings: `reads` the cards' (index, NVML reading) when the first card is
/// readable (else empty: no GPU figures), `names` their names (read for several cards only), the CPU load and RAM
/// (total, available) (nullopt: psutil raised), the disk rates and extra()'s figures
json assemble(const std::vector<std::pair<int, json>>& reads, const std::vector<json>& names,
              const std::optional<double>& load, const std::optional<std::pair<int64_t, int64_t>>& ram,
              const json& disk_read, const json& disk_write, const json& more) {
    json s = json::object();
    if (!reads.empty()) {
        json g = reads.front().second;
        if (reads.size() > 1) {
            auto vals = [&](const char* k) {
                std::vector<json> v;
                for (const auto& r : reads)
                    if (const json& x = jget(r.second, k); !x.is_null()) v.push_back(x);
                return v;
            };
            for (const char* k : {"mem_used", "mem_total", "power", "power_limit", "pcie_rx_mb", "pcie_tx_mb"}) {
                const auto v = vals(k);
                g[k] = v.empty() ? json(nullptr) : py_sum(v);
            }
            const auto u = vals("util");
            g["util"] = u.empty() ? json(nullptr) : json(py_sum(u).get<double>() / (double) u.size());
            const auto t = vals("temp");
            g["temp"] = t.empty() ? json(nullptr) : py_max(t);
            json each = json::array();
            for (size_t i = 0; i < reads.size(); ++i) {
                const json& r = reads[i].second;
                json one = {{"index", reads[i].first}, {"name", i < names.size() ? names[i] : json(nullptr)},
                            {"util", jget(r, "util")}, {"mem_used", jget(r, "mem_used")},
                            {"mem_total", jget(r, "mem_total")}, {"temp", jget(r, "temp")},
                            {"power", jget(r, "power")}};
                each.push_back(std::move(one));
            }
            s["gpus"] = std::move(each);
        }
        for (auto it = g.begin(); it != g.end(); ++it) s["gpu_" + it.key()] = it.value();
    }
    if (load) {
        s["cpu"] = *load;
        if (ram) {
            s["ram_used"] = ram->first - ram->second;
            s["ram_total"] = ram->first;
        }
    }
    s["disk_read_mb"] = disk_read;
    s["disk_write_mb"] = disk_write;
    if (more.is_object())
        for (auto it = more.begin(); it != more.end(); ++it) s[it.key()] = it.value();
    return s;
}

/// a sample into the sparklines' series (floats rounded to 2 places), the last kHistory of each
void record(json& hist, const json& s) {
    for (const char* k : kSeries) {
        const json& v = jget(s, k);
        json& h = hist[k];
        if (!h.is_array()) h = json::array();
        h.push_back(v.is_number_float() ? json(py_round(v.get<double>(), 2)) : v);
        if (h.size() > kHistory) h.erase(h.begin());
    }
}

/// the machine's facts ("static")
json machine_facts(const std::vector<std::pair<int, GpuReader>>& gpus) {
    json gpu_name = nullptr;
    if (gpus.front().second.ok()) {
        std::string names;
        for (size_t i = 0; i < gpus.size(); ++i) {
            const json n = gpus[i].second.name();
            names += (i ? " + " : "") + (truthy(n) ? n.get<std::string>() : std::string("?"));
        }
        gpu_name = names;
    }
#ifdef _WIN32
    const auto [sockets, numa] = win::topology();
    const json cpu = win::cpu_name(), cores = win::cpu_cores(), threads = win::cpu_threads();
#else
    const auto [sockets, numa] = procfs::topology("/sys");
    json cpu = procfs::cpu_name("/proc");
    if (cpu.is_null()) cpu = uname_p();
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    const json cores = procfs::cpu_cores("/proc", "/sys");
    const json threads = online >= 1 ? json((int64_t) online) : json(nullptr);
#endif
    // "psutil": the CPU, RAM and disk figures are psutil's (the web app's disk card asks for them)
    return {{"gpu_name", gpu_name}, {"gpu_vram_total_mib", nullptr}, {"gpu_count", (int64_t) gpus.size()},
            {"gpu_note", nullptr}, {"cpu_name", cpu}, {"cores", cores}, {"threads", threads}, {"sockets", sockets},
            {"numa_nodes", numa}, {"psutil", true}};
}

}  // namespace

// ---------------------------------------------------------------------------------------------- the sampler
struct Telemetry::Impl {
    std::function<json()> extra;
    std::vector<std::pair<int, GpuReader>> gpus;
    json facts;
    CpuPercent cpu;   // the sampler thread's
    DiskRate disk;
    std::mutex mu;    // now, hist
    json now = json::object(), hist = json::object();
    std::mutex stop_mu, join_mu;
    std::condition_variable stop_cv;
    bool stop = false;
    std::thread thread;

    json sample() {
        std::vector<std::pair<int, json>> reads;
        std::vector<json> names;
        if (gpus.front().second.ok()) {
            for (const auto& g : gpus) reads.emplace_back(g.first, g.second.read());
            if (reads.size() > 1)
                for (const auto& g : gpus) names.push_back(g.second.name());
        }
        const std::optional<double> load = cpu.percent();
        std::optional<std::pair<int64_t, int64_t>> ram;
        if (load) ram = os_virtual_memory();
        const auto raw = os_disk_counters();
        const auto [rd, wr] = disk.rates(raw, time_time());
        json more = nullptr;
        if (extra) {
            try {
                more = extra();
            } catch (...) {   // telemetry must never take the server down
            }
        }
        return assemble(reads, names, load, ram, rd, wr, more);
    }

    void loop() {
        std::unique_lock<std::mutex> lk(stop_mu);
        while (!stop) {
            lk.unlock();
            try {
                json s = sample();
                std::lock_guard<std::mutex> g(mu);
                record(hist, s);
                now = std::move(s);
            } catch (...) {   // nothing here may end the server (an exception out of a thread would): skip the second
            }
            lk.lock();
            stop_cv.wait_for(lk, std::chrono::seconds(1), [this] { return stop; });
        }
    }
};

Telemetry::Telemetry(std::function<json()> extra, int gpu_index, std::vector<int> gpu_indices)
    : impl_(std::make_shared<Impl>()) {
    Impl& m = *impl_;
    m.extra = std::move(extra);
    const std::vector<int> idx = gpu_indices.size() > 1 ? std::move(gpu_indices) : std::vector<int>{gpu_index};
    std::vector<std::pair<int, GpuReader>> all;
    for (const int i : idx) all.emplace_back(i, GpuReader(i));
    for (const auto& g : all)
        if (g.second.ok()) m.gpus.push_back(g);
    if (m.gpus.empty()) m.gpus.push_back(all.front());   // none readable: the first, for its count and its nulls
    m.facts = machine_facts(m.gpus);
    m.thread = std::thread([self = impl_] { self->loop(); });
}

Telemetry::~Telemetry() {
    close();
    if (impl_->thread.joinable()) impl_->thread.detach();   // closed from its own extra(): it ends by itself
}

json Telemetry::snapshot() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return {{"now", impl_->now}, {"history", impl_->hist}, {"static", impl_->facts}};
}

void Telemetry::close() {
    {
        std::lock_guard<std::mutex> lk(impl_->stop_mu);
        impl_->stop = true;
    }
    impl_->stop_cv.notify_all();
    std::lock_guard<std::mutex> lk(impl_->join_mu);
    if (impl_->thread.joinable() && impl_->thread.get_id() != std::this_thread::get_id()) impl_->thread.join();
}

int64_t Telemetry::free_vram_mib(int index) {
    const GpuReader g(index);
    const auto m = g.memory();   // read()'s mem_total and mem_used
    return m ? ((int64_t) m->total - (int64_t) m->used) >> 20 : -1;
}

}  // namespace lamina
