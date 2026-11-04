// src/server/process.cpp - child processes tied to the server's lifetime (process.hpp).
#include "process.hpp"

#include "log.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <filesystem>
#include <functional>
#include <future>
#include <mutex>
#include <stdexcept>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <fcntl.h>
#include <io.h>
#else
#include <cerrno>
#include <csignal>
#include <fcntl.h>
#include <sched.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
#endif

namespace lamina {
namespace {

constexpr size_t kTailLines = 40;   // stderr lines kept for a child's error message

/// splits a pipe's bytes into lines for a LineChannel (the line end, \n or \r\n, dropped)
struct LineSplitter {
    std::string part;
    bool cr = false;   ///< a lone \r ends a line too (Python's universal newlines: a progress bar's updates)
    void feed(const char* p, size_t n, LineChannel& out) {
        part.append(p, n);
        size_t start = 0;
        for (size_t i = 0; i < part.size(); ++i) {
            if (part[i] == '\n') {
                size_t end = i;
                if (end > start && part[end - 1] == '\r') --end;
                out.push(part.substr(start, end - start));
                start = i + 1;
            } else if (cr && part[i] == '\r') {
                if (i + 1 == part.size()) break;   // a \n may follow in the next piece
                if (part[i + 1] == '\n') continue;
                out.push(part.substr(start, i - start));
                start = i + 1;
            }
        }
        part.erase(0, start);
    }
    void end(LineChannel& out) {
        if (cr && !part.empty() && part.back() == '\r') part.pop_back();
        if (!part.empty()) out.push(part);
        part.clear();
        out.close();
    }
};

/// what Popen refuses before it starts anything: a NUL in an argument, an environment name with '='
void check_spawn(const SpawnOptions& o) {
    for (const auto& a : o.args)
        if (a.find('\0') != std::string::npos) throw std::invalid_argument("embedded null byte");
    if (o.env)
        for (const auto& [k, v] : *o.env) {
            if (k.empty() || k.find('=') != std::string::npos) throw std::invalid_argument("illegal environment variable name");
            if (k.find('\0') != std::string::npos || v.find('\0') != std::string::npos)
                throw std::invalid_argument("embedded null byte");
        }
}

}  // namespace

bool EnvLess::operator()(const std::string& a, const std::string& b) const {
#ifdef _WIN32
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(), [](char x, char y) {
        return std::toupper((unsigned char) x) < std::toupper((unsigned char) y);
    });
#else
    return a < b;
#endif
}

std::string exit_code_text(int code) {
#ifdef _WIN32
    return std::to_string((uint32_t) code);
#else
    return std::to_string(code);
#endif
}

#ifdef _WIN32
// ================================================================================================= Windows
namespace {

std::string os_error(DWORD code) {
    wchar_t* buf = nullptr;
    FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS, nullptr,
                   code, 0, (LPWSTR) &buf, 0, nullptr);
    std::string text = buf != nullptr ? narrow(buf) : "";
    if (buf != nullptr) LocalFree(buf);
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ' || text.back() == '.'))
        text.pop_back();   // Python's wording: no line end, no final period
    return "[WinError " + std::to_string(code) + "] " + text;
}

/// subprocess.list2cmdline: the quoting the MSVC runtime's argv parser reads back
std::string list2cmdline(const std::vector<std::string>& args) {
    std::string out;
    for (const auto& arg : args) {
        if (!out.empty()) out += ' ';
        const bool quote = arg.empty() || arg.find_first_of(" \t") != std::string::npos;
        if (quote) out += '"';
        size_t backslashes = 0;
        for (const char c : arg) {
            if (c == '\\') {
                ++backslashes;
            } else if (c == '"') {
                out.append(backslashes * 2, '\\');
                backslashes = 0;
                out += "\\\"";
            } else {
                out.append(backslashes, '\\');
                backslashes = 0;
                out += c;
            }
        }
        out.append(backslashes, '\\');
        if (quote) {
            out.append(backslashes, '\\');
            out += '"';
        }
    }
    return out;
}

bool no_throttle(HANDLE process) {
    struct Throttling {
        ULONG version, control_mask, state_mask;
    };
    // EXECUTION_SPEED and IGNORE_TIMER_RESOLUTION, both off (Windows 11); Windows 10 knows the first only
    for (const ULONG mask : {0x1UL | 0x4UL, 0x1UL}) {
        Throttling t{1, mask, 0};
        if (SetProcessInformation(process, (PROCESS_INFORMATION_CLASS) 4 /* ProcessPowerThrottling */, &t, sizeof t))
            return true;
    }
    return false;
}

HANDLE inheritable_null(bool write) {
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    return CreateFileW(L"NUL", write ? GENERIC_WRITE : GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa,
                       OPEN_EXISTING, 0, nullptr);
}

HANDLE inheritable_copy(DWORD which, bool write) {
    const HANDLE h = GetStdHandle(which);
    HANDLE d = nullptr;
    if (h != nullptr && h != INVALID_HANDLE_VALUE &&
        DuplicateHandle(GetCurrentProcess(), h, GetCurrentProcess(), &d, 0, TRUE, DUPLICATE_SAME_ACCESS))
        return d;
    return inheritable_null(write);
}

}  // namespace

struct Process::Impl {
    HANDLE process = nullptr, job = nullptr, in = nullptr, out = nullptr, err = nullptr;
    DWORD pid = 0;
    std::shared_ptr<LineChannel> lines = std::make_shared<LineChannel>();
    std::thread out_reader, err_reader;
    std::mutex in_mu, tail_mu;
    std::deque<std::string> tail;

    void read_out() {
        LineSplitter s;
        s.cr = cr_lines;
        char buf[65536];
        DWORD n = 0;
        while (ReadFile(out, buf, sizeof buf, &n, nullptr) && n > 0) s.feed(buf, n, *lines);
        s.end(*lines);
    }
    void read_err() {
        LineChannel sink;
        LineSplitter s;
        char buf[16384];
        DWORD n = 0;
        while (ReadFile(err, buf, sizeof buf, &n, nullptr) && n > 0) {
            s.feed(buf, n, sink);
            keep_err(sink);
        }
        s.end(sink);   // the last line, also without a line end
        keep_err(sink);
        err_done = true;
    }
    void keep_err(LineChannel& sink) {
        std::string line;
        while (sink.pop(line, 0) == 1) {
            if (on_err_line) on_err_line(line);
            std::lock_guard<std::mutex> lk(tail_mu);
            tail.push_back(line);
            while (tail.size() > kTailLines) tail.pop_front();
        }
    }
    std::function<void(const std::string&)> on_err_line;
    std::atomic<bool> err_done{false};
    bool cr_lines = false;
};

Process::Process(SpawnOptions o) : impl_(std::make_unique<Impl>()) {
    if (o.args.empty()) throw std::runtime_error("no program to start");
    check_spawn(o);
    Impl& p = *impl_;
    SECURITY_ATTRIBUTES sa{sizeof sa, nullptr, TRUE};
    HANDLE child[3] = {nullptr, nullptr, nullptr};
    auto close_child = [&] {
        for (HANDLE& h : child)
            if (h != nullptr && h != INVALID_HANDLE_VALUE) {
                CloseHandle(h);
                h = nullptr;
            }
    };
    auto pipe = [&](HANDLE& ours, HANDLE& theirs, bool child_reads) {
        HANDLE r = nullptr, w = nullptr;
        if (!CreatePipe(&r, &w, &sa, 0)) throw std::runtime_error(os_error(GetLastError()));
        ours = child_reads ? w : r;
        theirs = child_reads ? r : w;
        SetHandleInformation(ours, HANDLE_FLAG_INHERIT, 0);
    };
    try {
        if (o.in == SpawnOptions::pipe) pipe(p.in, child[0], true);
        else if (o.in == SpawnOptions::null) child[0] = inheritable_null(false);
        else child[0] = inheritable_copy(STD_INPUT_HANDLE, false);
        if (o.out == SpawnOptions::pipe) pipe(p.out, child[1], false);
        else if (o.out == SpawnOptions::null) child[1] = inheritable_null(true);
        else child[1] = inheritable_copy(STD_OUTPUT_HANDLE, true);
        if (o.err == SpawnOptions::pipe) {
            pipe(p.err, child[2], false);
        } else if (o.err == SpawnOptions::null) {
            child[2] = inheritable_null(true);
        } else if (o.err == SpawnOptions::file) {
            child[2] = CreateFileW(widen(o.err_file).c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE |
                                   FILE_SHARE_DELETE, &sa, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
            if (child[2] == INVALID_HANDLE_VALUE) throw std::runtime_error(os_error(GetLastError()) + ": " + o.err_file);
        } else {
            child[2] = inheritable_copy(STD_ERROR_HANDLE, true);
        }
    } catch (...) {
        close_child();
        if (p.in) CloseHandle(p.in);
        if (p.out) CloseHandle(p.out);
        if (p.err) CloseHandle(p.err);
        throw;
    }
    std::string cmdline;
    if (o.shell) {
        const char* comspec = std::getenv("COMSPEC");
        cmdline = std::string(comspec != nullptr ? comspec : "cmd.exe") + " /c \"" + o.args[0] + "\"";
    } else {
        cmdline = list2cmdline(o.args);
    }
    std::wstring wcmd = widen(cmdline);
    std::wstring block;   // the environment: sorted NAME=value strings, a double NUL at the end
    if (o.env) {
        for (const auto& [k, v] : *o.env) {
            block += widen(k + "=" + v);
            block.push_back(L'\0');
        }
        if (block.empty()) block.push_back(L'\0');
        block.push_back(L'\0');
    }
    SIZE_T attr_size = 0;
    InitializeProcThreadAttributeList(nullptr, 1, 0, &attr_size);
    std::vector<char> attr_buf(attr_size);
    auto* attrs = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(attr_buf.data());
    InitializeProcThreadAttributeList(attrs, 1, 0, &attr_size);
    // only its own three handles are inherited: another child's pipe end in it would keep that pipe open
    HANDLE list[3] = {child[0], child[1], child[2]};
    UpdateProcThreadAttribute(attrs, 0, PROC_THREAD_ATTRIBUTE_HANDLE_LIST, list, sizeof list, nullptr, nullptr);
    STARTUPINFOEXW si{};
    si.StartupInfo.cb = sizeof si;
    si.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
    si.StartupInfo.hStdInput = child[0];
    si.StartupInfo.hStdOutput = child[1];
    si.StartupInfo.hStdError = child[2];
    si.lpAttributeList = attrs;
    DWORD flags = CREATE_UNICODE_ENVIRONMENT | EXTENDED_STARTUPINFO_PRESENT;
    if (o.contain) flags |= CREATE_SUSPENDED;
    if (o.own_group) flags |= CREATE_NEW_PROCESS_GROUP;
    if (o.no_window) flags |= CREATE_NO_WINDOW;
    PROCESS_INFORMATION pi{};
    const std::wstring wcwd = widen(o.cwd);
    const BOOL ok = CreateProcessW(nullptr, wcmd.data(), nullptr, nullptr, TRUE, flags,
                                   o.env ? (LPVOID) block.data() : nullptr, o.cwd.empty() ? nullptr : wcwd.c_str(),
                                   &si.StartupInfo, &pi);
    const DWORD error = GetLastError();
    DeleteProcThreadAttributeList(attrs);
    close_child();
    if (!ok) {
        if (p.in) CloseHandle(p.in);
        if (p.out) CloseHandle(p.out);
        if (p.err) CloseHandle(p.err);
        p.in = p.out = p.err = nullptr;
        throw std::runtime_error(os_error(error));
    }
    p.process = pi.hProcess;
    p.pid = pi.dwProcessId;
    if (o.contain) {
        p.job = CreateJobObjectW(nullptr, nullptr);   // no security attributes: the handle is not inherited
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION info{};
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if (p.job != nullptr && (!SetInformationJobObject(p.job, JobObjectExtendedLimitInformation, &info, sizeof info) ||
                                 !AssignProcessToJobObject(p.job, pi.hProcess))) {
            CloseHandle(p.job);
            p.job = nullptr;
        }
        no_throttle(pi.hProcess);
        ResumeThread(pi.hThread);
    }
    CloseHandle(pi.hThread);
    p.cr_lines = o.cr_lines;
    if (p.out != nullptr) p.out_reader = std::thread([&p] { p.read_out(); });
    else p.lines->close();
    p.on_err_line = std::move(o.on_err_line);
    if (p.err != nullptr) p.err_reader = std::thread([&p] { p.read_err(); });
    else p.err_done = true;
}

Process::~Process() {
    Impl& p = *impl_;
    if (p.job != nullptr) TerminateJobObject(p.job, 1);   // also what it left running, which may hold its pipes
    else if (!poll()) kill();
    if (p.in != nullptr) CloseHandle(p.in);
    if (p.process != nullptr) WaitForSingleObject(p.process, 5000);
    if (p.out_reader.joinable()) p.out_reader.join();
    if (p.err_reader.joinable()) p.err_reader.join();
    if (p.out != nullptr) CloseHandle(p.out);
    if (p.err != nullptr) CloseHandle(p.err);
    if (p.job != nullptr) CloseHandle(p.job);   // kill-on-close: whatever it left running ends here
    if (p.process != nullptr) CloseHandle(p.process);
}

bool Process::write(std::string_view data) {
    Impl& p = *impl_;
    std::lock_guard<std::mutex> lk(p.in_mu);
    if (p.in == nullptr) return false;
    while (!data.empty()) {
        DWORD n = 0;
        if (!WriteFile(p.in, data.data(), (DWORD) std::min<size_t>(data.size(), 1 << 20), &n, nullptr)) return false;
        data.remove_prefix(n);
    }
    return true;
}

void Process::close_stdin() {
    Impl& p = *impl_;
    std::lock_guard<std::mutex> lk(p.in_mu);
    if (p.in != nullptr) CloseHandle(p.in);
    p.in = nullptr;
}

std::optional<int> Process::poll() {
    DWORD code = 0;
    if (impl_->process == nullptr || WaitForSingleObject(impl_->process, 0) != WAIT_OBJECT_0) return std::nullopt;
    GetExitCodeProcess(impl_->process, &code);
    return (int) code;
}

std::optional<int> Process::wait(double timeout_s) {
    const DWORD ms = timeout_s < 0 ? INFINITE : (DWORD) std::min(timeout_s * 1000.0, 4.0e9);
    if (WaitForSingleObject(impl_->process, ms) != WAIT_OBJECT_0) return std::nullopt;
    return poll();
}

void Process::kill() {
    Impl& p = *impl_;
    if (p.job != nullptr) TerminateJobObject(p.job, 1);
    else if (p.process != nullptr) TerminateProcess(p.process, 1);
}

long Process::pid() const { return (long) impl_->pid; }

Environment environment() {
    Environment env;
    wchar_t* block = GetEnvironmentStringsW();
    for (const wchar_t* s = block; s != nullptr && *s; s += wcslen(s) + 1) {
        const std::string kv = narrow(s);
        const size_t eq = kv.find('=', 1);   // "=C:=C:\..." (a drive's current directory) starts with '='
        if (eq == std::string::npos || kv[0] == '=') continue;
        env[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    if (block != nullptr) FreeEnvironmentStringsW(block);
    return env;
}

std::string which(const std::string& name, const Environment* env) {
    namespace fs = std::filesystem;
    const Environment here = env == nullptr ? environment() : Environment{};
    const Environment& e = env != nullptr ? *env : here;
    auto get = [&](const char* k) {
        const auto it = e.find(k);
        return it == e.end() ? std::string() : it->second;
    };
    std::vector<std::string> exts = split(get("PATHEXT").empty() ? ".COM;.EXE;.BAT;.CMD" : get("PATHEXT"), ";");
    // the found path in Windows' own form (cmd.exe reads "bin/x.cmd" as "bin" and a switch), as UTF-8
    auto found = [](fs::path p) { return u8_str(p.make_preferred()); };
    auto probe = [&](const fs::path& base) -> std::string {
        const std::string low = lower(u8_str(base.filename()));
        for (const auto& x : exts)
            if (!x.empty() && ends_with(low, lower(x))) {
                std::error_code ec;
                return fs::is_regular_file(base, ec) ? found(base) : "";
            }
        for (const auto& x : exts) {
            if (x.empty()) continue;
            fs::path c = base;
            c += x;
            std::error_code ec;
            if (fs::is_regular_file(c, ec)) return found(c);
        }
        return "";
    };
    const fs::path n = u8_path(name);
    if (name.find_first_of("/\\") != std::string::npos) return probe(n);
    // shutil.which on Windows: the current directory first, unless NoDefaultCurrentDirectoryInExePath is set
    std::vector<std::string> dirs;
    if (e.find("NoDefaultCurrentDirectoryInExePath") == e.end()) dirs.push_back(".");
    for (const auto& dir : split(get("PATH"), ";")) dirs.push_back(dir);
    for (const auto& dir : dirs) {
        if (dir.empty()) continue;
        const std::string hit = probe(u8_path(dir) / n);
        if (!hit.empty()) return hit;
    }
    return "";
}

bool no_throttle_self() { return no_throttle(GetCurrentProcess()); }

void redirect_stderr(const std::string& path) {
    const int saved = _dup(2);
    if (saved >= 0) {
        if (std::FILE* console = _fdopen(saved, "w")) set_err_stream(console);
    }
    const HANDLE h = CreateFileW(widen(path).c_str(), FILE_APPEND_DATA, FILE_SHARE_READ | FILE_SHARE_WRITE |
                                 FILE_SHARE_DELETE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (h == INVALID_HANDLE_VALUE) return;
    const int fd = _open_osfhandle((intptr_t) h, _O_APPEND | _O_WRONLY);
    if (fd < 0) {
        CloseHandle(h);
        return;
    }
    std::fflush(stderr);
    _dup2(fd, 2);
    _close(fd);
    SetStdHandle(STD_ERROR_HANDLE, (HANDLE) _get_osfhandle(2));
}

std::string self_dir() {
    std::wstring buf(32768, L' ');
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), (DWORD) buf.size());
    buf.resize(n);
    return u8_str(std::filesystem::path(buf).parent_path());
}

std::string temp_dir() {
    wchar_t buf[MAX_PATH + 2];
    const DWORD n = GetTempPathW(MAX_PATH + 1, buf);
    std::string s = narrow(std::wstring_view(buf, n));
    while (!s.empty() && (s.back() == '\\' || s.back() == '/')) s.pop_back();
    return s;
}

#else
// ================================================================================================= POSIX
namespace {

std::string os_error(int code) { return "[Errno " + std::to_string(code) + "] " + std::strerror(code); }

/// one long-lived thread forks every child: PR_SET_PDEATHSIG fires when the thread that forked ends
class Anchor {
public:
    Anchor() : t_([this] { loop(); }) { t_.detach(); }
    template <class F>
    auto call(F f) -> decltype(f()) {
        std::packaged_task<decltype(f())()> task(std::move(f));
        auto fut = task.get_future();
        {
            std::lock_guard<std::mutex> lk(mu_);
            q_.emplace_back([&task] { task(); });
        }
        cv_.notify_one();
        return fut.get();
    }

private:
    void loop() {
        for (;;) {
            std::function<void()> f;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&] { return !q_.empty(); });
                f = std::move(q_.front());
                q_.pop_front();
            }
            f();
        }
    }
    std::mutex mu_;
    std::condition_variable cv_;
    std::deque<std::function<void()>> q_;
    std::thread t_;
};

Anchor& anchor() {
    static Anchor* a = new Anchor();   // never destroyed: it must outlive every child
    return *a;
}

}  // namespace

struct Process::Impl {
    pid_t pid = -1;
    bool group = false;
    int in = -1, out = -1, err = -1;
    std::optional<int> code;
    std::mutex wait_mu, in_mu, tail_mu;
    std::shared_ptr<LineChannel> lines = std::make_shared<LineChannel>();
    std::thread out_reader, err_reader;
    std::deque<std::string> tail;

    void read_out() {
        LineSplitter s;
        s.cr = cr_lines;
        char buf[65536];
        for (ssize_t n; (n = ::read(out, buf, sizeof buf)) != 0;) {
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            s.feed(buf, (size_t) n, *lines);
        }
        s.end(*lines);
    }
    void read_err() {
        LineChannel sink;
        LineSplitter s;
        char buf[16384];
        for (ssize_t n; (n = ::read(err, buf, sizeof buf)) != 0;) {
            if (n < 0) {
                if (errno == EINTR) continue;
                break;
            }
            s.feed(buf, (size_t) n, sink);
            keep_err(sink);
        }
        s.end(sink);   // the last line, also without a line end
        keep_err(sink);
        err_done = true;
    }
    void keep_err(LineChannel& sink) {
        std::string line;
        while (sink.pop(line, 0) == 1) {
            if (on_err_line) on_err_line(line);
            std::lock_guard<std::mutex> lk(tail_mu);
            tail.push_back(line);
            while (tail.size() > kTailLines) tail.pop_front();
        }
    }
    std::function<void(const std::string&)> on_err_line;
    std::atomic<bool> err_done{false};
    bool cr_lines = false;
};

Process::Process(SpawnOptions o) : impl_(std::make_unique<Impl>()) {
    if (o.args.empty()) throw std::runtime_error("no program to start");
    check_spawn(o);
    Impl& p = *impl_;
    int in_pipe[2] = {-1, -1}, out_pipe[2] = {-1, -1}, err_pipe[2] = {-1, -1}, status[2] = {-1, -1};
    auto closeall = [](std::initializer_list<int*> fds) {
        for (int* fd : fds)
            if (*fd >= 0) {
                ::close(*fd);
                *fd = -1;
            }
    };
    // close-on-exec from the start: a child another thread starts meanwhile would otherwise hold these ends (a
    // replica's stdout write end kept open by an MCP server: no EOF when the replica dies); dup2 clears it for 0-2
    if ((o.in == SpawnOptions::pipe && ::pipe2(in_pipe, O_CLOEXEC) != 0) ||
        (o.out == SpawnOptions::pipe && ::pipe2(out_pipe, O_CLOEXEC) != 0) ||
        (o.err == SpawnOptions::pipe && ::pipe2(err_pipe, O_CLOEXEC) != 0) || ::pipe2(status, O_CLOEXEC) != 0) {
        const int e = errno;
        closeall({&in_pipe[0], &in_pipe[1], &out_pipe[0], &out_pipe[1], &err_pipe[0], &err_pipe[1], &status[0], &status[1]});
        throw std::runtime_error(os_error(e));
    }
    // everything the child needs, prepared before the fork (only async-signal-safe calls after it)
    std::vector<std::string> argv_s = o.shell ? std::vector<std::string>{"/bin/sh", "-c", o.args[0]} : o.args;
    std::vector<char*> argv;
    for (auto& a : argv_s) argv.push_back(a.data());
    argv.push_back(nullptr);
    std::vector<std::string> env_s;
    std::vector<char*> envp;
    if (o.env) {
        for (const auto& [k, v] : *o.env) env_s.push_back(k + "=" + v);
        for (auto& s : env_s) envp.push_back(s.data());
        envp.push_back(nullptr);
    }
    std::string exe = argv_s[0];
    if (exe.find('/') == std::string::npos) {
        const std::string found = which(exe, o.env ? &*o.env : nullptr);
        if (found.empty()) {
            closeall({&in_pipe[0], &in_pipe[1], &out_pipe[0], &out_pipe[1], &err_pipe[0], &err_pipe[1], &status[0], &status[1]});
            throw std::runtime_error(os_error(ENOENT) + ": '" + exe + "'");
        }
        exe = found;
    }
    int err_fd = -1;
    if (o.err == SpawnOptions::file) {
        err_fd = ::open(o.err_file.c_str(), O_WRONLY | O_APPEND | O_CREAT | O_CLOEXEC, 0644);
        if (err_fd < 0) {
            const int e = errno;
            closeall({&in_pipe[0], &in_pipe[1], &out_pipe[0], &out_pipe[1], &err_pipe[0], &err_pipe[1], &status[0], &status[1]});
            throw std::runtime_error(os_error(e) + ": '" + o.err_file + "'");
        }
    }
    const pid_t parent = ::getpid();
    const bool group = o.contain || o.own_group;
    const pid_t pid = anchor().call([&]() -> pid_t {
        const pid_t c = ::fork();
        if (c != 0) return c;
        // the child
        if (o.contain) {
            ::prctl(PR_SET_PDEATHSIG, SIGKILL);
            if (::getppid() != parent) ::_exit(127);   // the server ended before the call
        }
        if (group) ::setpgid(0, 0);
        if (!o.cpus.empty()) {   // a replica's own physical cores
            cpu_set_t set;
            CPU_ZERO(&set);
            for (const int c : o.cpus) CPU_SET(c, &set);
            ::sched_setaffinity(0, sizeof set, &set);
        }
        auto fd_to = [](int fd, int target) {
            if (fd >= 0 && fd != target) ::dup2(fd, target);
            else if (fd == target) ::fcntl(fd, F_SETFD, 0);   // already there: only its close-on-exec cleared
        };
        auto null_to = [](int target, int flags) {
            const int fd = ::open("/dev/null", flags);
            if (fd >= 0 && fd != target) {
                ::dup2(fd, target);
                ::close(fd);
            }
        };
        if (o.in == SpawnOptions::pipe) fd_to(in_pipe[0], 0);
        else if (o.in == SpawnOptions::null) null_to(0, O_RDONLY);
        if (o.out == SpawnOptions::pipe) fd_to(out_pipe[1], 1);
        else if (o.out == SpawnOptions::null) null_to(1, O_WRONLY);
        if (o.err == SpawnOptions::pipe) fd_to(err_pipe[1], 2);
        else if (o.err == SpawnOptions::null) null_to(2, O_WRONLY);
        else if (o.err == SpawnOptions::file) fd_to(err_fd, 2);
        for (int fd : {in_pipe[0], in_pipe[1], out_pipe[0], out_pipe[1], err_pipe[0], err_pipe[1]})
            if (fd > 2) ::close(fd);
        if (!o.cwd.empty() && ::chdir(o.cwd.c_str()) != 0) {
            const int e = errno;
            (void) !::write(status[1], &e, sizeof e);
            ::_exit(127);
        }
        if (o.env) ::execve(exe.c_str(), argv.data(), envp.data());
        else ::execv(exe.c_str(), argv.data());
        const int e = errno;
        (void) !::write(status[1], &e, sizeof e);
        ::_exit(127);
    });
    const int fork_errno = errno;
    if (err_fd >= 0) ::close(err_fd);
    closeall({&in_pipe[0], &out_pipe[1], &err_pipe[1], &status[1]});
    if (pid < 0) {
        closeall({&in_pipe[1], &out_pipe[0], &err_pipe[0], &status[0]});
        throw std::runtime_error(os_error(fork_errno));
    }
    int child_errno = 0;
    ssize_t got;
    do {
        got = ::read(status[0], &child_errno, sizeof child_errno);
    } while (got < 0 && errno == EINTR);
    ::close(status[0]);
    if (got == (ssize_t) sizeof child_errno) {
        ::waitpid(pid, nullptr, 0);
        closeall({&in_pipe[1], &out_pipe[0], &err_pipe[0]});
        throw std::runtime_error(os_error(child_errno) + ": '" + argv_s[0] + "'");
    }
    p.pid = pid;
    p.group = group;
    p.in = in_pipe[1];
    p.out = out_pipe[0];
    p.err = err_pipe[0];
    p.cr_lines = o.cr_lines;
    if (p.out >= 0) p.out_reader = std::thread([&p] { p.read_out(); });
    else p.lines->close();
    p.on_err_line = std::move(o.on_err_line);
    if (p.err >= 0) p.err_reader = std::thread([&p] { p.read_err(); });
    else p.err_done = true;
}

Process::~Process() {
    Impl& p = *impl_;
    // also what it left running, which may hold its pipes: the group is signalled while its leader is not reaped (its
    // id is ours until then)
    if (!poll()) kill();
    close_stdin();
    wait(5);
    if (p.out_reader.joinable()) p.out_reader.join();
    if (p.err_reader.joinable()) p.err_reader.join();
    if (p.out >= 0) ::close(p.out);
    if (p.err >= 0) ::close(p.err);
}

bool Process::write(std::string_view data) {
    Impl& p = *impl_;
    std::lock_guard<std::mutex> lk(p.in_mu);
    if (p.in < 0) return false;
    while (!data.empty()) {
        const ssize_t n = ::write(p.in, data.data(), data.size());
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        data.remove_prefix((size_t) n);
    }
    return true;
}

void Process::close_stdin() {
    Impl& p = *impl_;
    std::lock_guard<std::mutex> lk(p.in_mu);
    if (p.in >= 0) ::close(p.in);
    p.in = -1;
}

std::optional<int> Process::poll() {
    Impl& p = *impl_;
    std::lock_guard<std::mutex> lk(p.wait_mu);
    if (p.code || p.pid <= 0) return p.code;
    siginfo_t info{};
    if (::waitid(P_PID, (id_t) p.pid, &info, WEXITED | WNOHANG | WNOWAIT) == 0 && info.si_pid == p.pid) {
        // it has ended but is not reaped yet, so its id is still its group's: what it left running there ends now
        if (p.group) ::kill(-p.pid, SIGKILL);
        int st = 0;
        if (::waitpid(p.pid, &st, 0) == p.pid) p.code = WIFEXITED(st) ? WEXITSTATUS(st) : -WTERMSIG(st);
    }
    return p.code;
}

std::optional<int> Process::wait(double timeout_s) {
    const auto until = std::chrono::steady_clock::now() + std::chrono::duration<double>(timeout_s < 0 ? 1e9 : timeout_s);
    for (;;) {
        if (auto c = poll()) return c;
        if (std::chrono::steady_clock::now() >= until) return std::nullopt;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
}

void Process::kill() {
    Impl& p = *impl_;
    if (p.pid <= 0 || poll()) return;
    if (p.group) ::kill(-p.pid, SIGKILL);
    ::kill(p.pid, SIGKILL);
}

long Process::pid() const { return (long) impl_->pid; }

Environment environment() {
    Environment env;
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        const std::string kv = *e;
        const size_t eq = kv.find('=');
        if (eq != std::string::npos && eq > 0) env[kv.substr(0, eq)] = kv.substr(eq + 1);
    }
    return env;
}

std::string which(const std::string& name, const Environment* env) {
    namespace fs = std::filesystem;
    auto runnable = [](const fs::path& p) {
        std::error_code ec;
        return fs::is_regular_file(p, ec) && ::access(p.c_str(), X_OK) == 0;
    };
    if (name.find('/') != std::string::npos) return runnable(name) ? name : "";
    std::string path;
    if (env != nullptr) {
        const auto it = env->find("PATH");
        path = it == env->end() ? "" : it->second;
    } else if (const char* p = std::getenv("PATH")) {
        path = p;
    }
    for (const auto& dir : split(path, ":")) {
        const fs::path c = fs::path(dir.empty() ? "." : dir) / name;
        if (runnable(c)) return c.string();
    }
    return "";
}

bool no_throttle_self() { return false; }

void redirect_stderr(const std::string& path) {
    const int saved = ::dup(2);
    if (saved >= 0) {
        ::fcntl(saved, F_SETFD, FD_CLOEXEC);
        if (std::FILE* console = ::fdopen(saved, "w")) set_err_stream(console);
    }
    const int fd = ::open(path.c_str(), O_WRONLY | O_APPEND | O_CREAT, 0644);
    if (fd < 0) return;
    std::fflush(stderr);
    ::dup2(fd, 2);
    ::close(fd);
}

std::string self_dir() {
    std::error_code ec;
    const std::filesystem::path p = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::string(".") : p.parent_path().string();
}

std::string temp_dir() {
    const char* t = std::getenv("TMPDIR");
    std::string s = t != nullptr && *t ? t : "/tmp";
    while (s.size() > 1 && s.back() == '/') s.pop_back();
    return s;
}

#endif

// ================================================================================================= both
std::filesystem::path u8_path(std::string_view utf8) {
    return std::filesystem::path(std::u8string(reinterpret_cast<const char8_t*>(utf8.data()), utf8.size()));
}

std::string u8_str(const std::filesystem::path& p) {
    const std::u8string u = p.u8string();
    return std::string(u.begin(), u.end());
}

std::string Process::stderr_tail() {
    Impl& p = *impl_;
    std::lock_guard<std::mutex> lk(p.tail_mu);
    std::string out;
    for (const auto& l : p.tail) out += (out.empty() ? "" : "\n") + l;
    return out;
}

int Process::read_line(std::string& line, double timeout_s) { return impl_->lines->pop(line, timeout_s); }

bool Process::stderr_done() const { return impl_->err_done.load(); }

int64_t run_command(const json& cmd, double timeout_s, std::string& why) {
    SpawnOptions o;
    if (cmd.is_string()) {
        o.args = {cmd.get<std::string>()};
        o.shell = true;
    } else {
        for (const auto& x : cmd) o.args.push_back(x.is_string() ? x.get<std::string>() : py_dumps(x));
    }
    o.in = SpawnOptions::null;
    o.contain = false;   // as subprocess.run: a program it starts in the background may stay
    try {
        Process p(o);
#ifdef _WIN32
        if (const auto rc = p.wait(timeout_s)) return (int64_t) (uint32_t) *rc;   // Windows' codes are unsigned
#else
        if (const auto rc = p.wait(timeout_s)) return *rc;
#endif
        p.kill();
        p.wait(10);
        char buf[64];
        std::snprintf(buf, sizeof buf, "%g", timeout_s);
        why = std::string("it did not finish in ") + buf + " s (before_load_timeout_s)";
    } catch (const std::exception& e) {
        why = std::string("it could not run (") + e.what() + ")";
    }
    return -1;
}

}  // namespace lamina
