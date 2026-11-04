// src/server/main.cpp - lamina-server: the OpenAI, Anthropic and Responses APIs and the web app in front of the engine,
// which runs on a thread of this process (serve/server.py's main).
//
//   lamina-server --config lamina-<model>.json       the model the config names (written by setup)
//   lamina-server --engine mock --tokenizer <gguf>   a scripted engine (the tests')
#include "config.hpp"
#include "engine.hpp"
#include "frontend.hpp"
#include "http_server.hpp"
#include "images.hpp"
#include "keepawake.hpp"
#include "log.hpp"
#include "mcp.hpp"
#include "net.hpp"
#include "process.hpp"
#include "prometheus.hpp"
#include "replicas.hpp"
#include "service.hpp"
#include "telemetry.hpp"
#include "vision.hpp"

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <thread>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#else
#include <unistd.h>
#endif

using namespace lamina;

namespace {

std::atomic<int> g_stops{0};              // stop requests so far (a second Ctrl+C ends the engine at once)
std::atomic<const char*> g_reason{"sigint"};
std::atomic<bool> g_finished{false};      // the clean stop is over (a Windows console event waits for it)

void request_stop(const char* reason) {
    g_reason = reason;
    if (g_stops.fetch_add(1) >= 1) std::_Exit(130);   // a second Ctrl+C: no waiting for the engine to free its memory
}

#ifdef _WIN32
BOOL WINAPI on_console(DWORD event) {
    switch (event) {
        case CTRL_C_EVENT: request_stop("sigint"); return TRUE;
        case CTRL_BREAK_EVENT: request_stop("sigbreak"); return TRUE;
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            // the system ends the process a few seconds after this returns: wait for the clean stop (up to 4.5 s)
            request_stop(event == CTRL_CLOSE_EVENT ? "console-close" : event == CTRL_LOGOFF_EVENT ? "logoff" : "shutdown");
            for (int i = 0; i < 45 && !g_finished; ++i) Sleep(100);
            return TRUE;
        default: return FALSE;
    }
}
#else
void on_signal(int sig) { request_stop(sig == SIGTERM ? "sigterm" : sig == SIGHUP ? "sighup" : "sigint"); }
#endif

void install_stop_handlers() {
    // every ordinary way of being asked to stop takes Ctrl+C's path (QUIT to the engine, then exit), installed before
    // anything slow starts
#ifdef _WIN32
    SetConsoleCtrlHandler(on_console, TRUE);
#else
    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    struct sigaction old {};
    sigaction(SIGHUP, nullptr, &old);
    if (old.sa_handler != SIG_IGN) std::signal(SIGHUP, on_signal);   // nohup: a hangup ignored at start stays ignored
#endif
}

[[noreturn]] void usage_error(const std::string& msg) {
    say_err("usage: lamina-server --config FILE [--host HOST] [--port PORT] ... (--help)");
    say_err("lamina-server: error: " + msg);
    std::exit(2);
}

const char* kHelp =
    "lamina-server: the OpenAI, Anthropic and Responses APIs and the web app in front of the Lamina engine.\n"
    "\n"
    "  --config FILE          the run config (lamina-<model>.json, written by setup): the engine's arguments and the\n"
    "                         server's settings\n"
    "  --engine lamina|mock   lamina (the default) or mock (the tests' scripted engine, with --tokenizer)\n"
    "  --host HOST            the address to listen on: 127.0.0.1 = this PC only (the default), 0.0.0.0 = also other\n"
    "                         devices on your network (set an API key); also \"host\" in the config\n"
    "  --port PORT            the port (default 8095; 0: one the system picks); also \"port\" in the config\n"
    "  --api-key KEY          require this key on the API (Authorization: Bearer ... or x-api-key); also\n"
    "                         $LAMINA_API_KEY and \"api_key\"; several keys separated by commas\n"
    "  --lazy                 start unloaded (the engine and the image encoder start with the first request)\n"
    "  --idle-unload SECONDS  unload the model after this long without requests (\"idle_unload_s\")\n"
    "  --min-free-vram-mib N  load an unloaded model only when this much VRAM is free, else 503 (\"min_free_vram_mib\")\n"
    "  --before-load CMD      a command run before the model is loaded again (\"before_load\")\n"
    "  --parallel N           this many requests decode together in the engine's batch slots (\"parallel\")\n"
    "  --max-queue N          inference calls that may wait beyond the slots; 0 refuses overload (\"max_queue\")\n"
    "  --queue-timeout S      the longest wait for the first engine turn, 0.1..600 (needs --max-queue)\n"
    "  --shutdown-timeout S   on a stop, let running requests finish for up to S seconds (0..300)\n"
    "  --replicas N           data-parallel replicas: the \"gpu\" list cut into N groups, one engine each\n"
    "  --gpu N[,M]            the GPU(s) as nvidia-smi numbers them (\"gpu\")\n"
    "  --mcp-config FILE      MCP servers in Claude Desktop's format ({\"mcpServers\": {...}}) for the web app's chat\n"
    "  --api-monitor          the request monitor at /api-monitor: the last 100 requests kept in memory\n"
    "  --offline              no image URLs fetched, no MCP servers or before_load commands\n"
    "  --fit-max-tokens       shorten max_tokens to the context left (the default)\n"
    "  --no-fit-max-tokens    answer 400 when prompt + max_tokens does not fit the context\n"
    "  --ready-file FILE      write a JSON record of this instance once it serves (an absolute path, new file)\n"
    "  --open                 open the web app in the browser once the model is ready\n"
    "  --tokenizer GGUF       the mock engine's tokenizer and chat template (a model's .gguf)\n"
    "  --script TEXT          the mock engine's answer (repeatable: requests get them in turn, the last repeats)\n"
    "  --script-cycle         repeat the scripts by position instead of staying on the last\n"
    "  --script-delay S       seconds between the mock engine's tokens\n";

struct Args {
    std::string engine = "lamina";
    std::string config, host, ready_file, gpu, tokenizer, api_key, mcp_config, before_load;
    std::optional<int> port, parallel, max_queue_arg, replicas;
    std::optional<double> shutdown_timeout, queue_timeout_arg, idle_unload, script_delay;
    std::optional<int64_t> min_free_vram_mib;
    std::vector<std::string> scripts;
    bool script_cycle = false, offline = false, open = false, fit = false, no_fit = false, lazy = false, api_monitor = false;
    bool api_key_given = false, api_key_empty = false;
};

Args parse_args(int argc, char** argv) {
    Args a;
    if (const char* k = std::getenv("LAMINA_API_KEY")) {
        a.api_key = k;
        a.api_key_given = true;
        a.api_key_empty = strip(a.api_key).empty();
    }
    for (int i = 1; i < argc; ++i) {
        std::string o = argv[i], v;
        const size_t eq = o.find('=');
        const bool inline_value = starts_with(o, "--") && eq != std::string::npos;
        if (inline_value) {
            v = o.substr(eq + 1);
            o = o.substr(0, eq);
        }
        auto value = [&]() -> std::string {
            if (inline_value) return v;
            if (i + 1 >= argc) usage_error("argument " + o + ": expected one argument");
            return argv[++i];
        };
        auto number = [&](auto parse) {
            const std::string s = value();
            try {
                size_t used = 0;
                const auto n = parse(s, &used);
                if (used != s.size()) throw std::invalid_argument(s);
                return n;
            } catch (const std::exception&) {
                usage_error("argument " + o + ": invalid value: '" + s + "'");
            }
        };
        auto to_int = [](const std::string& s, size_t* u) { return std::stoi(s, u); };
        auto to_dbl = [](const std::string& s, size_t* u) { return std::stod(s, u); };
        auto to_ll = [](const std::string& s, size_t* u) { return (int64_t) std::stoll(s, u); };
        if (o == "-h" || o == "--help") {
            std::fputs(kHelp, stdout);
            std::exit(0);
        } else if (o == "--engine") {
            a.engine = value();
            if (a.engine != "mock" && a.engine != "lamina") usage_error("argument --engine: invalid choice: '" + a.engine + "'");
        } else if (o == "--config") a.config = value();
        else if (o == "--host") a.host = value();
        else if (o == "--port") a.port = number(to_int);
        else if (o == "--ready-file") a.ready_file = value();
        else if (o == "--shutdown-timeout") a.shutdown_timeout = number(to_dbl);
        else if (o == "--parallel") a.parallel = number(to_int);
        else if (o == "--max-queue") a.max_queue_arg = number(to_int);
        else if (o == "--queue-timeout") a.queue_timeout_arg = number(to_dbl);
        else if (o == "--offline") a.offline = true;
        else if (o == "--gpu") a.gpu = value();
        else if (o == "--tokenizer") a.tokenizer = value();
        else if (o == "--open") a.open = true;
        else if (o == "--fit-max-tokens") a.fit = true;
        else if (o == "--no-fit-max-tokens") a.no_fit = true;
        else if (o == "--api-key") {
            a.api_key = value();
            a.api_key_given = true;
            a.api_key_empty = strip(a.api_key).empty();
        } else if (o == "--mcp-config") a.mcp_config = value();
        else if (o == "--lazy") a.lazy = true;
        else if (o == "--api-monitor") a.api_monitor = true;
        else if (o == "--idle-unload") a.idle_unload = number(to_dbl);
        else if (o == "--min-free-vram-mib") a.min_free_vram_mib = number(to_ll);
        else if (o == "--before-load") a.before_load = value();
        else if (o == "--replicas") a.replicas = number(to_int);
        else if (o == "--script") a.scripts.push_back(value());
        else if (o == "--script-cycle") a.script_cycle = true;
        else if (o == "--script-delay") a.script_delay = number(to_dbl);
        else usage_error("unrecognized arguments: " + std::string(argv[i]));
    }
    if (a.engine == "lamina" && a.config.empty())
        usage_error("--config FILE is required: the run config (lamina-<model>.json, written by setup)");
    if (a.engine == "mock" && a.tokenizer.empty())
        usage_error("--engine mock needs --tokenizer <a model's .gguf> (its tokenizer and chat template)");
    return a;
}

void open_browser(const std::string& url) {
#ifdef _WIN32
    ShellExecuteW(nullptr, L"open", widen(url).c_str(), nullptr, nullptr, SW_SHOWNORMAL);
#else
    try {
        SpawnOptions o;
        o.args = {"xdg-open", url};
        o.in = SpawnOptions::null;
        o.out = SpawnOptions::null;
        o.err = SpawnOptions::null;
        o.contain = false;
        Process p(o);
        p.wait(-1);
    } catch (const std::exception&) {
    }
#endif
}

long self_pid() {
#ifdef _WIN32
    return (long) GetCurrentProcessId();
#else
    return (long) getpid();
#endif
}

std::string join(const std::vector<std::string>& v, const char* sep) {
    std::string out;
    for (const auto& s : v) out += (out.empty() ? "" : sep) + s;
    return out;
}

/// opt-in data-parallel replicas: one `lamina --serve` child per card group, each with its own cards, log and (Linux)
/// CPU cores, started together; the ReplicaEngine in front spreads the requests
std::shared_ptr<Engine> start_replicas(const json& cfg, const std::vector<std::vector<int>>& groups,
                                       const std::vector<std::string>& effort_end, bool lazy, bool offline, double silence) {
    const size_t n = groups.size();
    const auto cpus = cpu_sets(n, jget(cfg, "replica_cpus") != false);
    std::string where;
    for (size_t i = 0; i < n; ++i) {
        std::string g;
        for (const int c : groups[i]) g += (g.empty() ? "" : ",") + std::to_string(c);
        where += (where.empty() ? "" : " | ") + g;
    }
    say("[lamina] " + std::to_string(n) + " replicas on the cards " + where + (cpus ? " (own CPU cores each)" : "") +
        ": requests go to the least busy one, a conversation back to its own");
#ifdef _WIN32
    const std::string exe = u8_str(u8_path(self_dir()) / "lamina.exe");
#else
    const std::string exe = u8_str(u8_path(self_dir()) / "lamina");
#endif
    std::vector<std::shared_ptr<LaminaEngine>> engines(n);
    std::vector<std::string> errors(n);
    auto one = [&](size_t i) {
        const json rc = replica_cfg(cfg, i, groups[i]);
        try {
            std::vector<std::string> args = engine_args(rc);
            args.insert(args.end(), effort_end.begin(), effort_end.end());
            EngineProcess p{exe, jstr(rc, "cwd"), child_env(rc, offline), cpus ? (*cpus)[i] : std::vector<int>{}};
            engines[i] = std::make_shared<LaminaEngine>(args, jstr(rc, "log"), lazy, p);
            engines[i]->silence_s = silence;
        } catch (const std::exception& e) {
            errors[i] = e.what();
        }
    };
    const char* serial = std::getenv("LAMINA_REPLICA_START");
    if (serial != nullptr && std::string(serial) == "serial") {
        for (size_t i = 0; i < n; ++i) one(i);
    } else {
        std::vector<std::thread> ts;
        for (size_t i = 0; i < n; ++i) ts.emplace_back(one, i);
        for (auto& t : ts) t.join();
    }
    for (size_t i = 0; i < n; ++i)
        if (!errors[i].empty()) {
            for (auto& e : engines)
                if (e != nullptr) {
                    try {
                        e->close();
                    } catch (const std::exception&) {
                    }
                }
            throw std::runtime_error("replica " + std::to_string(i) + " did not start: " + errors[i]);
        }
    const json& lt = jget(cfg, "replica_load_tokens");
    return std::make_shared<ReplicaEngine>(engines, truthy(lt) && lt.is_number() ? lt.get<int64_t>() : 3000);
}

std::string server_names(const McpHub& hub) {
    std::string out;
    for (const auto& srv : hub.servers) out += (out.empty() ? "" : ", ") + srv->name;
    return out;
}

/// "k=v, ..." as Python prints a dict's items
std::string items(const json& d) {
    std::string out;
    for (auto it = d.begin(); it != d.end(); ++it)
        out += (out.empty() ? "" : ", ") + it.key() + "=" + (it.value().is_string() ? it.value().get<std::string>() : py_repr_value(it.value()));
    return out;
}

int run(int argc, char** argv) {
    no_throttle_self();   // #691: a minimized console window must not slow the engine down
    Args a = parse_args(argc, argv);
    install_stop_handlers();
    if (a.shutdown_timeout && !(*a.shutdown_timeout >= 0 && *a.shutdown_timeout <= 300))
        usage_error("--shutdown-timeout must be in 0..300 seconds");
    json cfg = json::object();
    try {
        if (!a.config.empty()) cfg = read_config(a.config);
    } catch (const ConfigError& e) {
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
    split_joined_args(cfg);
    std::string ready_path;
    if (!a.ready_file.empty()) {
        const std::filesystem::path p = u8_path(a.ready_file);
        std::error_code ec;
        if (!p.is_absolute()) usage_error("--ready-file must be an absolute path");
        if (!std::filesystem::is_directory(p.parent_path(), ec)) usage_error("--ready-file parent directory must exist");
        if (std::filesystem::exists(std::filesystem::symlink_status(p, ec)))
            usage_error("--ready-file already exists; verify and remove a stale file before reuse");
        ready_path = a.ready_file;
    }
    json max_queue = a.max_queue_arg ? json(*a.max_queue_arg) : jget(cfg, "max_queue");
    if (!max_queue.is_null() && (!max_queue.is_number_integer() || max_queue.get<int64_t>() < 0 || max_queue.get<int64_t>() > 4096))
        usage_error("--max-queue / max_queue must be a whole number in 0..4096");
    json queue_timeout = a.queue_timeout_arg ? json(*a.queue_timeout_arg) : jget(cfg, "queue_timeout");
    if (!queue_timeout.is_null() && (queue_timeout.is_boolean() || !queue_timeout.is_number() ||
                                     !std::isfinite(queue_timeout.get<double>()) || queue_timeout.get<double>() < 0.1 ||
                                     queue_timeout.get<double>() > 600))
        usage_error("--queue-timeout / queue_timeout must be finite seconds in 0.1..600");
    if (!queue_timeout.is_null() && max_queue.is_null()) usage_error("--queue-timeout requires --max-queue / max_queue to bound pending calls");
    if (a.parallel) {
        if (*a.parallel < 1 || *a.parallel > 1024) usage_error("--parallel must be in 1..1024; the engine may reduce it to the supported count");
        const json& args = jget(cfg, "args");
        if (args.is_array() && std::any_of(args.begin(), args.end(), [](const json& x) { return x == "--batch" || x == "--slots"; }))
            usage_error("--parallel conflicts with an explicit --batch/--slots in the saved engine args");
        cfg["parallel"] = *a.parallel;
    }
    lift_reasoning_effort_arg(cfg);
    if (a.replicas) cfg["replicas"] = *a.replicas;
    if (!a.gpu.empty()) {
        const std::string g(strip(a.gpu));
        cfg["gpu"] = !g.empty() && std::all_of(g.begin(), g.end(), ::isdigit) ? json(std::stoi(g)) : json(a.gpu);
    }
    std::string host = !a.host.empty() ? a.host : jstr(cfg, "host").empty() ? "127.0.0.1" : jstr(cfg, "host");
    int port = 8095;
    if (a.port) {
        port = *a.port;
    } else if (cfg.contains("port")) {
        if (!cfg["port"].is_number_integer() || cfg["port"].get<int64_t>() < 0 || cfg["port"].get<int64_t>() > 65535)
            usage_error("\"port\" in the config must be a whole number in 0..65535");
        port = (int) cfg["port"].get<int64_t>();
    }
    if (a.api_key_empty) {
        say_err("[lamina] an API key was given but it is empty: set a key, or leave it out");
        return 2;
    }
    json api_key;
    try {
        api_key = api_key_of(a.api_key_given ? json(a.api_key) : (cfg.contains("api_key") ? cfg["api_key"] : json("")));
    } catch (const std::invalid_argument& e) {
        say_err(std::string("[lamina] ") + e.what() + ": set a key, or leave it out");
        return 2;
    }
    const bool loopback = host == "127.0.0.1" || host == "localhost" || host == "::1";
    if (!loopback && !truthy(api_key)) {
        say_err("[lamina] a non-loopback bind requires an inference API key");
        return 2;
    }
    if (port != 0) {   // before the minutes of loading: is the port free?
        if (const int code = probe_listen(host, port)) usage_error(listen_problem(host, port, code));
    }
    // the model's file: its tokenizer and chat template (and, for the engine, its weights)
    const std::string gguf = a.engine == "lamina" ? model_file(cfg) : a.tokenizer;
    if (gguf.empty()) usage_error("the config's args name no model file (--native <gguf>)");
    std::unique_ptr<Tokenizer> tok;
    try {
        tok = Tokenizer::from_gguf(gguf);
    } catch (const std::exception& e) {
        say_err(std::string("[lamina] the model's tokenizer could not be read from ") + gguf + ": " + e.what());
        return 1;
    }
    std::shared_ptr<McpHub> hub;
    try {
        hub = hub_from_config(cfg, a.mcp_config);   // before the minutes of loading: a bad entry stops here
    } catch (const std::invalid_argument& e) {
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
    if (a.offline && (hub != nullptr || truthy(jget(cfg, "before_load")) || !a.before_load.empty()))
        usage_error("--offline requires MCP servers and before_load commands to be disabled in this profile");
    if (a.offline) {
        const std::vector<std::string> blocked = offline_blockers(cfg);
        if (!blocked.empty()) usage_error("--offline cannot be strict for this profile: " + join(blocked, "; "));
    }
    std::shared_ptr<Engine> engine;
    std::shared_ptr<Vision> vision;
    json sampling_defaults = json::object();
    std::optional<std::vector<std::string>> effort_end;
    std::vector<std::vector<int>> groups;
    bool lazy = false;
    try {
        if (a.engine == "lamina") {
            groups = gpu_groups(cfg);   // replicas: engines of their own, each a child with its cards and its log
            const std::string log = jstr(cfg, "log");
            if (groups.empty()) {
                apply_engine_env(cfg, a.offline);   // the engine runs in this process: its environment is this one
                if (!log.empty()) redirect_stderr(log);   // the engine's messages go to its log; the server's stay here
            }
            sampling_defaults = sampling_defaults_from_config(cfg);
            if (!sampling_defaults.empty()) say("[lamina] sampling defaults from the config: " + items(sampling_defaults));
            lazy = a.lazy || jget(cfg, "lazy_load") == true;
            if (truthy(jget(cfg, "vision"))) {
                say(lazy ? "vision encoder unloaded; it starts with the model ..." : "loading the vision encoder ...");
                json vcfg = cfg["vision"];   // relative paths are the config's cwd's, as for the engine
                for (const char* k : {"exe", "mmproj", "model"})
                    if (jget(vcfg, k).is_string() && !u8_path(vcfg[k].get<std::string>()).is_absolute())
                        vcfg[k] = u8_str(std::filesystem::absolute(u8_path(jstr(cfg, "cwd", ".")) / u8_path(vcfg[k].get<std::string>())));
                Environment venv = groups.empty() ? vision_env(cfg) : child_env(cfg, a.offline);
                const json& dev = jget(cfg["vision"], "cuda_device");
                if (!groups.empty() && !dev.is_null()) {   // #408: the encoder on a card of its own
                    venv["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
                    venv["CUDA_VISIBLE_DEVICES"] = dev.is_string() ? dev.get<std::string>() : py_dumps(dev);
                }
                vision = std::make_shared<Vision>(vcfg, log, venv, lazy);
            }
            say(lazy ? "model unloaded; the first request loads it ..." : "loading the model (the first start takes a minute or two) ...");
            std::vector<std::string> args = engine_args(cfg);
            const double silence = engine_silence_s(cfg);   // #481: checked before the (minutes-long) start
            effort_end = effort_end_args(cfg, *tok);        // #458
            if (effort_end) args.insert(args.end(), effort_end->begin(), effort_end->end());
            const std::string cwd = jstr(cfg, "cwd");
            if (!cwd.empty()) {
                std::error_code ec;
                std::filesystem::current_path(u8_path(cwd), ec);   // the engine's relative paths are the config's cwd's
                if (ec) throw ConfigError("\"cwd\": " + cwd + " is not a directory");
            }
            if (!groups.empty()) {
                engine = start_replicas(cfg, groups, effort_end.value_or(std::vector<std::string>{}), lazy, a.offline, silence);
            } else {
                auto le = std::make_shared<LaminaEngine>(args, log, lazy);
                le->silence_s = silence;
                engine = le;
            }
            warn_tight_ram(jget(engine->info, "arena_mib"));
        } else {
            std::vector<std::string> scripts = a.scripts;
            if (scripts.empty()) scripts.push_back("Thinking about it.</think>\n\nHello from the mock engine.");
            engine = std::make_shared<MockEngine>(*tok, scripts, 32768, a.script_delay.value_or(0.0), a.script_cycle);
        }
    } catch (const ConfigError& e) {
        say_err(std::string("[lamina] config ") + e.what());
        return 1;
    } catch (const ReplicaConfigError& e) {
        say_err(std::string("[lamina] config ") + e.what());
        return 1;
    } catch (const std::exception& e) {
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
    if (const std::string file = jstr(cfg, "chat_template_file"); !file.empty()) {   // in place of the GGUF's
        std::filesystem::path path = u8_path(file);
        if (!path.is_absolute()) path = u8_path(jstr(cfg, "cwd", ".")) / path;
        std::ifstream f(path, std::ios::binary);
        if (!f) {
            say_err("[lamina] config \"chat_template_file\": cannot read " + u8_str(path));
            return 1;
        }
        tok->chat_template.assign(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
        say("[lamina] chat template: " + u8_str(path));
    }
    auto tpl = std::make_shared<ChatTemplate>(tok->chat_template);
    const bool fit = a.fit || (!a.no_fit && jget(cfg, "fit_max_tokens") != false);
    Service svc(engine, *tok, tpl, default_model_name(cfg, tok->architecture), vision, sampling_defaults, fit);
    svc.offline = a.offline;
    if (!max_queue.is_null()) svc.max_queue = max_queue.get<int64_t>();
    if (!queue_timeout.is_null()) svc.queue_timeout_s = queue_timeout.get<double>();
    try {
        if (a.engine == "lamina") {   // the model card's sampling when nothing else is said
            svc.model_arch = jstr(cfg, "arch").empty() ? tok->architecture : jstr(cfg, "arch");
            json requested = jget(cfg, "parallel");
            if (requested.is_null()) {
                const json& args = jget(cfg, "args");
                for (size_t i = 0; args.is_array() && i + 1 < args.size(); ++i)
                    if ((args[i] == "--batch" || args[i] == "--slots") && args[i + 1].is_string()) {
                        try {
                            requested = std::stoi(args[i + 1].get<std::string>());
                        } catch (const std::exception&) {
                        }
                    }
            }
            const int64_t per_replica = requested.is_number_integer() && requested.get<int64_t>() > 0 ? requested.get<int64_t>() : 1;
            svc.requested_slots = groups.empty() ? per_replica : per_replica * (int64_t) groups.size();
        }
        svc.answer_room = jget(cfg, "answer_room") != false;
        svc.reasoning_close_retry = jget(cfg, "reasoning_close_retry") == true;   // #1053: opt-in
        svc.literal_markers = jget(cfg, "literal_markers") != false;
        if (const json& n = jget(cfg, "short_opening_tokens"); !n.is_null()) {
            if (!n.is_number_integer() || n.get<int64_t>() < 0)
                throw ConfigError("\"short_opening_tokens\" must be a whole number, 0 or more, not " + py_repr_value(n));
            svc.short_opening_tokens = n.get<int64_t>();
        }
        if (truthy(jget(cfg, "images"))) {   // image generation, set up by setup --images
            json icfg = cfg["images"];
            for (const char* k : {"exe", "diffusion_model", "vae", "llm", "out_dir"})
                if (jget(icfg, k).is_string() && !u8_path(icfg[k].get<std::string>()).is_absolute())
                    icfg[k] = u8_str(std::filesystem::absolute(u8_path(jstr(cfg, "cwd", ".")) / u8_path(icfg[k].get<std::string>())));
            svc.images = std::make_shared<images::ImageGen>(icfg, jstr(cfg, "log"));
            const std::vector<std::string> gone = svc.images->missing();
            if (!gone.empty())
                say("[lamina] image generation: missing " + join(gone, ", ") + " - requests fail until setup --images has run again");
            say("[lamina] image generation: " + py_str(svc.images->model) + " " + jstr(icfg, "quant") + " on " +
                (jstr(icfg, "backend").empty() ? "?" : jstr(icfg, "backend")) + " (POST /v1/images/generations; GPU policy " +
                svc.images->policy + ")");
        }
        const json& ps = jget(cfg, "prevent_sleep");
        if (!ps.is_null() && !ps.is_boolean())
            throw ConfigError("\"prevent_sleep\" must be true or false, not " + py_repr_value(ps));
        svc.keep_awake = KeepAwake::create(ps == true);   // #1727: opt-in, Windows only
        if (!jget(cfg, "reasoning_effort").is_null()) {   // #1641: the thinking level of requests that name none
            svc.config_effort = jstr(clean_shared_defaults({{"reasoning_effort", cfg["reasoning_effort"]}}), "reasoning_effort");
            if (!svc.config_effort.empty())
                say("[lamina] thinking level: " + svc.config_effort + " (reasoning_effort in the config; a request, or the Chat "
                    "settings shared with other apps, can set its own)");
        }
        svc.codex_thread_titles = jget(cfg, "codex_thread_titles") == true;       // #923: opt-in
        svc.codex_compaction_cache = jget(cfg, "codex_compaction_cache") == true; // #924: opt-in
        svc.set_aliases(jget(cfg, "aliases"));   // #297: other names the model answers to
        if (!svc.aliases.empty()) say("[lamina] model aliases: " + join(svc.aliases, ", "));
        svc.api_key = api_key;
        if (!truthy(svc.api_key) && cfg.contains("api_key"))   // #569: written, but empty: a warning, not a stop
            say_err("[lamina] the api_key in the config is empty: this server has no API key (anyone who can reach it can use "
                    "it); set one, or leave api_key out");
        svc.cors_origins = origins_of(jget(cfg, "cors_origins"), "cors_origins", true);
        svc.trusted_origins = origins_of(jget(cfg, "trusted_origins"), "trusted_origins", false);
        const char* env_hosts = std::getenv("LAMINA_ALLOWED_HOSTS");
        svc.allowed_hosts = allowed_hosts_of(jget(cfg, "allowed_hosts"), env_hosts != nullptr ? env_hosts : "");
        if (!svc.allowed_hosts.empty())
            say(std::find(svc.allowed_hosts.begin(), svc.allowed_hosts.end(), "*") != svc.allowed_hosts.end()
                    ? "[lamina] Host check off: any name reaches this server (allowed_hosts \"*\")"
                    : "[lamina] also answers to the host names " + join(svc.allowed_hosts, ", ") + " (allowed_hosts)");
        svc.api_monitor = a.api_monitor || jget(cfg, "api_monitor") == true;
        if (svc.api_monitor)
            say(std::string("[lamina] API request monitor on (/api-monitor): the last 100 requests' prompts and answers are kept "
                            "in memory") + (truthy(svc.api_key) ? "" : "; anyone who can reach this server can read them (no API key)"));
        if (!svc.cors_origins.empty())
            say("[lamina] CORS on /v1/* for " + join(svc.cors_origins, ", ") +
                (truthy(svc.api_key) || std::find(svc.cors_origins.begin(), svc.cors_origins.end(), "*") == svc.cors_origins.end()
                     ? "" : " - WARNING: any web page may use the model (no API key)"));
        svc.idle_unload_s = a.idle_unload ? *a.idle_unload : jget(cfg, "idle_unload_s").is_number() ? cfg["idle_unload_s"].get<double>() : 0.0;
        svc.min_free_vram_mib = a.min_free_vram_mib ? *a.min_free_vram_mib
                                : jget(cfg, "min_free_vram_mib").is_number() ? cfg["min_free_vram_mib"].get<int64_t>() : 0;
        svc.before_load = !a.before_load.empty() ? json(a.before_load) : truthy(jget(cfg, "before_load")) ? cfg["before_load"] : json(nullptr);
        const json& blt = jget(cfg, "before_load_timeout_s");
        if (truthy(blt)) {
            if (blt.is_number()) svc.before_load_timeout_s = blt.get<double>();
            else say("[lamina] \"before_load_timeout_s\" in the config is not a number; using " + py_g(svc.before_load_timeout_s) + " s");
        }
        svc.before_load_ignore_failure = jget(cfg, "before_load_ignore_failure") == true;
        svc.before_load_strict = jget(cfg, "before_load_strict") == true;
        const std::string mode = jget(cfg, "anthropic_thinking").is_string() && !cfg["anthropic_thinking"].get<std::string>().empty()
                                     ? cfg["anthropic_thinking"].get<std::string>() : "model";   // #278
        if (mode != "model" && mode != "on_request")
            throw ConfigError("anthropic_thinking must be \"model\" or \"on_request\", not " + py_repr_value(json(mode)));
        svc.anthropic_think_unasked = mode == "model";
        const json recovery = cfg.contains("tool_call_recovery") ? cfg["tool_call_recovery"] : json(false);
        if (!recovery.is_boolean()) throw ConfigError("\"tool_call_recovery\" must be true or false, not " + py_repr_value(recovery));
        svc.tool_call_recovery = recovery.get<bool>();
        const json etu = cfg.contains("env_to_user") ? cfg["env_to_user"] : json(false);
        if (!etu.is_boolean()) throw ConfigError("\"env_to_user\" must be true or false, not " + py_repr_value(etu));
        g_env_to_user = etu.get<bool>();
        const json rs = cfg.contains("repeat_stop_tokens") ? cfg["repeat_stop_tokens"] : json(256);   // #606: 0 turns it off
        if (!rs.is_number_integer() || rs.get<int64_t>() < 0)
            throw ConfigError("\"repeat_stop_tokens\" must be a whole number >= 0 (0 = off), not " + py_repr_value(rs));
        svc.repeat_stop_tokens = rs.get<int64_t>();
        svc.effort_end = effort_end.has_value();
        if (!jget(cfg, "reasoning_budget_tokens").is_null()) {   // #123: a default thinking budget for every request
            svc.reasoning_budget_tokens = cfg["reasoning_budget_tokens"];
            const auto budget = svc.reasoning_budget(json::object());
            if (budget && *budget)
                say("[lamina] thinking budget: " + std::to_string(*budget) + " tokens (reasoning_budget_tokens; a request can set its own)");
        }
        json loop = cfg.contains("reasoning_loop_recovery") ? cfg["reasoning_loop_recovery"] : json(false);   // #728
        if (loop == true) loop = "recover";
        if (loop != false && loop != "stop" && loop != "recover")
            throw ConfigError("\"reasoning_loop_recovery\" must be false, \"stop\" or \"recover\", not " + py_repr_value(loop));
        svc.reasoning_loop_recovery = loop.is_string() ? loop.get<std::string>() : "";
        svc.gpu_indices = engine_gpus(cfg);   // the Monitor reads every card the engine uses (#112), the first its own (#51)
        if (!groups.empty()) {               // replicas: every card of every group
            svc.gpu_indices.clear();
            for (const auto& g : groups) svc.gpu_indices.insert(svc.gpu_indices.end(), g.begin(), g.end());
            svc.replica_gpu_groups = groups;
        }
        svc.gpu_index = svc.gpu_indices.empty() ? 0 : svc.gpu_indices[0];
        svc.backend = jget(cfg, "backend");
    } catch (const ConfigError& e) {
        say_err(std::string("[lamina] config ") + e.what());
        return 1;
    } catch (const std::invalid_argument& e) {
        say_err(std::string("[lamina] config ") + e.what());
        return 1;
    }
    if (!a.config.empty()) {
        svc.config_path = a.config;   // #564: the web page's Settings view
        // the Chat settings shared with other apps, from last time
        std::filesystem::path sp = u8_path(a.config);
        sp.replace_extension("");
        svc.shared_path = u8_str(sp) + ".shared-settings.json";
        try {
            std::ifstream f(u8_path(svc.shared_path), std::ios::binary);
            json saved;
            if (f && py_loads(std::string(std::istreambuf_iterator<char>(f), {}), saved)) {
                const json shared = svc.set_shared(saved, false);
                if (!shared.empty()) say("[lamina] other apps use the Chat settings: " + items(shared));
            }
        } catch (const std::exception&) {
            svc.set_shared(json::object(), false);
        }
    }
    if (hub != nullptr) {
        svc.mcp = hub;
        say("[lamina] starting " + std::to_string(hub->servers.size()) + " MCP server" + (hub->servers.size() != 1 ? "s" : "") +
            " for the web app's chat: " + server_names(*hub));
        hub->start();
    }
    // the HTTP side
    svc.host_names = host_names_for(host, svc.allowed_hosts, svc.trusted_origins);
    svc.telemetry = std::make_shared<Telemetry>(
        [&svc] { return json{{"tok_s", svc.tok_s()}, {"tok_s_mean", svc.tok_s_mean()}, {"prefill_tok_s_mean", svc.prefill_tok_s_mean()}}; },
        svc.gpu_index, svc.gpu_indices);
    HttpServer http(svc);
    http.latencies = std::make_shared<prometheus::Latencies>();
    auto lat = http.latencies;
    svc.observe_latencies = [lat](std::optional<double> ttft, int64_t n, double decode_s, double total_s) {
        lat->observe(ttft, n, decode_s, total_s);
    };
    try {
        http.bind(host, port);
    } catch (const std::exception& e) {
        usage_error(e.what());
    }
    http.start();
    svc.start_idle_unload();
    const int bound = http.port();
    const std::string here = host == "0.0.0.0" || host.empty() || host == "::" ? "127.0.0.1" : host;
    say("ready: http://" + here + ":" + std::to_string(bound) + "/v1  (OpenAI: /v1/chat/completions, Anthropic: /v1/messages, "
        "context " + std::to_string(engine->max_context()) + " tokens" + (vision != nullptr ? ", images on" : "") +
        (truthy(svc.api_key) ? ", API key required" : "") + ")");
    say("       open http://" + here + ":" + std::to_string(bound) + "/ in a browser to chat; close this window to stop the model");
    if (!loopback) {   // issue #26: reachable from other devices - at which address, and what can still block it
        const std::vector<std::string> ips = lan_addresses();
        for (const auto& ip : ips)
            say("       from other devices: http://" + ip + ":" + std::to_string(bound) + "/   (API: http://" + ip + ":" +
                std::to_string(bound) + "/v1)");
        if (ips.empty()) say("       from other devices: http://<this PC's IP address>:" + std::to_string(bound) + "/");
        if (!truthy(svc.api_key))
            say("       WARNING: no API key - anyone on your network can use this model. Add \"api_key\": \"...\" to the config "
                "(clients send it as their API key; the web page asks for it)");
#ifdef _WIN32
        say("       nothing arrives? Windows Firewall blocks it until allowed: accept its prompt for lamina-server, or run in an "
            "admin PowerShell:\n         New-NetFirewallRule -DisplayName \"Lamina " + std::to_string(bound) + "\" -Direction "
            "Inbound -Protocol TCP -LocalPort " + std::to_string(bound) + " -Action Allow -Profile Private\n       (and set "
            "this network to Private in Windows' network settings)");
#endif
    }
    if (a.open && jget(cfg, "open_browser") != false) {   // #609: the config's "open_browser": false wins
        const std::string url = "http://" + std::string(host == "0.0.0.0" || host.empty() ? "127.0.0.1" : host) + ":" + std::to_string(bound) + "/";
        std::thread([url] { open_browser(url); }).detach();
    }
    if (!ready_path.empty()) {
        const std::string rh = here.find(':') != std::string::npos && !starts_with(here, "[") ? "[" + here + "]" : here;
        json record = {{"api_state", "ready"}, {"api_url", "http://" + rh + ":" + std::to_string(bound)}, {"bound_host", host},
                       {"bound_port", bound}, {"created_at", now_s()}, {"instance_id", svc.instance_id},
                       {"model_id", svc.model}, {"model_state", lazy ? "unloaded" : "ready"}, {"pid", self_pid()},
                       {"schema_version", 1}};   // sorted keys, as json.dumps(sort_keys=True)
        std::FILE* f = std::fopen(ready_path.c_str(), "wx");
        if (f == nullptr || std::fputs((py_dumps(record, false) + "\n").c_str(), f) < 0 || std::fclose(f) != 0) {
            say_err("[lamina] could not write ready file (OSError)");
            http.stop();
            engine->close();
            return 6;
        }
    }
    while (g_stops.load() == 0) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    say("\n[lamina] stopping (Ctrl+C again to end the engine at once) ...");
    bool drained = true;
    if (a.shutdown_timeout) {
        int64_t pending = svc.begin_drain();
        if (pending) say("[lamina] draining " + std::to_string(pending) + " inference call(s) for up to " + py_g(*a.shutdown_timeout) + " s ...");
        pending = svc.wait_for_drain(*a.shutdown_timeout);
        if (pending) {
            const int64_t cancelled = svc.cancel_owned_requests();
            say("[lamina] drain deadline expired; cancelling " + std::to_string(cancelled) + " owned request(s) ...");
            svc.wait_for_drain(5);
            drained = false;
        }
    } else {   // no time given to finish: the running and the queued requests stop now (a console close allows 4.5 s)
        svc.begin_drain();
        if (const int64_t n = svc.cancel_owned_requests()) say("[lamina] stopping " + std::to_string(n) + " request(s) ...");
        svc.wait_for_drain(5);
    }
    http.stop();
    if (svc.telemetry != nullptr) svc.telemetry->close();
    try {
        engine->close();
    } catch (const EngineStuck& e) {
        say_err(std::string("[lamina] ") + e.what());
        std::_Exit(1);
    }
    vision.reset();   // its directory goes with it (#914)
    if (hub != nullptr) hub->close();
    if (!ready_path.empty()) {   // only this instance's own record
        std::ifstream f(u8_path(ready_path), std::ios::binary);
        json rec;
        if (f && py_loads(std::string(std::istreambuf_iterator<char>(f), {}), rec) && jstr(rec, "instance_id") == svc.instance_id &&
            jget(rec, "pid") == self_pid()) {
            f.close();
            std::error_code ec;
            std::filesystem::remove(u8_path(ready_path), ec);
        }
    }
    g_finished = true;
    say("[lamina] stopped");
    return drained ? 0 : 8;
}

}  // namespace

int main(int argc, char** argv) {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
#endif
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        say_err(std::string("[lamina] ") + e.what());
        return 1;
    }
}
