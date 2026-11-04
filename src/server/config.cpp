// src/server/config.cpp - the run config as the server reads it (config.hpp).
#include "config.hpp"

#include "log.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <regex>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace lamina {
namespace {

constexpr int kParallelMax = 8;   // the engine's batch window holds at most 8 rows (kVerifyMaxT)

void set_env(const std::string& k, const std::string& v) {
#ifdef _WIN32
    _putenv_s(k.c_str(), v.c_str());
#else
    setenv(k.c_str(), v.c_str(), 1);
#endif
}

std::string arg_str(const json& a) { return a.is_string() ? a.get<std::string>() : py_dumps(a); }

std::vector<std::string> args_of(const json& cfg) {
    std::vector<std::string> out;
    if (jget(cfg, "args").is_array())
        for (const auto& a : cfg["args"]) out.push_back(arg_str(a));
    return out;
}

bool has(const std::vector<std::string>& args, const char* flag) { return std::find(args.begin(), args.end(), flag) != args.end(); }

/// (total, available) RAM in bytes; zeros when it cannot be read
std::pair<double, double> ram() {
#ifdef _WIN32
    MEMORYSTATUSEX m{};
    m.dwLength = sizeof m;
    if (GlobalMemoryStatusEx(&m)) return {(double) m.ullTotalPhys, (double) m.ullAvailPhys};
    return {0, 0};
#else
    std::ifstream f("/proc/meminfo");
    std::string key;
    double total = 0, avail = 0, kb;
    std::string unit;
    while (f >> key >> kb >> unit) {
        if (key == "MemTotal:") total = kb * 1024;
        if (key == "MemAvailable:") avail = kb * 1024;
    }
    return {total, avail};
#endif
}

/// an expert profile's (layers, experts per layer) from its header (tools/make_profile.py's format)
std::optional<std::pair<uint32_t, uint32_t>> profile_shape(const std::string& path) {
    std::ifstream f(u8_path(path), std::ios::binary | std::ios::ate);
    if (!f) return std::nullopt;
    const std::streamoff size = f.tellg();
    f.seekg(0);
    unsigned char head[24];
    if (!f.read((char*) head, sizeof head) || std::memcmp(head, "STRP", 4) != 0) return std::nullopt;
    auto u32 = [&](int i) { return (uint32_t) head[4 + 4 * i] | (uint32_t) head[5 + 4 * i] << 8 | (uint32_t) head[6 + 4 * i] << 16 | (uint32_t) head[7 + 4 * i] << 24; };
    const uint32_t nl = u32(1), ne = u32(2), n = u32(4);
    if (size < 24 + 4 * (std::streamoff) n) return std::nullopt;
    return std::make_pair(nl, ne);
}

std::string here_path(const json& cfg, const std::string& p) {
    const std::filesystem::path x = u8_path(p);
    if (x.is_absolute()) return p;
    const std::string cwd = jstr(cfg, "cwd", ".");
    return u8_str(u8_path(cwd.empty() ? "." : cwd) / x);
}

}  // namespace

json read_config(const std::string& path) {
    std::ifstream f(u8_path(path), std::ios::binary);
    if (!f) throw ConfigError(path + " cannot be read");
    std::string text((std::istreambuf_iterator<char>(f)), {});
    if (starts_with(text, "\xEF\xBB\xBF")) text.erase(0, 3);   // Notepad adds a BOM
    json cfg;
    std::string err;
    if (!py_loads(text, cfg, &err))
        throw ConfigError(path + " is not valid JSON: " + err + ". Check the commas between the entries of \"args\" (each "
                          "option and each value is its own quoted entry, e.g. \"--ple-io\", \"mmap\",) and that no entry is "
                          "followed by a comma at the end of a list.");
    if (!cfg.is_object()) throw ConfigError(path + " is not a JSON object");
    return cfg;
}

void split_joined_args(json& cfg) {
    if (!jget(cfg, "args").is_array()) return;
    static const std::regex joined(R"(^(--[A-Za-z0-9][A-Za-z0-9-]*)(?:\s+|=)(\S.*)$)");
    static const std::regex flag(R"(^--[A-Za-z0-9][A-Za-z0-9-]*$)");
    json out = json::array();
    std::vector<std::string> changed;
    for (const auto& a : cfg["args"]) {
        std::smatch m;
        const std::string s = a.is_string() ? a.get<std::string>() : "";
        const bool value_place = !out.empty() && out.back().is_string() && std::regex_match(out.back().get_ref<const std::string&>(), flag);
        if (!a.is_string() || !std::regex_match(s, m, joined) || value_place) {
            out.push_back(a);
            continue;
        }
        out.push_back(m[1].str());
        out.push_back(std::string(strip(m[2].str())));
        changed.push_back(s);
    }
    if (changed.empty()) return;
    cfg["args"] = out;
    std::string list;
    for (const auto& c : changed) list += (list.empty() ? "" : ", ") + py_repr_value(json(c));
    say("[lamina] the config's args hold " + list + " as one element each: split into the flag and its value (write them "
        "as two elements, e.g. \"--ple-io\", \"mmap\")");
}

void lift_reasoning_effort_arg(json& cfg) {
    if (!jget(cfg, "args").is_array()) return;
    json out = json::array(), value;
    const json& args = cfg["args"];
    for (size_t i = 0; i < args.size();) {
        const std::string a = args[i].is_string() ? args[i].get<std::string>() : "";
        if (a == "--reasoning-effort" && i + 1 < args.size()) {
            value = args[i + 1];
            i += 2;
        } else if (starts_with(a, "--reasoning-effort=")) {
            value = a.substr(a.find('=') + 1);
            ++i;
        } else {
            out.push_back(args[i]);
            ++i;
        }
    }
    if (value.is_null()) return;
    cfg["args"] = out;
    if (jget(cfg, "reasoning_effort").is_null()) cfg["reasoning_effort"] = value;
    say("[lamina] '--reasoning-effort' in the config's args is not an engine option: taken out of them. The thinking level "
        "is the config's 'reasoning_effort' key (outside 'args'); it is used as " + py_repr_value(cfg["reasoning_effort"]));
}

std::vector<int> gpu_list(const json& cfg) {
    const json& g = jget(cfg, "gpu");
    std::vector<int> out;
    if (g.is_null() || (g.is_string() && g.get_ref<const std::string&>().empty())) return out;
    std::vector<std::string> items;
    if (g.is_array()) {
        for (const auto& x : g) items.push_back(arg_str(x));
    } else {
        items = split(arg_str(g), ",");
    }
    for (const auto& x : items) {
        const std::string s(strip(x));
        if (s.empty()) continue;
        try {
            out.push_back(std::stoi(s));
        } catch (const std::exception&) {
            throw ConfigError("\"gpu\": " + py_repr_value(json(s)) + " is not a card number");
        }
    }
    return out;
}

std::vector<GpuInfo> nvidia_gpus() {
    std::vector<GpuInfo> out;
    try {
        SpawnOptions o;
        o.args = {"nvidia-smi", "--query-gpu=index,name,pci.bus_id,uuid", "--format=csv,noheader"};
        o.in = SpawnOptions::null;
        o.out = SpawnOptions::pipe;
        o.err = SpawnOptions::null;
        Process p(o);
        std::string line;
        while (p.read_line(line, 30) == 1) {
            std::vector<std::string> f = split(line, ",");
            for (auto& x : f) x = std::string(strip(x));
            if (f.size() == 4 && !f[0].empty() && std::all_of(f[0].begin(), f[0].end(), ::isdigit))
                out.push_back({std::stoi(f[0]), f[1], f[2], f[3]});
        }
        p.wait(30);
    } catch (const std::exception&) {
    }
    return out;
}

int find_gpu(const std::string& spec_in, const std::vector<GpuInfo>& gpus) {
    // a number (or, when no card has it, a word of a name: "3090"), a UUID or its start, a PCI bus id, or a name or a
    // part of one (a whole name first), as src/core/gpu_select.cpp reads them
    const std::string s(strip_chars(strip(spec_in), "\"'"));
    const std::string low = lower(s);
    const bool num = !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
    if (num)
        for (const auto& g : gpus)
            if (g.index == std::stoi(s)) return g.index;
    static const std::regex bus_re(R"(^(?:([0-9a-fA-F]+):)?([0-9a-fA-F]+):([0-9a-fA-F]+)(?:\.[0-9a-fA-F]+)?$)");
    auto bus = [](const std::string& text) -> std::optional<std::tuple<long, long, long>> {
        std::smatch m;
        if (!std::regex_match(text, m, bus_re)) return std::nullopt;
        auto hex = [&](int i) { return m[i].matched && m[i].length() ? std::stol(m[i].str(), nullptr, 16) : 0L; };
        return std::make_tuple(hex(1), hex(2), hex(3));
    };
    std::vector<const GpuInfo*> hits;
    if (starts_with(low, "gpu-")) {
        for (const auto& g : gpus)
            if (starts_with(lower(g.uuid), low)) hits.push_back(&g);
    } else if (bus(s)) {
        for (const auto& g : gpus)
            if (bus(g.bus_id) == bus(s)) hits.push_back(&g);
    } else {
        for (const auto& g : gpus)
            if (lower(g.name) == low) hits.push_back(&g);
        if (hits.empty()) {
            for (const auto& g : gpus) {
                const std::string name = lower(g.name);
                if (num) {   // a number as a word of a name: not a part of a longer number or word
                    for (size_t at = name.find(low); at != std::string::npos; at = name.find(low, at + 1)) {
                        const bool left = at == 0 || !std::isalnum((unsigned char) name[at - 1]);
                        const bool right = at + low.size() >= name.size() || !std::isalnum((unsigned char) name[at + low.size()]);
                        if (left && right) {
                            hits.push_back(&g);
                            break;
                        }
                    }
                } else if (name.find(low) != std::string::npos) {
                    hits.push_back(&g);
                }
            }
        }
    }
    return hits.size() == 1 ? hits[0]->index : -1;
}

std::vector<int> engine_gpus(const json& cfg) {
    const std::vector<int> listed = gpu_list(cfg);
    if (!listed.empty()) return listed;
    const std::vector<std::string> args = args_of(cfg);
    auto values = [&](std::initializer_list<const char*> flags) -> std::optional<std::vector<std::string>> {
        std::optional<size_t> at;
        for (size_t i = 0; i + 1 < args.size(); ++i)
            for (const char* f : flags)
                if (args[i] == f) at = i;
        if (!at) return std::nullopt;
        std::vector<std::string> out;
        for (const auto& x : split(args[*at + 1], ",")) {
            const std::string v(strip(x));
            if (!v.empty() && !starts_with(v, "-")) out.push_back(v);
        }
        return out;
    };
    const auto main_spec = values({"--main-gpu"});
    const std::vector<std::string> tiers = values({"--second-gpu", "--tier-gpus"}).value_or(std::vector<std::string>{});
    if ((!main_spec || main_spec->empty()) && tiers.empty()) return {};
    std::optional<std::vector<GpuInfo>> known;
    auto index = [&](const std::string& spec) {
        if (!spec.empty() && std::all_of(spec.begin(), spec.end(), ::isdigit) && !known) return std::stoi(spec);
        if (!known) known = nvidia_gpus();
        return find_gpu(spec, *known);
    };
    int main_gpu = main_spec && !main_spec->empty() ? index((*main_spec)[0]) : 0;
    if (main_gpu < 0) main_gpu = 0;
    std::vector<int> out = {main_gpu};
    if (tiers.size() == 1 && tiers[0] == "auto") {
        if (!known) known = nvidia_gpus();
        for (const auto& g : *known)
            if (g.index != main_gpu) out.push_back(g.index);
        return out;
    }
    for (const auto& t : tiers) {
        const int i = index(t);
        if (i >= 0 && std::find(out.begin(), out.end(), i) == out.end() && out.size() <= 3) out.push_back(i);   // kMaxGpuTiers
    }
    return out;
}

std::optional<std::vector<std::string>> effort_end_args(const json& cfg, const Tokenizer& tok) {
    const json& pos = cfg.contains("effort_position") ? cfg["effort_position"] : json("start");
    if (pos != "start" && pos != "end")
        throw ConfigError("\"effort_position\" must be \"start\" (the default) or \"end\", not " + py_repr_value(pos));
    if (pos == "start") return std::nullopt;
    const std::vector<int32_t> role = tok.encode("system", true);
    if (role.size() != 1) {
        say("[lamina] effort_position \"end\" needs the role \"system\" to be one token: the reasoning effort stays at the "
            "top of the prompt");
        return std::nullopt;
    }
    say("[lamina] effort_position end: a request's non-default reasoning effort goes right before the answer, so changing it "
        "keeps the cached conversation");
    return std::vector<std::string>{"--tail-role-token", std::to_string(role[0])};
}

double engine_silence_s(const json& cfg) {
    const json& v = jget(cfg, "engine_silence_s");
    if (v.is_null()) return 300.0;
    if (v.is_boolean() || !v.is_number() || v.get<double>() < 0)
        throw ConfigError("\"engine_silence_s\" must be a number of seconds >= 0 (0 = no limit), not " + py_repr_value(v));
    return v.get<double>();
}

std::vector<std::string> engine_args(const json& cfg) {
    std::vector<std::string> args = args_of(cfg);
    // #1322: a "vision" section without --vision in the args advertised images and then refused every picture
    if (jget(cfg, "vision").is_object() && !has(args, "--vision")) {
        args.push_back("--vision");
        std::string note = "added --vision";
        if (truthy(jget(cfg["vision"], "gpu")) && !has(args, "--vram-reserve-mib")) {
            args.insert(args.end(), {"--vram-reserve-mib", "700"});
            note += " and --vram-reserve-mib 700";
        }
        say("[lamina] the config has a \"vision\" section but no --vision in its args: " + note + " (run setup again with "
            "--vision to write them)");
    }
    // #465 (opt-in): "parallel": N - up to N requests decode together in the engine's batch slots
    const json& n = jget(cfg, "parallel");
    if (!has(args, "--batch") && !has(args, "--slots") && !n.is_null() && n != false) {
        if (n.is_boolean() || !n.is_number_integer()) {
            say("[lamina] \"parallel\" must be a whole number of requests (2.." + std::to_string(kParallelMax) + "), not " +
                py_repr_value(n) + ": ignored");
        } else if (n.get<int64_t>() > 1) {
            const char* mtp = std::getenv("LAMINA_BATCH_MTP");
            if (n.get<int64_t>() > kParallelMax && !has(args, "--batch-mtp") && (mtp == nullptr || !*mtp || std::string(mtp) == "0"))
                say("[lamina] \"parallel\": " + std::to_string(n.get<int64_t>()) + " - the engine runs at most " +
                    std::to_string(kParallelMax) + " requests together; it will use " + std::to_string(kParallelMax));
            args.insert(args.end(), {"--batch", std::to_string(n.get<int64_t>())});
        }
    }
    // #477 (opt-in): "expert_profile_save": the engine saves what its adaptive tier learned there, and the next start
    // begins from it when it is a profile of the same model
    const json& save = jget(cfg, "expert_profile_save");
    if (!save.is_string() || strip(save.get<std::string>()).empty() || has(args, "--expert-profile-save")) return args;
    args.insert(args.end(), {"--expert-profile-save", save.get<std::string>()});
    const json& every = jget(cfg, "expert_profile_save_every");
    if (every.is_number() && every.get<double>() >= 0)
        args.insert(args.end(), {"--expert-profile-save-every", every.is_number_integer() ? std::to_string(every.get<int64_t>())
                                                                                           : py_repr(every.get<double>())});
    for (size_t i = 0; i + 2 < args.size(); ++i)
        if (args[i] == "--expert-profile") {
            const auto learned = profile_shape(here_path(cfg, save.get<std::string>()));
            const auto base = profile_shape(here_path(cfg, args[i + 1]));
            if (learned && learned == base) args[i + 1] = save.get<std::string>();
            break;
        }
    return args;
}

std::string model_file(const json& cfg) {
    const std::vector<std::string> args = args_of(cfg);
    for (size_t i = 0; i + 1 < args.size(); ++i)
        if (args[i] == "--native") return here_path(cfg, args[i + 1]);
    return "";
}

const std::vector<std::pair<std::string, std::string>>& offline_env() {
    // what the common runtimes honour (Hugging Face, Transformers, pip, the proxies): an accidental call fails fast
    static const std::vector<std::pair<std::string, std::string>> env = {
        {"HF_HUB_OFFLINE", "1"}, {"TRANSFORMERS_OFFLINE", "1"}, {"HF_HUB_DISABLE_TELEMETRY", "1"}, {"DO_NOT_TRACK", "1"},
        {"PIP_NO_INDEX", "1"}, {"PIP_DISABLE_PIP_VERSION_CHECK", "1"}, {"NO_PROXY", "*"}, {"no_proxy", "*"},
        {"HTTP_PROXY", ""}, {"HTTPS_PROXY", ""}, {"http_proxy", ""}, {"https_proxy", ""}, {"LAMINA_OFFLINE", "1"}};
    return env;
}

void apply_engine_env(const json& cfg, bool offline) {
    const std::vector<int> gpus = gpu_list(cfg);
    if (!gpus.empty()) {   // issue #51: the GPU(s) as nvidia-smi numbers them (CUDA's own order can differ)
        std::string v;
        for (const int g : gpus) v += (v.empty() ? "" : ",") + std::to_string(g);
        set_env("CUDA_DEVICE_ORDER", "PCI_BUS_ID");
        set_env("CUDA_VISIBLE_DEVICES", v);
    }
    const json& env = jget(cfg, "env");   // engine settings the config carries
    if (env.is_object())
        for (auto it = env.begin(); it != env.end(); ++it) set_env(it.key(), arg_str(it.value()));
    if (offline)   // last, so the config's own "env" cannot remove the policy
        for (const auto& [k, v] : offline_env()) set_env(k, v);
}

Environment child_env(const json& cfg, bool offline) {
    Environment env = environment();
    const std::vector<int> gpus = gpu_list(cfg);
    if (!gpus.empty()) {   // issue #51: the GPU(s) as nvidia-smi numbers them
        std::string v;
        for (const int g : gpus) v += (v.empty() ? "" : ",") + std::to_string(g);
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
        env["CUDA_VISIBLE_DEVICES"] = v;
    }
    const json& extra = jget(cfg, "env");
    if (extra.is_object())
        for (auto it = extra.begin(); it != extra.end(); ++it) env[it.key()] = arg_str(it.value());
    if (offline)
        for (const auto& [k, v] : offline_env()) env[k] = v;
    prepend_lib_dirs(cfg, env);
    return env;
}

void prepend_lib_dirs(const json& cfg, Environment& env) {
    std::string dirs;
    const json& lib = jget(cfg, "lib_dirs");
    if (lib.is_array())
        for (const auto& d : lib) {
            std::error_code ec;
            if (d.is_string() && std::filesystem::is_directory(u8_path(d.get<std::string>()), ec))
#ifdef _WIN32
                dirs += (dirs.empty() ? "" : ";") + d.get<std::string>();
#else
                dirs += (dirs.empty() ? "" : ":") + d.get<std::string>();
#endif
        }
    if (!dirs.empty()) {
#ifdef _WIN32
        const char* var = "PATH";
        const char* sep = ";";
#else
        const char* var = "LD_LIBRARY_PATH";
        const char* sep = ":";
#endif
        const auto it = env.find(var);
        env[var] = it == env.end() || it->second.empty() ? dirs : dirs + sep + it->second;
    }
}

Environment vision_env(const json& cfg) {
    Environment env = environment();
    prepend_lib_dirs(cfg, env);   // a CUDA encoder's runtime (its RUNPATH is its own folder)
    const json& dev = jget(jget(cfg, "vision"), "cuda_device");
    if (dev.is_null()) return env;
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";   // #408: the encoder on a card of its own
    env["CUDA_VISIBLE_DEVICES"] = arg_str(dev);
    return env;
}

std::vector<std::string> offline_blockers(const json& cfg) {
    std::vector<std::string> needs;
    const json& servers = truthy(jget(cfg, "mcp_servers")) ? cfg["mcp_servers"] : jget(cfg, "mcpServers");
    if (truthy(servers))
        needs.push_back("the config declares " + std::to_string(servers.is_object() || servers.is_array() ? servers.size() : 1) +
                        " MCP server(s), which are started as subprocesses and may fetch tools");
    for (const char* key : {"before_load", "beforeLoad"})
        if (truthy(jget(cfg, key)))
            needs.push_back(std::string("the config declares a ") + key + " command, which runs before the model loads");
    for (const char* key : {"images", "vision"}) {
        const json& section = jget(cfg, key);
        if (!section.is_object()) continue;
        for (const char* field : {"remote_sources", "allowed_hosts", "url"})
            if (truthy(jget(section, field)))
                needs.push_back(std::string("the config's ") + key + "." + field + " permits network sources");
    }
    return needs;
}

void warn_tight_ram(const json& arena_mib) {
    if (!arena_mib.is_number_integer() || arena_mib.get<int64_t>() <= 0) return;
    const double total = ram().first;
    if (total <= 0) return;
    const double left = total / 1073741824.0 - (double) arena_mib.get<int64_t>() / 1024;
    if (left >= 6) return;
    char buf[256];
    std::snprintf(buf, sizeof buf, "[lamina] WARNING: RAM is tight - the model's experts take %.1f GB of this PC's %.0f GB, "
                  "leaving %.1f GB for everything else. ", (double) arena_mib.get<int64_t>() / 1024, total / 1073741824.0, left);
#ifdef _WIN32
    say(std::string(buf) + "Windows will slow down (paging to disk). Close other programs, or set up the smaller quant "
        "(setup --model iq3_s).");
#else
    say(std::string(buf) + "Linux may stop the engine in the middle of an answer. Close other programs, or set up the "
        "smaller quant (setup --model iq3_s).");
#endif
}

}  // namespace lamina
