// src/server/config.hpp - the run config (lamina-<model>.json) as the server reads it: the engine's arguments and what
// the config's other keys add to them, the GPUs it names, the environment the engine runs in, the offline policy.
#pragma once

#include "json.hpp"
#include "process.hpp"
#include "tokenizer.hpp"

#include <optional>
#include <string>
#include <vector>

namespace lamina {

/// a config that cannot be used: the server does not start (main prints "[lamina] config <what>")
struct ConfigError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

/// the file's JSON (a BOM allowed: Notepad writes one); a mistake is said with its place in the file
json read_config(const std::string& path);
/// #2267: an option written as one element of "args" ("--ple-io mmap", "--ple-io=mmap") is split into two (said once)
void split_joined_args(json& cfg);
/// #1641: "--reasoning-effort X" in the args is the config's reasoning_effort, not an engine option (said once)
void lift_reasoning_effort_arg(json& cfg);
/// the config's "gpu": one card, or several ([0, 2] or "0,2"), numbered as nvidia-smi numbers them
std::vector<int> gpu_list(const json& cfg);
struct GpuInfo {
    int index;
    std::string name, bus_id, uuid;
};
/// the NVIDIA cards as nvidia-smi lists them (empty without nvidia-smi)
std::vector<GpuInfo> nvidia_gpus();
/// the card `spec` names, read as the engine reads --main-gpu / --second-gpu; -1 when none or several match
int find_gpu(const std::string& spec, const std::vector<GpuInfo>& gpus);
/// the cards the engine works on (the config's "gpu", else --main-gpu and --second-gpu), the first its main one
std::vector<int> engine_gpus(const json& cfg);
/// #458: --tail-role-token for "effort_position": "end" (nullopt: the effort stays at the top); ConfigError for a
/// value other than "start" / "end"
std::optional<std::vector<std::string>> effort_end_args(const json& cfg, const Tokenizer& tok);
/// #481: seconds the engine may print nothing during a request (0: no limit); ConfigError for a bad value
double engine_silence_s(const json& cfg);
/// the engine's arguments: the config's, with what "vision", "parallel" and "expert_profile_save" ask for
std::vector<std::string> engine_args(const json& cfg);
/// the model file of the engine's arguments (--native), relative ones the config's cwd's
std::string model_file(const json& cfg);
/// the engine's environment, set in this process before the engine starts: "gpu" (CUDA_DEVICE_ORDER and
/// CUDA_VISIBLE_DEVICES), the config's "env", the offline policy
void apply_engine_env(const json& cfg, bool offline);
/// a child engine's environment (a replica): this process's with the config's "gpu", "env", "lib_dirs" and the offline
/// policy
Environment child_env(const json& cfg, bool offline);
/// the image encoder's environment: this process's with the config's "lib_dirs", on its own card when the vision
/// section names "cuda_device"
Environment vision_env(const json& cfg);
/// the config's "lib_dirs" (the existing ones) before env's library search path: LD_LIBRARY_PATH, PATH on Windows
void prepend_lib_dirs(const json& cfg, Environment& env);
/// the variables every child gets under --offline
const std::vector<std::pair<std::string, std::string>>& offline_env();
/// why this config cannot be strictly offline (empty: it can)
std::vector<std::string> offline_blockers(const json& cfg);
/// the experts' RAM (INFO arena_mib) leaves less than ~6 GB for everything else: a warning
void warn_tight_ram(const json& arena_mib);

}  // namespace lamina
