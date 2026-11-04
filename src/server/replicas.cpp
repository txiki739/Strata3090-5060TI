// src/server/replicas.cpp - data-parallel replicas (replicas.hpp).
#include "replicas.hpp"

#include "log.hpp"

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <set>
#include <thread>

#ifndef _WIN32
#include <sched.h>
#endif

namespace lamina {
namespace {

constexpr size_t kBlock = 64;            // tokens per hashed block of a prompt
constexpr size_t kIndexMax = 262144;     // block hashes kept (about 17M tokens of prompts)
constexpr double kHealCooldownS = 60.0;  // a replica that died is started again at most this often

thread_local LaminaEngine* tl_current = nullptr;   // the replica this thread's request runs on
thread_local uint64_t tl_replica_dones = 0;        // the DONEs this thread's requests read, on whichever replica

std::vector<int> cards(const json& g) {
    std::vector<std::string> items;
    if (g.is_null() || (g.is_string() && g.get_ref<const std::string&>().empty())) return {};
    if (g.is_array()) {
        for (const auto& x : g) items.push_back(x.is_string() ? x.get<std::string>() : py_dumps(x));
    } else {
        items = split(g.is_string() ? g.get<std::string>() : py_dumps(g), ",");
    }
    std::vector<int> out;
    for (const auto& x : items) {
        const std::string s(strip(x));
        if (s.empty()) continue;
        try {
            size_t used = 0;
            out.push_back(std::stoi(s, &used));
            if (used != s.size()) throw std::invalid_argument(s);
        } catch (const std::exception&) {
            throw ReplicaConfigError("cards must be numbers, not " + py_repr_value(g));
        }
    }
    return out;
}

std::string suffixed(const json& path, size_t i) {
    if (!path.is_string() || path.get_ref<const std::string&>().empty()) return path.is_string() ? path.get<std::string>() : "";
    const std::string p = path.get<std::string>();
    const std::filesystem::path x = u8_path(p);
    const std::string ext = u8_str(x.extension());
    return p.substr(0, p.size() - ext.size()) + ".r" + std::to_string(i) + ext;
}

std::string list_str(const std::vector<int>& v) {
    std::string out = "[";
    for (size_t k = 0; k < v.size(); ++k) out += (k ? ", " : "") + std::to_string(v[k]);
    return out + "]";
}

}  // namespace

std::vector<std::vector<int>> gpu_groups(const json& cfg) {
    const json& spec = jget(cfg, "replicas");
    if (spec.is_null() || spec == false) return {};
    const std::vector<int> all = cards(jget(cfg, "gpu"));
    if (spec.is_boolean() || !(spec.is_number_integer() || spec.is_array()))
        throw ReplicaConfigError("\"replicas\" must be a number or a list of {\"gpus\": [...]}, not " + py_repr_value(spec));
    if (spec.is_number_integer()) {
        const int64_t n = spec.get<int64_t>();
        if (n <= 1) return {};
        if (all.empty())
            throw ReplicaConfigError("\"replicas\": " + std::to_string(n) + " needs the \"gpu\" list to cut into groups (e.g. "
                                     "\"gpu\": [0, 1, 2, 3])");
        if ((int64_t) all.size() % n)
            throw ReplicaConfigError("\"replicas\": " + std::to_string(n) + " does not divide the " + std::to_string(all.size()) +
                                     " cards " + list_str(all) + " evenly: name the groups (\"replicas\": [{\"gpus\": [0, 1]}, "
                                     "...]) or change the count");
        const size_t each = all.size() / (size_t) n;
        std::vector<std::vector<int>> out;
        for (size_t i = 0; i < (size_t) n; ++i) out.emplace_back(all.begin() + (ptrdiff_t) (i * each), all.begin() + (ptrdiff_t) ((i + 1) * each));
        return out;
    }
    std::vector<std::vector<int>> groups;
    for (const auto& e : spec) groups.push_back(cards(e.is_object() ? jget(e, "gpus") : e));
    if (std::any_of(groups.begin(), groups.end(), [](const auto& g) { return g.empty(); }))
        throw ReplicaConfigError("every replica needs its cards: {\"gpus\": [0, 1]}");
    std::set<int> seen;
    size_t total = 0;
    for (const auto& g : groups) {
        seen.insert(g.begin(), g.end());
        total += g.size();
    }
    if (seen.size() != total) {
        std::string s = "[";
        for (size_t k = 0; k < groups.size(); ++k) s += (k ? ", " : "") + list_str(groups[k]);
        throw ReplicaConfigError("a card is in two replicas: " + s + "]");
    }
    if (groups.size() < 2) return {};
    return groups;
}

json replica_cfg(const json& cfg, size_t i, const std::vector<int>& gpus) {
    json c = cfg;
    const json& spec = jget(cfg, "replicas");
    const json entry = spec.is_array() && i < spec.size() && spec[i].is_object() ? spec[i] : json::object();
    c.erase("replicas");
    c["gpu"] = gpus.size() > 1 ? json(gpus) : json(gpus[0]);
    c["log"] = suffixed(jget(cfg, "log"), i);
    if (truthy(jget(cfg, "expert_profile_save"))) c["expert_profile_save"] = suffixed(cfg["expert_profile_save"], i);
    if (truthy(jget(entry, "env"))) {
        json env = jget(c, "env").is_object() ? c["env"] : json::object();
        for (auto it = entry["env"].begin(); it != entry["env"].end(); ++it)
            env[it.key()] = it.value().is_string() ? it.value().get<std::string>() : py_dumps(it.value());
        c["env"] = env;
    }
    if (truthy(jget(entry, "args"))) {
        json args = jget(c, "args").is_array() ? c["args"] : json::array();
        for (const auto& a : entry["args"]) args.push_back(a.is_string() ? a.get<std::string>() : py_dumps(a));
        c["args"] = args;
    }
    return c;
}

std::optional<std::vector<std::vector<int>>> cpu_sets(size_t n, bool enabled) {
#ifdef _WIN32
    (void) n;
    (void) enabled;
    return std::nullopt;
#else
    if (!enabled || n < 2) return std::nullopt;
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) != 0) return std::nullopt;
    std::map<std::pair<int, int>, std::vector<int>> cores;   // (package, core) -> its CPUs
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &set)) continue;
        const std::string t = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream pkg(t + "physical_package_id"), core(t + "core_id");
        int p = -1, c = -1;
        if (!(pkg >> p) || !(core >> c)) {
            say("[lamina] replicas: no separate CPU cores (cannot read physical-core topology for CPU " + std::to_string(cpu) + ")");
            return std::nullopt;
        }
        cores[{p, c}].push_back(cpu);
    }
    std::vector<std::vector<int>> available;
    for (auto& [k, v] : cores) available.push_back(v);
    std::sort(available.begin(), available.end(), [](const auto& a, const auto& b) { return a.front() < b.front(); });
    if (available.size() < n) {
        say("[lamina] replicas: no separate CPU cores (" + std::to_string(n) + " instances need at least " + std::to_string(n) +
            " free physical cores, but only " + std::to_string(available.size()) + " are available)");
        return std::nullopt;
    }
    const size_t size = available.size() / n, extra = available.size() % n;
    std::vector<std::vector<int>> out;
    size_t start = 0;
    for (size_t i = 0; i < n; ++i) {
        const size_t end = start + size + (i < extra ? 1 : 0);
        std::vector<int> g;
        for (size_t k = start; k < end; ++k) g.insert(g.end(), available[k].begin(), available[k].end());
        std::sort(g.begin(), g.end());
        out.push_back(g);
        start = end;
    }
    return out;
#endif
}

// ---------------------------------------------------------------------------------------------- Router
std::vector<uint64_t> Router::chain(const std::vector<int32_t>& ids) {
    std::vector<uint64_t> out;
    uint64_t h = 1469598103934665603ull;
    for (size_t k = 0; k + kBlock <= ids.size(); k += kBlock) {
        for (size_t j = k; j < k + kBlock; ++j) h = (h ^ (uint32_t) ids[j]) * 1099511628211ull;   // FNV-1a, chained
        h ^= h >> 29;
        out.push_back(h);
    }
    return out;
}

std::vector<int64_t> Router::matched(const std::vector<int32_t>& ids) {
    std::vector<int64_t> got(n_, 0);
    uint64_t alive = n_ >= 64 ? ~0ull : (1ull << n_) - 1;
    std::lock_guard<std::mutex> lk(mu_);
    const std::vector<uint64_t> hs = chain(ids);
    for (size_t k = 0; k < hs.size(); ++k) {
        const auto it = blocks_.find(hs[k]);
        const uint64_t m = it == blocks_.end() ? 0 : it->second.first & alive;
        if (!m) break;
        alive = m;
        for (size_t r = 0; r < n_; ++r)
            if (m >> r & 1) got[r] = (int64_t) ((k + 1) * kBlock);
    }
    return got;
}

void Router::note(size_t r, const std::vector<int32_t>& ids) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const uint64_t h : chain(ids)) {
        auto it = blocks_.find(h);
        uint64_t mask = 1ull << r;
        if (it != blocks_.end()) {
            mask |= it->second.first;
            order_.erase(it->second.second);
            blocks_.erase(it);
        }
        order_.push_back(h);
        blocks_[h] = {mask, std::prev(order_.end())};
    }
    while (blocks_.size() > kIndexMax) {
        blocks_.erase(order_.front());
        order_.pop_front();
    }
}

void Router::forget(size_t r) {
    std::lock_guard<std::mutex> lk(mu_);
    const uint64_t bit = ~(1ull << r);
    for (auto it = blocks_.begin(); it != blocks_.end();) {
        it->second.first &= bit;
        if (it->second.first == 0) {
            order_.erase(it->second.second);
            it = blocks_.erase(it);
        } else {
            ++it;
        }
    }
}

// ---------------------------------------------------------------------------------------------- ReplicaEngine
ReplicaEngine::ReplicaEngine(std::vector<std::shared_ptr<LaminaEngine>> engines, int64_t load_tokens)
    : engines_(std::move(engines)), router_(engines_.size()), load_tokens_(load_tokens) {
    if (engines_.size() < 2) throw std::invalid_argument("ReplicaEngine needs two or more engines");
    inflight_.assign(engines_.size(), 0);
    routed_.assign(engines_.size(), 0);
    healing_.assign(engines_.size(), false);
    heal_at_.assign(engines_.size(), 0.0);
    for (size_t k = 0; k < engines_.size(); ++k) run_locks_.push_back(std::make_unique<std::timed_mutex>());
    model_path = engines_[0]->model_path;
    refresh_info();
}

void ReplicaEngine::refresh_info() {
    json i = engines_[0]->info();
    i["replicas"] = engines_.size();
    i["replicas_alive"] = std::count_if(engines_.begin(), engines_.end(), [](const auto& e) { return e->alive(); });
    set_info(std::move(i));
}

bool ReplicaEngine::alive() {
    return std::any_of(engines_.begin(), engines_.end(), [](const auto& e) { return e->alive(); });
}

int64_t ReplicaEngine::max_context() const {
    int64_t best = 0;
    for (const auto& e : engines_)
        if (e->max_context() > 0 && e->alive() && (best == 0 || e->max_context() < best)) best = e->max_context();
    return best;
}

int64_t ReplicaEngine::known_ctx() const {
    int64_t best = 0;
    for (const auto& e : engines_)
        if (e->known_ctx() > 0 && (best == 0 || e->known_ctx() < best)) best = e->known_ctx();
    return best;
}

bool ReplicaEngine::starting() const {
    bool any_alive = false, any_starting = false;
    for (const auto& e : engines_) {
        any_alive = any_alive || e->alive();
        any_starting = any_starting || e->starting();
    }
    return !any_alive && any_starting;
}

int ReplicaEngine::batch() const {
    int n = 0;
    for (const auto& e : engines_) n += std::max(1, e->batch());
    return n;
}

bool ReplicaEngine::unloaded() const {
    return std::all_of(engines_.begin(), engines_.end(), [](const auto& e) { return e->unloaded(); });
}

std::string ReplicaEngine::death_note() {
    for (const auto& e : engines_)
        if (!e->alive()) return e->death_note();
    return "";
}

json ReplicaEngine::last_figures() const { return tl_current != nullptr ? tl_current->last_figures() : engines_[0]->last_figures(); }
uint64_t ReplicaEngine::done_count() const { return tl_replica_dones; }
std::vector<int32_t> ReplicaEngine::drained_tokens() const {
    return tl_current != nullptr ? tl_current->drained_tokens() : std::vector<int32_t>{};
}

json ReplicaEngine::slots_view() {
    json out = json::array();
    int base = 0;
    for (size_t k = 0; k < engines_.size(); ++k) {
        const int n = engines_[k]->batch();
        json view = n ? engines_[k]->slots_view() : json(nullptr);
        if (!view.is_array()) {
            std::lock_guard<std::mutex> lk(mu_);
            view = json::array({{{"slot", 0}, {"state", inflight_[k] ? "busy" : "idle"}, {"held_tokens", 0}}});
        }
        for (auto s : view) {
            s["slot"] = base + s["slot"].get<int>();
            s["replica"] = k;
            out.push_back(s);
        }
        base += std::max(1, n);
    }
    return out;
}

std::pair<int, int> ReplicaEngine::admission() {
    std::pair<int, int> sum{0, 0};
    for (const auto& e : engines_) {
        const auto [w, a] = e->admission();
        sum.first += w;
        sum.second += a;
    }
    return sum;
}

json ReplicaEngine::replica_view() {
    json out = json::array();
    std::lock_guard<std::mutex> lk(mu_);
    for (size_t k = 0; k < engines_.size(); ++k)
        out.push_back({{"replica", k}, {"alive", engines_[k]->alive()}, {"running", inflight_[k]}, {"routed", routed_[k]},
                       {"batch", engines_[k]->batch()}});
    return out;
}

json ReplicaEngine::cache_view() {
    json sum = nullptr;
    for (const auto& e : engines_) {
        const json v = e->cache_view();
        if (v.is_null()) continue;
        if (sum.is_null()) sum = json::object();
        for (auto it = v.begin(); it != v.end(); ++it)
            sum[it.key()] = sum.value(it.key(), (int64_t) 0) + it.value().get<int64_t>();
    }
    return sum;
}

void ReplicaEngine::load() {
    // the replicas that are not running start (together); the others serve when one does not
    std::vector<size_t> dead;
    for (size_t k = 0; k < engines_.size(); ++k)
        if (!engines_[k]->alive()) dead.push_back(k);
    std::vector<std::string> errors(engines_.size());
    std::vector<std::thread> ts;
    for (const size_t k : dead)
        ts.emplace_back([this, k, &errors] {
            try {
                engines_[k]->load();
                router_.forget(k);
            } catch (const std::exception& e) {
                errors[k] = e.what();
            }
        });
    for (auto& t : ts) t.join();
    refresh_info();
    for (size_t k = 0; k < engines_.size(); ++k)
        if (!errors[k].empty()) {
            if (!alive()) throw EngineDied(errors[k]);
            say("[lamina] replica " + std::to_string(k) + " did not start (" + errors[k] + "); the others serve");
        }
}

void ReplicaEngine::unload() {
    for (const auto& e : engines_) e->unload();
    for (size_t k = 0; k < engines_.size(); ++k) router_.forget(k);
    refresh_info();
}

void ReplicaEngine::close() {
    std::exception_ptr err;
    for (const auto& e : engines_) {
        try {
            e->close();
        } catch (...) {
            if (!err) err = std::current_exception();
        }
    }
    if (err) std::rethrow_exception(err);
}

int ReplicaEngine::route(const std::vector<int32_t>& ids) {
    // the lowest cost among the living ones (its running count raised); a tie goes to the least loaded
    const std::vector<int64_t> got = router_.matched(ids);
    int pick = -1;
    size_t n_alive = 0;
    {
        std::lock_guard<std::mutex> lk(mu_);
        std::tuple<int64_t, int64_t, int64_t> best{};
        for (size_t k = 0; k < engines_.size(); ++k) {
            if (!engines_[k]->alive()) continue;
            ++n_alive;
            const auto key = std::make_tuple((int64_t) ids.size() - got[k] + load_tokens_ * inflight_[k], inflight_[k], routed_[k]);
            if (pick < 0 || key < best) {
                pick = (int) k;
                best = key;
            }
        }
        if (pick >= 0) {
            ++inflight_[(size_t) pick];
            ++routed_[(size_t) pick];
        }
    }
    if (pick >= 0 && n_alive < engines_.size()) heal();
    return pick;
}

void ReplicaEngine::heal() {
    // a replica that died starts again in the background (the others keep serving)
    const double now = mono();
    for (size_t k = 0; k < engines_.size(); ++k) {
        LaminaEngine& e = *engines_[k];
        {
            std::lock_guard<std::mutex> lk(mu_);
            if (e.alive() || e.unloaded() || e.starting() || healing_[k] || now < heal_at_[k]) continue;
            heal_at_[k] = now + kHealCooldownS;
            healing_[k] = true;
        }
        std::thread([this, k] {
            try {
                say("[lamina] replica " + std::to_string(k) + " had stopped; starting it again");
                engines_[k]->load();
                router_.forget(k);
                say("[lamina] replica " + std::to_string(k) + " is running again");
            } catch (const std::exception& x) {
                say("[lamina] replica " + std::to_string(k) + " did not start again: " + x.what());
            }
            std::lock_guard<std::mutex> lk(mu_);
            healing_[k] = false;
        }).detach();
    }
}

void ReplicaEngine::generate(const std::vector<int32_t>& ids, int64_t max_new, const json& sampling, Cancel& cancel,
                             const std::string& embeddings, const std::function<bool(int32_t)>& on_token,
                             const std::function<bool()>& on_beat) {
    const int k = route(ids);
    if (k < 0) throw EngineDied("no replica is running; this request was not sent");
    LaminaEngine& e = *engines_[(size_t) k];
    tl_current = &e;
    uint64_t dones_before = 0;   // read once the replica runs this request alone, or counts this thread's DONEs
    bool counting = false;
    std::vector<int32_t> produced;
    bool locked = false, nonfinite = false;
    auto finish = [&] {
        if (counting) tl_replica_dones += e.done_count() - dones_before;
        if (locked) run_locks_[(size_t) k]->unlock();
        {
            std::lock_guard<std::mutex> lk(mu_);
            --inflight_[(size_t) k];
        }
        if (nonfinite) {
            router_.forget((size_t) k);   // #879: the engine dropped every cached conversation
        } else if (!produced.empty()) {
            std::vector<int32_t> all = ids;
            all.insert(all.end(), produced.begin(), produced.end());
            router_.note((size_t) k, all);
        }
        update_progress([mean = e.progress().tok_s_mean](Progress& p) { p.tok_s_mean = mean; });
        refresh_info();
    };
    try {
        if (cancel.replica_selected) cancel.replica_selected(k);
        router_.note((size_t) k, ids);
        if (!e.batch()) {   // no slots: this replica takes one request at a time
            while (!run_locks_[(size_t) k]->try_lock_for(std::chrono::milliseconds(250)))
                if (cancel.is_set()) {
                    finish();
                    return;
                }
            locked = true;
        }
        dones_before = e.done_count();
        counting = true;
        e.generate(ids, max_new, sampling, cancel, embeddings, [&](int32_t t) {
            produced.push_back(t);
            return on_token(t);
        }, on_beat);
    } catch (const EngineNonFinite&) {
        nonfinite = true;
        finish();
        throw;
    } catch (...) {
        finish();
        throw;
    }
    finish();
}

}  // namespace lamina
