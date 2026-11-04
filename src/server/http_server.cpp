// src/server/http_server.cpp - the HTTP side of lamina-server (http_server.hpp).
#include "http_server.hpp"

#include "api.hpp"
#include "frontend.hpp"
#include "images.hpp"
#include "log.hpp"
#include "mcp.hpp"
#include "net.hpp"
#include "prometheus.hpp"
#include "responses.hpp"
#include "runconfig.hpp"
#include "structured.hpp"
#include "vision.hpp"
#include "web_assets.hpp"

#include <httplib.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace lamina {
namespace {

constexpr size_t kChunkedBodyMax = 256u << 20;   // a million-token conversation with room to spare
constexpr size_t kRecordMax = 262144;           // the monitor keeps this much of a request's text
constexpr size_t kControlBodyMax = 65536;

/// explicit model controls must not interrupt running or queued requests: a 409
struct ModelBusy : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Anthropic clients add /v1/messages to their base URL: a base that already ends in /v1 posts /v1/v1/messages
const std::map<std::string, std::string> kV1V1 = {{"/v1/v1/messages", "/v1/messages"},
                                                  {"/v1/v1/messages/count_tokens", "/v1/messages/count_tokens"}};

/// the first n characters of UTF-8 text
std::string head_chars(const std::string& s, size_t n) {
    size_t i = 0, c = 0;
    while (i < s.size() && c < n) {
        size_t k = 1;
        utf8_at(s, i, &k);
        i += k;
        ++c;
    }
    return s.substr(0, i);
}

/// int() of a request field: an int, a float (truncated), a bool or a string of digits; RequestError otherwise
int64_t py_int(const json& v) {
    // Python's int has no bound: a value past int64's is the same "as many as fit" for every limit read with this
    constexpr int64_t kMax = std::numeric_limits<int64_t>::max(), kMin = std::numeric_limits<int64_t>::min();
    if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
    if (v.is_number_unsigned()) return (int64_t) std::min<uint64_t>(v.get<uint64_t>(), (uint64_t) kMax);
    if (v.is_number_integer()) return v.get<int64_t>();
    if (v.is_number_float()) {
        const double d = v.get<double>();
        if (!std::isfinite(d)) throw RequestError("cannot convert float " + py_repr(d) + " to integer");
        if (d >= 9223372036854775808.0) return kMax;
        if (d < -9223372036854775808.0) return kMin;
        return (int64_t) std::trunc(d);
    }
    if (v.is_string()) {
        const std::string s(strip(v.get<std::string>()));
        size_t i = s.size() && (s[0] == '-' || s[0] == '+') ? 1 : 0;
        bool ok = i < s.size();
        for (size_t j = i; j < s.size(); ++j) ok = ok && ((s[j] >= '0' && s[j] <= '9') || (s[j] == '_' && j > i && j + 1 < s.size()));
        if (ok) {
            const std::string digits = replace_all(s.substr(s[0] == '+' ? 1 : 0), "_", "");
            int64_t out = 0;
            if (std::from_chars(digits.data(), digits.data() + digits.size(), out).ec == std::errc::result_out_of_range)
                return digits[0] == '-' ? kMin : kMax;
            return out;
        }
        throw RequestError("invalid literal for int() with base 10: " + py_repr_value(v));
    }
    throw RequestError("int() argument must be a string, a bytes-like object or a real number, not '" +
                       std::string(v.is_null() ? "NoneType" : v.is_array() ? "list" : "dict") + "'");
}

/// `or` of request fields as Python reads them: the first truthy one
const json& first_truthy(std::initializer_list<const json*> vs) {
    static const json zero = 0;
    for (const json* v : vs)
        if (truthy(*v)) return *v;
    return zero;
}

/// #1997: the `wait` of POST /unload?wait=...: absent / 0 / false: 0; 1 / true / yes: `dflt`; seconds above 1
double unload_wait(const std::string* value, double dflt) {
    if (value == nullptr) return 0.0;
    const std::string v = lower(strip(*value));
    if (v.empty() || v == "0" || v == "false" || v == "no" || v == "off") return 0.0;
    if (v == "1" || v == "true" || v == "yes" || v == "on") return dflt;
    try {
        return std::min(3600.0, std::max(0.0, std::stod(v)));
    } catch (const std::exception&) {
        return dflt;
    }
}

// ---- logit_bias (serve/logit_bias.py): the engine has no token biases, so a non-empty one is refused before streaming
json normalize_bias(const json& value, int64_t vocab) {
    json out = json::object();
    if (value.is_null()) return out;
    std::vector<std::pair<json, json>> entries;
    const bool pairs = value.is_array();
    if (value.is_object()) {
        for (auto it = value.begin(); it != value.end(); ++it) entries.emplace_back(json(it.key()), it.value());
    } else if (pairs) {
        for (const auto& e : value) {
            if (!e.is_array() || e.size() != 2) throw RequestError("logit_bias: each entry must be [token_id, bias]");
            entries.emplace_back(e[0], e[1]);
        }
    } else {
        throw RequestError("logit_bias: expected an object or a list of [token_id, bias] pairs");
    }
    int64_t bans = 0;
    for (auto& [token, bias] : entries) {
        std::string t = token.is_string() ? token.get<std::string>() : token.is_number_integer() ? std::to_string(token.get<int64_t>()) : "";
        if (t.empty() || !std::all_of(t.begin(), t.end(), ::isdigit) || t.size() > 10)
            throw RequestError("logit_bias: token IDs must be non-negative integers");
        const int64_t id = std::stoll(t);
        if (id > 2147483647 || (vocab > 0 && id >= vocab)) throw RequestError("logit_bias: token ID outside the model vocabulary");
        if (out.contains(std::to_string(id))) throw RequestError("logit_bias: duplicate token ID");
        json b = bias;
        if (pairs && b.is_boolean() && !b.get<bool>()) b = -100.0;
        if (b.is_boolean() || !b.is_number() || !std::isfinite(b.get<double>()) || b.get<double>() < -100 || b.get<double>() > 100)
            throw RequestError("logit_bias: biases must be finite numbers between -100 and 100; false is a pair-list ban");
        out[std::to_string(id)] = b.get<double>();
        bans += b.get<double>() == -100;
    }
    if (vocab > 0 && bans == vocab) throw RequestError("logit_bias: cannot ban the entire vocabulary");
    return out;
}

void validate_bias(const json& req, const Engine& engine, int64_t vocab) {
    if (normalize_bias(jget(req, "logit_bias"), vocab).empty()) return;
    if (jget(engine.info, "logit_bias") != 1)
        throw RequestError("logit_bias: this engine does not support token biases; use a supporting CUDA/HIP build");
    if (engine.batch()) throw RequestError("logit_bias: continuous batching is not supported; run without --batch");
}

/// stops a request's disconnect watcher
struct Watch {
    Cancel stop;
    std::thread t;
    ~Watch() {
        stop.set();
        if (t.joinable()) t.join();
    }
};

}  // namespace

size_t body_limit() {
    // the most a request body may hold (read into memory); a larger one is answered 413 before it is read
    const char* v = std::getenv("LAMINA_MAX_BODY_MIB");
    long long mib = 0;
    if (v != nullptr) {
        try {
            mib = std::stoll(v);
        } catch (const std::exception&) {
            mib = 0;
        }
    }
    return mib > 0 ? (size_t) mib << 20 : kChunkedBodyMax;
}

// ============================================================================================ one request
namespace {

/// the API key a request sends: "Authorization: Bearer KEY", else x-api-key (Anthropic's)
std::string given_key(const httplib::Request& req) {
    const std::string auth = req.get_header_value("Authorization");
    return lower(auth.substr(0, 7)) == "bearer " ? std::string(strip(auth.substr(7))) : req.get_header_value("x-api-key");
}

class Exchange {
public:
    Exchange(Service& svc, HttpServer& server, const httplib::Request& req, httplib::Response& res)
        : svc(svc), server(server), req(req), res(res) {
        path = req.path;
        while (!path.empty() && path.back() == '/') path.pop_back();
    }
    /// every way out, an exception past do_post's handlers too: what the request held is given back
    ~Exchange() {
        try {
            finish();
        } catch (...) {
        }
    }

    Service& svc;
    HttpServer& server;
    const httplib::Request& req;
    httplib::Response& res;
    std::string path;
    std::shared_ptr<json> record;          // #332: this request's monitor record, when the monitor is on
    std::shared_ptr<Cancel> cancel;        // (before watch: its thread reads it until watch is gone)
    std::shared_ptr<Watch> watch;          // #430 #431: cancels the request when its client hangs up
    bool reserved = false, streaming = false, finished = false;

    std::string header(const char* name) const { return req.get_header_value(name); }
    const std::string* param(const char* name) const {
        static thread_local std::string v;
        if (!req.has_param(name)) return nullptr;
        v = req.get_param_value(name);
        return &v;
    }

    void note(const json& values) {
        if (record == nullptr) return;
        std::lock_guard<std::mutex> lk(svc.status_mu);
        for (auto it = values.begin(); it != values.end(); ++it) (*record)[it.key()] = it.value();
    }

    void cors() {
        // #321: CORS headers for an API path and an origin the config lists in cors_origins, nothing otherwise
        if (svc.cors_origins.empty() || !starts_with(path, "/v1/")) return;
        const std::string origin(rstrip_chars(header("Origin"), "/"));
        std::string allow;
        if (std::find(svc.cors_origins.begin(), svc.cors_origins.end(), "*") != svc.cors_origins.end()) {
            allow = "*";
        } else if (!origin.empty() && std::find(svc.cors_origins.begin(), svc.cors_origins.end(), origin) != svc.cors_origins.end()) {
            allow = origin;
            res.set_header("Vary", "Origin");
        } else {
            return;
        }
        res.set_header("Access-Control-Allow-Origin", allow);
        res.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
        const std::string asked = header("Access-Control-Request-Headers");   // "*" does not cover Authorization
        res.set_header("Access-Control-Allow-Headers",
                       asked.empty() ? "Authorization, Content-Type, x-api-key, anthropic-version, anthropic-beta" : asked);
        res.set_header("Access-Control-Max-Age", "600");
    }

    void reply(int code, const json& obj, const json& headers = json::object(), std::optional<int> retry_after = std::nullopt) {
        if (record != nullptr) {
            std::lock_guard<std::mutex> lk(svc.status_mu);
            (*record)["http_status"] = code;
            const std::string raw = py_dumps(obj, false, false, 2);
            (*record)["response"] = head_chars(raw, kRecordMax);
            (*record)["response_truncated"] = utf8_len(raw) > kRecordMax;
            for (const char* key : {"error", "usage", "timings"})
                if (obj.is_object() && obj.contains(key)) (*record)[key] = obj[key];
        }
        res.status = code;
        for (auto it = headers.begin(); it != headers.end(); ++it) res.set_header(it.key(), py_str(it.value()));
        if (retry_after) res.set_header("Retry-After", std::to_string(*retry_after));
        cors();
        res.set_content(py_dumps(obj, false), "application/json");
    }

    void raw(int code, std::string body, const char* type, const char* cache = nullptr) {
        res.status = code;
        if (cache != nullptr) res.set_header("Cache-Control", cache);
        res.set_content(std::move(body), type);
    }

    bool authorized() {
        if (!truthy(svc.api_key) || key_matches(given_key(req), svc.api_key)) return true;
        reply(401, {{"error", {{"type", "authentication_error"}, {"message", "missing or wrong API key"}}}});
        return false;
    }

    /// a web page of another origin sent this (its Origin is neither this server's address nor a trusted origin)
    bool foreign_origin() {
        const std::string origin(rstrip_chars(header("Origin"), "/"));
        if (origin.empty()) return false;
        const size_t sep = origin.find("://");
        const std::string rest = sep == std::string::npos ? origin : origin.substr(sep + 3);
        return rest != header("Host") &&
               std::find(svc.trusted_origins.begin(), svc.trusted_origins.end(), origin) == svc.trusted_origins.end();
    }

    /// without an API key, a /v1 POST from a browser page of another site would burn GPU time: an Origin must name an
    /// allowed page, and the body must be JSON then
    bool foreign_page() {
        const std::string origin = header("Origin");
        if (truthy(svc.api_key) || origin.empty()) return false;
        std::vector<std::string> origins = svc.trusted_origins;
        origins.insert(origins.end(), svc.cors_origins.begin(), svc.cors_origins.end());
        if (!origin_allowed(origin, header("Host"), svc.host_names, origins)) {
            say("[lamina] refused an API request from the web page " + py_repr_value(json(origin)) + " (no API key; add its "
                "host to \"allowed_hosts\" or its origin to \"cors_origins\" in the config)");
            reply(403, {{"error", {{"type", "forbidden"},
                                   {"message", "web pages of " + origin + " may not use this server without an API key; set "
                                               "\"api_key\", or add the page's host to \"allowed_hosts\" (or its origin to "
                                               "\"cors_origins\") in the config"}}}});
            return true;
        }
        if (!starts_with(header("Content-Type"), "application/json")) {
            reply(415, {{"error", {{"message", "send application/json"}}}});
            return true;
        }
        return false;
    }

    /// only JSON (no form can send it without a preflight, which is never granted) and no foreign Origin
    bool own_page(const std::string& what) {
        if (!starts_with(header("Content-Type"), "application/json")) {
            reply(415, {{"error", {{"message", "send application/json"}}}});
            return false;
        }
        if (foreign_origin()) {
            reply(403, {{"error", {{"message", what + " only from Lamina's own page (or an origin in the config's "
                                                      "trusted_origins)"}}}});
            return false;
        }
        return true;
    }

    /// a web page of another origin must not have a file on this computer read as an image
    void no_local_images(const json& messages) {
        if (!foreign_origin()) return;
        for (const auto& src : images_of(messages))
            if (!starts_with(src, "data:") && !starts_with(src, "http://") && !starts_with(src, "https://"))
                throw RequestError("a web page of another origin cannot have a file on this computer read as an image: send "
                                   "it as a data: URL (or add the page's origin to the config's trusted_origins)");
    }

    bool control_body() {
        if (req.body.size() <= kControlBodyMax) return true;
        reply(413, {{"error", {{"message", "control request body is limited to 64 KiB"}}}});
        return false;
    }

    /// #430 #431: the request is cancelled as soon as its client hangs up (checked every 0.5 s)
    Cancel& watch_client() {
        cancel = std::make_shared<Cancel>();
        {
            std::lock_guard<std::mutex> lk(svc.status_mu);
            svc.active_cancels.insert(cancel.get());
        }
        watch = std::make_shared<Watch>();
        Watch* w = watch.get();
        Cancel* c = cancel.get();
        w->t = std::thread([this, w, c] {
            while (!w->stop.wait(0.5) && !c->is_set()) {
                if (req.is_connection_closed()) {
                    note({{"outcome", "disconnected"}});
                    svc.note_client_disconnect(*c);
                    return;
                }
            }
        });
        return *cancel;
    }

    /// the end of a POST (Python's finally): what this request held is given back, its record closed
    void finish() {
        if (finished) return;
        finished = true;
        if (reserved) svc.release_inference();
        svc.drop_embeddings();   // the images' file of a request that never got to run()
        watch.reset();
        if (cancel != nullptr) {
            std::lock_guard<std::mutex> lk(svc.status_mu);
            svc.active_cancels.erase(cancel.get());
        }
        if (record != nullptr) {
            std::lock_guard<std::mutex> lk(svc.status_mu);
            json& r = *record;
            r["wallclock_s"] = py_round(mono_s() - r["_clock"].get<double>(), 3);
            r["finished_at"] = now_s();
            r["state"] = truthy(jget(r, "error")) ? json("error") : r.contains("outcome") ? r["outcome"] : json("completed");
        }
        svc.end_request();
    }

    /// what the monitor keeps of the answer, without changing it
    void capture(const std::string& api, const json& item) {
        if (record == nullptr) return;
        std::string content, reasoning;
        json usage, timings;
        if (api == "openai") {
            const json& d = item["choices"][0]["delta"];
            content = jstr(d, "content");
            reasoning = jstr(d, "reasoning_content");
            usage = jget(item, "usage");
            timings = jget(item, "timings");
        } else if (api == "responses") {
            const std::string kind = jstr(item, "type");
            if (kind == "response.output_text.delta") content = jstr(item, "delta");
            if (kind == "response.reasoning_text.delta") reasoning = jstr(item, "delta");
            usage = jget(jget(item, "response"), "usage");
        } else {
            const json& d = jget(item, "delta");
            content = jstr(d, "text");
            reasoning = jstr(d, "thinking");
            usage = jget(item, "usage");
        }
        std::lock_guard<std::mutex> lk(svc.status_mu);
        json& r = *record;
        for (const auto& [name, value] : {std::pair<const char*, const std::string&>{"output", content}, {"reasoning", reasoning}}) {
            if (value.empty()) continue;
            const std::string combined = r[name].get<std::string>() + value;
            r[name] = head_chars(combined, kRecordMax);
            const std::string t = std::string(name) + "_truncated";
            r[t] = r[t].get<bool>() || utf8_len(combined) > kRecordMax;
        }
        if (truthy(usage)) r["usage"] = usage;
        if (truthy(timings)) r["timings"] = timings;
    }

    void sse_headers() {
        note({{"http_status", 200}});
        res.status = 200;
        res.set_header("Cache-Control", "no-cache");
        res.set_header("X-Accel-Buffering", "no");   // nginx and similar proxies pass each event at once
        cors();
    }
};

/// a streamed answer: the provider runs the generation on the connection's thread after the route returned; the
/// Exchange's state outlives the route through this
struct Stream {
    std::shared_ptr<Exchange> x;
    std::function<void(httplib::DataSink&)> body;
};

void stream(std::shared_ptr<Exchange> x, std::function<void(httplib::DataSink&)> body) {
    x->streaming = true;
    x->sse_headers();
    auto s = std::make_shared<Stream>(Stream{x, std::move(body)});
    x->res.set_chunked_content_provider(
        "text/event-stream",
        [s](size_t, httplib::DataSink& sink) {
            try {
                s->body(sink);
            } catch (const std::exception& e) {   // nothing may escape into the HTTP library
                say(std::string("[lamina] the stream ended with an error: ") + e.what());
            }
            sink.done();
            return true;
        },
        [s](bool) { s->x->finish(); });
}

bool write(httplib::DataSink& sink, const std::string& data) { return sink.write(data.data(), data.size()); }

}  // namespace

// ============================================================================================ the server
// the routes, by method (below)
void do_get(Service& svc, HttpServer& server, const httplib::Request& req, httplib::Response& res);
void do_post(Service& svc, HttpServer& server, const httplib::Request& req, httplib::Response& res);
void do_options(Service& svc, HttpServer& server, const httplib::Request& req, httplib::Response& res);

HttpServer::HttpServer(Service& svc) : svc_(svc), http_(std::make_unique<httplib::Server>()) {
    auto& h = *http_;
    h.set_payload_max_length(body_limit());
    h.set_read_timeout(60, 0);
    h.set_write_timeout(60, 0);
    // on Windows SO_REUSEADDR lets a second server bind a port that is already serving (requests then land on either);
    // without it a forgotten second start fails loudly
    h.set_socket_options([](socket_t sock) {
#ifdef _WIN32
        BOOL on = TRUE;
        setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char*) &on, sizeof on);
#else
        int on = 1;
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &on, sizeof on);
#endif
    });
    h.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        // with an API key a POST's body is read only once its key is right (here, before the body: up to 256 MiB of it)
        if (req.method == "POST" && truthy(svc_.api_key) && !key_matches(given_key(req), svc_.api_key)) {
            Exchange x(svc_, *this, req, res);
            x.authorized();   // the 401, as a handler gives it
            res.set_header("Connection", "close");   // the unread body is not drained
            return httplib::Server::HandlerResponse::Handled;
        }
        // without an API key every request first passes the Host check (DNS rebinding); with a key a rebinding page
        // cannot authenticate, so tunnels and proxies that pass their own name keep working
        const std::string host = req.get_header_value("Host");
        const bool any = std::find(svc_.allowed_hosts.begin(), svc_.allowed_hosts.end(), "*") != svc_.allowed_hosts.end();
        if (truthy(svc_.api_key) || host_allowed(host, svc_.host_names, any)) return httplib::Server::HandlerResponse::Unhandled;
        say("[lamina] refused a request for Host " + py_repr_value(json(host)) + " from " + req.remote_addr +
            ": not a name this server answers to (add it to \"allowed_hosts\" in the config or LAMINA_ALLOWED_HOSTS, or "
            "set an API key)");
        res.status = 403;
        res.set_content(py_dumps({{"error", {{"type", "forbidden"},
                                             {"message", "Host " + py_repr_value(json(host)) + " is not allowed (DNS rebinding "
                                                         "protection). Reaching Lamina under this name on purpose? Add it to "
                                                         "\"allowed_hosts\" in the config (lamina-<model>.json) or to the "
                                                         "LAMINA_ALLOWED_HOSTS environment variable, or set an API key "
                                                         "(\"api_key\"), which turns this check off"}}}}, false),
                        "application/json");
        return httplib::Server::HandlerResponse::Handled;
    });
    h.set_error_handler([](const httplib::Request&, httplib::Response& res) {
        if (res.status == 413 && res.body.empty())
            res.set_content(py_dumps({{"error", {{"type", "invalid_request_error"},
                                                 {"message", "the request body is larger than " + std::to_string(body_limit() >> 20) +
                                                             " MiB (LAMINA_MAX_BODY_MIB raises the limit)"}}}}, false),
                            "application/json");
    });
    h.set_exception_handler([](const httplib::Request&, httplib::Response& res, std::exception_ptr ep) {
        std::string what = "unknown error";
        try {
            std::rethrow_exception(ep);
        } catch (const std::exception& e) {
            what = e.what();
        } catch (...) {
        }
        say("[lamina] 500: " + what);
        res.status = 500;
        res.set_content(py_dumps({{"error", {{"type", "server_error"}, {"message", what}}}}, false), "application/json");
    });
    auto route = [this](void (*fn)(Service&, HttpServer&, const httplib::Request&, httplib::Response&)) {
        return [this, fn](const httplib::Request& req, httplib::Response& res) { fn(svc_, *this, req, res); };
    };
    h.Get(".*", route(&do_get));
    h.Post(".*", route(&do_post));
    h.Options(".*", route(&do_options));
}

HttpServer::~HttpServer() { stop(); }

void HttpServer::bind(const std::string& host, int port) {
    if (port == 0) {
        port_ = http_->bind_to_any_port(host);
        if (port_ <= 0) throw std::runtime_error("cannot listen on " + host + ": no free port");
        return;
    }
    if (!http_->bind_to_port(host, port)) {
        const int code = probe_listen(host, port);
        throw std::runtime_error(listen_problem(host, port, code));
    }
    port_ = port;
}

void HttpServer::start() {
    thread_ = std::thread([this] { http_->listen_after_bind(); });
}

void HttpServer::stop() {
    if (http_ != nullptr) http_->stop();
    if (thread_.joinable()) thread_.join();
}

// ============================================================================================ GET
namespace {

void get_props(Exchange& x) {
    Service& svc = x.svc;
    const std::string* m = x.param("model");
    const std::string model = m != nullptr ? *m : svc.model;
    const auto names = svc.model_names();
    if (std::find(names.begin(), names.end(), model) == names.end()) {
        x.reply(404, {{"error", {{"message", "model not found"}}}});
        return;
    }
    if (!svc.loaded() && !svc.engine->unloaded()) {
        x.reply(503, {{"error", {{"message", "the engine is not running"}}}});
        return;
    }
    json defaults = model_card_sampling(svc.model_arch, true);
    for (auto it = svc.sampling_defaults.begin(); it != svc.sampling_defaults.end(); ++it) defaults[it.key()] = it.value();
    const json shared = svc.shared();
    for (auto it = shared.begin(); it != shared.end(); ++it) defaults[it.key()] = it.value();
    json params = json::object();
    for (auto it = defaults.begin(); it != defaults.end(); ++it) {
        static const std::set<std::string> keep = {"temperature", "top_p", "top_k", "min_p", "seed", "repetition_penalty",
                                                   "presence_penalty", "frequency_penalty", "penalty_last_n"};
        if (!keep.count(it.key())) continue;
        const std::string k = it.key() == "repetition_penalty" ? "repeat_penalty" : it.key() == "penalty_last_n" ? "repeat_last_n" : it.key();
        params[k] = it.value();
    }
    params["n_predict"] = shared.contains("max_tokens") ? shared["max_tokens"] : json(-1);
    json props = {{"default_generation_settings", {{"n_ctx", svc.reported_ctx()}, {"params", params}}},
                  {"total_slots", std::max(1, svc.engine->batch())},   // #1004: the batch slots
                  {"model_alias", svc.model}, {"chat_template", svc.tpl->source()}, {"chat_template_caps", svc.tpl->caps()},
                  {"modalities", {{"vision", svc.vision != nullptr}}}, {"models_autoload", svc.engine->auto_load()},
                  {"is_sleeping", !svc.loaded()}};
    if (!svc.engine->model_path.empty()) props["model_path"] = svc.engine->model_path;
    const json& version = jget(svc.engine->info, "version");
    if (truthy(version)) props["build_info"] = "Lamina " + py_str(version);
    x.reply(200, props);
}

void get_models(Exchange& x) {
    Service& svc = x.svc;
    bool loaded = svc.loaded();
    json model = {{"id", svc.model}, {"object", "model"}, {"status", {{"value", "loaded"}}},
                  {"meta", {{"n_ctx", svc.reported_ctx()}}},
                  {"architecture", {{"input_modalities", svc.vision != nullptr ? json::array({"text", "image"}) : json::array({"text"})},
                                    {"output_modalities", json::array({"text"})}}}};
    if (!loaded && (svc.idle_unload_s || svc.engine->unloaded())) {
        model["status"] = {{"value", "unloaded"}};   // like llama-server's router: listed, loads on use
        loaded = true;
    }
    if (!svc.aliases.empty()) model["aliases"] = svc.aliases;   // #297: each alias listed under its own id too
    json data = json::array({model});
    for (const auto& a : svc.aliases) {
        json m = model;
        m["id"] = a;
        m["alias_of"] = svc.model;
        data.push_back(m);
    }
    if (!loaded) data = json::array();
    if (svc.images != nullptr)   // the image model: POST /v1/images/generations
        data.push_back({{"id", svc.images->model}, {"object", "model"}, {"status", {{"value", "loaded"}}},
                        {"architecture", {{"input_modalities", json::array({"text"})}, {"output_modalities", json::array({"image"})}}}});
    x.reply(200, {{"object", "list"}, {"data", data}});
}

void get_slots(Exchange& x) {
    Service& svc = x.svc;
    const bool loaded = svc.engine->alive();
    bool busy;
    int64_t in_use;
    {
        std::lock_guard<std::mutex> lk(svc.status_mu);
        busy = truthy(jget(svc.status, "busy"));
        // llama.cpp's name for what the slot's context holds (#562: a context meter divides it by n_ctx)
        in_use = jget(svc.status, "prompt_tokens").is_number() ? svc.status["prompt_tokens"].get<int64_t>() : 0;
    }
    json slots = json::array();
    const json view = svc.engine->batch() ? svc.engine->slots_view() : json(nullptr);
    if (view.is_array()) {
        const int64_t n_ctx = svc.reported_ctx();
        for (const auto& s : view) {
            const json& pt = truthy(jget(s, "prompt_tokens")) ? s["prompt_tokens"] : jget(s, "held_tokens");
            slots.push_back({{"id", s["slot"]}, {"n_ctx", n_ctx}, {"is_processing", jstr(s, "state") != "idle"},
                             {"n_prompt_tokens", pt.is_number() ? pt.get<int64_t>() : 0}});
        }
    } else {
        slots.push_back({{"id", 0}, {"n_ctx", svc.reported_ctx()}, {"is_processing", busy}, {"n_prompt_tokens", in_use}});
    }
    x.reply(200, loaded ? slots : json::array());
}

void get_status(Exchange& x) {
    Service& svc = x.svc;
    if (!x.authorized()) return;   // #212: it shows the end of the last answer
    json s;
    {
        std::lock_guard<std::mutex> lk(svc.status_mu);
        s = svc.status;
    }
    if (truthy(jget(s, "busy")) && jget(s, "started").is_number()) {
        s["elapsed_s"] = py_round(now_s() - s["started"].get<double>(), 1);
        if (truthy(jget(s, "first_token"))) {
            s["tokens_per_s"] = py_round(svc.tok_s(), 1);
            s["tokens_per_s_mean"] = py_round(svc.tok_s_mean(), 1);
        }
    }
    s.erase("started");
    s.erase("first_token");
    x.reply(200, s);
}

void serve_asset(Exchange& x, const std::string& name, const char* type, const char* cache) {
    const WebAsset* a = web_asset(name);
    if (a == nullptr) {
        x.reply(404, {{"error", {{"message", "not found"}}}});
        return;
    }
    x.raw(200, std::string(a->data), type, cache);
}

}  // namespace

void do_get(Service& svc, HttpServer& server, const httplib::Request& req, httplib::Response& res) {
    Exchange x(svc, server, req, res);
    const std::string& path = x.path;
    if (starts_with(path, "/fonts/")) {   // the web app's font (Outfit, OFL); the page falls back to the system font
        const std::string name = path.substr(7);
        if (name.find_first_of("/\\") != std::string::npos || !ends_with(name, ".woff2")) {
            x.reply(404, {{"error", {{"message", "not found"}}}});
            return;
        }
        serve_asset(x, "fonts/" + name, "font/woff2", "max-age=86400");
        return;
    }
    if (starts_with(path, "/web/")) {   // the web app's own files: same origin, no CDN
        const std::string name = path.substr(5);
        const std::string ext = name.find('.') == std::string::npos ? "" : name.substr(name.rfind('.'));
        const char* type = ext == ".css" ? "text/css; charset=utf-8" : ext == ".js" ? "text/javascript; charset=utf-8"
                           : ext == ".svg" ? "image/svg+xml" : nullptr;
        if (name.find_first_of("/\\") != std::string::npos || type == nullptr) {
            x.reply(404, {{"error", {{"message", "not found"}}}});
            return;
        }
        serve_asset(x, name, type, "no-cache");
        return;
    }
    if (path == "/metrics/prometheus") {
        if (x.authorized()) x.raw(200, svc.prometheus(), "text/plain; version=0.0.4; charset=utf-8");
        return;
    }
    if (path == "/metrics") {
        std::string query;
        for (const auto& [k, v] : req.params) query += (query.empty() ? "" : "&") + k + "=" + v;
        if (prometheus::wants_prometheus(x.header("Accept"), query)) {
            if (x.authorized())   // a scraper: the text format, vLLM's names
                x.raw(200, prometheus::render(svc.metrics(), server.latencies->snapshot()), prometheus::kContentType);
            return;
        }
        // the last 12 requests; ?requests=all every one kept (the Monitor's "Show all", issue #35)
        const std::string* all = x.param("requests");
        if (x.authorized()) x.reply(200, svc.metrics(all != nullptr && *all == "all"));
        return;
    }
    if (path == "/api/requests" && svc.api_monitor) {
        if (!x.authorized()) return;
        const std::string* id = x.param("id");
        const json records = svc.request_records(id != nullptr ? *id : "");
        if (records.is_null()) x.reply(404, {{"error", {{"message", "request no longer retained"}}}});
        else if (id != nullptr) x.reply(200, records);
        else x.reply(200, {{"requests", records}, {"retention", 100}, {"persistent", false}, {"loaded", svc.loaded()},
                           {"auto_load", svc.engine->auto_load()}});
        return;
    }
    if (path == "/settings") {
        if (x.authorized()) {
            const json shared = svc.shared();
            x.reply(200, {{"shared", truthy(shared)}, {"defaults", shared}});
        }
        return;
    }
    if (path == "/config") {   // #564: the Settings view's keys of the run config
        if (!x.authorized()) return;
        if (svc.config_path.empty()) {
            x.reply(404, {{"error", {{"message", "this server was started without a run config"}}}});
            return;
        }
        json cfg;
        try {
            cfg = runconfig::load(svc.config_path);
        } catch (const std::exception& e) {
            x.reply(500, {{"error", {{"type", "server_error"}, {"message", std::string("the run config cannot be read: ") + e.what()}}}});
            return;
        }
        x.reply(200, runconfig::view(cfg, svc.config_path));
        return;
    }
    if (path == "/mcp") {   // the MCP servers, their state and tools (the web app's switch and Monitor card)
        if (x.authorized()) x.reply(200, svc.mcp != nullptr ? svc.mcp->status() : json{{"servers", json::array()}, {"tools", 0}});
        return;
    }
    if (path.empty() || (path == "/api-monitor" && svc.api_monitor)) {
        // never in another site's frame: the pages act with this server's rights (its MCP tools)
        res.set_header("Content-Security-Policy", "frame-ancestors 'none'");
        res.set_header("X-Frame-Options", "DENY");
        serve_asset(x, path.empty() ? "index.html" : "monitor.html", "text/html; charset=utf-8", nullptr);
        return;
    }
    if (path == "/health" || path == "/api/health") {
        x.reply(200, {{"status", "ok"}, {"max_context", svc.reported_ctx()}, {"model", svc.model}, {"images", svc.vision != nullptr},
                      {"api_key", truthy(svc.api_key)}, {"image_generation", svc.images != nullptr}, {"loaded", svc.loaded()},
                      {"service", "lamina"}});
    } else if (starts_with(path, "/v1/images/files/")) {
        // response_format url: the name (32 random hex digits) is the access, as an OpenAI image URL is
        const auto f = svc.images != nullptr ? svc.images->file(path.substr(17)) : std::nullopt;
        std::string body;
        if (f) {
            std::ifstream in(*f, std::ios::binary);
            body.assign(std::istreambuf_iterator<char>(in), {});
        }
        if (!f || body.empty()) {
            x.reply(404, {{"error", {{"message", "not found (the server keeps the newest images only)"}}}});
            return;
        }
        x.raw(200, std::move(body), "image/png", "private, max-age=86400");
    } else if (path == "/status") {
        get_status(x);
    } else if (path == "/v1/models" || path == "/models") {
        if (x.authorized()) get_models(x);
    } else if (path == "/props") {
        if (x.authorized()) get_props(x);
    } else if (path == "/slots") {
        if (x.authorized()) get_slots(x);
    } else if (path == "/v1/status") {
        if (x.authorized()) x.reply(200, svc.v1_status());
    } else {
        x.reply(404, {{"error", {{"message", "not found"}}}});
    }
}

void do_options(Service& svc, HttpServer& server, const httplib::Request& req, httplib::Response& res) {
    // a CORS preflight: no credentials come with it, so no API key; the headers only for cors_origins
    Exchange x(svc, server, req, res);
    res.status = 204;
    x.cors();
}

// ============================================================================================ POST
namespace {

/// the image job's turn on the GPU (serve/images.py's gpu_turn): the service's turn (with batch slots, once none is
/// generating), and the text engine given room per the config's policy; -> the policy used
std::string image_turn(Service& svc, images::ImageGen& gen, const images::Job& job, Cancel* cancel,
                       const std::function<void()>& work) {
    {
        std::lock_guard<std::mutex> lk(svc.status_mu);
        svc.status["queued"] = svc.status["queued"].get<int64_t>() + 1;
    }
    bool queued = true;
    auto unqueue = [&] {
        if (!queued) return;
        queued = false;
        std::lock_guard<std::mutex> lk(svc.status_mu);
        svc.status["queued"] = svc.status["queued"].get<int64_t>() - 1;
    };
    std::string policy;
    try {
        const auto turn = svc.turn(cancel);
        unqueue();
        while (svc.slots_busy()) {   // running parallel requests finish first
            if (cancel != nullptr && cancel->is_set()) throw ClientGone();
            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }
        const int64_t need = images::need_mib(gen.cfg, job.width, job.height);
        // auto -> keep or unload: the text engine stays unless the job's VRAM is not free beside it
        if (!svc.engine->alive() || !svc.engine->auto_load()) {
            policy = "keep";
        } else if (gen.policy != "auto") {
            policy = gen.policy;
        } else if (truthy(jget(gen.cfg, "uma"))) {
            policy = "keep";   // an APU: the job and the engine share the RAM either way
        } else {
            const auto free = svc.free_vram_mib();
            policy = !free || *free >= need ? "keep" : "unload";
        }
        {
            std::lock_guard<std::mutex> lk(svc.status_mu);
            for (const auto& [k, v] : std::vector<std::pair<const char*, json>>{{"busy", true}, {"phase", "generating an image"},
                                                                                  {"started", now_s()}, {"first_token", nullptr},
                                                                                  {"prompt_tokens", nullptr}, {"generated", nullptr},
                                                                                  {"max_tokens", nullptr}})
                svc.status[k] = v;
        }
        const bool unloaded = policy == "unload";
        if (unloaded) {
            svc.engine->unload();
            if (svc.vision != nullptr) svc.vision->unload();
        }
        auto give_back = [&] {
            if (unloaded && gen.reload == "eager") {
                try {
                    svc.ensure_loaded();   // the turn is still held
                } catch (const std::exception& e) {
                    gen.say(std::string("image job: the model did not load again yet (") + e.what() + "); the next request loads it");
                }
            }
            std::lock_guard<std::mutex> lk(svc.status_mu);
            svc.status["busy"] = false;
            svc.last_request_at = now_s();
        };
        try {
            work();
        } catch (...) {
            give_back();
            throw;
        }
        give_back();
    } catch (...) {
        unqueue();
        throw;
    }
    return policy;
}

void post_images(Exchange& x, const json& req) {
    Service& svc = x.svc;
    if (svc.images == nullptr) {
        x.reply(404, {{"error", {{"type", "invalid_request_error"}, {"code", "images_not_set_up"},
                                 {"message", "image generation is not set up on this server (run setup with --images qwen-image)"}}}});
        return;
    }
    images::ImageGen& gen = *svc.images;
    const images::Job job = images::parse_request(req, gen.cfg);   // ImageRequestError: a 400
    {
        std::lock_guard<std::mutex> lk(svc.status_mu);   // (the idle unload reads it under the lock)
        svc.last_request_at = now_s();
    }
    Cancel& cancel = x.watch_client();
    say("[lamina] image: " + std::to_string(job.width) + "x" + std::to_string(job.height) + ", n=" + std::to_string(job.n) +
        ", " + std::to_string(job.steps) + " steps, seed " + std::to_string(job.seed));
    std::vector<std::string> pictures;
    json figures;
    std::string policy;
    try {
        policy = image_turn(svc, gen, job, &cancel, [&] { std::tie(pictures, figures) = gen.run(job, &cancel); });
    } catch (const images::ImageJobFailed& e) {
        x.reply(500, {{"error", {{"type", "server_error"}, {"code", "image_generation_failed"}, {"message", e.what()}}}});
        return;
    }
    figures["gpu_policy"] = policy;
    char buf[64];
    std::snprintf(buf, sizeof buf, "%.1f", figures["seconds"].get<double>());
    say("[lamina] image done in " + std::string(buf) + " s (" + policy + ")");
    std::string host = x.header("Host");
    if (host.empty()) host = "127.0.0.1:" + std::to_string(x.server.port());
    const std::string scheme = lower(x.header("X-Forwarded-Proto")) == "https" ? "https" : "http";
    x.reply(200, gen.response(pictures, job, figures, scheme + "://" + host));
}

void post_settings(Exchange& x) {
    // they change what every client gets, so only the app's own page may set them
    if (!x.own_page("settings can be changed")) return;
    json req;
    std::string err;
    if (!py_loads(x.req.body.empty() ? "{}" : x.req.body, req, &err)) {
        x.reply(400, {{"error", {{"type", "invalid_request_error"}, {"message", err}}}});
        return;
    }
    if (!req.is_object() || !req.contains("defaults")) {
        x.reply(400, {{"error", {{"type", "invalid_request_error"},
                                 {"message", "the body must be {\"defaults\": {...}} to set them, or {\"defaults\": null} to clear them"}}}});
        return;
    }
    json shared;
    try {
        shared = x.svc.set_shared(req["defaults"]);
    } catch (const std::invalid_argument& e) {
        x.reply(400, {{"error", {{"type", "invalid_request_error"}, {"message", e.what()}}}});
        return;
    }
    if (truthy(shared)) {
        std::string list;
        for (auto it = shared.begin(); it != shared.end(); ++it) list += (list.empty() ? "" : ", ") + it.key() + "=" + py_str(it.value());
        say("[lamina] other apps now use the Chat settings: " + list);
    } else {
        say("[lamina] other apps use their own settings again");
    }
    x.reply(200, {{"shared", truthy(shared)}, {"defaults", shared}});
}

void post_config(Exchange& x) {
    // #564: a few documented keys of the run config, from Lamina's own page only; every other key stays as it is
    if (!x.own_page("the run config can be changed")) return;
    Service& svc = x.svc;
    if (svc.config_path.empty()) {
        x.reply(404, {{"error", {{"message", "this server was started without a run config"}}}});
        return;
    }
    json view;
    std::vector<std::string> changed;
    std::string bak;
    try {
        json req;
        std::string err;
        if (!py_loads(x.req.body.empty() ? "{}" : x.req.body, req, &err)) throw RequestError(err);
        std::lock_guard<std::mutex> lk(svc.config_mu);
        const json cfg = runconfig::load(svc.config_path);
        json updated;
        std::tie(updated, changed) = runconfig::apply(cfg, req.is_object() ? jget(req, "set") : json(nullptr));
        if (!changed.empty()) bak = runconfig::save(svc.config_path, updated);
        view = runconfig::view(updated, svc.config_path);
    } catch (const std::invalid_argument& e) {
        x.reply(400, {{"error", {{"type", "invalid_request_error"}, {"message", e.what()}}}});
        return;
    } catch (const std::exception& e) {
        x.reply(500, {{"error", {{"type", "server_error"}, {"message", std::string("the run config cannot be written: ") + e.what()}}}});
        return;
    }
    if (!changed.empty()) {
        std::string list;
        for (const auto& c : changed) list += (list.empty() ? "" : ", ") + c;
        say("[lamina] the Settings view changed " + list + " in " + u8_str(u8_path(svc.config_path).filename()) +
            " (the earlier file: " + u8_str(u8_path(bak).filename()) + "); used from the next start");
    }
    view["changed"] = changed;
    x.reply(200, view);
}

void post_openai(std::shared_ptr<Exchange> xp, json req) {
    Exchange& x = *xp;
    Service& svc = x.svc;
    req = svc.with_shared(req, "openai");
    normalize_bias(jget(req, "logit_bias"), svc.tok.n_vocab());
    Converted conv = openai_to_messages(req);
    json messages = std::move(conv.messages), tools = std::move(conv.tools);
    const json kw = std::move(conv.kwargs);
    x.no_local_images(messages);
    if (tool_choice_of(jget(req, "tool_choice")).first == "none") tools = nullptr;   // as the Responses route: none offered
    const std::optional<std::string> force = forced_call(jget(req, "tool_choice"), tools);   // a bad value: a 400 now
    auto [formatted, validator] = prepare_format(jget(req, "response_format"), messages, false, xp->req.body,
                                                 {"response_format", "json_schema", "schema"});
    messages = std::move(formatted);
    if (validator != nullptr && (truthy(tools) || truthy(jget(req, "lamina_mcp"))))
        throw RequestError("structured response_format with tools/MCP is not supported");
    svc.load();
    const int64_t max_req = py_int(first_truthy({&jget(req, "max_completion_tokens"), &jget(req, "max_tokens")}));   // 0/-1: the rest
    int64_t max_new = max_req;
    validate_bias(req, *svc.engine, svc.tok.n_vocab());
    bool use_mcp = jget(req, "lamina_mcp") == true && svc.mcp != nullptr;   // the web app's opt-in
    std::set<std::string> own;   // #592: a second line of defence
    if (tools.is_array())
        for (const auto& t : tools)
            if (t.is_object() && jget(t, "name").is_string()) own.insert(t["name"].get<std::string>());
    json extra = json::array();
    if (use_mcp) {
        if (!x.own_page("MCP tools can be used")) return;   // tools run with the user's rights on this PC
        svc.mcp->wait(10);                                   // servers still starting (only right after start)
        extra = svc.mcp->template_tools(own);                // the request's own tools win a name clash
        use_mcp = !extra.empty();
        json all = tools.is_array() ? tools : json::array();
        for (const auto& t : extra) all.push_back(t);
        tools = all.empty() ? json(nullptr) : all;
    }
    if (force && use_mcp) throw RequestError("a forced tool_choice with MCP tools is not supported");
    svc.reasoning_budget(req);       // a bad value is a 400 before anything is sent
    stop_strings(req);               // ... and so is a bad stop / stop_sequences
    check_request_sampling(req);     // ... and a sampling field of the wrong type
    Cancel& cancel = x.watch_client();   // #430 #431 (#1999: before prepare, which can take long)
    Prepared p = svc.prepare(messages, tools, kw, max_new, force, &req, &cancel);
    debug_request("openai", req, messages, tools, p.max_new, p.thinking, p.ids.size());
    std::set<std::string> mcp_names;
    for (const auto& t : extra) mcp_names.insert(jstr(t, "name"));
    auto runner = std::make_shared<Runner>();
    if (use_mcp) {
        auto hub = svc.mcp;
        *runner = [&svc, hub, messages, tools, kw, p, max_req, req, cancel = x.cancel, mcp_names](const Sink& sink) {
            run_with_mcp(svc, hub, messages, tools, kw, p.ids, p.thinking, p.max_new, max_req, req, cancel, mcp_names, sink);
        };
    }
    std::shared_ptr<const Validator> keep_validator = validator;
    auto produce = [xp, req, p, tools, force, runner, use_mcp, keep_validator](const ChunkSink& out) {
        Exchange& x = *xp;
        auto captured = [&](const json* c) {
            if (c != nullptr) x.capture("openai", *c);
            return out(c);
        };
        if (keep_validator != nullptr) {
            StructuredBuffer buf(keep_validator.get(), captured);
            const ChunkSink in = buf.sink();
            bool gone = false;
            openai_chunks(x.svc, req, p.ids, p.thinking, tools, p.max_new, *x.cancel, nullptr, force,
                          [&](const json* c) { return in(c) || !(gone = true); });
            if (!gone) buf.finish();
        } else {
            openai_chunks(x.svc, req, p.ids, p.thinking, tools, p.max_new, *x.cancel, use_mcp ? runner.get() : nullptr, force, captured);
        }
    };
    if (!truthy(jget(req, "stream"))) {
        std::vector<json> chunks;
        produce([&](const json* c) {
            if (c != nullptr) chunks.push_back(*c);
            return true;
        });
        x.reply(200, openai_collect(chunks));
        return;
    }
    stream(xp, [xp, produce](httplib::DataSink& sink) {
        Exchange& x = *xp;
        bool gone = false;
        auto fail = [&](const json& err) {
            x.note({{"error", err["error"]}});
            write(sink, "data: " + py_dumps(err) + "\n\ndata: [DONE]\n\n");
        };
        try {
            produce([&](const json* c) {
                const bool ok = write(sink, c == nullptr ? std::string(": keep-alive\n\n")   // an SSE comment
                                                         : "data: " + py_dumps(*c, false) + "\n\n");
                if (!ok) gone = true;
                return ok;
            });
            if (gone) {
                x.note({{"outcome", "disconnected"}});
                x.svc.note_client_disconnect(*x.cancel);   // the client went away: the engine stops
                return;
            }
            write(sink, "data: [DONE]\n\n");
        } catch (const EngineDied& e) {   // mid-stream: said, then the stream ends properly
            fail({{"error", {{"type", "server_error"}, {"message", std::string(e.what()) + "; the next request restarts it"}}}});
        } catch (const QueueWaitExpired& e) {
            fail({{"error", {{"type", "rate_limit_error"}, {"code", "queue_wait_timeout"}, {"message", e.what()}}}});
        } catch (const StructuredOutputError& e) {
            fail({{"error", {{"type", "structured_output_failed"}, {"code", "structured_output_failed"}, {"message", e.what()}}}});
        } catch (const std::invalid_argument& e) {   // the engine's ERR after the stream started: no 400 now
            fail({{"error", {{"type", "server_error"}, {"message", e.what()}}}});
        }
    });
}

void post_anthropic(std::shared_ptr<Exchange> xp, json req) {
    Exchange& x = *xp;
    Service& svc = x.svc;
    svc.load();
    req = svc.with_shared(req, "anthropic");
    Converted conv = anthropic_to_messages(req, svc.anthropic_think_unasked);
    json messages = std::move(conv.messages), tools = std::move(conv.tools);
    const json kw = std::move(conv.kwargs);
    x.no_local_images(messages);
    if (tool_choice_of(jget(req, "tool_choice")).first == "none") tools = nullptr;   // {"type": "none"}: none offered
    const std::optional<std::string> force = forced_call(jget(req, "tool_choice"), tools);   // "any" / {"type": "tool"}
    const int64_t max_new = py_int(first_truthy({&jget(req, "max_tokens")}));   // 0/-1: the rest of the context
    svc.reasoning_budget(req);
    stop_strings(req);
    check_request_sampling(req);
    Cancel& cancel = x.watch_client();
    Prepared p = svc.prepare(messages, tools, kw, max_new, force, &req, &cancel);
    debug_request("anthropic", req, messages, tools, p.max_new, p.thinking, p.ids.size());
    auto produce = [xp, req, p, tools, force](const EventSink& out) {
        Exchange& x = *xp;
        anthropic_events(x.svc, req, p.ids, p.thinking, tools, p.max_new, *x.cancel, force,
                         [&](const std::string* name, const json* e) {
                             if (e != nullptr) x.capture("anthropic", *e);
                             return out(name, e);
                         });
    };
    if (!truthy(jget(req, "stream"))) {
        std::vector<std::pair<std::string, json>> events;
        produce([&](const std::string* name, const json* e) {
            if (name != nullptr) events.emplace_back(*name, *e);
            return true;
        });
        x.reply(200, anthropic_collect(events));
        return;
    }
    stream(xp, [xp, produce](httplib::DataSink& sink) {
        Exchange& x = *xp;
        bool gone = false;
        auto fail = [&](const json& err) {
            x.note({{"error", err["error"]}});
            write(sink, "event: error\ndata: " + py_dumps(err) + "\n\n");
        };
        try {
            produce([&](const std::string* name, const json* e) {
                const bool ok = write(sink, name == nullptr ? std::string(": keep-alive\n\n")
                                                            : "event: " + *name + "\ndata: " + py_dumps(*e, false) + "\n\n");
                if (!ok) gone = true;
                return ok;
            });
            if (gone) {
                x.note({{"outcome", "disconnected"}});
                x.svc.note_client_disconnect(*x.cancel);
            }
        } catch (const EngineDied& e) {   // mid-stream: Anthropic's error event
            fail({{"type", "error"}, {"error", {{"type", "api_error"}, {"message", std::string(e.what()) + "; the next request restarts it"}}}});
        } catch (const QueueWaitExpired& e) {
            fail({{"type", "error"}, {"error", {{"type", "rate_limit_error"}, {"code", "queue_wait_timeout"}, {"message", e.what()}}}});
        } catch (const std::invalid_argument& e) {   // the engine's ERR after the stream started
            fail({{"type", "error"}, {"error", {{"type", "api_error"}, {"message", e.what()}}}});
        }
    });
}

void post_count_tokens(Exchange& x, json req) {
    // Anthropic's token count (Claude Code's context figures): the prompt this server would read, rendered and tokenized
    Service& svc = x.svc;
    req = svc.with_shared(req, "anthropic");
    Converted conv = anthropic_to_messages(req, svc.anthropic_think_unasked);
    stop_strings(req);
    check_request_sampling(req);
    x.reply(200, {{"input_tokens", svc.encode_prompt(conv.messages, conv.tools, conv.kwargs, false).size()}});
}

/// well-formed UTF-8: no overlong forms, surrogates or code points past U+10FFFF
bool valid_utf8(const std::string& s) {
    for (size_t i = 0; i < s.size();) {
        const unsigned char c = (unsigned char) s[i];
        const size_t n = c < 0x80 ? 0 : c >= 0xC2 && c <= 0xDF ? 1 : c >= 0xE0 && c <= 0xEF ? 2 : c >= 0xF0 && c <= 0xF4 ? 3 : 9;
        if (n == 9 || i + n >= s.size() + (n == 0)) return false;
        for (size_t k = 1; k <= n; ++k)
            if (((unsigned char) s[i + k] & 0xC0) != 0x80) return false;
        const unsigned char d = n > 0 ? (unsigned char) s[i + 1] : 0;
        if ((c == 0xE0 && d < 0xA0) || (c == 0xED && d > 0x9F) || (c == 0xF0 && d < 0x90) || (c == 0xF4 && d > 0x8F))
            return false;
        i += n + 1;
    }
    return true;
}

void post_tokenize(Exchange& x, const json& req) {
    // llama-server's: a text's ids, the control tokens' texts read as the tokens unless "parse_special" is false (this
    // vocabulary adds no BOS: "add_special" changes nothing); "with_pieces": each id with its text, or its bytes when
    // they are no whole UTF-8 characters
    const json& content = jget(req, "content");
    if (!content.is_string()) throw RequestError("content must be a string");
    const json& special = jget(req, "parse_special");
    const std::vector<int32_t> ids = x.svc.tok.encode(content.get<std::string>(), !special.is_boolean() || special.get<bool>());
    json tokens = json::array();
    for (const int32_t id : ids) {
        if (!truthy(jget(req, "with_pieces"))) {
            tokens.push_back(id);
            continue;
        }
        const std::string& piece = x.svc.tok.token_bytes(id);
        json bytes = json::array();
        for (const char c : piece) bytes.push_back((int) (unsigned char) c);
        tokens.push_back({{"id", id}, {"piece", valid_utf8(piece) ? json(piece) : bytes}});
    }
    x.reply(200, {{"tokens", tokens}});
}

void post_detokenize(Exchange& x, const json& req) {
    // llama-server's: the text of a list of ids
    const json& tokens = jget(req, "tokens");
    if (!tokens.is_array()) throw RequestError("tokens must be a list of token ids");
    std::vector<int32_t> ids;
    for (const json& t : tokens) {
        if (!t.is_number_integer() || t.get<int64_t>() < 0 || t.get<int64_t>() >= x.svc.tok.n_vocab())
            throw RequestError("tokens must be a list of token ids below " + std::to_string(x.svc.tok.n_vocab()));
        ids.push_back((int32_t) t.get<int64_t>());
    }
    x.reply(200, {{"content", x.svc.tok.decode(ids)}});
}

void post_apply_template(Exchange& x, json req) {
    // llama-server's: the prompt a chat request is read as (its messages, tools, response format and template
    // variables, as POST /v1/chat/completions takes them)
    Service& svc = x.svc;
    req = svc.with_shared(req, "openai");
    Converted conv = openai_to_messages(req);
    if (tool_choice_of(jget(req, "tool_choice")).first == "none") conv.tools = nullptr;
    auto formatted = prepare_format(jget(req, "response_format"), conv.messages, false, x.req.body,
                                    {"response_format", "json_schema", "schema"}).first;
    x.reply(200, {{"prompt", svc.render_prompt(formatted, conv.tools, conv.kwargs)}});
}

void post_choose(Exchange& x, const json& req) {
    x.no_local_images(openai_to_messages(req).messages);
    Cancel& cancel = x.watch_client();
    x.reply(200, x.svc.choose(req, cancel));
}

// ---- /v1/responses (#451): OpenAI's Responses API, stateless, on the chat path
void responses_ready(std::shared_ptr<Exchange> xp, const json& req, std::vector<json> events) {
    if (!truthy(jget(req, "stream"))) {
        xp->reply(200, responses::collect(events));
        return;
    }
    stream(xp, [xp, events](httplib::DataSink& sink) {
        for (const auto& e : events)
            if (!write(sink, "event: " + jstr(e, "type") + "\ndata: " + py_dumps(e, false) + "\n\n")) {
                xp->note({{"outcome", "disconnected"}});
                return;
            }
    });
}

void post_responses(std::shared_ptr<Exchange> xp, json req) {
    Exchange& x = *xp;
    Service& svc = x.svc;
    std::optional<std::vector<json>> title;
    try {
        if (svc.codex_thread_titles) title = responses::thread_title_events(req, svc.model_for(req));
    } catch (const responses::ResponsesError& e) {
        x.reply(e.status, e.body());
        return;
    }
    if (title) {
        responses_ready(xp, req, *title);
        return;
    }
    Cancel& cancel = x.watch_client();   // #430 #431 (#1999: before prepare, which can take long)
    Prepared p;
    json tools;
    std::shared_ptr<responses::Assembler> asm_;
    std::shared_ptr<const Validator> validator;
    try {
        responses::check_request(req);
        json messages = responses::input_messages(req);
        responses::RequestTools rt = responses::request_tools(req);
        tools = rt.tools;
        const json kw = responses::template_kwargs(req, {{"reasoning_effort", svc.effort_default()}});
        try {
            x.no_local_images(messages);   // as on the chat route: no file read for a foreign page
        } catch (const std::invalid_argument& e) {
            throw responses::ResponsesError(e.what(), "input");
        }
        try {
            auto [m, v] = prepare_format(responses::text_format(req), messages, truthy(tools), xp->req.body,
                                         {"text", "format", "schema"});
            messages = std::move(m);
            validator = v;
        } catch (const std::invalid_argument& e) {
            throw responses::ResponsesError(replace_all(e.what(), "response_format", "text.format"), "text.format");
        }
        std::vector<std::string> fresh;
        {
            std::lock_guard<std::mutex> lk(svc.status_mu);
            for (const auto& s : rt.skipped)
                if (svc.hosted_tools_noted.insert(s).second) fresh.push_back(s);
        }
        if (!fresh.empty()) {
            std::sort(fresh.begin(), fresh.end());
            std::string list;
            for (const auto& s : fresh) list += (list.empty() ? "" : ", ") + s;
            say("[lamina] /v1/responses: left out the hosted tools " + list + " (this server cannot run them)");
        }
        if (const json shared = svc.shared(); jget(req, "max_output_tokens").is_null() && shared.contains("max_tokens"))
            req["max_output_tokens"] = shared["max_tokens"];   // the shared Chat settings
        try {
            svc.reasoning_budget(req);
        } catch (const std::invalid_argument& e) {
            throw responses::ResponsesError(e.what(), "reasoning_budget_tokens");
        }
        try {
            stop_strings(req);
        } catch (const std::invalid_argument& e) {
            throw responses::ResponsesError(e.what(), "stop");
        }
        try {
            check_request_sampling(req);   // the request is also the sampling dict
        } catch (const std::invalid_argument& e) {
            const std::string m = e.what();
            throw responses::ResponsesError(m, m.substr(0, m.find(':')));
        }
        svc.load();
        // #924 (opt-in): a Codex compaction request is rendered with its conversation's tools, so its prompt starts as
        // the cached one did; the parser and the response keep the request's own tools
        json shown = svc.codex_compaction_cache ? responses::prompt_tools(req, tools) : tools;
        const json& mo = jget(req, "max_output_tokens");
        const int64_t max_out = truthy(mo) ? py_int(mo) : 0;
        try {
            try {
                p = svc.prepare(messages, shown, kw, max_out, std::nullopt, &req, &cancel);
            } catch (const RequestError&) {
                if (shown == tools) throw;
                shown = tools;   // too long with the kept tools: the request as sent
                p = svc.prepare(messages, shown, kw, max_out, std::nullopt, &req, &cancel);
            }
        } catch (const responses::ResponsesError&) {
            throw;
        } catch (const std::invalid_argument& e) {   // too long for the context, an image without vision
            const std::string m = e.what();
            throw responses::ResponsesError(m, "input", m.find("context") != std::string::npos ? "context_length_exceeded" : "");
        }
        if (svc.codex_compaction_cache) responses::prompt_made(req, shown);
        debug_request("responses", req, messages, tools, p.max_new, p.thinking, p.ids.size());
        const json& include = jget(req, "include");
        bool encrypted = false;
        if (include.is_array())
            for (const auto& i : include) encrypted = encrypted || i == "reasoning.encrypted_content";
        asm_ = std::make_shared<responses::Assembler>(req, svc.model_for(req), (int64_t) p.ids.size(), rt.names, encrypted,
                                                      validator != nullptr);
    } catch (const ClientGone&) {
        x.note({{"outcome", "disconnected"}});
        return;
    } catch (const responses::ResponsesError& e) {
        x.reply(e.status, e.body());
        return;
    } catch (const ModelBusy& e) {
        x.reply(409, responses::error_body(e.what(), "server_error", nullptr, "model_busy"));
        return;
    } catch (const GpuBusy& e) {
        x.reply(503, responses::error_body(e.what(), "server_error", nullptr, "server_error"));
        return;
    } catch (const EngineStarting& e) {
        x.reply(503, responses::error_body(e.what(), "server_error", nullptr, "server_error"));
        return;
    } catch (const EngineDied& e) {
        x.reply(503, responses::error_body(e.what(), "server_error", nullptr, "server_error"));
        return;
    }
    auto produce = [xp, req, p, tools, asm_, validator](const std::function<bool(const json*)>& out) {
        Exchange& x = *xp;
        auto emit = [&](const json& e) {
            x.capture("responses", e);
            return out(&e);
        };
        for (const auto& e : asm_->start())
            if (!emit(e)) return;
        json done;
        bool gone = false;
        x.svc.run(p.ids, p.thinking, tools, p.max_new, req, *x.cancel, std::nullopt,
                  [&](const std::string& kind, const json& v, const Event* ev) -> bool {
                      if (kind == "ping") return out(nullptr) || !(gone = true);
                      if (kind == "event") {
                          for (const auto& e : asm_->feed(*ev))
                              if (!emit(e)) return !(gone = true);
                      } else if (kind == "done") {
                          done = v;
                      }
                      return true;
                  });
        if (gone || done.is_null() || jstr(done, "finish") == "cancel" || x.cancel->is_set()) return;
        std::function<std::string(const std::string&, const std::string&)> check;
        if (validator != nullptr)
            check = [validator](const std::string& text, const std::string& finish) { return validated_json(text, validator.get(), finish); };
        for (const auto& e : asm_->finish(done, check))
            if (!emit(e)) return;
    };
    if (!truthy(jget(req, "stream"))) {
        std::vector<json> items;
        try {
            produce([&](const json* e) {
                items.push_back(e != nullptr ? *e : json(nullptr));
                return true;
            });
        } catch (const StructuredOutputError& e) {
            x.reply(502, responses::error_body(e.what(), "server_error", nullptr, "structured_output_failed"));
            return;
        } catch (const QueueWaitExpired& e) {
            x.reply(429, responses::error_body(e.what(), "server_error", nullptr, "queue_wait_timeout"), json::object(), 1);
            return;
        } catch (const EngineDied& e) {
            x.reply(503, responses::error_body(std::string(e.what()) + "; the next request restarts it", "server_error", nullptr, "server_error"));
            return;
        } catch (const EngineNonFinite& e) {   // #879, twice: not the request's fault
            x.reply(503, responses::error_body(e.what(), "server_error", nullptr, "server_error"));
            return;
        } catch (const ToolCallError& e) {
            x.reply(502, responses::error_body(e.what(), "server_error", nullptr, "tool_call_failed"));
            return;
        } catch (const std::invalid_argument& e) {   // the engine's ERR line
            x.reply(500, responses::error_body(e.what(), "server_error", nullptr, "server_error"));
            return;
        }
        x.reply(200, responses::collect(items));
        return;
    }
    stream(xp, [xp, produce, asm_](httplib::DataSink& sink) {
        Exchange& x = *xp;
        double last = mono_s();
        bool gone = false;
        auto send = [&](const json& e) {
            return write(sink, "event: " + jstr(e, "type") + "\ndata: " + py_dumps(e, false) + "\n\n");
        };
        auto failed = [&](const std::string& message, const std::string& code) {
            const json e = asm_->failed(message, code);
            x.note({{"error", e["response"]["error"]}});
            send(e);
        };
        try {
            produce([&](const json* e) {
                bool ok;
                if (e == nullptr) {
                    // Codex's idle timeout counts events, not SSE comments: while a long prompt is read, a
                    // response.in_progress every 15 s tells it the server is still working
                    if (mono_s() - last >= 15) {
                        ok = send(asm_->in_progress());
                        last = mono_s();
                    } else {
                        ok = write(sink, ": keep-alive\n\n");
                    }
                } else {
                    ok = send(*e);
                    last = mono_s();
                }
                if (!ok) gone = true;
                return ok;
            });
            if (gone) {
                x.note({{"outcome", "disconnected"}});
                x.svc.note_client_disconnect(*x.cancel);
            }
        } catch (const EngineDied& e) {   // mid-stream: response.failed, then the stream ends
            failed(std::string(e.what()) + "; the next request restarts it", "server_error");
        } catch (const QueueWaitExpired& e) {
            failed(e.what(), "queue_wait_timeout");
        } catch (const StructuredOutputError& e) {
            failed(e.what(), "structured_output_failed");
        } catch (const std::invalid_argument& e) {   // the engine's ERR after the stream started
            failed(e.what(), "server_error");
        }
    });
}

json error_for(const std::string& path, const std::string& message, const char* kind, const char* code) {
    if (path == "/v1/responses") return responses::error_body(message, kind, nullptr, code != nullptr ? json(code) : json(nullptr));
    json e = {{"type", kind}};
    if (code != nullptr) e["code"] = code;
    e["message"] = message;
    return {{"error", e}};
}

}  // namespace

void do_post(Service& svc, HttpServer& server, const httplib::Request& req, httplib::Response& res) {
    auto xp = std::make_shared<Exchange>(svc, server, req, res);
    Exchange& x = *xp;
    if (!x.authorized()) return;
    std::string path = x.path;   // issue #55: Claude Code posts /v1/messages?beta=true
    if (kV1V1.count(path)) path = kV1V1.at(path);
    x.path = path;
    if (starts_with(path, "/v1/") && x.foreign_page()) return;
    if (path == "/settings") {
        post_settings(x);
        return;
    }
    if (path == "/config") {
        post_config(x);
        return;
    }
    if (path == "/unload" || path == "/load") {
        if (!x.control_body()) return;
        // JSON from Lamina's own page only, as /settings: else a plain form POST from any site unloads the model
        if (!x.own_page("the model can be loaded or unloaded")) return;
        if (path == "/unload") {   // give the GPU back now (between requests)
            try {
                const double wait_s = unload_wait(x.param("wait"), Service::kUnloadWaitS);
                const std::string r = wait_s ? svc.unload_waiting(wait_s) : svc.unload();
                x.reply(r == "busy" ? 409 : 200, {{"status", r}});
            } catch (const EngineStuck& e) {
                x.reply(503, {{"error", {{"type", "server_error"}, {"message", e.what()}}}});
            }
        } else {   // load now, e.g. ahead of a request
            try {
                svc.load();
                x.reply(200, {{"status", "loaded"}});
            } catch (const GpuBusy& e) {
                x.reply(503, {{"error", {{"type", "server_error"}, {"message", e.what()}}}},
                        e.retry_after ? json{{"Retry-After", e.retry_after}} : json::object());
            }
        }
        return;
    }
    try {
        if (path == "/v1/chat/completions" || path == "/v1/messages" || path == "/v1/responses") {
            const auto r = svc.reserve_inference();
            if (!r) {
                x.reply(503, error_for(path, "this instance is draining; retry against the next instance", "server_error", "server_draining"),
                        json::object(), 1);
                return;
            }
            if (!*r) {
                x.reply(429, path == "/v1/responses"
                                 ? responses::error_body("the inference admission limit is full; retry later", "server_error", nullptr, "admission_full")
                                 : json{{"error", {{"type", "rate_limit_error"}, {"code", "admission_full"},
                                                   {"message", "the inference admission limit is full; retry later"}}}},
                        json::object(), 1);
                return;
            }
            x.reserved = true;
        }
        json body;
        std::string err;
        if (!py_loads(req.body.empty() ? "{}" : req.body, body, &err)) throw RequestError(err);
        if (!body.is_object()) throw RequestError("send a JSON object");
        if (starts_with(path, "/v1/responses/")) {   // retrieve/delete/cancel/compact: nothing is stored
            x.reply(404, responses::error_body("this server keeps no responses (stateless); send the whole conversation to "
                                               "POST /v1/responses", "invalid_request_error", nullptr, "not_found"));
            x.finish();
            return;
        }
        if (path == "/v1/load" || path == "/v1/unload") {
            if (!x.own_page("the model can be loaded or unloaded")) {
                x.finish();
                return;
            }
            if (!body["model"].is_null() && body["model"] != svc.model) {
                x.reply(404, {{"error", {{"message", "model not found"}}}});
                x.finish();
                return;
            }
            std::string result;
            if (path == "/v1/unload") {
                const double wait_s = unload_wait(x.param("wait"), Service::kUnloadWaitS);
                result = wait_s ? svc.unload_waiting(wait_s) : svc.unload();
                if (result == "busy") throw ModelBusy("a request is running or queued");
            } else {
                auto turn = svc.try_turn();
                if (!turn.owns_lock()) throw ModelBusy("a request is running or queued");
                {
                    std::lock_guard<std::mutex> lk(svc.status_mu);
                    if (truthy(jget(svc.status, "busy")) || truthy(jget(svc.status, "queued")))
                        throw ModelBusy("a request is running or queued");
                }
                svc.ensure_loaded();
                {
                    std::lock_guard<std::mutex> lk(svc.status_mu);
                    svc.last_request_at = now_s();
                }
                result = "loaded";
            }
            json out = {{"status", result}};
            const json st = svc.v1_status();
            for (auto it = st.begin(); it != st.end(); ++it) out[it.key()] = it.value();
            x.reply(200, out);
            x.finish();
            return;
        }
        if (path == "/v1/chat/completions" || path == "/v1/messages" || path == "/v1/responses" || path == "/v1/choose")
            x.record = svc.begin_request(path, body);
        if (path == "/v1/responses") post_responses(xp, std::move(body));
        else if (path == "/v1/chat/completions") post_openai(xp, std::move(body));
        else if (path == "/v1/messages") post_anthropic(xp, std::move(body));
        else if (path == "/v1/messages/count_tokens") post_count_tokens(x, std::move(body));
        else if (path == "/tokenize") post_tokenize(x, body);
        else if (path == "/detokenize") post_detokenize(x, body);
        else if (path == "/apply-template") post_apply_template(x, std::move(body));
        else if (path == "/v1/choose") post_choose(x, body);
        else if (path == "/v1/images/generations") post_images(x, body);
        else x.reply(404, {{"error", {{"message", "not found"}}}});
    } catch (const EngineNonFinite& e) {   // #879, twice: not the request's fault
        say(std::string("[lamina] 503: ") + e.what());
        x.reply(503, error_for(path, e.what(), "server_error", nullptr));
    } catch (const ToolCallError& e) {   // the model's output, not the request
        say(std::string("[lamina] 502: ") + e.what());
        x.reply(502, error_for(path, e.what(), "server_error", "tool_call_failed"));
    } catch (const std::invalid_argument& e) {   // Python's ValueError
        say(std::string("[lamina] 400 invalid request: ") + e.what());
        x.reply(400, path == "/v1/responses" ? responses::error_body(e.what())
                                             : json{{"error", {{"type", "invalid_request_error"}, {"message", e.what()}}}});
    } catch (const ModelBusy& e) {
        x.reply(409, {{"error", {{"type", "model_busy"}, {"message", e.what()}}}});
    } catch (const QueueWaitExpired& e) {
        const json error = {{"type", "rate_limit_error"}, {"code", "queue_wait_timeout"}, {"message", e.what()}};
        x.note({{"error", error}});
        x.reply(429, path == "/v1/responses" ? responses::error_body(e.what(), "server_error", nullptr, "queue_wait_timeout")
                                             : json{{"error", error}},
                json::object(), 1);
    } catch (const StructuredOutputError& e) {
        x.reply(502, {{"error", {{"type", "structured_output_failed"}, {"code", "structured_output_failed"}, {"message", e.what()}}}});
    } catch (const GpuBusy& e) {   // #1997: a refusal worth retrying
        x.reply(503, {{"error", {{"type", "server_error"}, {"message", e.what()}}}},
                e.retry_after ? json{{"Retry-After", e.retry_after}} : json::object());
    } catch (const EngineStarting& e) {
        x.reply(503, {{"error", {{"type", "server_error"}, {"message", e.what()}}}});
    } catch (const json::exception& e) {
        // a body that is JSON but of the wrong shape: a 400 that says so instead of a dropped connection
        if (x.streaming) throw;
        say(std::string("[lamina] 400 malformed request (") + e.what() + ")");
        const std::string m = std::string("the request is malformed (") + e.what() + ")";
        x.reply(400, path == "/v1/responses" ? responses::error_body(m) : json{{"error", {{"type", "invalid_request_error"}, {"message", m}}}});
    } catch (const EngineDied& e) {   // before the answer started (not streamed)
        x.reply(503, {{"error", {{"type", "server_error"}, {"message", std::string(e.what()) + "; the next request restarts it"}}}});
    } catch (const EngineStuck& e) {   // an unload or restart that could not end the engine
        x.reply(503, {{"error", {{"type", "server_error"}, {"message", e.what()}}}});
    } catch (const ClientGone&) {   // #1999: the client left while the request was prepared
        x.note({{"outcome", "disconnected"}});
    }
    if (!x.streaming) x.finish();
}

}  // namespace lamina
