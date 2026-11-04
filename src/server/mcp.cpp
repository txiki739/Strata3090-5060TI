// src/server/mcp.cpp - tools from MCP servers for the web app's chats (mcp.hpp; serve/mcp.py's port): the stdio and
// Streamable HTTP transports, a server's start (initialize, every page of tools/list) and calls, the hub's
// namespaced tool list and the config's checks. JSON-RPC 2.0 with nothing but the platform layer; no MCP SDK.
#include "mcp.hpp"

#include "http_client.hpp"
#include "log.hpp"
#include "process.hpp"
#include "sync.hpp"

#include <atomic>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>

#ifndef _WIN32
#include <sys/stat.h>
#endif

namespace lamina {
namespace {

constexpr std::string_view kBom = "\xEF\xBB\xBF";

/// format(x, "g")
std::string fmt_g(double x) {
    char buf[64];
    std::snprintf(buf, sizeof buf, "%g", x);
    return buf;
}

/// format(n, ",")
std::string commas(unsigned long long n) {
    std::string s = std::to_string(n);
    for (int i = (int) s.size() - 3; i > 0; i -= 3) s.insert((size_t) i, ",");
    return s;
}

/// the first n characters (code points) of UTF-8 text
std::string first_chars(std::string_view s, size_t n) {
    size_t i = 0;
    for (size_t k = 0; i < s.size() && k < n; ++k) {
        size_t len = 1;
        utf8_at(s, i, &len);
        i += len;
    }
    return std::string(s.substr(0, i));
}

/// json.loads of a str: as py_loads, but a leading BOM is refused (nlohmann's parser would skip it)
bool loads_str(std::string_view text, json& out, std::string* err = nullptr) {
    if (starts_with(text, kBom)) {
        if (err != nullptr) *err = "Unexpected UTF-8 BOM (decode using utf-8-sig): line 1 column 1 (char 0)";
        return false;
    }
    return py_loads(text, out, err);
}

/// the request id a message answers: Python compares ids with ==, so 1.0 and true answer request 1 as well
std::optional<long long> answer_id(const json& id) {
    if (id.is_boolean()) return id.get<bool>() ? 1 : 0;
    if (id.is_number_unsigned()) {
        const auto u = id.get<unsigned long long>();
        if (u <= (unsigned long long) LLONG_MAX) return (long long) u;
        return std::nullopt;
    }
    if (id.is_number_integer()) return id.get<long long>();
    if (id.is_number_float()) {
        const double d = id.get<double>();
        if (std::isfinite(d) && d == std::floor(d) && std::fabs(d) < 9.2e18) return (long long) d;
    }
    return std::nullopt;
}

/// float() of a setting
double num(const json& v) {
    if (v.is_number()) return v.get<double>();
    if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
    return 0;
}

/// int() of a setting (a float cut toward zero), within 64 bits
long long to_int(const json& v) {
    if (v.is_number_unsigned()) {
        const auto u = v.get<unsigned long long>();
        return u > (unsigned long long) LLONG_MAX ? LLONG_MAX : (long long) u;
    }
    if (v.is_number_integer()) return v.get<long long>();
    if (v.is_number_float()) {
        const double d = std::trunc(v.get<double>());
        return d >= 9.2e18 ? LLONG_MAX : d <= -9.2e18 ? LLONG_MIN : (long long) d;
    }
    if (v.is_boolean()) return v.get<bool>() ? 1 : 0;
    return 0;
}

/// re.sub(r"[^A-Za-z0-9_-]", "_", name) or "x"
std::string clean(std::string_view name) {
    std::string out;
    for (size_t i = 0; i < name.size();) {
        size_t n = 1;
        const uint32_t c = utf8_at(name, i, &n);
        const bool keep = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' || c == '-';
        out += keep ? (char) c : '_';
        i += n;
    }
    return out.empty() ? "x" : out;
}

/// a tool's own name, as a string (the spec says it is one)
std::string tool_name(const json& t) { return py_str(jget(t, "name")); }

// ------------------------------------------------------------------------------------------------ a request
/// one request waiting for its response
struct Slot {
    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    json msg;
    std::optional<std::string> error;

    void finish(json m) {
        {
            std::lock_guard<std::mutex> lk(mu);
            msg = std::move(m);
            done = true;
        }
        cv.notify_all();
    }
    void fail(std::string e) {
        {
            std::lock_guard<std::mutex> lk(mu);
            error = std::move(e);
            done = true;
        }
        cv.notify_all();
    }
    bool wait(double seconds) {
        std::unique_lock<std::mutex> lk(mu);
        return cv.wait_for(lk, std::chrono::duration<double>(seconds), [&] { return done; });
    }
};

/// waits for a response, looking at the chat's stop button and the deadline a few times a second
json wait_slot(Slot& slot, double timeout, Cancel* cancel, const std::string& what) {
    const double deadline = mono() + timeout;
    while (!slot.wait(0.05)) {
        if (cancel != nullptr && cancel->is_set()) throw McpCancelled("stopped");
        if (mono() > deadline) throw McpTimeout(what + " timed out after " + fmt_g(timeout) + " s");
    }
    std::lock_guard<std::mutex> lk(slot.mu);
    if (slot.error) throw McpError(*slot.error);
    const json& msg = slot.msg;
    if (msg.contains("error")) {
        const json& e = msg["error"];
        std::string text = "error", code = "None";
        if (e.is_object()) {
            if (truthy(jget(e, "message"))) text = py_str(jget(e, "message"));
            code = py_str(jget(e, "code"));
        } else if (const std::string s = py_str(e); !s.empty()) {
            text = s;
        }
        throw McpError(text + " (code " + code + ")");
    }
    const json& result = jget(msg, "result");
    return truthy(result) ? result : json::object();
}

/// an event stream read as it arrives, until the event that answers `rid` (it may carry notifications first)
struct EventReader {
    long long rid = 0;
    std::string part;                 // a line not ended yet
    std::vector<std::string> data;    // the data lines of the event so far
    json found;                       // the answer, once read

    /// true once the answer is in
    bool feed(std::string_view chunk) {
        part.append(chunk);
        size_t start = 0;
        for (size_t nl; (nl = part.find('\n', start)) != std::string::npos; start = nl + 1)
            if (line(std::string_view(part).substr(start, nl - start))) return true;
        part.erase(0, start);
        return false;
    }
    bool line(std::string_view raw) {
        const std::string text = decode_replace(raw);
        const std::string_view l = rstrip_chars(text, "\r\n");
        if (starts_with(l, "data:")) {
            data.emplace_back(lstrip_chars(l.substr(5), " "));
            return false;
        }
        if (!l.empty() || data.empty()) return false;
        std::string joined;
        for (size_t i = 0; i < data.size(); ++i) joined += (i ? "\n" : "") + data[i];
        data.clear();
        json m;
        if (!loads_str(joined, m)) return false;
        const json one = m.is_array() ? m : json::array({m});
        for (const auto& x : one) {
            if (x.is_object() && answer_id(jget(x, "id")) == rid && !x.contains("method")) {
                found = x;
                return true;
            }
        }
        return false;
    }
};

}  // namespace

json mcp_defaults() {
    return {{"timeout_s", 60.0}, {"max_result_chars", 20000}, {"max_rounds", 8}, {"start_timeout_s", 120.0}};
}

// ------------------------------------------------------------------------------------------------ stdio
struct StdioTransport::Impl {
    std::string name;
    json cfg;
    std::unique_ptr<Process> proc;
    std::mutex mu;   // pending, next_id, protocol, the last stdout line that was not JSON
    std::map<long long, std::shared_ptr<Slot>> pending;
    long long next_id = 0;
    std::string protocol = kMcpProtocol;
    std::string junk, junk_after;   // that line, and the stderr tail when it came (which of the two is newer)
    std::atomic<bool> ended{false};
    bool reader_done = false;       // under mu
    bool writer_done = false;       // under mu
    std::condition_variable reader_cv;   // and writer_done
    // the lines to the server, written by write_loop: a server that stops reading holds that thread in its write,
    // not a request (it times out, a Stop ends it) or close (it ends the server then)
    std::shared_ptr<LineChannel> lines = std::make_shared<LineChannel>();

    void read();
    void write_loop();
    void dispatch(const json& m);
    void send(const json& msg);
    std::string tail();
    void cancelled(long long rid, const std::string& reason);
};

StdioTransport::StdioTransport(std::string name, json cfg) : impl_(std::make_shared<Impl>()) {
    impl_->name = std::move(name);
    impl_->cfg = std::move(cfg);
}

StdioTransport::~StdioTransport() {
    impl_->lines->close();
    if (impl_->proc) impl_->proc->kill();   // with what it started; the reader ends with its output, the writer's write fails
}

void StdioTransport::start() {
    Impl& s = *impl_;
    Environment env = environment();
    const json& extra = jget(s.cfg, "env");
    if (extra.is_object())
        for (auto it = extra.begin(); it != extra.end(); ++it) env[it.key()] = py_str(it.value());
    const std::string command = py_str(jget(s.cfg, "command"));
    // on Windows npx, uvx and friends are .cmd files that CreateProcess finds only with their extension
    std::string exe;
    try {
        exe = which(command, &env);
    } catch (const std::exception&) {
    }
    SpawnOptions o;
    o.args.push_back(exe.empty() ? command : exe);
    const json& args = jget(s.cfg, "args");
    if (args.is_array())
        for (const auto& a : args) o.args.push_back(py_str(a));
    const json& cwd = jget(s.cfg, "cwd");
    if (truthy(cwd)) o.cwd = py_str(cwd);
    o.env = std::move(env);
    o.in = o.out = o.err = SpawnOptions::pipe;
    o.no_window = true;   // no console window of its own on Windows
    if (const char* debug = std::getenv("LAMINA_DEBUG"); debug != nullptr && *debug) {
        const std::string name = s.name;
        o.on_err_line = [name](const std::string& line) {
            if (!strip(line).empty()) say("[mcp " + name + "] " + std::string(rstrip(line)));
        };
    }
    try {
        s.proc = std::make_unique<Process>(std::move(o));
    } catch (const std::exception& e) {
        throw McpError("could not start " + py_repr_str(command) + ": " + e.what());
    }
    std::thread([self = impl_] { self->read(); }).detach();
    std::thread([self = impl_] { self->write_loop(); }).detach();
}

bool StdioTransport::alive() {
    const Impl& s = *impl_;
    return s.proc && !s.proc->poll() && !s.ended;
}

void StdioTransport::Impl::read() {
    std::string raw;
    while (proc->read_line(raw, -1) == 1) {
        const std::string text = decode_replace(raw);
        const std::string line(strip(text));
        if (line.empty()) continue;
        json msg;
        if (!loads_str(line, msg)) {   // a server that prints to stdout; the spec forbids it
            const std::string err_now = proc->stderr_tail();
            std::lock_guard<std::mutex> lk(mu);
            junk = line;
            junk_after = err_now;
            continue;
        }
        if (msg.is_array()) {
            for (const auto& m : msg)
                if (m.is_object()) dispatch(m);
        } else if (msg.is_object()) {
            dispatch(msg);
        }
    }
    ended = true;
    // the server's last log line says why it stopped: its stderr is let in first (the pipe ends with the process),
    // or the error raced it and said only "the server stopped"
    const double until = mono() + 1.2;
    proc->wait(1.0);
    for (std::string before = proc->stderr_tail(); mono() < until;) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        std::string after = proc->stderr_tail();
        if (after == before) break;
        before = std::move(after);
    }
    const std::optional<int> code = proc->poll();
    const std::string err = "the server stopped" + (code ? " (exit code " + exit_code_text(*code) + ")" : std::string()) + tail();
    std::map<long long, std::shared_ptr<Slot>> slots;
    {
        std::lock_guard<std::mutex> lk(mu);
        slots.swap(pending);
    }
    for (auto& entry : slots) entry.second->fail(err);
    {
        std::lock_guard<std::mutex> lk(mu);
        reader_done = true;
    }
    reader_cv.notify_all();
}

std::string StdioTransport::Impl::tail() {
    if (!proc) return "";
    const std::string err = proc->stderr_tail();
    std::string last, stdout_line;
    {
        std::lock_guard<std::mutex> lk(mu);
        stdout_line = junk;
        if (!junk.empty() && err == junk_after) last = junk;   // nothing on stderr since
    }
    if (last.empty()) {
        for (const auto& l : split(err, "\n")) {
            const std::string text = decode_replace(l);
            const std::string_view line = rstrip(text);
            if (!line.empty()) last = std::string(line);
        }
        if (last.empty()) last = stdout_line;
    }
    return last.empty() ? "" : ": " + first_chars(last, 300);
}

void StdioTransport::Impl::dispatch(const json& m) {
    if (m.contains("method")) {
        if (m.contains("id")) {   // a request from the server: we offer no client features
            const json& method = m["method"];
            try {
                if (method.is_string() && method.get<std::string>() == "ping")
                    send({{"jsonrpc", "2.0"}, {"id", m["id"]}, {"result", json::object()}});
                else
                    send({{"jsonrpc", "2.0"}, {"id", m["id"]},
                          {"error", {{"code", -32601}, {"message", "method not supported: " + py_str(method)}}}});
            } catch (const McpError&) {
            }
        }
        return;   // notifications (log messages, list changes): not used
    }
    std::shared_ptr<Slot> slot;
    if (const auto id = answer_id(jget(m, "id"))) {
        std::lock_guard<std::mutex> lk(mu);
        if (const auto it = pending.find(*id); it != pending.end()) {
            slot = it->second;
            pending.erase(it);
        }
    }
    if (slot) slot->finish(m);   // an answer to a call that timed out is simply dropped
}

void StdioTransport::Impl::send(const json& msg) {
    if (!proc || ended) throw McpError("the server stopped" + tail());
    lines->push(py_dumps(msg, false) + "\n");   // a write that fails ends the input; the reader then ends
}

void StdioTransport::Impl::write_loop() {
    std::string line;
    while (lines->pop(line, -1) == 1)
        if (!proc->write(line)) break;
    proc->close_stdin();   // close's end of the input, or a server that stopped reading it
    {
        std::lock_guard<std::mutex> lk(mu);
        writer_done = true;
    }
    reader_cv.notify_all();
}

void StdioTransport::Impl::cancelled(long long rid, const std::string& reason) {
    {
        std::lock_guard<std::mutex> lk(mu);
        pending.erase(rid);
    }
    try {   // tell the server to stop working on it
        send({{"jsonrpc", "2.0"}, {"method", "notifications/cancelled"}, {"params", {{"requestId", rid}, {"reason", reason}}}});
    } catch (const McpError&) {
    }
}

json StdioTransport::request(const std::string& method, const json& params, double timeout, Cancel* cancel) {
    Impl& s = *impl_;
    if (!alive()) throw McpError("the server is not running" + s.tail());
    auto slot = std::make_shared<Slot>();
    long long rid;
    {
        std::lock_guard<std::mutex> lk(s.mu);
        rid = ++s.next_id;
        s.pending[rid] = slot;
    }
    try {
        s.send({{"jsonrpc", "2.0"}, {"id", rid}, {"method", method}, {"params", params}});
        return wait_slot(*slot, timeout, cancel, method);
    } catch (const McpTimeout& e) {
        s.cancelled(rid, e.what());
        throw;
    } catch (const McpCancelled& e) {
        s.cancelled(rid, e.what());
        throw;
    }
}

void StdioTransport::notify(const std::string& method, const json& params) {
    json msg = {{"jsonrpc", "2.0"}, {"method", method}};
    if (!params.is_null()) msg["params"] = params;
    impl_->send(msg);
}

void StdioTransport::close() {
    Impl& s = *impl_;
    if (!s.proc) return;
    s.lines->close();   // the writer closes the input after what is queued; one held in a write is ended below
    {
        std::unique_lock<std::mutex> lk(s.mu);
        s.reader_cv.wait_for(lk, std::chrono::seconds(2), [&] { return s.writer_done; });
    }
    if (!s.proc->wait(2)) s.proc->kill();   // npx.cmd -> node: the whole tree
    // its reader ends with the output (Python closes the pipe under it; here its end is awaited, briefly: a program
    // it started may hold the pipe open)
    std::unique_lock<std::mutex> lk(s.mu);
    s.reader_cv.wait_for(lk, std::chrono::milliseconds(1500), [&] { return s.reader_done; });
}

std::string StdioTransport::protocol() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->protocol;
}

void StdioTransport::set_protocol(const std::string& p) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->protocol = p;
}

// ------------------------------------------------------------------------------------------------ Streamable HTTP
struct HttpTransport::Impl {
    std::string name, url;
    HttpHeaders headers;    // the config's
    mutable std::mutex mu;  // session, protocol, next_id, broken
    std::string session, protocol;
    long long next_id = 0;
    std::optional<std::string> broken;

    HttpHeaders request_headers() const;
    json exchange(const json& msg, long long rid, double timeout, bool answer);
    void post(const json& msg, long long rid, double timeout, Slot* slot);
};

HttpTransport::HttpTransport(std::string name, const json& cfg) : impl_(std::make_shared<Impl>()) {
    impl_->name = std::move(name);
    impl_->url = py_str(jget(cfg, "url"));
    const json& h = jget(cfg, "headers");
    if (h.is_object())
        for (auto it = h.begin(); it != h.end(); ++it) impl_->headers.emplace_back(it.key(), py_str(it.value()));
}

HttpHeaders HttpTransport::Impl::request_headers() const {
    HttpHeaders h = {{"Content-Type", "application/json"}, {"Accept", "application/json, text/event-stream"},
                     {"User-Agent", "lamina"}};
    auto set = [&h](const std::string& k, const std::string& v) {   // urllib keys them case-insensitively: the last wins
        for (auto& [hk, hv] : h)
            if (lower(hk) == lower(k)) {
                hv = v;
                return;
            }
        h.emplace_back(k, v);
    };
    for (const auto& [k, v] : headers) set(k, v);
    std::lock_guard<std::mutex> lk(mu);
    if (!session.empty()) set("Mcp-Session-Id", session);
    if (!protocol.empty()) set("MCP-Protocol-Version", protocol);
    return h;
}

/// one POST -> the response that answers `rid` (null for a notification); McpError in this module's words, any
/// other exception for a server that could not be reached
json HttpTransport::Impl::exchange(const json& msg, long long rid, double timeout, bool answer) {
    EventReader events;
    events.rid = rid;
    std::string body;
    const HttpReply r = http_request("POST", url, request_headers(), py_dumps(msg, false), timeout, 0,
                                     [&](std::string_view chunk) {
                                         body.append(chunk);
                                         // a notification's answer is not read; a stream may stay open after the
                                         // event that answers
                                         return answer && !events.feed(chunk);
                                     });
    if (r.status < 200 || r.status >= 300) {
        const std::string detail = first_chars(decode_replace(body), 300);
        {
            std::lock_guard<std::mutex> lk(mu);
            if (r.status == 404 && !session.empty()) broken = "the server ended the session";   // a restart starts a new one
        }
        throw McpError("HTTP " + std::to_string(r.status) + " " + r.reason + (detail.empty() ? "" : ": " + detail));
    }
    if (const std::string sid = r.header("mcp-session-id"); !sid.empty()) {
        std::lock_guard<std::mutex> lk(mu);
        session = sid;
    }
    if (!answer) return nullptr;   // a notification: 202, nothing to read
    json found;
    if (r.header("content-type").find("text/event-stream") != std::string::npos) {
        found = events.found;
    } else {
        const auto space = [](char c) { return c == ' ' || (c >= '\t' && c <= '\r'); };   // bytes.strip()'s
        std::string_view b = body;
        while (!b.empty() && space(b.front())) b.remove_prefix(1);
        while (!b.empty() && space(b.back())) b.remove_suffix(1);
        if (!b.empty()) {
            std::string_view text = body;
            if (starts_with(text, kBom)) text.remove_prefix(kBom.size());   // json.loads of bytes: utf-8-sig
            json parsed;
            std::string err;
            if (!loads_str(text, parsed, &err)) throw std::runtime_error(err);
            const json all = parsed.is_array() ? parsed : json::array({parsed});
            for (const auto& m : all)
                if (m.is_object() && answer_id(jget(m, "id")) == rid && !m.contains("method")) found = m;
        }
    }
    if (found.is_null()) throw McpError("the server's answer did not contain a response");
    return found;
}

void HttpTransport::Impl::post(const json& msg, long long rid, double timeout, Slot* slot) {
    std::optional<std::string> error;
    json found;
    try {
        found = exchange(msg, rid, timeout, slot != nullptr);
    } catch (const McpError& e) {
        error = e.what();
    } catch (const std::exception& e) {
        error = "could not reach " + url + ": " + e.what();
    }
    if (slot == nullptr) {
        if (error) throw McpError(*error);
        return;
    }
    if (error) slot->fail(*error);
    else slot->finish(std::move(found));
}

bool HttpTransport::alive() {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return !impl_->broken;
}

json HttpTransport::request(const std::string& method, const json& params, double timeout, Cancel* cancel) {
    std::shared_ptr<Impl> s = impl_;
    long long rid;
    {
        std::lock_guard<std::mutex> lk(s->mu);
        if (s->broken) throw McpError(*s->broken);
        rid = ++s->next_id;
    }
    auto slot = std::make_shared<Slot>();
    json msg = {{"jsonrpc", "2.0"}, {"id", rid}, {"method", method}, {"params", params}};
    // the POST runs on its own thread so the stop button and the deadline are noticed while it waits
    std::thread([s, slot, msg = std::move(msg), rid, timeout] { s->post(msg, rid, timeout + 5, slot.get()); }).detach();
    auto cancelled = [&](const char* reason) {
        json note = {{"jsonrpc", "2.0"}, {"method", "notifications/cancelled"},
                     {"params", {{"requestId", rid}, {"reason", reason}}}};
        std::thread([s, note = std::move(note)] {
            try {
                s->post(note, 0, 10, nullptr);
            } catch (const std::exception&) {
            }
        }).detach();
    };
    try {
        return wait_slot(*slot, timeout, cancel, method);
    } catch (const McpTimeout& e) {
        cancelled(e.what());
        throw;
    } catch (const McpCancelled& e) {
        cancelled(e.what());
        throw;
    }
}

void HttpTransport::notify(const std::string& method, const json& params) {
    json msg = {{"jsonrpc", "2.0"}, {"method", method}};
    if (!params.is_null()) msg["params"] = params;
    impl_->post(msg, 0, 10, nullptr);
}

void HttpTransport::close() {
    if (session().empty()) return;
    try {   // the server may not allow it
        http_request("DELETE", impl_->url, impl_->request_headers(), "", 3);
    } catch (const std::exception&) {
    }
}

std::string HttpTransport::protocol() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->protocol;
}

void HttpTransport::set_protocol(const std::string& p) {
    std::lock_guard<std::mutex> lk(impl_->mu);
    impl_->protocol = p;
}

std::string HttpTransport::session() const {
    std::lock_guard<std::mutex> lk(impl_->mu);
    return impl_->session;
}

// ------------------------------------------------------------------------------------------------ one server
McpServer::McpServer(std::string name_, json cfg_, const json& settings_)
    : name(std::move(name_)), cfg(std::move(cfg_)), kind(truthy(jget(cfg, "url")) ? "http" : "stdio"), settings(settings_) {}

std::shared_ptr<McpTransport> McpServer::transport() const {
    std::lock_guard<std::mutex> lk(state_mu);
    return transport_;
}

bool McpServer::start() {
    std::lock_guard<std::mutex> lk(lock_);
    return start_locked();
}

bool McpServer::start_locked(Cancel* cancel) {
    if (const auto old = transport()) old->close();
    std::shared_ptr<McpTransport> t;
    if (kind == "http") t = std::make_shared<HttpTransport>(name, cfg);
    else t = std::make_shared<StdioTransport>(name, cfg);
    {
        std::lock_guard<std::mutex> lk(state_mu);
        status = "starting";
        error.reset();
        transport_ = t;
    }
    last_start_ = mono();
    const double timeout = num(jget(settings, "start_timeout_s"));
    try {
        t->start();
        const json res = t->request("initialize",
                                    {{"protocolVersion", kMcpProtocol},
                                     {"capabilities", json::object()},
                                     {"clientInfo", {{"name", "lamina"}, {"title", "Lamina"}, {"version", "1"}}}},
                                    timeout, cancel);
        const json& version = jget(res, "protocolVersion");
        t->set_protocol(truthy(version) ? py_str(version) : std::string(kMcpProtocol));
        json server_info = json::object();
        const json& si = jget(res, "serverInfo");
        if (si.is_object())
            for (auto it = si.begin(); it != si.end(); ++it)
                if (it.key() == "name" || it.key() == "title" || it.key() == "version") server_info[it.key()] = it.value();
        server_info["protocol"] = t->protocol();
        {
            std::lock_guard<std::mutex> lk(state_mu);
            info = server_info;
        }
        t->notify("notifications/initialized");
        json listed = json::array(), cursor;
        for (int page_no = 0; page_no < 100; ++page_no) {   // pages; a server that never stops paging is cut off
            json page;
            try {
                page = t->request("tools/list", truthy(cursor) ? json{{"cursor", cursor}} : json::object(), timeout, cancel);
            } catch (const McpError& e) {
                // a server without tools (prompts or resources only)
                if (std::string_view(e.what()).find("code -32601") != std::string_view::npos && listed.empty()) break;
                throw;
            }
            const json& page_tools = jget(page, "tools");
            if (page_tools.is_array())
                for (const auto& x : page_tools)
                    if (x.is_object() && truthy(jget(x, "name"))) listed.push_back(x);
            cursor = jget(page, "nextCursor");
            if (!truthy(cursor)) break;
        }
        {
            std::lock_guard<std::mutex> lk(state_mu);
            tools = listed;
            status = "ready";
        }
        std::string names;
        for (size_t i = 0; i < listed.size() && i < 8; ++i) names += (i ? ", " : "") + tool_name(listed[i]);
        if (listed.size() > 8) names += ", ...";
        say("[lamina] MCP server " + py_repr_str(name) + ": " + std::to_string(listed.size()) + " tool" +
            (listed.size() != 1 ? "s" : "") + (listed.empty() ? "" : " (" + names + ")"));
        return true;
    } catch (const McpCancelled&) {   // the requester left: not a failed start, the next call starts it again
        {
            std::lock_guard<std::mutex> lk(state_mu);
            status = "stopped";   // (its last tool list kept: routes() sends the next call here)
        }
        t->close();
        throw;
    } catch (const std::exception& e) {   // McpError (anything else would be a failed start as well)
        bool listed;   // its tools from an earlier start stay: a call to one starts it again (at most every 10 s)
        {
            std::lock_guard<std::mutex> lk(state_mu);
            status = "failed";
            error = e.what();
            listed = !tools.empty();
        }
        t->close();
        say("[lamina] MCP server " + py_repr_str(name) + " did not start: " + e.what() +
            (listed ? ". Its tools stay listed; a call to one tries again." : ". Its tools are left out; the chat works without them."));
        return false;
    }
}

json McpServer::call(const std::string& tool, const json& arguments, double timeout, Cancel* cancel) {
    std::shared_ptr<McpTransport> t;
    {
        std::lock_guard<std::mutex> lk(lock_);   // (another call's start may hold it a while)
        if (cancel != nullptr && cancel->is_set()) throw McpCancelled("stopped");
        t = transport();
        if (!t || !t->alive()) {
            // it stopped since its last call (a crash, an expired session): start it again; one that failed to start
            // is tried again at most every 10 s, so a broken command does not run on every call
            std::string was;
            std::optional<std::string> why;
            {
                std::lock_guard<std::mutex> st(state_mu);
                was = status;
                why = error;
            }
            if (was == "failed" && mono() - last_start_ < 10)
                throw McpError("the server is not running (" + (why && !why->empty() ? *why : std::string("it stopped")) + ")");
            say("[lamina] MCP server " + py_repr_str(name) + " had stopped; starting it again");
            if (!start_locked(cancel)) {
                std::lock_guard<std::mutex> st(state_mu);
                throw McpError("the server could not be started again: " + (error ? *error : std::string("None")));
            }
            t = transport();
        }
    }
    try {
        if (cancel != nullptr && cancel->is_set()) throw McpCancelled("stopped");   // never sent after a Stop
        return t->request("tools/call", {{"name", tool}, {"arguments", arguments}}, timeout, cancel);
    } catch (const McpCancelled&) {
        throw;
    } catch (const McpError& e) {
        if (!t->alive()) {
            std::lock_guard<std::mutex> st(state_mu);
            status = "stopped";
            error = e.what();
        }
        throw;
    }
}

void McpServer::close() {
    std::lock_guard<std::mutex> lk(lock_);
    if (const auto t = transport()) t->close();
    std::lock_guard<std::mutex> st(state_mu);
    if (status == "ready" || status == "starting") status = "stopped";
}

// ------------------------------------------------------------------------------------------------ the hub
std::string result_text(const json& result) {
    std::vector<std::string> parts;
    const json& content = jget(result, "content");
    if (content.is_array()) {
        for (const auto& block : content) {
            if (!block.is_object()) continue;
            const json& type = jget(block, "type");
            const std::string k = type.is_string() ? type.get<std::string>() : "";   // another type matches nothing
            if (k == "text") {
                parts.push_back(block.contains("text") ? py_str(block["text"]) : "");
            } else if (k == "resource") {
                const json& res = jget(block, "resource");   // not an object: read as {}
                if (res.is_object() && res.contains("text"))
                    parts.push_back(py_str(res["text"]));
                else
                    parts.push_back("[resource " + (res.contains("uri") ? py_str(res["uri"]) : "") + " (" +
                                    (res.contains("mimeType") ? py_str(res["mimeType"]) : "binary") + "), not shown]");
            } else if (k == "resource_link") {
                const json& n = jget(block, "name");
                parts.push_back(replace_all("[resource link: " + (truthy(n) ? py_str(n) : "") + " " +
                                                (block.contains("uri") ? py_str(block["uri"]) : "") + "]",
                                            "  ", " "));
            } else if (k == "image" || k == "audio") {
                const json& data = jget(block, "data");
                size_t len = 0;   // len(data or "")
                if (data.is_string()) len = utf8_len(data.get_ref<const std::string&>());
                else if (data.is_array() || data.is_object()) len = data.size();
                parts.push_back("[" + k + " (" + (block.contains("mimeType") ? py_str(block["mimeType"]) : "?") + ", " +
                                commas(len * 3 / 4) + " bytes): not shown to the model]");
            } else {
                parts.push_back(py_dumps(block, false));
            }
        }
    }
    if (parts.empty() && result.is_object() && result.contains("structuredContent") && !result["structuredContent"].is_null())
        parts.push_back(py_dumps(result["structuredContent"], false));
    std::string out;
    for (size_t i = 0; i < parts.size(); ++i) out += (i ? "\n" : "") + parts[i];
    return out;
}

McpHub::McpHub(const json& servers_cfg, const json& settings_) : settings(mcp_defaults()) {
    if (settings_.is_object())
        for (auto it = settings_.begin(); it != settings_.end(); ++it) settings[it.key()] = it.value();
    if (servers_cfg.is_object())
        for (auto it = servers_cfg.begin(); it != servers_cfg.end(); ++it)
            servers.push_back(std::make_unique<McpServer>(it.key(), it.value(), settings));
}

McpHub::~McpHub() {
    bool was_closed;
    {
        std::lock_guard<std::mutex> lk(mu_);
        was_closed = closed_;
    }
    if (!was_closed) close();
    for (auto& t : threads_)
        if (t.joinable()) t.join();
}

void McpHub::start(bool wait_all) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (const auto& s : servers) {
            ++starting_;
            threads_.emplace_back([this, srv = s.get()] {
                try {
                    srv->start();
                } catch (const std::exception&) {
                }
                {
                    std::lock_guard<std::mutex> done(mu_);
                    --starting_;
                }
                cv_.notify_all();
            });
        }
    }
    if (wait_all) wait(-1);
}

void McpHub::wait(double timeout_s) {
    std::unique_lock<std::mutex> lk(mu_);
    const auto done = [&] { return starting_ == 0; };
    if (timeout_s < 0) cv_.wait(lk, done);
    else cv_.wait_for(lk, std::chrono::duration<double>(timeout_s), done);
}

McpRoutes McpHub::routes() {
    McpRoutes out;
    for (const auto& s : servers) {
        json listed;
        {
            std::lock_guard<std::mutex> lk(s->state_mu);   // a failed one: the tools of its last start, if any
            if (s->status != "ready" && s->status != "stopped" && s->status != "failed") continue;
            listed = s->tools;
        }
        for (const auto& t : listed) {
            std::string n = (clean(s->name) + "__" + clean(tool_name(t))).substr(0, 64);
            for (int k = 2; out.count(n) != 0; ++k) n = n.substr(0, 60) + "_" + std::to_string(k);   // two names that differ only in cleaned characters
            out[n] = {s.get(), tool_name(t)};
        }
    }
    std::lock_guard<std::mutex> lk(mu_);
    routes_ = McpRoutes(out);   // ordered_map cannot be copy-assigned (its keys are const)
    return out;
}

json McpHub::template_tools(const std::set<std::string>& exclude) {
    json out = json::array();
    for (const auto& [n, route] : routes()) {
        if (exclude.count(n) != 0) continue;
        json t;
        {
            std::lock_guard<std::mutex> lk(route.first->state_mu);
            for (const auto& x : route.first->tools)
                if (tool_name(x) == route.second) {
                    t = x;
                    break;
                }
        }
        if (t.is_null()) continue;   // its tools changed since (a restart)
        const json& schema = jget(t, "inputSchema");
        const json& description = jget(t, "description");
        const json& title = jget(t, "title");
        out.push_back({{"name", n},
                       {"description", truthy(description) ? py_str(description) : truthy(title) ? py_str(title) : ""},
                       {"parameters", schema.is_object() && !schema.empty() ? schema
                                                                             : json{{"type", "object"}, {"properties", json::object()}}}});
    }
    return out;
}

json McpHub::openai_tools() {
    json out = json::array();
    for (const auto& t : template_tools()) out.push_back({{"type", "function"}, {"function", t}});
    return out;
}

json McpHub::call(const std::string& name, const json& arguments, Cancel* cancel) {
    const double t0 = mono();
    std::pair<McpServer*, std::string> route{nullptr, name};
    bool cached = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (const auto it = routes_.find(name); it != routes_.end()) {
            route = it->second;
            cached = true;
        }
    }
    if (!cached) {
        const McpRoutes fresh = routes();
        if (const auto it = fresh.find(name); it != fresh.end()) route = it->second;
    }
    McpServer* server = route.first;
    json out = {{"server", server != nullptr ? json(server->name) : json(nullptr)}, {"tool", route.second}};
    std::string text;
    bool ok = false;
    if (server == nullptr) {
        text = "error: there is no tool named " + py_repr_str(name);
    } else {
        try {
            const json res = server->call(route.second, arguments.is_object() ? arguments : json::object(),
                                          num(jget(settings, "timeout_s")), cancel);
            text = result_text(res);
            ok = !truthy(jget(res, "isError"));
            if (!ok) text = "error: " + (text.empty() ? std::string("the tool reported an error") : text);
        } catch (const McpCancelled&) {
            throw;
        } catch (const McpError& e) {
            text = std::string("error: ") + e.what();
        }
    }
    const long long cap = to_int(jget(settings, "max_result_chars"));
    const size_t full = utf8_len(text);
    const bool truncated = cap > 0 && full > (unsigned long long) cap;
    if (truncated)
        text = first_chars(text, (size_t) cap) + "\n\n[... truncated: the tool returned " + commas(full) +
               " characters; only the first " + commas((unsigned long long) cap) + " are shown]";
    out["ok"] = ok;
    out["text"] = text;
    out["chars"] = full;
    out["truncated"] = truncated;
    out["ms"] = (long long) ((mono() - t0) * 1000);
    return out;
}

json McpHub::status() {
    const McpRoutes r = routes();
    json list = json::array();
    for (const auto& s : servers) {
        std::map<std::string, std::string> names;   // a tool's own name -> the name the model sees (the last one)
        for (const auto& [n, route] : r)
            if (route.first == s.get()) names[route.second] = n;
        json listed, inf;
        std::string st;
        std::optional<std::string> err;
        {
            std::lock_guard<std::mutex> lk(s->state_mu);
            st = s->status;
            err = s->error;
            listed = s->tools;
            inf = s->info;
        }
        json tl = json::array();
        for (const auto& t : listed) {
            const auto it = names.find(tool_name(t));
            const json& description = jget(t, "description");
            tl.push_back({{"name", it != names.end() ? json(it->second) : jget(t, "name")},
                          {"tool", jget(t, "name")},
                          {"description", first_chars(truthy(description) ? py_str(description) : "", 300)}});
        }
        list.push_back({{"name", s->name},
                        {"transport", s->kind},
                        {"status", st},
                        {"error", err ? json(*err) : json(nullptr)},
                        {"info", inf},
                        {"tools", tl}});
    }
    return {{"servers", list},
            {"tools", r.size()},
            {"settings", {{"timeout_s", jget(settings, "timeout_s")},
                          {"max_result_chars", jget(settings, "max_result_chars")},
                          {"max_rounds", jget(settings, "max_rounds")}}}};
}

void McpHub::close() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        closed_ = true;
    }
    for (const auto& s : servers) {
        try {   // closing is best effort
            s->close();
        } catch (const std::exception&) {
        }
    }
}

// ------------------------------------------------------------------------------------------------ config
namespace {

json check_server(const std::string& name, const json& cfg, const std::string& where) {
    const std::string who = "[lamina] " + where + ": MCP server " + py_repr_str(name);
    if (!cfg.is_object()) throw McpConfigError(who + " must be an object with \"command\" or \"url\"");
    if (truthy(jget(cfg, "url"))) {
        const json& type = jget(cfg, "type");
        if (type.is_string() && type.get<std::string>() == "sse")
            throw McpConfigError(who + " uses the old SSE transport (\"type\": \"sse\"); Lamina speaks Streamable HTTP - "
                                       "most servers offer it at /mcp");
        const json& headers = jget(cfg, "headers");
        if (truthy(headers) && !headers.is_object()) throw McpConfigError(who + ": \"headers\" must be an object");
        return cfg;
    }
    const json& command = jget(cfg, "command");
    if (!command.is_string() || strip(command.get_ref<const std::string&>()).empty())
        throw McpConfigError(who + " needs \"command\" (a program to start) or \"url\"");
    const json& args = jget(cfg, "args");
    if (truthy(args) && !args.is_array()) throw McpConfigError(who + ": \"args\" must be a list");
    const json& env = jget(cfg, "env");
    if (truthy(env) && !env.is_object()) throw McpConfigError(who + ": \"env\" must be an object");
    return cfg;
}

/// open(path, encoding="utf-8-sig") read whole; false with Python's OSError text ("[Errno 2] No such file or
/// directory: 'path'") when it cannot be
bool read_file(const std::string& path, std::string& text, std::string& error) {
#ifdef _WIN32
    FILE* f = _wfopen(u8_path(path).c_str(), L"rb");
#else
    FILE* f = std::fopen(path.c_str(), "rb");
    struct stat st;
    if (f != nullptr && fstat(fileno(f), &st) == 0 && S_ISDIR(st.st_mode)) {   // Python's open() refuses a directory
        std::fclose(f);
        f = nullptr;
        errno = EISDIR;
    }
#endif
    if (f == nullptr) {
        const int e = errno;
        error = "[Errno " + std::to_string(e) + "] " + std::strerror(e) + ": " + py_repr_str(path);
        return false;
    }
    char buf[65536];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
    const bool failed = std::ferror(f) != 0;
    const int e = errno;
    std::fclose(f);
    if (failed) error = "[Errno " + std::to_string(e) + "] " + std::strerror(e);
    else if (starts_with(text, kBom)) text.erase(0, kBom.size());
    return !failed;
}

}  // namespace

json servers_from(const json& block, const std::string& where) {
    if (block.is_null()) return json::object();
    if (!block.is_object()) throw McpConfigError("[lamina] " + where + ": the MCP servers must be an object {\"name\": {...}}");
    json out = json::object();
    for (auto it = block.begin(); it != block.end(); ++it) {
        const json& disabled = jget(it.value(), "disabled");
        if (disabled.is_boolean() && disabled.get<bool>()) continue;
        out[it.key()] = check_server(it.key(), it.value(), where);
    }
    return out;
}

json settings_from(const json& cfg) {
    json out = json::object();
    const json& block = jget(cfg, "mcp");
    if (!truthy(block)) return out;
    if (!block.is_object()) throw McpConfigError("[lamina] config mcp=" + py_repr_value(block) + ": expected an object");
    for (auto it = block.begin(); it != block.end(); ++it) {
        const std::string& key = it.key();
        const json& value = it.value();
        const bool number = is_num(value);
        const std::string shown = "[lamina] config mcp." + key + "=" + py_repr_value(value);
        if (key == "timeout_s" || key == "start_timeout_s") {
            if (!number || value.get<double>() <= 0) throw McpConfigError(shown + ": expected a number of seconds > 0");
            out[key] = value.get<double>();
        } else if (key == "max_result_chars" || key == "max_rounds") {
            const double d = number ? value.get<double>() : 0;
            if (!number || (value.is_number_float() && d != std::trunc(d)) || d < 1)
                throw McpConfigError(shown + ": expected a positive integer");
            if (!value.is_number_float()) out[key] = value;
            else if (d < 9.2e18) out[key] = (long long) d;
            else if (d < 1.8e19) out[key] = (unsigned long long) d;
            else out[key] = d;   // beyond 64 bits (Python keeps the int)
        } else {
            say(shown + ": unknown key, ignored");
        }
    }
    return out;
}

std::shared_ptr<McpHub> hub_from_config(const json& cfg, const std::string& mcp_config_path) {
    json servers = json::object();
    auto add = [&servers](const json& block) {
        for (auto it = block.begin(); it != block.end(); ++it) servers[it.key()] = it.value();
    };
    add(servers_from(jget(cfg, "mcp_servers"), "config mcp_servers"));
    add(servers_from(jget(cfg, "mcpServers"), "config mcpServers"));
    if (!mcp_config_path.empty()) {
        const std::string where = "--mcp-config " + mcp_config_path;
        std::string text, err;
        json data;
        if (!read_file(mcp_config_path, text, err) || !loads_str(text, data, &err))
            throw McpConfigError("[lamina] " + where + ": " + err);
        json block;
        if (data.is_object()) block = data.contains("mcpServers") ? data["mcpServers"] : jget(data, "mcp_servers");
        if (block.is_null()) throw McpConfigError("[lamina] " + where + ": expected {\"mcpServers\": {...}}");
        add(servers_from(block, where));
    }
    if (servers.empty()) return nullptr;
    return std::make_shared<McpHub>(servers, settings_from(cfg));
}

}  // namespace lamina
