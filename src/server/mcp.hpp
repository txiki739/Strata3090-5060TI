// src/server/mcp.hpp - tools from MCP (Model Context Protocol) servers for the web app's chats (serve/mcp.py's port).
//
// The user lists servers in the run config ("mcp_servers", or Claude Desktop's "mcpServers" block pasted as is) or
// in a separate file (--mcp-config). Each one is a program Lamina starts (command, args, env, cwd: the stdio
// transport, newline-delimited JSON-RPC 2.0 on its stdin/stdout) or an address (url, headers: the Streamable HTTP
// transport, JSON-RPC POSTed, answered as JSON or as an event stream).
//
// McpHub owns every configured server: it starts them in the background, offers their tools to the model as
// function tools named <server>__<tool> (two servers may both have a `search`) and routes a call back to the server
// that has it. A server that does not start is left out with one log line; the chat works without it. One that
// stops later (a crash) is started again on its next call. A result reaches the model as text: the MCP content
// blocks joined, capped at max_result_chars with a note saying so (a directory tree or a web page can be megabytes).
// A failed call - the tool's own error, a crash, a timeout - is a result that starts with "error:", so the model can
// react instead of the chat breaking.
#pragma once

#include "json.hpp"
#include "sync.hpp"

#include <condition_variable>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lamina {

/// a server answered with an error, stopped, or could not be reached
struct McpError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct McpTimeout : McpError {
    using McpError::McpError;
};
/// the chat was stopped while the call ran
struct McpCancelled : McpError {
    using McpError::McpError;
};
/// an MCP entry of the config that is not what it must be: the server does not start (Python's SystemExit)
struct McpConfigError : std::invalid_argument {
    using std::invalid_argument::invalid_argument;
};

/// the MCP revision Lamina asks for; the server's answer is used as given
inline constexpr const char* kMcpProtocol = "2025-06-18";
/// the "mcp" block's defaults: timeout_s, max_result_chars, max_rounds, start_timeout_s
json mcp_defaults();

/// how a server is reached
class McpTransport {
public:
    virtual ~McpTransport() = default;
    virtual void start() = 0;
    virtual bool alive() = 0;
    /// a JSON-RPC request -> its result ({} when it is empty). McpTimeout after `timeout` s and McpCancelled once
    /// `cancel` is set (the server is told to stop working on it); McpError for an error answer or a server that
    /// stopped
    virtual json request(const std::string& method, const json& params, double timeout, Cancel* cancel = nullptr) = 0;
    /// a notification (params null: none); McpError when it cannot be sent
    virtual void notify(const std::string& method, const json& params = nullptr) = 0;
    virtual void close() = 0;
    /// the MCP revision: sent as MCP-Protocol-Version once negotiated ("": not yet)
    virtual std::string protocol() const = 0;
    virtual void set_protocol(const std::string& p) = 0;
};

/// a server Lamina starts: JSON-RPC messages one per line on its stdin and stdout; its stderr is its log
class StdioTransport final : public McpTransport {
public:
    StdioTransport(std::string name, json cfg);
    /// ends the server (and what it started) if it still runs
    ~StdioTransport() override;
    StdioTransport(const StdioTransport&) = delete;
    StdioTransport& operator=(const StdioTransport&) = delete;
    void start() override;
    bool alive() override;
    json request(const std::string& method, const json& params, double timeout, Cancel* cancel = nullptr) override;
    void notify(const std::string& method, const json& params = nullptr) override;
    /// as the spec says: its stdin closed, a moment to exit, then it (and what it started) is ended
    void close() override;
    std::string protocol() const override;
    void set_protocol(const std::string& p) override;
    struct Impl;

private:
    std::shared_ptr<Impl> impl_;   // shared with the thread that reads its output
};

/// a server at an address (Streamable HTTP): every message is a POST; the answer is JSON or an event stream that
/// carries it. The session id the server hands out at initialize rides on every later request.
class HttpTransport final : public McpTransport {
public:
    HttpTransport(std::string name, const json& cfg);
    HttpTransport(const HttpTransport&) = delete;
    HttpTransport& operator=(const HttpTransport&) = delete;
    void start() override {}
    bool alive() override;
    json request(const std::string& method, const json& params, double timeout, Cancel* cancel = nullptr) override;
    void notify(const std::string& method, const json& params = nullptr) override;
    /// ends the session on the server (it may not allow that)
    void close() override;
    std::string protocol() const override;
    void set_protocol(const std::string& p) override;
    /// the session id the server handed out ("": none)
    std::string session() const;
    struct Impl;

private:
    std::shared_ptr<Impl> impl_;   // shared with the threads that POST
};

/// one configured server: started once (and again after it stopped), with the tools it listed
class McpServer {
public:
    /// `settings`: the hub's, which outlive the server
    McpServer(std::string name, json cfg, const json& settings);
    McpServer(const McpServer&) = delete;
    McpServer& operator=(const McpServer&) = delete;

    const std::string name;
    const json cfg;
    const std::string kind;   ///< "http" (it has a url) or "stdio"
    const json& settings;

    // idle -> starting -> ready | failed; ready -> stopped when it ends. status, error, tools and info are read and
    // written under state_mu: the hub reads them while a start runs.
    std::string status = "idle";
    std::optional<std::string> error;
    json tools = json::array();   ///< what it listed: objects with a name
    json info = json::object();   ///< name, title and version of its serverInfo, and the protocol
    mutable std::mutex state_mu;

    /// initialize, then every page of tools/list; false (said in one log line) when it did not start
    bool start();
    /// tools/call -> the MCP result; a server that stopped is started again first (one that failed to start: at most
    /// every 10 s)
    json call(const std::string& tool, const json& arguments, double timeout, Cancel* cancel = nullptr);
    void close();
    /// the transport of its last start (null before the first)
    std::shared_ptr<McpTransport> transport() const;

private:
    /// `cancel`: a call's start again, given up (McpCancelled, the server left stopped) when its requester leaves
    bool start_locked(Cancel* cancel = nullptr);
    std::mutex lock_;                          ///< a start, and a call's look at the transport
    std::shared_ptr<McpTransport> transport_;  ///< changed under lock_ and state_mu
    double last_start_ = 0;
};

/// the model's tool name -> (its server, the tool's own name), in the order the model sees them
using McpRoutes = nlohmann::ordered_map<std::string, std::pair<McpServer*, std::string>>;

/// every configured server, and the merged, namespaced tool list the model sees
class McpHub {
public:
    /// servers: {name: config}; settings: over mcp_defaults()
    explicit McpHub(const json& servers, const json& settings = json::object());
    /// closes the servers (unless close() did) and waits for the threads that start them
    ~McpHub();
    McpHub(const McpHub&) = delete;
    McpHub& operator=(const McpHub&) = delete;

    json settings;
    std::vector<std::unique_ptr<McpServer>> servers;   ///< in the config's order

    /// every server started on a thread of its own (npx may download a package first: that must not delay the chat)
    void start(bool wait = false);
    /// waits (up to timeout_s in all; < 0: no limit) for servers that are still starting
    void wait(double timeout_s);
    /// a server that stopped after it was ready keeps its tools here: a call starts it again
    McpRoutes routes();
    /// the tools in the chat template's form ({name, description, parameters}); `exclude`: names the request brought
    /// itself (those win)
    json template_tools(const std::set<std::string>& exclude = {});
    json openai_tools();
    /// one tool -> {server, tool, ok, text (what the model reads, capped), chars (its full length), truncated, ms}.
    /// Errors become text starting with "error:"; only McpCancelled is thrown.
    json call(const std::string& name, const json& arguments, Cancel* cancel = nullptr);
    /// GET /mcp: every server, its state and its tools (for the web app)
    json status();
    void close();

private:
    std::mutex mu_;
    McpRoutes routes_;   ///< the last routes(), where call() looks first
    std::vector<std::thread> threads_;
    std::condition_variable cv_;
    int starting_ = 0;
    bool closed_ = false;
};

/// an MCP tool result as the text the model reads: the text blocks as they are, other blocks named
std::string result_text(const json& result);
/// {"name": {...}} with the entries checked; "disabled": true leaves one out. `where` names the block in messages
json servers_from(const json& block, const std::string& where);
/// the run config's optional "mcp" block: timeout_s, max_result_chars, max_rounds, start_timeout_s
json settings_from(const json& cfg);
/// the run config's "mcp_servers" (or "mcpServers") plus the servers in the --mcp-config file (Claude Desktop's
/// format: {"mcpServers": {...}}); a name in both takes the file's entry. nullptr when there are none
std::shared_ptr<McpHub> hub_from_config(const json& cfg, const std::string& mcp_config_path = "");

}  // namespace lamina
