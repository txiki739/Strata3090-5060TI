// src/server/vision.cpp - the resident image encoder (vision.hpp).
#include "vision.hpp"

#include "http_client.hpp"
#include "log.hpp"
#include "service.hpp"

#include "hash/hash.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO
#define STBI_FAILURE_USERMSG
#ifdef _MSC_VER
#pragma warning(push, 0)
#endif
#include "stb/stb_image.h"
#ifdef _MSC_VER
#pragma warning(pop)
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <wincodec.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace lamina {
namespace fs = std::filesystem;
namespace {

const std::optional<double> kReadyS = timeout_env("LAMINA_VISION_READY_S", 300.0);
const std::optional<double> kEncodeS = timeout_env("LAMINA_VISION_ENCODE_S", 300.0);

std::string read_file(const fs::path& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open");
    return std::string(std::istreambuf_iterator<char>(f), {});
}

void write_file(const fs::path& p, const std::string& data) {
    std::ofstream f(p, std::ios::binary | std::ios::trunc);
    f.write(data.data(), (std::streamsize) data.size());
    if (!f) throw std::runtime_error("the file " + p.string() + " could not be written");
}

/// base64 as Python's b64decode reads it (characters outside the alphabet are skipped)
std::string b64decode(std::string_view s) {
    auto val = [](unsigned char c) -> int {
        if (c >= 'A' && c <= 'Z') return c - 'A';
        if (c >= 'a' && c <= 'z') return c - 'a' + 26;
        if (c >= '0' && c <= '9') return c - '0' + 52;
        if (c == '+') return 62;
        if (c == '/') return 63;
        return -1;
    };
    std::string out;
    out.reserve(s.size() * 3 / 4);
    uint32_t acc = 0;
    int bits = 0;
    size_t n = 0, pad = 0;
    for (const unsigned char c : s) {
        if (c == '=') {
            ++pad;
            continue;
        }
        const int v = val(c);
        if (v < 0) continue;
        if (pad) throw std::invalid_argument("Discontinuous padding not allowed");
        acc = (acc << 6) | (uint32_t) v;
        bits += 6;
        ++n;
        if (bits >= 8) {
            bits -= 8;
            out.push_back((char) ((acc >> bits) & 0xff));
        }
    }
    if (n % 4 == 1)
        throw std::invalid_argument("Invalid base64-encoded string: number of data characters (" + std::to_string(n) +
                                    ") cannot be 1 more than a multiple of 4");
    if (n % 4 != 0 && (n % 4) + pad < 4) throw std::invalid_argument("Incorrect padding");
    return out;
}

bool is_jpeg(const std::string& d) { return d.size() >= 3 && (unsigned char) d[0] == 0xff && (unsigned char) d[1] == 0xd8 && (unsigned char) d[2] == 0xff; }
bool passes(const std::string& d) {
    return starts_with(d, std::string_view("\x89PNG\r\n\x1a\n", 8)) || starts_with(d, "BM") || starts_with(d, "GIF87a") ||
           starts_with(d, "GIF89a");
}

/// a JPEG's EXIF Orientation (1 when it has none): stb_image ignores the tag (#1229)
int exif_orientation(const std::string& d) {
    auto u16 = [&](size_t at, bool le) -> uint32_t {
        if (at + 2 > d.size()) return 0;
        const auto a = (unsigned char) d[at], b = (unsigned char) d[at + 1];
        return le ? (uint32_t) (a | b << 8) : (uint32_t) (a << 8 | b);
    };
    auto u32 = [&](size_t at, bool le) -> uint32_t { return le ? u16(at, le) | u16(at + 2, le) << 16 : u16(at, le) << 16 | u16(at + 2, le); };
    size_t at = 2;
    while (at + 4 <= d.size() && (unsigned char) d[at] == 0xff) {
        const auto marker = (unsigned char) d[at + 1];
        if (marker == 0xd9 || marker == 0xda) break;   // the end, or the image data
        const uint32_t len = u16(at + 2, false);
        if (len < 2) break;
        if (marker == 0xe1 && d.compare(at + 4, 6, std::string("Exif\0\0", 6)) == 0) {
            const size_t tiff = at + 10;
            if (tiff + 8 > d.size()) return 1;
            const bool le = d[tiff] == 'I';
            const size_t ifd = tiff + u32(tiff + 4, le);
            const uint32_t count = u16(ifd, le);
            for (uint32_t i = 0; i < count; ++i) {
                const size_t e = ifd + 2 + 12 * (size_t) i;
                if (e + 12 > d.size()) break;
                if (u16(e, le) == 0x0112) return (int) u16(e + 8, le);
            }
            return 1;
        }
        at += 2 + len;
    }
    return 1;
}

/// a picture over kImagePixelsMax, as Pillow words it
std::string too_many_pixels(uint64_t w, uint64_t h) {
    return "Image size (" + std::to_string(w * h) + " pixels) exceeds limit of " + std::to_string(kImagePixelsMax) +
           " pixels, could be decompression bomb DOS attack.";
}

struct Pixels {
    int w = 0, h = 0;
    std::vector<unsigned char> rgb;   // 3 bytes a pixel, rows top to bottom
};

/// RGBA pixels -> RGB, transparent areas white (not black)
Pixels on_white(const unsigned char* rgba, int w, int h) {
    Pixels p{w, h, std::vector<unsigned char>((size_t) w * h * 3)};
    for (size_t i = 0; i < (size_t) w * h; ++i) {
        const unsigned a = rgba[4 * i + 3];
        for (int c = 0; c < 3; ++c) p.rgb[3 * i + c] = (unsigned char) ((rgba[4 * i + c] * a + 255 * (255 - a) + 127) / 255);
    }
    return p;
}

/// Pillow's exif_transpose for orientations 2-8
Pixels upright(const Pixels& s, int orientation) {
    const bool swap = orientation >= 5;
    Pixels o{swap ? s.h : s.w, swap ? s.w : s.h, std::vector<unsigned char>(s.rgb.size())};
    for (int y = 0; y < o.h; ++y)
        for (int x = 0; x < o.w; ++x) {
            int sx = x, sy = y;
            switch (orientation) {
                case 2: sx = s.w - 1 - x; break;
                case 3: sx = s.w - 1 - x; sy = s.h - 1 - y; break;
                case 4: sy = s.h - 1 - y; break;
                case 5: sx = y; sy = x; break;
                case 6: sx = y; sy = s.h - 1 - x; break;
                case 7: sx = s.w - 1 - y; sy = s.h - 1 - x; break;
                case 8: sx = s.w - 1 - y; sy = x; break;
                default: break;
            }
            std::memcpy(&o.rgb[3 * ((size_t) y * o.w + x)], &s.rgb[3 * ((size_t) sy * s.w + sx)], 3);
        }
    return o;
}

/// a 24-bit BMP (the decoder reads it; the converted picture's lossless form)
std::string bmp(const Pixels& p) {
    const size_t row = ((size_t) p.w * 3 + 3) & ~(size_t) 3, size = 54 + row * p.h;
    std::string out(size, '\0');
    auto put32 = [&](size_t at, uint32_t v) {
        for (int i = 0; i < 4; ++i) out[at + i] = (char) (v >> (8 * i));
    };
    out[0] = 'B';
    out[1] = 'M';
    put32(2, (uint32_t) size);
    put32(10, 54);
    put32(14, 40);
    put32(18, (uint32_t) p.w);
    put32(22, (uint32_t) p.h);   // positive: rows bottom to top
    out[26] = 1;
    out[28] = 24;
    put32(34, (uint32_t) (row * p.h));
    for (int y = 0; y < p.h; ++y) {
        char* dst = out.data() + 54 + row * (size_t) (p.h - 1 - y);
        const unsigned char* src = &p.rgb[(size_t) y * p.w * 3];
        for (int x = 0; x < p.w; ++x) {
            dst[3 * x] = (char) src[3 * x + 2];
            dst[3 * x + 1] = (char) src[3 * x + 1];
            dst[3 * x + 2] = (char) src[3 * x];
        }
    }
    return out;
}

#ifdef _WIN32
/// the formats stb_image does not read (WebP, TIFF, ICO, HEIF ...) through Windows' own codecs
std::optional<Pixels> wic_decode(const std::string& data, std::string& why) {
    const HRESULT init = CoInitializeEx(nullptr, COINIT_MULTITHREADED);
    struct Uninit {
        bool on;
        ~Uninit() {
            if (on) CoUninitialize();
        }
    } uninit{SUCCEEDED(init)};
    IWICImagingFactory* factory = nullptr;
    IWICStream* stream = nullptr;
    IWICBitmapDecoder* decoder = nullptr;
    IWICBitmapFrameDecode* frame = nullptr;
    IWICBitmapSource* rgba = nullptr;
    auto release = [&] {
        for (IUnknown* u : std::initializer_list<IUnknown*>{rgba, frame, decoder, stream, factory})
            if (u != nullptr) u->Release();
    };
    std::optional<Pixels> out;
    UINT w = 0, h = 0;
    if (SUCCEEDED(CoCreateInstance(CLSID_WICImagingFactory, nullptr, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&factory))) &&
        SUCCEEDED(factory->CreateStream(&stream)) &&
        SUCCEEDED(stream->InitializeFromMemory((BYTE*) data.data(), (DWORD) data.size())) &&
        SUCCEEDED(factory->CreateDecoderFromStream(stream, nullptr, WICDecodeMetadataCacheOnDemand, &decoder)) &&
        SUCCEEDED(decoder->GetFrame(0, &frame)) && SUCCEEDED(frame->GetSize(&w, &h)) && w > 0 && h > 0 &&
        (uint64_t) w * h <= kImagePixelsMax &&
        SUCCEEDED(WICConvertBitmapSource(GUID_WICPixelFormat32bppRGBA, frame, &rgba)) && SUCCEEDED(rgba->GetSize(&w, &h)) &&
        w > 0 && h > 0) {
        std::vector<unsigned char> px((size_t) w * h * 4);
        if (SUCCEEDED(rgba->CopyPixels(nullptr, w * 4, (UINT) px.size(), px.data()))) out = on_white(px.data(), (int) w, (int) h);
    }
    if (!out) why = (uint64_t) w * h > kImagePixelsMax ? too_many_pixels(w, h) : "not a format this computer's image codecs read";
    release();
    return out;
}
#endif

std::string vision_start_error(const std::string& line, const std::string& log) {
    const std::string text(strip(line));
    std::string tail;
    if (!log.empty()) {
        std::ifstream f(u8_path(log), std::ios::binary | std::ios::ate);
        if (f) {
            const std::streamoff end = f.tellg();
            f.seekg(std::max<std::streamoff>(0, end - 16384));
            tail.assign(std::istreambuf_iterator<char>(f), {});
        }
    }
    if (lower(text + tail).find("busy or unavailable") != std::string::npos)
        return "the vision encoder did not start: the GPU refused a second CUDA context (\"CUDA-capable device(s) "
               "is/are busy or unavailable\"). This is usually the compute mode Exclusive_Process: only one process "
               "may use the card, and the engine already holds it. Check with: nvidia-smi --query-gpu=compute_mode "
               "--format=csv; set it back with: sudo nvidia-smi -c DEFAULT (or run the encoder on another card). "
               "First line from the encoder: " + (text.empty() ? std::string("(none)") : text);
    return "the vision encoder did not start: " + text;
}

/// #480: a directory for the encoder's files whose path has no space (the system temp directory while it has none,
/// then the working directory, then the system drive's root; the temp directory as the last resort)
std::string work_dir() {
    std::random_device rd;
    auto make = [&](const fs::path& parent) -> std::optional<fs::path> {
        for (int i = 0; i < 16; ++i) {
            char name[32];
            std::snprintf(name, sizeof name, "lamina-vision-%08x", (unsigned) rd());
            std::error_code ec;
            const fs::path d = parent / name;
            if (fs::create_directory(d, ec)) return d;
            if (ec) return std::nullopt;
        }
        return std::nullopt;
    };
    std::vector<fs::path> parents = {u8_path(temp_dir())};
    std::error_code ec;
    parents.push_back(fs::current_path(ec));
#ifdef _WIN32
    const char* drive = std::getenv("SystemDrive");
    parents.push_back(u8_path(std::string(drive != nullptr ? drive : "C:") + "\\"));
#endif
    for (const auto& parent : parents) {
        const auto d = make(parent);
        if (!d) continue;
        if (u8_str(*d).find(' ') == std::string::npos) return u8_str(*d);
        fs::remove(*d, ec);
    }
    const auto d = make(u8_path(temp_dir()));
    if (!d) throw std::runtime_error("no directory for the image encoder's files could be made");
    return u8_str(*d);
}

}  // namespace

std::string combined_embeddings_path(const std::string& vision_dir, uintmax_t nbytes) {
    const std::string name = "req-" + random_hex(12) + ".sve";
#ifndef _WIN32
    std::error_code ec;
    const fs::path shm = "/dev/shm";
    if (fs::is_directory(shm, ec) && ::access("/dev/shm", W_OK) == 0) {
        const fs::space_info sp = fs::space(shm, ec);
        if (!ec && sp.available > 2 * nbytes + (16u << 20))
            return "/dev/shm/lamina-vision-" + std::to_string(::getpid()) + "-" + name;
    }
#else
    (void) nbytes;
#endif
    return u8_str(u8_path(vision_dir) / name);
}

void write_temporary(const std::string& path, const std::vector<std::string>& parts) {
#ifdef _WIN32
    const std::wstring w = u8_path(path).wstring();
    const HANDLE h = CreateFileW(w.c_str(), GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr, CREATE_ALWAYS,
                                 FILE_ATTRIBUTE_TEMPORARY, nullptr);
    if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("the image embeddings could not be written: " + path);
    struct Close {
        HANDLE h;
        ~Close() { CloseHandle(h); }
    } close{h};
    for (const auto& part : parts) {
        const std::string data = read_file(u8_path(part));
        DWORD n = 0;
        if (!WriteFile(h, data.data(), (DWORD) data.size(), &n, nullptr) || n != data.size())
            throw std::runtime_error("the image embeddings could not be written: " + path);
    }
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
    if (fd < 0) throw std::runtime_error("the image embeddings could not be written: " + path);
    for (const auto& part : parts) {
        const std::string data = read_file(part);
        for (size_t at = 0; at < data.size();) {
            const ssize_t n = ::write(fd, data.data() + at, data.size() - at);
            if (n <= 0) {
                ::close(fd);
                throw std::runtime_error("the image embeddings could not be written: " + path);
            }
            at += (size_t) n;
        }
    }
    ::close(fd);
#endif
}

std::optional<double> timeout_env(const char* name, double dflt) {
    double v = dflt;
    if (const char* s = std::getenv(name)) {
        try {
            v = std::stod(s);
        } catch (const std::exception&) {
            v = dflt;
        }
    }
    if (v <= 0) return std::nullopt;
    return v;
}

bool network_path(const std::string& path) {
    const std::string p = replace_all(path, "/", "\\");
    return starts_with(p, "\\\\") || starts_with(p, "\\??\\");
}

std::vector<std::string> vision_reader_args(const json& cfg) {
    // the encoder runs in its own directory (#480): a path with a directory part is made absolute
    auto absolute = [](const std::string& p) {
        const fs::path x = u8_path(p);
        return x.has_parent_path() ? u8_str(fs::absolute(x)) : p;
    };
    std::vector<std::string> args = {absolute(jstr(cfg, "exe")), "--mmproj", absolute(jstr(cfg, "mmproj")), "--model",
                                      absolute(jstr(cfg, "model"))};
    auto num = [](const json& v) { return v.is_number_integer() ? std::to_string(v.get<int64_t>()) : py_dumps(v); };
    if (truthy(jget(cfg, "gpu"))) args.push_back("--gpu");
    if (truthy(jget(cfg, "threads"))) args.insert(args.end(), {"--threads", num(cfg["threads"])});
    if (truthy(jget(cfg, "max_tokens"))) args.insert(args.end(), {"--max-tokens", num(cfg["max_tokens"])});
    if (truthy(jget(cfg, "min_tokens"))) args.insert(args.end(), {"--min-tokens", num(cfg["min_tokens"])});   // #767
    return args;
}

Vision::Vision(const json& cfg, std::string log, std::optional<Environment> env, bool lazy)
    : dir(work_dir()), args_(vision_reader_args(cfg)), log_(std::move(log)), env_(std::move(env)) {
    if (!lazy) start();
}

Vision::~Vision() {
    close();
    std::error_code ec;
    fs::remove_all(u8_path(dir), ec);
}

void Vision::start() {
    SpawnOptions o;
    o.args = args_;
    o.cwd = dir;   // the ENC lines name the files relative to it (#480: no space in the line)
    o.env = env_;
    o.in = SpawnOptions::pipe;
    o.out = SpawnOptions::pipe;
    if (log_.empty()) {
        o.err = SpawnOptions::null;
    } else {
        o.err = SpawnOptions::file;
        o.err_file = log_;
    }
    o.own_group = true;
    proc_ = std::make_unique<Process>(o);
    std::string line;
    try {
        read_line(line, kReadyS.value_or(-1), "the vision encoder did not start");
    } catch (const std::runtime_error&) {   // silent for VISION_READY_S: ended, and not left half-started
        proc_.reset();
        stopped_ = true;
        throw;
    }
    if (!starts_with(line, "READY")) {
        proc_.reset();   // killed
        stopped_ = true;
        throw std::runtime_error(vision_start_error(line, log_));
    }
    stopped_ = false;
}

int Vision::read_line(std::string& line, double timeout_s, const std::string& what) {
    // #1317: on a timeout the encoder is killed, so the next request starts a fresh one
    const int r = proc_->read_line(line, timeout_s);
    if (r == 0) {
        proc_->kill();
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.0f", timeout_s);
        throw std::runtime_error(what + ": it said nothing for " + buf + " s");
    }
    if (r < 0) line.clear();
    return r;
}

bool Vision::alive() { return proc_ != nullptr && !stopped_ && !proc_->poll(); }

void Vision::unload() {
    close();
    stopped_ = true;
}

void Vision::restart() {
    if (proc_ != nullptr) proc_->kill();
    proc_.reset();
    stopped_ = true;
    start();
}

void Vision::close() {
    std::unique_ptr<Process> p = std::move(proc_);
    stopped_ = true;
    if (p == nullptr) return;
    if (p->write("QUIT\n") && p->wait(10)) return;
    p->kill();
    p->wait(10);
}

std::string Vision::load(const std::string& source) {
    if (starts_with(source, "data:")) {
        const size_t comma = source.find(',');
        try {
            if (comma == std::string::npos) throw std::invalid_argument("list index out of range");
            return b64decode(std::string_view(source).substr(comma + 1));
        } catch (const std::invalid_argument& e) {
            throw std::invalid_argument(std::string("the image's data: URL could not be read (") + e.what() + ")");
        }
    }
    if (starts_with(source, "http://") || starts_with(source, "https://")) return download(source);
    const bool file_url = starts_with(source, "file://");
    const std::string path = file_url ? source.substr(7) : source;
    // refused before anything looks at the path: file://host/share/x.png is \\host\share\x.png ("file://C:/x.png", a
    // drive, is a client's spelling of file:///C:/x.png)
    std::string host;
    if (file_url) host = path.substr(0, path.find_first_of("/\\"));
    const bool drive = host.size() == 2 && std::isalpha((unsigned char) host[0]) && host[1] == ':';
    if (network_path(path) || (!host.empty() && !drive && lower(host) != "localhost"))
        throw std::invalid_argument("an image file must be on this computer: network paths (\\\\host\\share, "
                                    "//host/share, file://host/...) are not read");
    std::error_code ec;
    if (!path.empty() && fs::is_regular_file(u8_path(path), ec)) {
        if (const uintmax_t size = fs::file_size(u8_path(path), ec); !ec && size > kImageBytesMax)
            throw std::invalid_argument("the image file " + path + " is over " + std::to_string(kImageBytesMax >> 20) + " MiB");
        try {
            return read_file(u8_path(path));
        } catch (const std::exception& e) {
            throw std::invalid_argument("the image could not be read from " + path + " (" + e.what() + ")");
        }
    }
    throw std::invalid_argument("an image must be a data: URL, an http(s) URL or a local file path");
}

std::string Vision::download(const std::string& url) {
    // at most kImageBytesMax: a huge or endless response is not read whole; one that cannot be read is a 400 that says so
    const std::string too_big = "the image URL's file is over " + std::to_string(kImageBytesMax >> 20) + " MiB";
    HttpReply r;
    try {
        r = http_request("GET", url, {{"User-Agent", "lamina"}}, "", 60, kImageBytesMax);
    } catch (const HttpError& e) {
        if (std::string(e.what()).find("larger than") != std::string::npos) throw std::invalid_argument(too_big);
        throw std::invalid_argument(std::string("the image URL could not be read: ") + e.what());
    }
    if (r.status >= 400)
        throw std::invalid_argument("the image URL could not be read: HTTP Error " + std::to_string(r.status) + ": " + r.reason);
    const std::string size = r.header("content-length");
    if (!size.empty() && std::all_of(size.begin(), size.end(), ::isdigit) && std::stoull(size) > kImageBytesMax)
        throw std::invalid_argument(too_big);
    return r.body;
}

std::string Vision::normalize(const std::string& data) {
    // a header over the limit is refused before a decoder holds its pixels: this process's, or the encoder's for the
    // formats that pass through
    int w = 0, h = 0, n = 0;
    if (stbi_info_from_memory((const stbi_uc*) data.data(), (int) data.size(), &w, &h, &n) &&
        (uint64_t) w * (uint64_t) h > kImagePixelsMax)
        throw std::invalid_argument("the image could not be read (" + too_many_pixels((uint64_t) w, (uint64_t) h) + ")");
    const bool jpeg = is_jpeg(data);
    const int orientation = jpeg ? exif_orientation(data) : 1;
    if ((jpeg && (orientation == 0 || orientation == 1 || orientation > 8)) || passes(data)) return data;
    unsigned char* px = stbi_load_from_memory((const stbi_uc*) data.data(), (int) data.size(), &w, &h, &n, 4);
    std::optional<Pixels> p;
    if (px != nullptr) {
        p = on_white(px, w, h);
        stbi_image_free(px);
    } else {
        std::string why = stbi_failure_reason() != nullptr ? stbi_failure_reason() : "unknown format";
#ifdef _WIN32
        p = wic_decode(data, why);
#endif
        if (!p) throw std::invalid_argument("the image could not be read (" + why + ")");
    }
    if (jpeg) *p = upright(*p, orientation);   // the camera's orientation tag, applied (#1229)
    return bmp(*p);
}

std::vector<std::pair<std::string, int64_t>> Vision::encode_all(const std::vector<std::pair<std::string, bool>>& sources,
                                                                Cancel* cancel) {
    std::vector<std::pair<std::string, int64_t>> out;
    for (const auto& [src, is_bytes] : sources) {
        if (cancel != nullptr && cancel->is_set()) throw ClientGone();
        std::vector<std::string> keep;
        for (const auto& [p, n] : out) keep.push_back(p);
        out.push_back(encode(src, is_bytes, keep));
    }
    return out;
}

std::pair<std::string, int64_t> Vision::encode(const std::string& source, bool is_bytes, const std::vector<std::string>& keep) {
    const std::string data = normalize(is_bytes ? source : load(source));
    const std::string key = hash_sha256_hex(data.data(), data.size()).substr(0, 32);
    std::lock_guard<std::mutex> lk(mu_);
    for (auto it = cache_.begin(); it != cache_.end(); ++it)
        if (it->first == key) {
            auto hit = *it;   // most recently used last
            cache_.erase(it);
            cache_.push_back(hit);
            return hit.second;
        }
    const fs::path img = u8_path(dir) / (key + ".img"), out = u8_path(dir) / (key + ".sve");
    write_file(img, data);
    std::string line;
    struct Unlink {   // #352: also when the encoder's pipe is gone
        fs::path p;
        ~Unlink() {
            std::error_code ec;
            fs::remove(p, ec);
        }
    } unlink{img};
    if (proc_ == nullptr || !proc_->write("ENC " + u8_str(img.filename()) + " " + u8_str(out.filename()) + "\n")) {
        stopped_ = true;
        throw std::invalid_argument("the image could not be read: the vision encoder stopped");
    }
    try {
        read_line(line, kEncodeS.value_or(-1), "the image could not be read");
    } catch (const std::runtime_error& e) {   // silent for VISION_ENCODE_S: ended (#1317); alive() is false now
        stopped_ = true;
        throw std::invalid_argument(e.what());
    }
    line = std::string(strip(line));
    if (!starts_with(line, "OK"))
        throw std::invalid_argument("the image could not be read: " +
                                    (starts_with(line, "ERR") ? line.substr(std::min<size_t>(4, line.size())) : "the vision encoder stopped"));
    const std::vector<std::string> f = split(line, " ");
    int64_t ntok = 0;
    try {
        ntok = f.size() > 1 ? std::stoll(f[1]) : 0;
    } catch (const std::exception&) {
        throw std::invalid_argument("the image could not be read: " + line);
    }
    const std::pair<std::string, int64_t> result{u8_str(out), ntok};
    cache_.push_back({key, result});
    std::vector<std::string> held = keep;
    held.push_back(u8_str(out));
    for (auto it = cache_.begin(); it != cache_.end() && cache_.size() > 64;) {   // oldest first, never one in use
        if (std::find(held.begin(), held.end(), it->second.first) == held.end()) {
            std::error_code ec;
            fs::remove(u8_path(it->second.first), ec);
            it = cache_.erase(it);
        } else {
            ++it;
        }
    }
    return result;
}

}  // namespace lamina
