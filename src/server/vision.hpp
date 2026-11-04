// src/server/vision.hpp - the resident image encoder (serve/server.py's Vision).
//
// lamina-vision (llama.cpp's mtmd and the model's mmproj file) reads "ENC <image> <out>" lines and writes each
// picture's embeddings; the results are cached by the picture's hash, so a conversation that sends the same picture
// every turn (most clients do) encodes it once.
#pragma once

#include "json.hpp"
#include "process.hpp"
#include "sync.hpp"

#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace lamina {

/// an image URL or file is read up to this much (the web app attaches pictures of up to 20 MB)
inline constexpr size_t kImageBytesMax = 32u << 20;
/// the pixels a picture the server decodes may have: Pillow's limit (twice Image.MAX_IMAGE_PIXELS), where it
/// refuses a decompression bomb
inline constexpr uint64_t kImagePixelsMax = 178956970;

class Vision {
public:
    /// the config's "vision" section (exe, mmproj, model, gpu, threads, max_tokens, min_tokens), the log its stderr
    /// is appended to ("": none), its environment; started now unless `lazy`
    Vision(const json& cfg, std::string log = "", std::optional<Environment> env = std::nullopt, bool lazy = false);
    /// for good (#914): the encoder ends and its directory (one .sve file per picture) goes with it
    ~Vision();
    bool alive();
    /// stops the encoder (its memory goes back); the encoded pictures stay cached on disk
    void unload();
    /// starts it again after an unload, or after it died; the cache is kept
    void restart();
    void close();
    /// a picture's bytes from a data: URL, an http(s) URL or a local file, a URL's or file's at most kImageBytesMax
    /// (std::invalid_argument when it cannot)
    static std::string load(const std::string& source);
    /// an image URL's bytes, at most kImageBytesMax (std::invalid_argument when it cannot be read)
    static std::string download(const std::string& url);
    /// the formats lamina-vision's decoder reads pass through; any other is converted, a JPEG stored turned (its EXIF
    /// Orientation) is turned upright (std::invalid_argument when it cannot be read, or has over kImagePixelsMax)
    static std::string normalize(const std::string& data);
    /// encode() for every picture of one request: the files of its earlier pictures are never evicted for a later
    /// one (#1072); a request whose client left encodes no more (#1999: ClientGone)
    std::vector<std::pair<std::string, int64_t>> encode_all(const std::vector<std::pair<std::string, bool>>& sources,
                                                            Cancel* cancel);
    /// (its embeddings file, its image tokens); `source`: what load() reads, or the picture's bytes (`is_bytes`);
    /// `keep`: files the caller still needs
    std::pair<std::string, int64_t> encode(const std::string& source, bool is_bytes = false,
                                           const std::vector<std::string>& keep = {});
    std::string dir;   ///< the encoder's files: a path without a space (#480: the ENC and GENI lines split at spaces)

private:
    void start();
    int read_line(std::string& line, double timeout_s, const std::string& what);

    std::vector<std::string> args_;
    std::string log_;
    std::optional<Environment> env_;
    std::unique_ptr<Process> proc_;
    bool stopped_ = true;
    std::mutex mu_;
    std::vector<std::pair<std::string, std::pair<std::string, int64_t>>> cache_;   ///< key -> (file, tokens), oldest first
};

/// the config's vision section as its args: the encoder runs in its own directory, so its paths are made absolute
std::vector<std::string> vision_reader_args(const json& cfg);
/// a path to another computer (\\host\share, //host/share, \\?\UNC\..., \??\UNC\...): Windows would sign in to it
bool network_path(const std::string& path);
/// #874: where one request's combined image-embeddings file goes: on Linux /dev/shm when it has room (the engine
/// reads it back at once), else the encoder's own directory
std::string combined_embeddings_path(const std::string& vision_dir, uintmax_t nbytes);
/// #874: `parts` (files) one after the other into `path`; on Windows as a short-lived file (FILE_ATTRIBUTE_TEMPORARY),
/// which normally never reaches the drive before it is deleted
void write_temporary(const std::string& path, const std::vector<std::string>& parts);
/// a timeout from the environment (LAMINA_VISION_READY_S ...): seconds, nullopt for no limit (0 or less)
std::optional<double> timeout_env(const char* name, double dflt);

}  // namespace lamina
