// tools/vision/gguf_meta.h - open a model GGUF for its vocabulary only, whatever its weight tensors are.
//
// The picture reader needs the text model for one thing: its tokenizer (and the header values that go with it).  llama.cpp's
// loader reads the whole tensor directory first and refuses a file with a tensor type it does not know - and a Gyro
// quantization (TQK6 / TQK7, ids 145 and 146) and other engine-private types are exactly that - so even a vocab-only
// load failed with "cannot open the text model".  lamina_open_text_model() therefore copies the key/value header of the
// GGUF (everything before the tensor directory) into a small temporary GGUF without tensors, and loads THAT vocab-only.
// The weights are never read, mapped or type-checked, and the split keys are dropped so the loader does not go looking
// for the other shards.  If the header cannot be read (a type this reader does not know), it falls back to loading the
// file itself, as before.
#pragma once

#include "gguf_header.h"
#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define LAMINA_GETPID _getpid
#else
#include <unistd.h>
#define LAMINA_GETPID getpid
#endif

namespace lamina_gguf {

// Copy the key/value header of `src` into `dst` as a GGUF with no tensors.  False (and `err`) when the header holds
// something this reader does not handle.
inline bool copy_meta(const std::string& src, const std::string& dst, std::string& err) {
    FILE* f = std::fopen(src.c_str(), "rb");
    if (!f) { err = "cannot open " + src; return false; }
    struct Closer { FILE* f; ~Closer() { std::fclose(f); } } closer{f};
    gguf_context* out = read_header(f, err);
    if (!out) return false;
    struct Freer { gguf_context* g; ~Freer() { gguf_free(g); } } freer{out};
    if (!gguf_write_to_file(out, dst.c_str(), true)) { err = "cannot write " + dst; return false; }
    return true;
}

}  // namespace lamina_gguf

// The text model for the picture reader: vocabulary and header only.  `how` says which way it was opened.
inline llama_model* lamina_open_text_model(const std::string& model, std::string* how = nullptr) {
    llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    const char* tmpdir = std::getenv("TMPDIR");
    if (!tmpdir || !*tmpdir) tmpdir = std::getenv("TEMP");
    if (!tmpdir || !*tmpdir) tmpdir = std::getenv("TMP");
    if (!tmpdir || !*tmpdir) tmpdir = "/tmp";
    const std::string meta = std::string(tmpdir) + "/lamina-vision-meta-" + std::to_string((long) LAMINA_GETPID()) + ".gguf";
    std::string err;
    llama_model* m = nullptr;
    if (lamina_gguf::copy_meta(model, meta, err)) {
        m = llama_model_load_from_file(meta.c_str(), mp);
        if (how) *how = m ? "header only" : "header copy was not accepted";
    } else if (how) *how = "header copy failed: " + err;
    std::remove(meta.c_str());
    if (!m) {   // the file as it is (a type of header this reader does not handle, or an llama.cpp that wants more)
        m = llama_model_load_from_file(model.c_str(), mp);
        if (how) *how += m ? ", loaded the file itself" : ", the file itself did not load either";
    }
    return m;
}
