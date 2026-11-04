// src/ngram/ple_reader_test.cpp - plan v0.3 P2: the direct SSD row reader against ground truth.
//
//   ple_reader_test --selftest [--dir D]        synthetic table file; CPU and disk only, no model, no GPU
//   ple_reader_test --gguf SHARD2 [--rows N]    the real table: Direct vs Mmap bytes for N random rows (+ the
//                                               16 rows of every token in --tokens FILE), with read latencies
//
// Every row the synthetic table holds encodes its own index, so a wrong offset, a straddle mishandled or a
// dedup slot mixed up shows as a mismatch rather than as plausible data.
#include "lamina/kernels/ngram.hpp"
#include "lamina/ngram/ple_reader.hpp"
#include "lamina/platform/direct_file.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace ng = lamina::ngram;
namespace k = lamina::kernels;
using lamina::platform::now_us;

namespace {

int g_fail = 0;
#define CHECK(c, ...)                                            \
    do {                                                         \
        if (!(c)) {                                              \
            std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
            std::fprintf(stderr, __VA_ARGS__);                   \
            std::fprintf(stderr, "\n");                          \
            ++g_fail;                                            \
        }                                                        \
    } while (0)

constexpr uint64_t HEADER = 192;   // the real shard's data offset, so rows are misaligned the same way

void expected_row(uint32_t row, uint32_t rb, uint8_t* out) {
    for (uint32_t b = 0; b < rb; ++b) out[b] = (uint8_t) ((row * 2654435761u + b * 97u) >> 7);
    std::memcpy(out, &row, 4);
}

bool make_table(const std::string& path, uint32_t rows, uint32_t rb) {
    std::ofstream f(path, std::ios::binary);
    std::vector<uint8_t> head(HEADER, 0xAB);
    f.write((const char*) head.data(), (std::streamsize) head.size());
    std::vector<uint8_t> r(rb);
    for (uint32_t i = 0; i < rows; ++i) {
        expected_row(i, rb, r.data());
        f.write((const char*) r.data(), rb);
    }
    return (bool) f;
}

bool check_rows(ng::PleReader& rd, const std::vector<uint32_t>& rows, uint32_t n_rows, uint32_t rb, const char* what) {
    std::vector<uint8_t> out(rows.size() * rb, 0xCC);
    std::string err;
    const auto t = rd.issue(rows.data(), rows.size(), out.data());
    if (!rd.collect(t, err)) { CHECK(false, "%s: collect failed: %s", what, err.c_str()); return false; }
    std::vector<uint8_t> want(rb);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (rows[i] >= n_rows) std::memset(want.data(), 0, rb);
        else expected_row(rows[i], rb, want.data());
        if (std::memcmp(want.data(), &out[i * rb], rb) != 0) {
            CHECK(false, "%s: row %u (index %zu) differs", what, rows[i], i);
            return false;
        }
    }
    return true;
}

/// BOTH FORMATS OF THE TABLE, END TO END: IQ4_NL and Q8_0 (the unsloth files), the types `PleTable::open` takes. A
/// minimal PLE-only GGUF holds `rows` rows of that format's row size, each filled with bytes that encode the row, and
/// every I/O mode must return exactly what `ple_dequant_row` makes of those bytes - so the row width, the offset and
/// the type have to be right for both formats, not only for IQ4_NL.
struct Format {
    const char* name;
    uint32_t gguf_type;
    uint32_t row_bytes;
};
constexpr Format kFormats[] = {{"IQ4_NL", 20, k::PLE_ROW_BYTES}, {"Q8_0", 8, k::PLE_ROW_BYTES_Q8_0}};
constexpr Format kQ5_0 = {"Q5_0", 6, 110};   // a table type other GGUFs have, which `open` refuses

constexpr k::PleIo kModes[] = {k::PleIo::Direct, k::PleIo::Mmap, k::PleIo::Ram};
const char* mode_name(k::PleIo m) { return m == k::PleIo::Direct ? "direct" : m == k::PleIo::Mmap ? "mmap" : "ram"; }

/// `path` opened as `io` says; in Ram mode the table is read into RAM first, so the rows come from there.
bool open_table(k::PleTable& t, const std::string& path, const k::PleIoOptions& io, std::string& err) {
    if (!t.open(path, err, io)) return false;
    if (io.mode != k::PleIo::Ram) return true;
    if (!t.start_ram_load(err)) return false;
    t.wait_ram_load();
    if (!t.ram_ready()) { err = "the table was not read into RAM"; return false; }
    return true;
}

// `claimed_rows` is the row count the header states; `rows` rows are actually written (equal unless a test lies)
std::string write_format_table(const std::string& dir, const Format& f, uint32_t rows, uint64_t claimed_rows = 0) {
    const std::string path = dir + "/ple_format_selftest.gguf";
    std::ofstream out(path, std::ios::binary);
    if (!out) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return {}; }
    const auto le = [](auto value, int bytes) {
        std::string s((size_t) bytes, '\0');
        for (int i = 0; i < bytes; ++i) s[(size_t) i] = (char) ((value >> (8 * i)) & 0xff);
        return s;
    };
    const auto str = [&](const char* s) { return le((uint64_t) std::strlen(s), 8) + std::string(s); };
    std::string head = "GGUF" + le(3u, 4) + le(1ull, 8) + le(1ull, 8);
    head += str("general.architecture") + le(8u, 4) + str("lamina-ple");
    head += str("per_layer_token_embd.weight") + le(2u, 4);
    head += le((uint64_t) k::PLE_HEAD_DIM, 8) + le(claimed_rows != 0 ? claimed_rows : (uint64_t) rows, 8);
    head += le(f.gguf_type, 4) + le(0ull, 8);
    while (head.size() % 32) head += '\0';
    out.write(head.data(), (std::streamsize) head.size());
    std::vector<uint8_t> r(f.row_bytes);
    for (uint32_t i = 0; i < rows; ++i) {
        expected_row(i, f.row_bytes, r.data());
        out.write((const char*) r.data(), (std::streamsize) f.row_bytes);
    }
    out.close();
    return out ? path : std::string();
}

void format_round_trip(const std::string& dir, const Format& f, uint32_t rows) {
    const std::string path = write_format_table(dir, f, rows);
    if (path.empty()) { CHECK(false, "%s: no table was written", f.name); return; }
    std::mt19937 rng(13);
    std::vector<uint32_t> want(rows);
    for (uint32_t i = 0; i < rows; ++i) want[i] = rng() % rows;
    want[0] = 0;
    want[1] = rows - 1;
    for (k::PleIo mode : kModes) {
        const char* name = mode_name(mode);
        k::PleTable t;
        std::string err;
        k::PleIoOptions io;
        io.mode = mode;
        io.max_inflight = 8;
        io.cache_rows = 0;
        CHECK(open_table(t, path, io, err), "%s: open in %s mode: %s", f.name, name, err.c_str());
        if (g_fail) break;
        CHECK(t.rows() == rows, "%s: rows %llu, not %u", f.name, (unsigned long long) t.rows(), rows);
        std::vector<uint8_t> raw(f.row_bytes);
        float got[k::PLE_HEAD_DIM], ref[k::PLE_HEAD_DIM];
        for (uint32_t r : want) {
            t.read_row(r, got);
            expected_row(r, f.row_bytes, raw.data());
            k::ple_dequant_row((int) f.gguf_type, raw.data(), ref);
            if (std::memcmp(got, ref, sizeof got) != 0) {
                CHECK(false, "%s: %s: row %u differs from the format's own dequantizer", f.name, name, r);
                break;
            }
        }
        // the 16-row path a token takes, one batch
        float tok[16 * k::PLE_HEAD_DIM];
        uint32_t rows16[16];
        for (int h = 0; h < 16; ++h) rows16[h] = want[(size_t) h + 2];
        t.gather(rows16, tok);
        for (int h = 0; h < 16; ++h) {
            expected_row(rows16[h], f.row_bytes, raw.data());
            k::ple_dequant_row((int) f.gguf_type, raw.data(), ref);
            CHECK(std::memcmp(tok + (size_t) h * k::PLE_HEAD_DIM, ref, sizeof ref) == 0, "%s: %s: gather head %d differs", f.name, name, h);
        }
        // a prompt chunk: gather_batch over more rows than one prefetch window (4096), one of them past the table's end
        {
            const size_t nt = 700, nr = nt * 16;     // 11200 rows, three windows
            std::vector<uint32_t> br(nr);
            for (size_t i = 0; i < nr; ++i) br[i] = want[i % want.size()];
            br[5000] = rows + 7;                     // out of range: zeros, as read_row gives
            std::vector<float> bo(nr * k::PLE_HEAD_DIM);
            std::string e2;
            CHECK(t.gather_batch(br.data(), nt, bo.data(), e2), "%s: %s: gather_batch: %s", f.name, name, e2.c_str());
            size_t bad = 0;
            for (size_t i = 0; i < nr && !g_fail; ++i) {
                if (br[i] >= rows) {
                    for (int d = 0; d < k::PLE_HEAD_DIM; ++d) bad += bo[i * k::PLE_HEAD_DIM + d] != 0.0f;
                    continue;
                }
                expected_row(br[i], f.row_bytes, raw.data());
                k::ple_dequant_row((int) f.gguf_type, raw.data(), ref);
                bad += std::memcmp(bo.data() + i * k::PLE_HEAD_DIM, ref, sizeof ref) != 0;
            }
            CHECK(bad == 0, "%s: %s: gather_batch differs in %zu rows", f.name, name, bad);
        }
    }
    std::filesystem::remove(path);
}

/// A table whose header does not match the file is refused with a message, never read past its end: one row short,
/// one row long (a single-tensor shard must be filled exactly), and a row count that wraps the 64-bit size; so is a
/// table of a type this reader has no dequantizer for.
void bad_tables_are_refused(const std::string& dir) {
    struct Case { const Format* f; const char* what; uint32_t rows; uint64_t claimed; };
    std::vector<Case> cases;
    for (const Format& f : kFormats) {
        cases.push_back({&f, "claims one row more than the file holds", 500, 501});
        cases.push_back({&f, "claims one row less than the file holds", 500, 499});
        cases.push_back({&f, "claims a row count that wraps 2^64", 8, (uint64_t) 0x4000000000000000ull / 3 * 2});
    }
    cases.push_back({&kQ5_0, "is of another type", 500, 0});
    for (const auto& c : cases) {
        const std::string path = write_format_table(dir, *c.f, c.rows, c.claimed);
        if (path.empty()) { CHECK(false, "%s: no table was written", c.f->name); continue; }
        for (k::PleIo mode : kModes) {
            k::PleTable t;
            std::string err;
            k::PleIoOptions io;
            io.mode = mode;
            io.cache_rows = 0;
            CHECK(!t.open(path, err, io), "%s: a table that %s was accepted (%s)", c.f->name, c.what, mode_name(mode));
            CHECK(!err.empty(), "%s: refused without a message (%s)", c.f->name, c.what);
            CHECK(!t.is_open(), "%s: left open after refusing (%s)", c.f->name, c.what);
        }
        std::filesystem::remove(path);
    }
}

/// The read-ahead slots (`ahead`, the verify window's tokens): in every I/O mode a slot gives the bytes the mapped
/// reader gives for its rows and holds exactly those rows until they are collected, a slot still holding rows takes
/// others, the slots work beside a token in `issue` and are collected in any order, and each row is asked for once.
void ahead_slots_hold_their_rows(const std::string& dir) {
    const Format& f = kFormats[0];
    const std::string path = write_format_table(dir, f, 4096);
    if (path.empty()) { CHECK(false, "cannot write the read-ahead table"); return; }
    constexpr int S = k::PleTable::kAheadSlots, kTokens = 32, kReplaced = 20, kIssued = 30;
    const auto keys = [](int id) {                  // token `id`'s 16 rows
        std::vector<uint32_t> r;
        for (int h = 0; h < k::PLE_N_HEADS; ++h) r.push_back((uint32_t) (id * kTokens + h));
        return r;
    };
    const auto requests = [](const k::PleTable& t) {
        unsigned long long n = 0;
        const std::string s = t.io_report();
        return std::sscanf(s.c_str(), "ple io: %llu rows", &n) == 1 ? (uint64_t) n : ~0ull;
    };
    const size_t tok = (size_t) k::PLE_N_HEADS * k::PLE_HEAD_DIM;
    std::vector<float> ref(kTokens * tok);
    {
        k::PleTable mapped;
        k::PleIoOptions mio;
        mio.mode = k::PleIo::Mmap;
        std::string err;
        if (!mapped.open(path, err, mio)) { CHECK(false, "mapped open: %s", err.c_str()); return; }
        for (int id = 0; id < kTokens; ++id) mapped.gather(keys(id).data(), ref.data() + id * tok);
    }
    struct Arm { const char* name; k::PleIo mode; uint64_t cache; bool worker; };
    const Arm arms[] = {{"direct", k::PleIo::Direct, 0, false}, {"direct, worker", k::PleIo::Direct, 0, true},
                        {"direct, row cache", k::PleIo::Direct, 256, true}, {"mmap", k::PleIo::Mmap, 0, true},
                        {"ram", k::PleIo::Ram, 0, true}};
    const float sentinel = -12345.5f;
    std::vector<float> out(tok + 2);
    for (const Arm& a : arms) {
        k::PleTable t;
        k::PleIoOptions io;
        io.mode = a.mode;
        io.cache_rows = a.cache;
        io.io_thread = a.worker;
        io.max_inflight = 8;
        std::string err;
        if (!open_table(t, path, io, err)) { CHECK(false, "%s: open: %s", a.name, err.c_str()); continue; }
        const uint64_t rq = requests(t);
        for (int s = 0; s < S; ++s) t.ahead(s, keys(s).data());
        for (int s = 0; s < S; ++s)
            CHECK(t.ahead_holds(s, keys(s).data()) && !t.ahead_holds(s, keys(s + 1).data()), "%s: slot %d holds other rows", a.name, s);
        t.ahead(3, keys(kReplaced).data());          // a slot still holding rows takes others
        CHECK(t.ahead_holds(3, keys(kReplaced).data()) && !t.ahead_holds(3, keys(3).data()), "%s: slot 3 kept its earlier rows", a.name);
        CHECK(t.issue(keys(kIssued).data()), "%s: issue beside the slots", a.name);
        std::fill(out.begin(), out.end(), sentinel);
        CHECK(t.collect(out.data() + 1, err), "%s: collect: %s", a.name, err.c_str());
        CHECK(std::memcmp(out.data() + 1, ref.data() + kIssued * tok, tok * sizeof(float)) == 0, "%s: the issued token differs", a.name);
        for (int s = S - 1; s >= 0; --s) {           // collected in reverse order
            const int id = s == 3 ? kReplaced : s;
            std::fill(out.begin(), out.end(), sentinel);
            CHECK(t.ahead_collect(s, out.data() + 1, err), "%s: slot %d: %s", a.name, s, err.c_str());
            CHECK(std::memcmp(out.data() + 1, ref.data() + id * tok, tok * sizeof(float)) == 0,
                  "%s: slot %d differs from the mapped reader", a.name, s);
            CHECK(out.front() == sentinel && out.back() == sentinel, "%s: slot %d wrote outside the output", a.name, s);
            CHECK(!t.ahead_holds(s, keys(id).data()), "%s: slot %d still holds its rows after the collect", a.name, s);
        }
        err.clear();
        CHECK(!t.ahead_collect(0, out.data(), err) && !err.empty(), "%s: an empty slot was collected", a.name);
        t.ahead(-1, keys(0).data());                 // slots out of range are ignored
        t.ahead(S, keys(0).data());
        CHECK(!t.ahead_holds(S, keys(0).data()) && !t.ahead_collect(S, out.data(), err), "%s: slot %d was used", a.name, S);
        if (a.mode == k::PleIo::Direct)              // S slots, the replaced one's new rows and the issued token
            CHECK(requests(t) - rq == (uint64_t) (S + 2) * k::PLE_N_HEADS, "%s: %llu rows asked for, want %d", a.name,
                  (unsigned long long) (requests(t) - rq), (S + 2) * k::PLE_N_HEADS);
    }
    std::filesystem::remove(path);
}

void all_formats_round_trip(const std::string& dir) {
    bad_tables_are_refused(dir);
    ahead_slots_hold_their_rows(dir);
    for (const Format& f : kFormats) format_round_trip(dir, f, 2000);
    if (g_fail == 0) {
        std::printf("ple formats");
        for (const Format& f : kFormats) std::printf(" %s", f.name);
        std::printf(" (direct, mmap, ram) vs ple_dequant_row, bad tables refused, read-ahead slots: OK\n");
    }
}

// row_bytes: ng::ROW_BYTES (90, IQ4_NL) and k::PLE_ROW_BYTES_Q8_0 (170), the table's two row sizes, through the
// reader's one generic row_bytes path.
int selftest(const std::string& dir, uint32_t rb) {
    const uint32_t N = 500000;                          // 45-85 MB: large enough for thousands of distinct pages
    const std::string path = dir + "/ple_reader_selftest_" + std::to_string(rb) + ".bin";
    if (!make_table(path, N, rb)) { std::fprintf(stderr, "cannot write %s\n", path.c_str()); return 2; }
    std::mt19937 rng(7);
    for (bool thr : {false, true})
    for (uint64_t cache : {0ull, 4096ull}) {
        for (uint32_t inflight : {1u, 8u, 64u}) {
            ng::PleReader rd;
            std::string err;
            CHECK(rd.open(path, HEADER, N, inflight, cache, err, thr, rb), "open: %s", err.c_str());
            // decode-shaped tickets: 16 random rows
            for (int t = 0; t < 200; ++t) {
                std::vector<uint32_t> rows(16);
                for (auto& r : rows) r = rng() % N;
                check_rows(rd, rows, N, rb, "decode");
            }
            // rows that straddle a 4 KiB boundary: byte offset of row r is HEADER + rb * r
            std::vector<uint32_t> straddle;
            for (uint32_t r = 0; r < N && straddle.size() < 64; ++r) {
                const uint64_t a = HEADER + (uint64_t) r * rb;
                if (a / 4096 != (a + rb - 1) / 4096) straddle.push_back(r);
            }
            check_rows(rd, straddle, N, rb, "straddle");
            // duplicates, neighbours on one page, the first and last rows, and out-of-range rows
            check_rows(rd, {5, 5, 6, 7, 5, 0, N - 1, N, 0xFFFFFFFFu, 44, 45}, N, rb, "dedup/edges");
            // a prefill-shaped ticket much larger than the in-flight window
            std::vector<uint32_t> bulk(20000);
            for (auto& r : bulk) r = rng() % N;
            check_rows(rd, bulk, N, rb, "bulk");
            // two tickets in flight at once, collected in reverse order
            std::vector<uint32_t> a(16), b(16);
            for (auto& r : a) r = rng() % N;
            for (auto& r : b) r = rng() % N;
            std::vector<uint8_t> oa(16 * rb), ob(16 * rb);
            const auto ta = rd.issue(a.data(), 16, oa.data());
            const auto tb = rd.issue(b.data(), 16, ob.data());
            CHECK(rd.collect(tb, err) && rd.collect(ta, err), "two tickets: %s", err.c_str());
            std::vector<uint8_t> want(rb);
            for (int i = 0; i < 16; ++i) {
                expected_row(a[i], rb, want.data());
                CHECK(!std::memcmp(want.data(), &oa[i * rb], rb), "ticket a row %d", i);
                expected_row(b[i], rb, want.data());
                CHECK(!std::memcmp(want.data(), &ob[i * rb], rb), "ticket b row %d", i);
            }
            if (cache > 0) CHECK(rd.stats().cache_hits > 0, "the row cache never hit");
            CHECK(rd.cache_size() <= rd.cache_capacity(), "row cache exceeded its bound");
        }
    }
    // fault injection: a 3 ms delay must be observed, and must not change the bytes
    for (bool thr : {false, true}) {
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 0, err, thr, rb), "open: %s", err.c_str());
        rd.set_injected_delay_us(3000);
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        const double t0 = now_us();
        check_rows(rd, rows, N, rb, "delayed");
        CHECK(now_us() - t0 >= 3000, "injected delay not observed (%.0f us)", now_us() - t0);
        CHECK(rd.stats().late_injected > 0, "no read was held back");
    }
    // keep-alive: while rows are asked for, a page goes out after `period` without a read; it stops once the
    // window after the last issue has passed, starts again with the next issue (even one the row cache serves),
    // and never shows up as a row read or changes a row
    {
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 4096, err, true, rb), "open: %s", err.c_str());
        rd.set_keepalive(20.0, 0.5);
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        check_rows(rd, rows, N, rb, "keep-alive, first ticket");
        const uint64_t reads0 = rd.snapshot().reads;
        const auto nap = [](int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); };
        nap(300);
        const uint64_t k1 = rd.snapshot().keepalive_reads;
        CHECK(k1 >= 8 && k1 <= 20, "keep-alive: %llu reads in 300 ms at a 20 ms period", (unsigned long long) k1);
        CHECK(rd.snapshot().reads == reads0, "keep-alive reads were counted as row reads");
        nap(600);                                          // the 0.5 s window has passed
        const uint64_t k2 = rd.snapshot().keepalive_reads;
        nap(300);
        const uint64_t k3 = rd.snapshot().keepalive_reads;
        CHECK(k3 == k2, "keep-alive went on after the window (%llu -> %llu)", (unsigned long long) k2,
              (unsigned long long) k3);
        CHECK(k2 <= k1 + 20, "keep-alive: %llu reads by the end of a 0.5 s window", (unsigned long long) k2);
        check_rows(rd, rows, N, rb, "keep-alive, cached ticket");   // the row cache serves all 16: no row read
        CHECK(rd.snapshot().reads == reads0, "cached rows were read again");
        nap(200);
        CHECK(rd.snapshot().keepalive_reads >= k3 + 5, "the keep-alive did not start again (%llu -> %llu)",
              (unsigned long long) k3, (unsigned long long) rd.snapshot().keepalive_reads);
        std::vector<uint32_t> more(2000);
        for (auto& r : more) r = rng() % N;
        check_rows(rd, more, N, rb, "keep-alive, rows afterwards");
        rd.set_keepalive(0, 0.5);                           // off
        const uint64_t k4 = rd.snapshot().keepalive_reads;
        nap(200);
        CHECK(rd.snapshot().keepalive_reads <= k4 + 1, "keep-alive went on after it was turned off");
        rd.reset_stats();
        CHECK(rd.snapshot().keepalive_reads == 0 && rd.snapshot().keepalive_us_max == 0, "reset_stats kept keep-alive counts");
    }
    {   // the caller's thread does the reads: no worker, so no keep-alive
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 0, err, false, rb), "open: %s", err.c_str());
        rd.set_keepalive(20.0, 1.0);
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        check_rows(rd, rows, N, rb, "keep-alive, caller thread");
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        CHECK(rd.snapshot().keepalive_reads == 0, "keep-alive without the worker thread");
    }
    // a failed read: its ticket and every later one report the error; none waits for reads that never go out
    for (bool thr : {false, true}) {
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 0, err, thr, rb), "open: %s", err.c_str());
        rd.set_injected_failures(1);
        std::vector<uint32_t> rows(16);
        std::vector<uint8_t> out(16 * rb);
        for (int k = 0; k < 3; ++k) {
            for (auto& r : rows) r = rng() % N;
            err.clear();
            CHECK(!rd.collect(rd.issue(rows.data(), 16, out.data()), err) && !err.empty(),
                  "ticket %d after a failed read (%s): no error", k, thr ? "worker" : "caller thread");
        }
    }
    {   // an open that fails once the file is open: close() waits for no read, as none went out
        ng::PleReader rd;
        std::string err;
        CHECK(rd.open(path, HEADER, N, 16, 0, err, true, rb), "open: %s", err.c_str());
        CHECK(!rd.open(path, HEADER, (uint64_t) N * 4, 16, 0, err, true, rb), "a table past the file's end opened");
        CHECK(rd.open(path, HEADER, N, 16, 0, err, true, rb), "open after a failed one: %s", err.c_str());
        std::vector<uint32_t> rows(16);
        for (auto& r : rows) r = rng() % N;
        check_rows(rd, rows, N, rb, "after a failed open");
    }
    std::filesystem::remove(path);
    std::printf("ple_reader selftest (row_bytes=%u): %s\n", rb, g_fail ? "FAILED" : "OK");
    return g_fail ? 1 : 0;
}

int real(const std::string& gguf, int n_random, const std::string& tokens_path, uint32_t inflight, bool direct_first,
         bool direct_only, bool sync_submit) {
    k::PleTable mm, direct;
    std::string err;
    k::PleIoOptions mo;
    mo.mode = k::PleIo::Mmap;
    k::PleIoOptions dopt;
    dopt.cache_rows = 0;                               // measure the SSD, not the cache
    dopt.max_inflight = inflight;
    dopt.io_thread = !sync_submit;
    // `--direct-only` measures the direct path with NO mapping of the file alive anywhere in the process: a
    // live section on the same file forces the file system to keep cached and non-cached views coherent.
    if (!direct_only && !mm.open(gguf, err, mo)) { std::fprintf(stderr, "mmap open: %s\n", err.c_str()); return 2; }
    if (!direct.open(gguf, err, dopt)) { std::fprintf(stderr, "direct open: %s\n", err.c_str()); return 2; }
    std::vector<std::vector<uint32_t>> tickets;
    std::mt19937_64 rng(11);
    for (int t = 0; t < n_random / 16; ++t) {
        std::vector<uint32_t> r(16);
        for (auto& x : r) x = (uint32_t) (rng() % direct.rows());
        tickets.push_back(r);
    }
    if (!tokens_path.empty()) {
        std::ifstream f(tokens_path);
        std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        for (char& c : text) if (c == ',') c = ' ';
        std::istringstream in(text);
        std::vector<int32_t> ids;
        for (int32_t v; in >> v;) ids.push_back(v);
        const k::PleConsts C = k::ple_artifact_consts();
        for (size_t i = 0; i < ids.size(); ++i) {
            const int32_t prev[2] = {i >= 2 ? ids[i - 2] : k::TOKEN_NULL, i >= 1 ? ids[i - 1] : k::TOKEN_NULL};
            std::vector<uint32_t> r(16);
            k::ngram_rows(&ids[i], prev, 1, C, r.data());
            tickets.push_back(r);
        }
    }
    std::vector<float> a(k::NG_N_EMBD), b(k::NG_N_EMBD);
    double t_mm = 0, t_dir = 0, t_issue = 0;
    for (const auto& r : tickets) {
        for (int pass = 0; pass < 2; ++pass) {
            const bool do_direct = (pass == 0) == direct_first;
            const double t0 = now_us();
            if (!do_direct) {
                if (direct_only) continue;
                mm.gather(r.data(), a.data());
                t_mm += now_us() - t0;
            } else {
                const double ti = now_us();
                const bool issued = direct.issue(r.data());
                t_issue += now_us() - ti;
                if (!issued || !direct.collect(b.data(), err)) {
                    std::fprintf(stderr, "direct: %s\n", err.c_str());
                    return 1;
                }
                t_dir += now_us() - t0;
            }
        }
        if (!direct_only)
            CHECK(!std::memcmp(a.data(), b.data(), a.size() * sizeof(float)), "token rows differ between mmap and direct");
    }
    std::printf("tokens %zu: mmap %.1f us/token, direct %.1f us/token (issue on this thread %.1f us)\n%s\n",
                tickets.size(), t_mm / (double) tickets.size(), t_dir / (double) tickets.size(),
                t_issue / (double) tickets.size(), direct.io_report().c_str());
    std::printf("ple_reader real-table check: %s\n", g_fail ? "FAILED" : "OK (bit-identical)");
    return g_fail ? 1 : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::string gguf, dir = std::filesystem::temp_directory_path().string(), tokens;
    int rows = 20000;
    uint32_t inflight = 64;
    bool direct_first = false;
    bool direct_only = false;
    bool sync_submit = false;
    bool self = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--selftest") self = true;
        else if (a == "--dir" && i + 1 < argc) dir = argv[++i];
        else if (a == "--gguf" && i + 1 < argc) gguf = argv[++i];
        else if (a == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (a == "--tokens" && i + 1 < argc) tokens = argv[++i];
        else if (a == "--inflight" && i + 1 < argc) inflight = (uint32_t) std::atoi(argv[++i]);
        else if (a == "--direct-first") direct_first = true;
        else if (a == "--direct-only") direct_only = true;
        else if (a == "--sync") sync_submit = true;
        else { std::fprintf(stderr, "usage: ple_reader_test --selftest [--dir D] | --gguf SHARD2 [--rows N] [--tokens F]\n"); return 2; }
    }
    if (self) {
        all_formats_round_trip(dir);               // both formats, every I/O mode, vs ple_dequant_row
        if (g_fail != 0) return 1;
        const int r90 = selftest(dir, ng::ROW_BYTES);
        const int r170 = selftest(dir, k::PLE_ROW_BYTES_Q8_0);
        return r90 != 0 ? r90 : r170;
    }
    if (!gguf.empty()) return real(gguf, rows, tokens, inflight, direct_first, direct_only, sync_submit);
    std::fprintf(stderr, "nothing to do\n");
    return 2;
}
