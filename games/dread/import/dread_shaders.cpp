// Metroid Dread on-device .bshdat ("MSHD") Maxwell shader extractor.
// Verified across all 478 /system/shd/*.bshdat files in Base 1.0.0:
// 1,342 NVN control sections (0x98761234) paired 1:1 with 1,342 Maxwell graphics code blocks (0x12345678)
// -> 801 unique vertex + fragment stages (0 compute, 0 geometry, 0 tessellation).
#include "dread_shaders.h"

#include <sys/stat.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <direct.h>
#endif

namespace dread {

namespace {

struct Fail : std::runtime_error {
    using std::runtime_error::runtime_error;
};

constexpr uint32_t kCtrlMagic = 0x98761234u;
constexpr uint32_t kCodeGraphics = 0x12345678u;
constexpr uint32_t kCodeCompute = 0x12345679u;

inline uint32_t le32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
inline uint32_t be32(const uint8_t* p) {
    return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}
inline uint64_t up256(uint64_t x) { return (x + 0xffu) & ~uint64_t(0xffu); }

// First 16 hex chars of SHA-1 (matching spirv_harness naming convention).
std::string sha1_16(const uint8_t* data, size_t n) {
    uint32_t h[5] = {0x67452301u, 0xEFCDAB89u, 0x98BADCFEu, 0x10325476u, 0xC3D2E1F0u};
    auto rol = [](uint32_t x, int s) { return (x << s) | (x >> (32 - s)); };
    auto block = [&](const uint8_t* p) {
        uint32_t w[80];
        for (int i = 0; i < 16; i++) w[i] = be32(p + 4 * i);
        for (int i = 16; i < 80; i++) w[i] = rol(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
        uint32_t a = h[0], b = h[1], c = h[2], d = h[3], e = h[4];
        for (int i = 0; i < 80; i++) {
            uint32_t f, k;
            if (i < 20) f = (b & c) | (~b & d), k = 0x5A827999u;
            else if (i < 40) f = b ^ c ^ d, k = 0x6ED9EBA1u;
            else if (i < 60) f = (b & c) | (b & d) | (c & d), k = 0x8F1BBCDCu;
            else f = b ^ c ^ d, k = 0xCA62C1D6u;
            uint32_t t = rol(a, 5) + f + e + k + w[i];
            e = d, d = c, c = rol(b, 30), b = a, a = t;
        }
        h[0] += a, h[1] += b, h[2] += c, h[3] += d, h[4] += e;
    };
    size_t i = 0;
    for (; i + 64 <= n; i += 64) block(data + i);
    uint8_t tail[128] = {};
    size_t rest = n - i;
    std::memcpy(tail, data + i, rest);
    tail[rest] = 0x80;
    size_t len = (rest + 1 + 8 <= 64) ? 64 : 128;
    uint64_t bits = uint64_t(n) * 8;
    for (int j = 0; j < 8; j++) tail[len - 1 - j] = uint8_t(bits >> (8 * j));
    block(tail);
    if (len == 128) block(tail + 64);
    char out[41];
    for (int j = 0; j < 5; j++) std::snprintf(out + 8 * j, 9, "%08x", h[j]);
    return std::string(out, 16);
}

void make_dir(const std::string& p) {
#if defined(_WIN32)
    _mkdir(p.c_str());
#else
    mkdir(p.c_str(), 0755);
#endif
}

void write_file(const std::string& path, const uint8_t* p, size_t n) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(p, 1, n, f) != n) {
        if (f) std::fclose(f);
        throw Fail("cannot write " + path + " (" + std::strerror(errno) + ")");
    }
    if (std::fclose(f) != 0) throw Fail("cannot close " + path);
}

bool is_zero_range(const std::vector<uint8_t>& d, size_t lo, size_t hi) {
    for (size_t i = lo; i < hi; i++)
        if (d[i] != 0) return false;
    return true;
}

bool ctrl_fits_code(const std::vector<uint8_t>& d, size_t p, size_t c) {
    if (c + 0x880 > d.size() || p + 0x34 > d.size()) return false;
    uint32_t s = le32(d.data() + c + 0x704);
    uint32_t start = le32(d.data() + c + 0x708);
    uint32_t plen = le32(d.data() + c + 0x6f8);
    uint32_t csize = le32(d.data() + c + 0x6fc);
    uint32_t coff = le32(d.data() + c + 0x700);
    bool compute = le32(d.data() + p) == kCodeCompute;
    if (p + s > d.size()) return false;
    if (start != (compute ? 0x100u : 0x30u)) return false;
    if (start + plen > s) return false;
    if (csize) {
        if (coff != up256(start + plen) || s != up256(coff + csize)) return false;
    } else {
        if (s != up256(start + plen)) return false;
    }
    return true;
}

std::string stage_of(const std::vector<uint8_t>& d, size_t p, size_t c) {
    if (le32(d.data() + p) == kCodeCompute || le32(d.data() + c + 0x708) == 0) return "compute";
    uint32_t t = (le32(d.data() + p + 0x30) >> 10) & 0xFu;
    return t == 1 ? "vertex" : t == 5 ? "pixel" : "other";
}

// Extract all validated (code_off, ctrl_off) pairs from a single .bshdat file.
std::vector<std::pair<size_t, size_t>> bshdat_pairs(const std::vector<uint8_t>& d) {
    std::vector<size_t> ctrls, codes;
    for (size_t o = 0; o + 0x880 <= d.size(); o += 4) {
        if (le32(d.data() + o) == kCtrlMagic) {
            uint32_t s = le32(d.data() + o + 0x704);
            if (s > 0 && s < (1u << 24)) ctrls.push_back(o);
        }
    }
    for (size_t o = 0; o + 0x30 <= d.size();) {
        if ((d[o] == 0x78 || d[o] == 0x79) && d[o + 1] == 0x56 && d[o + 2] == 0x34 && d[o + 3] == 0x12 &&
            is_zero_range(d, o + 4, o + 0x30)) {
            codes.push_back(o);
            o += 0x30;
        } else {
            o++;
        }
    }
    std::vector<std::pair<size_t, size_t>> out;
    if (ctrls.size() == codes.size()) {
        for (size_t i = 0; i < codes.size(); i++) {
            if (ctrl_fits_code(d, codes[i], ctrls[i])) out.push_back({codes[i], ctrls[i]});
        }
        return out;
    }
    // Fallback pairing if a future update has unmatched counts: match each code block to the nearest preceding ctrl.
    for (size_t p : codes) {
        size_t best_c = SIZE_MAX;
        for (size_t c : ctrls) {
            if (c < p && ctrl_fits_code(d, p, c)) best_c = c;
        }
        if (best_c != SIZE_MAX) out.push_back({p, best_c});
    }
    return out;
}

void collect_bshdat_files(const nxi::GameImage& game, const std::string& dir,
                          std::vector<std::pair<std::string, nxi::GameImage::Entry>>& out) {
    for (const auto& [name, entry] : game.list(dir)) {
        std::string full = dir.empty() ? name : (dir + "/" + name);
        if (entry.dir) {
            collect_bshdat_files(game, full, out);
        } else if (name.size() > 7 && name.compare(name.size() - 7, 7, ".bshdat") == 0) {
            out.push_back({full, entry});
        }
    }
}

}  // namespace

long extract_shaders(const nxi::GameImage& game, const std::string& out_dir,
                     const std::function<void(int, int)>& progress, std::string& error) {
    try {
        make_dir(out_dir);
        std::vector<std::pair<std::string, nxi::GameImage::Entry>> files;
        collect_bshdat_files(game, "system/shd", files);
        if (files.empty()) collect_bshdat_files(game, "", files);
        if (files.empty()) throw Fail("no .bshdat shader archives found in RomFS (is this Metroid Dread?)");

        std::sort(files.begin(), files.end(), [](const auto& a, const auto& b) { return a.first < b.first; });

        // Put all deduplicated Metroid Dread shaders into a single harness batch group ("dread_shd")
        // so spirv_harness --batch translates them with minimal directory/manifest overhead.
        const std::string group = "dread_shd";
        const std::string gdir = out_dir + "/" + group;
        make_dir(gdir);

        std::string manifest =
            "id\tstage\tcode_off\tcode_size\tctrl_off\tctrl_size\tprogram_start\tprogram_bytes\tconst_off\tconst_size\t"
            "code_sha1_16\tcode_file\tctrl_file\n";

        std::set<std::string> seen;
        long written = 0;
        const int total_steps = int(files.size());
        char line[512];

        for (size_t i = 0; i < files.size(); i++) {
            if (progress) progress(int(i), total_steps);
            const auto& [path, entry] = files[i];
            std::vector<uint8_t> data(size_t(entry.size));
            if (game.read(entry, 0, data.data(), data.size()) != data.size())
                throw Fail("failed reading RomFS shader archive: " + path);

            for (auto [p, c] : bshdat_pairs(data)) {
                uint32_t s = le32(data.data() + c + 0x704);
                if (p + s > data.size()) continue;
                std::string h = sha1_16(data.data() + p, size_t(s));
                if (!seen.insert(h).second) continue;

                size_t clen = std::min<size_t>(0x880, data.size() - c);
                std::string cf = "ctrl_" + sha1_16(data.data() + c, clen) + ".bin";
                std::string stage = stage_of(data, p, c);
                write_file(gdir + "/code_" + h + ".bin", data.data() + p, size_t(s));
                write_file(gdir + "/" + cf, data.data() + c, clen);

                std::snprintf(
                    line, sizeof line,
                    "%s/0x%llx/%s\t%s\t0x%llx\t0x%x\t0x%llx\t0x%x\t0x%x\t0x%x\t0x%x\t0x%x\t%s\tcode_%s.bin\t%s\n",
                    group.c_str(), (unsigned long long)p, stage.c_str(), stage.c_str(), (unsigned long long)p, s,
                    (unsigned long long)c, 0x880u, le32(data.data() + c + 0x708), le32(data.data() + c + 0x6f8),
                    le32(data.data() + c + 0x700), le32(data.data() + c + 0x6fc), h.c_str(), h.c_str(), cf.c_str());
                manifest += line;
                written++;
            }
        }

        if (written == 0) throw Fail("0 valid Maxwell shader blocks extracted from .bshdat archives");
        write_file(gdir + "/manifest.tsv", reinterpret_cast<const uint8_t*>(manifest.data()), manifest.size());
        if (progress) progress(total_steps, total_steps);
        return written;
    } catch (const Fail& e) {
        error = e.what();
    } catch (const std::bad_alloc&) {
        error = "out of memory while extracting shaders";
    }
    return -1;
}

}  // namespace dread
