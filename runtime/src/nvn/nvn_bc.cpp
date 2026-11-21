// BC texture decoding on the CPU (docs/nvn/ANDROID-PORT.md D1, DEBUG-2.md #85): for GPUs without
// textureCompressionBC (Mali, most Android phones but Adreno and Xclipse), a BC texture is decoded when it is uploaded
// and stored as RGBA8 (BC1-3, BC7), R8 (BC4) or RG8 (BC5), UNORM, SNORM or sRGB as the BC format was. D1's ASTC
// side table (transcoded on the Mac) is meant to replace this for the game's own files; this stays the fallback.
// BC6H (HDR, NVN 0x4f SF16 / 0x50 UF16) is decoded to RGBA16F (VK_FORMAT_R16G16B16A16_SFLOAT, alpha 1.0).
// NX_FORCE_BC_DECODE=1: decode even where the GPU has BC (checking it on the Mac against the native path).
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "nvn_vk.h"

namespace nx::nvn {

namespace {

// Derived from a reference decoder (Pillow) with blocks whose subsets decode to distinct colours (DEBUG-2.md #85).
// Bit i (2 subsets) or bits 2i..2i+1 (3 subsets): the subset of pixel i.
const uint16_t kPart2[64] = {
    0xcccc, 0x8888, 0xeeee, 0xecc8, 0xc880, 0xfeec, 0xfec8, 0xec80,
    0xc800, 0xffec, 0xfe80, 0xe800, 0xffe8, 0xff00, 0xfff0, 0xf000,
    0xf710, 0x008e, 0x7100, 0x08ce, 0x008c, 0x7310, 0x3100, 0x8cce,
    0x088c, 0x3110, 0x6666, 0x366c, 0x17e8, 0x0ff0, 0x718e, 0x399c,
    0xaaaa, 0xf0f0, 0x5a5a, 0x33cc, 0x3c3c, 0x55aa, 0x9696, 0xa55a,
    0x73ce, 0x13c8, 0x324c, 0x3bdc, 0x6996, 0xc33c, 0x9966, 0x0660,
    0x0272, 0x04e4, 0x4e40, 0x2720, 0xc936, 0x936c, 0x39c6, 0x639c,
    0x9336, 0x9cc6, 0x817e, 0xe718, 0xccf0, 0x0fcc, 0x7744, 0xee22,
};
const uint32_t kPart3[64] = {
    0xaa685050, 0x6a5a5040, 0x5a5a4200, 0x5450a0a8, 0xa5a50000, 0xa0a05050,
    0x5555a0a0, 0x5a5a5050, 0xaa550000, 0xaa555500, 0xaaaa5500, 0x90909090,
    0x94949494, 0xa4a4a4a4, 0xa9a59450, 0x2a0a4250, 0xa5945040, 0x0a425054,
    0xa5a5a500, 0x55a0a0a0, 0xa8a85454, 0x6a6a4040, 0xa4a45000, 0x1a1a0500,
    0x0050a4a4, 0xaaa59090, 0x14696914, 0x69691400, 0xa08585a0, 0xaa821414,
    0x50a4a450, 0x6a5a0200, 0xa9a58000, 0x5090a0a8, 0xa8a09050, 0x24242424,
    0x00aa5500, 0x24924924, 0x24499224, 0x50a50a50, 0x500aa550, 0xaaaa4444,
    0x66660000, 0xa5a0a5a0, 0x50a050a0, 0x69286928, 0x44aaaa44, 0x66666600,
    0xaa444444, 0x54a854a8, 0x95809580, 0x96969600, 0xa85454a8, 0x80959580,
    0xaa141414, 0x96960000, 0xaaaa1414, 0xa05050a0, 0xa0a5a5a0, 0x96000000,
    0x40804080, 0xa9a8a9a8, 0xaaaaaa44, 0x2a4a5254,
};
const uint8_t kAnchor2[64] = {15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 2, 8, 2, 2, 8, 8, 15, 2, 8, 2, 2, 8, 8, 2, 2, 15, 15, 6, 8, 2, 8, 15, 15, 2, 8, 2, 2, 2, 15, 15, 6, 6, 2, 6, 8, 15, 15, 2, 2, 15, 15, 15, 15, 15, 2, 2, 15};
const uint8_t kAnchor3a[64] = {3, 3, 15, 15, 8, 3, 15, 15, 8, 8, 6, 6, 6, 5, 3, 3, 3, 3, 8, 15, 3, 3, 6, 10, 5, 8, 8, 6, 8, 5, 15, 15, 8, 15, 3, 5, 6, 10, 8, 15, 15, 3, 15, 5, 15, 15, 15, 15, 3, 15, 5, 5, 5, 8, 5, 10, 5, 10, 8, 13, 15, 12, 3, 3};
const uint8_t kAnchor3b[64] = {15, 8, 8, 3, 15, 15, 3, 8, 15, 15, 15, 15, 15, 15, 15, 8, 15, 8, 15, 3, 15, 8, 15, 8, 3, 15, 6, 10, 15, 15, 10, 8, 15, 3, 15, 10, 10, 8, 9, 10, 6, 15, 8, 15, 3, 6, 6, 8, 15, 3, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 3, 15, 15, 8};

struct Mode {
    uint8_t subsets, partition_bits, rotation_bits, index_select_bits, color_bits, alpha_bits, endpoint_pbits,
        shared_pbits, index_bits, index2_bits;
};
const Mode kModes[8] = {{3, 4, 0, 0, 4, 0, 1, 0, 3, 0}, {2, 6, 0, 0, 6, 0, 0, 1, 3, 0}, {3, 6, 0, 0, 5, 0, 0, 0, 2, 0},
                        {2, 6, 0, 0, 7, 0, 1, 0, 2, 0}, {1, 0, 2, 1, 5, 6, 0, 0, 2, 3}, {1, 0, 2, 0, 7, 8, 0, 0, 2, 2},
                        {1, 0, 0, 0, 7, 7, 1, 0, 4, 0}, {2, 6, 0, 0, 5, 5, 1, 0, 2, 0}};
const uint8_t kWeights2[4] = {0, 21, 43, 64};
const uint8_t kWeights3[8] = {0, 9, 18, 27, 37, 46, 55, 64};
const uint8_t kWeights4[16] = {0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64};

const uint8_t* weights(uint32_t bits) { return bits == 2 ? kWeights2 : bits == 3 ? kWeights3 : kWeights4; }

struct Reader {
    uint64_t lo, hi;
    uint32_t pos;
    Reader(const uint8_t* b, uint32_t p) : pos(p) {
        std::memcpy(&lo, b, 8);
        std::memcpy(&hi, b + 8, 8);
    }
    inline uint32_t get(uint32_t n) {
        if (!n) return 0;
        const uint32_t p = pos;
        pos = p + n;
        const uint32_t mask = (1u << n) - 1u;
        if (p + n <= 64u) return uint32_t(lo >> p) & mask;
        if (p >= 64u) return uint32_t(hi >> (p - 64u)) & mask;
        return uint32_t((lo >> p) | (hi << (64u - p))) & mask;
    }
};

void bc7(const uint8_t* block, uint8_t out[16][4]) {
    uint32_t mode = 0;
    while (mode < 8 && !(block[0] >> mode & 1)) mode++;
    if (mode == 8) {  // reserved: transparent black
        std::memset(out, 0, 64);
        return;
    }
    const Mode& m = kModes[mode];
    Reader r{block, mode + 1};
    uint32_t part = r.get(m.partition_bits), rotation = r.get(m.rotation_bits), select = r.get(m.index_select_bits);
    uint32_t ends = m.subsets * 2;
    uint8_t ep[6][4] = {};
    for (uint32_t c = 0; c < 3; c++)
        for (uint32_t e = 0; e < ends; e++) ep[e][c] = uint8_t(r.get(m.color_bits));
    for (uint32_t e = 0; e < ends; e++) ep[e][3] = uint8_t(m.alpha_bits ? r.get(m.alpha_bits) : 255);
    uint8_t pbit[6] = {};
    if (m.endpoint_pbits)
        for (uint32_t e = 0; e < ends; e++) pbit[e] = uint8_t(r.get(1));
    if (m.shared_pbits)
        for (uint32_t s = 0; s < m.subsets; s++) pbit[2 * s] = pbit[2 * s + 1] = uint8_t(r.get(1));
    bool has_p = m.endpoint_pbits || m.shared_pbits;
    for (uint32_t e = 0; e < ends; e++)
        for (uint32_t c = 0; c < 4; c++) {
            uint32_t bits = c < 3 ? m.color_bits : m.alpha_bits;
            if (!bits) continue;  // alpha of an opaque mode: 255 already
            uint32_t v = ep[e][c];
            if (has_p) v = (v << 1) | pbit[e], bits++;
            v <<= 8 - bits;
            ep[e][c] = uint8_t(v | (v >> bits));
        }
    auto subset = [&](uint32_t i) -> uint32_t {
        if (m.subsets == 2) return kPart2[part] >> i & 1;
        if (m.subsets == 3) return kPart3[part] >> (2 * i) & 3;
        return 0;
    };
    auto anchor = [&](uint32_t i) {
        if (i == 0) return true;
        if (m.subsets == 2) return i == kAnchor2[part];
        if (m.subsets == 3) return i == kAnchor3a[part] || i == kAnchor3b[part];
        return false;
    };
    uint32_t idx[16], idx2[16] = {};
    for (uint32_t i = 0; i < 16; i++) idx[i] = r.get(m.index_bits - (anchor(i) ? 1 : 0));
    if (m.index2_bits)
        for (uint32_t i = 0; i < 16; i++) idx2[i] = r.get(m.index2_bits - (i == 0 ? 1 : 0));
    for (uint32_t i = 0; i < 16; i++) {
        uint32_t s = subset(i);
        const uint8_t* e0 = ep[2 * s];
        const uint8_t* e1 = ep[2 * s + 1];
        uint32_t ci = idx[i], ai = idx[i], cb = m.index_bits, ab = m.index_bits;
        if (m.index2_bits) {  // modes 4 and 5: a second index set, for alpha (or for colour when selected)
            ai = idx2[i], ab = m.index2_bits;
            if (select) std::swap(ci, ai), std::swap(cb, ab);
        }
        const uint8_t* wc = weights(cb);
        const uint8_t* wa = weights(ab);
        for (uint32_t c = 0; c < 3; c++) out[i][c] = uint8_t(((64 - wc[ci]) * e0[c] + wc[ci] * e1[c] + 32) >> 6);
        out[i][3] = uint8_t(((64 - wa[ai]) * e0[3] + wa[ai] * e1[3] + 32) >> 6);
        if (rotation) std::swap(out[i][3], out[i][rotation - 1]);
    }
}

// BC1's colour part (also BC2/BC3's, where it always has four colours). `alpha`: BC1 with a transparent index 3.
void bc1_colors(const uint8_t* b, uint8_t out[16][4], bool four, bool alpha) {
    uint16_t c0 = uint16_t(b[0] | b[1] << 8), c1 = uint16_t(b[2] | b[3] << 8);
    uint8_t pal[4][4];
    auto expand = [](uint16_t c, uint8_t* p) {
        uint32_t r = c >> 11, g = (c >> 5) & 63, bl = c & 31;
        p[0] = uint8_t(r << 3 | r >> 2), p[1] = uint8_t(g << 2 | g >> 4), p[2] = uint8_t(bl << 3 | bl >> 2), p[3] = 255;
    };
    expand(c0, pal[0]);
    expand(c1, pal[1]);
    bool opaque4 = four || c0 > c1;
    for (int c = 0; c < 3; c++) {
        if (opaque4) {
            pal[2][c] = uint8_t((2 * pal[0][c] + pal[1][c]) / 3);
            pal[3][c] = uint8_t((pal[0][c] + 2 * pal[1][c]) / 3);
        } else {
            pal[2][c] = uint8_t((pal[0][c] + pal[1][c]) / 2);
            pal[3][c] = 0;
        }
    }
    pal[2][3] = 255;
    pal[3][3] = opaque4 || !alpha ? 255 : 0;
    uint32_t bits = uint32_t(b[4] | b[5] << 8 | b[6] << 16 | uint32_t(b[7]) << 24);
    for (int i = 0; i < 16; i++) std::memcpy(out[i], pal[bits >> (2 * i) & 3], 4);
}

// BC4 (and BC3's alpha, BC5's channels): eight values between two endpoints. Signed: -127..127 as SNORM bytes.
void bc4(const uint8_t* b, uint8_t out[16], bool is_signed) {
    int v[8];
    if (is_signed) {
        v[0] = std::max<int>(int8_t(b[0]), -127), v[1] = std::max<int>(int8_t(b[1]), -127);
    } else {
        v[0] = b[0], v[1] = b[1];
    }
    if (v[0] > v[1]) {
        for (int k = 1; k < 7; k++) v[k + 1] = ((7 - k) * v[0] + k * v[1]) / 7;
    } else {
        for (int k = 1; k < 5; k++) v[k + 1] = ((5 - k) * v[0] + k * v[1]) / 5;
        v[6] = is_signed ? -127 : 0, v[7] = is_signed ? 127 : 255;
    }
    uint64_t bits = 0;
    for (int k = 0; k < 6; k++) bits |= uint64_t(b[2 + k]) << (8 * k);
    for (int i = 0; i < 16; i++) out[i] = uint8_t(v[bits >> (3 * i) & 7]);
}

inline uint32_t bc6_bits(const uint8_t* b, uint32_t lo, uint32_t n) {
    if (!n) return 0;
    uint64_t w0, w1;
    std::memcpy(&w0, b, 8);
    std::memcpy(&w1, b + 8, 8);
    const uint32_t mask = (1u << n) - 1u;
    if (lo + n <= 64u) return uint32_t(w0 >> lo) & mask;
    if (lo >= 64u) return uint32_t(w1 >> (lo - 64u)) & mask;
    return uint32_t((w0 >> lo) | (w1 << (64u - lo))) & mask;
}

uint32_t bc6_rev(uint32_t v, uint32_t n) {
    uint32_t out = 0;
    for (uint32_t i = 0; i < n; i++) {
        if ((v >> i) & 1u) out |= 1u << (n - 1u - i);
    }
    return out;
}

int32_t bc6_sxt(uint32_t v, uint32_t bits) {
    uint32_t mask = (1u << bits) - 1u;
    v &= mask;
    uint32_t sign = 1u << (bits - 1u);
    return (v & sign) ? int32_t(v | ~mask) : int32_t(v);
}

int32_t bc6_unq(int32_t val, uint32_t prec, bool is_signed) {
    if (!is_signed) {
        uint32_t u = uint32_t(val);
        if (prec >= 15u) return int32_t(u);
        if (u == 0u) return 0;
        if (u == (1u << prec) - 1u) return 0xFFFF;
        return int32_t(((u << 16) + 0x8000u) >> prec);
    }
    if (prec >= 16u) return val;
    bool neg = val < 0;
    uint32_t av = uint32_t(neg ? -val : val);
    uint32_t u = av == 0u ? 0u
                          : (av >= (1u << (prec - 1u)) - 1u ? 0x7FFFu
                                                            : ((av << 15) + 0x4000u) >> (prec - 1u));
    return neg ? -int32_t(u) : int32_t(u);
}

uint16_t bc6_finish(int32_t comp, bool is_signed) {
    if (!is_signed) return uint16_t((uint32_t(comp) * 31u) >> 6);
    int32_t scaled = comp < 0 ? -(((-comp) * 31) >> 5) : (comp * 31) >> 5;
    return scaled < 0 ? uint16_t(0x8000u | uint32_t(-scaled)) : uint16_t(scaled);
}

void bc6h(const uint8_t* b, bool is_signed, uint16_t out[16][4]) {
    uint32_t m2 = b[0] & 3u, m5 = b[0] & 31u;
    bool two = true, trans = true;
    uint32_t ep_bits = 10, db[3] = {5, 5, 5};
    uint32_t rw = 0, gw = 0, bw = 0, rx = 0, gx = 0, bx = 0, ry = 0, gy = 0, by = 0, rz = 0, gz = 0, bz = 0;

    if (m2 == 0) {
        uint32_t gy4 = bc6_bits(b, 2, 1), by4 = bc6_bits(b, 3, 1), bz4 = bc6_bits(b, 4, 1);
        rw = bc6_bits(b, 5, 10), gw = bc6_bits(b, 15, 10), bw = bc6_bits(b, 25, 10);
        rx = bc6_bits(b, 35, 5);
        uint32_t gz4 = bc6_bits(b, 40, 1);
        gy = bc6_bits(b, 41, 4) | (gy4 << 4), gx = bc6_bits(b, 45, 5);
        uint32_t bz0 = bc6_bits(b, 50, 1);
        gz = bc6_bits(b, 51, 4) | (gz4 << 4), bx = bc6_bits(b, 55, 5);
        uint32_t bz1 = bc6_bits(b, 60, 1);
        by = bc6_bits(b, 61, 4) | (by4 << 4), ry = bc6_bits(b, 65, 5);
        uint32_t bz2 = bc6_bits(b, 70, 1);
        rz = bc6_bits(b, 71, 5);
        uint32_t bz3 = bc6_bits(b, 76, 1);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3) | (bz4 << 4);
    } else if (m2 == 1) {
        ep_bits = 7, db[0] = 6, db[1] = 6, db[2] = 6;
        uint32_t gy5 = bc6_bits(b, 2, 1), gz4 = bc6_bits(b, 3, 1), gz5 = bc6_bits(b, 4, 1);
        rw = bc6_bits(b, 5, 7);
        uint32_t bz0 = bc6_bits(b, 12, 1), bz1 = bc6_bits(b, 13, 1), by4 = bc6_bits(b, 14, 1);
        gw = bc6_bits(b, 15, 7);
        uint32_t by5 = bc6_bits(b, 22, 1), bz2 = bc6_bits(b, 23, 1), gy4 = bc6_bits(b, 24, 1);
        bw = bc6_bits(b, 25, 7);
        uint32_t bz3 = bc6_bits(b, 32, 1), bz5 = bc6_bits(b, 33, 1), bz4 = bc6_bits(b, 34, 1);
        rx = bc6_bits(b, 35, 6), gy = bc6_bits(b, 41, 4) | (gy4 << 4) | (gy5 << 5);
        gx = bc6_bits(b, 45, 6), gz = bc6_bits(b, 51, 4) | (gz4 << 4) | (gz5 << 5);
        bx = bc6_bits(b, 55, 6), by = bc6_bits(b, 61, 4) | (by4 << 4) | (by5 << 5);
        ry = bc6_bits(b, 65, 6), rz = bc6_bits(b, 71, 6);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3) | (bz4 << 4) | (bz5 << 5);
    } else if (m5 == 2) {
        ep_bits = 11, db[0] = 5, db[1] = 4, db[2] = 4;
        rw = bc6_bits(b, 5, 10) | (bc6_bits(b, 40, 1) << 10);
        gw = bc6_bits(b, 15, 10) | (bc6_bits(b, 49, 1) << 10);
        bw = bc6_bits(b, 25, 10) | (bc6_bits(b, 59, 1) << 10);
        rx = bc6_bits(b, 35, 5), gy = bc6_bits(b, 41, 4), gx = bc6_bits(b, 45, 4);
        uint32_t bz0 = bc6_bits(b, 50, 1);
        gz = bc6_bits(b, 51, 4), bx = bc6_bits(b, 55, 4);
        uint32_t bz1 = bc6_bits(b, 60, 1);
        by = bc6_bits(b, 61, 4), ry = bc6_bits(b, 65, 5);
        uint32_t bz2 = bc6_bits(b, 70, 1);
        rz = bc6_bits(b, 71, 5);
        uint32_t bz3 = bc6_bits(b, 76, 1);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3);
    } else if (m5 == 6) {
        ep_bits = 11, db[0] = 4, db[1] = 5, db[2] = 4;
        rw = bc6_bits(b, 5, 10) | (bc6_bits(b, 39, 1) << 10);
        gw = bc6_bits(b, 15, 10) | (bc6_bits(b, 50, 1) << 10);
        bw = bc6_bits(b, 25, 10) | (bc6_bits(b, 59, 1) << 10);
        rx = bc6_bits(b, 35, 4);
        uint32_t gz4 = bc6_bits(b, 40, 1);
        gy = bc6_bits(b, 41, 4) | (bc6_bits(b, 75, 1) << 4), gx = bc6_bits(b, 45, 5);
        gz = bc6_bits(b, 51, 4) | (gz4 << 4), bx = bc6_bits(b, 55, 4);
        uint32_t bz1 = bc6_bits(b, 60, 1);
        by = bc6_bits(b, 61, 4), ry = bc6_bits(b, 65, 4);
        uint32_t bz0 = bc6_bits(b, 69, 1), bz2 = bc6_bits(b, 70, 1);
        rz = bc6_bits(b, 71, 4);
        uint32_t bz3 = bc6_bits(b, 76, 1);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3);
    } else if (m5 == 10) {
        ep_bits = 11, db[0] = 4, db[1] = 4, db[2] = 5;
        rw = bc6_bits(b, 5, 10) | (bc6_bits(b, 39, 1) << 10);
        gw = bc6_bits(b, 15, 10) | (bc6_bits(b, 49, 1) << 10);
        bw = bc6_bits(b, 25, 10) | (bc6_bits(b, 60, 1) << 10);
        rx = bc6_bits(b, 35, 4);
        uint32_t by4 = bc6_bits(b, 40, 1);
        gy = bc6_bits(b, 41, 4), gx = bc6_bits(b, 45, 4);
        uint32_t bz0 = bc6_bits(b, 50, 1);
        gz = bc6_bits(b, 51, 4), bx = bc6_bits(b, 55, 5);
        by = bc6_bits(b, 61, 4) | (by4 << 4), ry = bc6_bits(b, 65, 4);
        uint32_t bz1 = bc6_bits(b, 69, 1), bz2 = bc6_bits(b, 70, 1);
        rz = bc6_bits(b, 71, 4);
        uint32_t bz4 = bc6_bits(b, 75, 1), bz3 = bc6_bits(b, 76, 1);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3) | (bz4 << 4);
    } else if (m5 == 14) {
        ep_bits = 9, db[0] = 5, db[1] = 5, db[2] = 5;
        rw = bc6_bits(b, 5, 9);
        uint32_t by4 = bc6_bits(b, 14, 1);
        gw = bc6_bits(b, 15, 9);
        uint32_t gy4 = bc6_bits(b, 24, 1);
        bw = bc6_bits(b, 25, 9);
        uint32_t bz4 = bc6_bits(b, 34, 1);
        rx = bc6_bits(b, 35, 5);
        uint32_t gz4 = bc6_bits(b, 40, 1);
        gy = bc6_bits(b, 41, 4) | (gy4 << 4), gx = bc6_bits(b, 45, 5);
        uint32_t bz0 = bc6_bits(b, 50, 1);
        gz = bc6_bits(b, 51, 4) | (gz4 << 4), bx = bc6_bits(b, 55, 5);
        uint32_t bz1 = bc6_bits(b, 60, 1);
        by = bc6_bits(b, 61, 4) | (by4 << 4), ry = bc6_bits(b, 65, 5);
        uint32_t bz2 = bc6_bits(b, 70, 1);
        rz = bc6_bits(b, 71, 5);
        uint32_t bz3 = bc6_bits(b, 76, 1);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3) | (bz4 << 4);
    } else if (m5 == 18) {
        ep_bits = 8, db[0] = 6, db[1] = 5, db[2] = 5;
        rw = bc6_bits(b, 5, 8);
        uint32_t gz4 = bc6_bits(b, 13, 1), by4 = bc6_bits(b, 14, 1);
        gw = bc6_bits(b, 15, 8);
        uint32_t bz2 = bc6_bits(b, 23, 1), gy4 = bc6_bits(b, 24, 1);
        bw = bc6_bits(b, 25, 8);
        uint32_t bz3 = bc6_bits(b, 33, 1), bz4 = bc6_bits(b, 34, 1);
        rx = bc6_bits(b, 35, 6), gy = bc6_bits(b, 41, 4) | (gy4 << 4), gx = bc6_bits(b, 45, 5);
        uint32_t bz0 = bc6_bits(b, 50, 1);
        gz = bc6_bits(b, 51, 4) | (gz4 << 4), bx = bc6_bits(b, 55, 5);
        uint32_t bz1 = bc6_bits(b, 60, 1);
        by = bc6_bits(b, 61, 4) | (by4 << 4), ry = bc6_bits(b, 65, 6), rz = bc6_bits(b, 71, 6);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3) | (bz4 << 4);
    } else if (m5 == 22) {
        ep_bits = 8, db[0] = 5, db[1] = 6, db[2] = 5;
        rw = bc6_bits(b, 5, 8);
        uint32_t bz0 = bc6_bits(b, 13, 1), by4 = bc6_bits(b, 14, 1);
        gw = bc6_bits(b, 15, 8);
        uint32_t gy5 = bc6_bits(b, 23, 1), gy4 = bc6_bits(b, 24, 1);
        bw = bc6_bits(b, 25, 8);
        uint32_t gz5 = bc6_bits(b, 33, 1), bz4 = bc6_bits(b, 34, 1);
        rx = bc6_bits(b, 35, 5);
        uint32_t gz4 = bc6_bits(b, 40, 1);
        gy = bc6_bits(b, 41, 4) | (gy4 << 4) | (gy5 << 5);
        gx = bc6_bits(b, 45, 6), gz = bc6_bits(b, 51, 4) | (gz4 << 4) | (gz5 << 5);
        bx = bc6_bits(b, 55, 5);
        uint32_t bz1 = bc6_bits(b, 60, 1);
        by = bc6_bits(b, 61, 4) | (by4 << 4), ry = bc6_bits(b, 65, 5);
        uint32_t bz2 = bc6_bits(b, 70, 1);
        rz = bc6_bits(b, 71, 5);
        uint32_t bz3 = bc6_bits(b, 76, 1);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3) | (bz4 << 4);
    } else if (m5 == 26) {
        ep_bits = 8, db[0] = 5, db[1] = 5, db[2] = 6;
        rw = bc6_bits(b, 5, 8);
        uint32_t bz1 = bc6_bits(b, 13, 1), by4 = bc6_bits(b, 14, 1);
        gw = bc6_bits(b, 15, 8);
        uint32_t by5 = bc6_bits(b, 23, 1), gy4 = bc6_bits(b, 24, 1);
        bw = bc6_bits(b, 25, 8);
        uint32_t bz5 = bc6_bits(b, 33, 1), bz4 = bc6_bits(b, 34, 1);
        rx = bc6_bits(b, 35, 5);
        uint32_t gz4 = bc6_bits(b, 40, 1);
        gy = bc6_bits(b, 41, 4) | (gy4 << 4), gx = bc6_bits(b, 45, 5);
        uint32_t bz0 = bc6_bits(b, 50, 1);
        gz = bc6_bits(b, 51, 4) | (gz4 << 4), bx = bc6_bits(b, 55, 6);
        by = bc6_bits(b, 61, 4) | (by4 << 4) | (by5 << 5), ry = bc6_bits(b, 65, 5);
        uint32_t bz2 = bc6_bits(b, 70, 1);
        rz = bc6_bits(b, 71, 5);
        uint32_t bz3 = bc6_bits(b, 76, 1);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3) | (bz4 << 4) | (bz5 << 5);
    } else if (m5 == 30) {
        trans = false, ep_bits = 6, db[0] = 6, db[1] = 6, db[2] = 6;
        rw = bc6_bits(b, 5, 6);
        uint32_t gz4 = bc6_bits(b, 11, 1), bz0 = bc6_bits(b, 12, 1), bz1 = bc6_bits(b, 13, 1), by4 = bc6_bits(b, 14, 1);
        gw = bc6_bits(b, 15, 6);
        uint32_t gy5 = bc6_bits(b, 21, 1), by5 = bc6_bits(b, 22, 1), bz2 = bc6_bits(b, 23, 1), gy4 = bc6_bits(b, 24, 1);
        bw = bc6_bits(b, 25, 6);
        uint32_t gz5 = bc6_bits(b, 31, 1), bz3 = bc6_bits(b, 32, 1), bz5 = bc6_bits(b, 33, 1), bz4 = bc6_bits(b, 34, 1);
        rx = bc6_bits(b, 35, 6), gy = bc6_bits(b, 41, 4) | (gy4 << 4) | (gy5 << 5);
        gx = bc6_bits(b, 45, 6), gz = bc6_bits(b, 51, 4) | (gz4 << 4) | (gz5 << 5);
        bx = bc6_bits(b, 55, 6), by = bc6_bits(b, 61, 4) | (by4 << 4) | (by5 << 5);
        ry = bc6_bits(b, 65, 6), rz = bc6_bits(b, 71, 6);
        bz = bz0 | (bz1 << 1) | (bz2 << 2) | (bz3 << 3) | (bz4 << 4) | (bz5 << 5);
    } else if (m5 == 3) {
        two = false, trans = false, ep_bits = 10, db[0] = 10, db[1] = 10, db[2] = 10;
        rw = bc6_bits(b, 5, 10), gw = bc6_bits(b, 15, 10), bw = bc6_bits(b, 25, 10);
        rx = bc6_bits(b, 35, 10), gx = bc6_bits(b, 45, 10), bx = bc6_bits(b, 55, 10);
    } else if (m5 == 7) {
        two = false, ep_bits = 11, db[0] = 9, db[1] = 9, db[2] = 9;
        rw = bc6_bits(b, 5, 10) | (bc6_bits(b, 44, 1) << 10);
        gw = bc6_bits(b, 15, 10) | (bc6_bits(b, 54, 1) << 10);
        bw = bc6_bits(b, 25, 10) | (bc6_bits(b, 64, 1) << 10);
        rx = bc6_bits(b, 35, 9), gx = bc6_bits(b, 45, 9), bx = bc6_bits(b, 55, 9);
    } else if (m5 == 11) {
        two = false, ep_bits = 12, db[0] = 8, db[1] = 8, db[2] = 8;
        rw = bc6_bits(b, 5, 10) | (bc6_rev(bc6_bits(b, 43, 2), 2) << 10);
        gw = bc6_bits(b, 15, 10) | (bc6_rev(bc6_bits(b, 53, 2), 2) << 10);
        bw = bc6_bits(b, 25, 10) | (bc6_rev(bc6_bits(b, 63, 2), 2) << 10);
        rx = bc6_bits(b, 35, 8), gx = bc6_bits(b, 45, 8), bx = bc6_bits(b, 55, 8);
    } else if (m5 == 15) {
        two = false, ep_bits = 16, db[0] = 4, db[1] = 4, db[2] = 4;
        rw = bc6_bits(b, 5, 10) | (bc6_rev(bc6_bits(b, 39, 6), 6) << 10);
        gw = bc6_bits(b, 15, 10) | (bc6_rev(bc6_bits(b, 49, 6), 6) << 10);
        bw = bc6_bits(b, 25, 10) | (bc6_rev(bc6_bits(b, 59, 6), 6) << 10);
        rx = bc6_bits(b, 35, 4), gx = bc6_bits(b, 45, 4), bx = bc6_bits(b, 55, 4);
    } else {
        for (int i = 0; i < 16; i++) out[i][0] = 0, out[i][1] = 0, out[i][2] = 0, out[i][3] = 0x3C00;
        return;
    }

    int32_t ep[4][3] = {
        {int32_t(rw), int32_t(gw), int32_t(bw)},
        {int32_t(rx), int32_t(gx), int32_t(bx)},
        {int32_t(ry), int32_t(gy), int32_t(by)},
        {int32_t(rz), int32_t(gz), int32_t(bz)},
    };
    uint32_t n_ep = two ? 4u : 2u, mask = (1u << ep_bits) - 1u;
    if (is_signed) {
        for (uint32_t c = 0; c < 3; c++) ep[0][c] = bc6_sxt(uint32_t(ep[0][c]), ep_bits);
    }
    if (trans) {
        for (uint32_t i = 1; i < n_ep; i++) {
            for (uint32_t c = 0; c < 3; c++) {
                uint32_t sum = (uint32_t(ep[0][c]) + uint32_t(bc6_sxt(uint32_t(ep[i][c]), db[c]))) & mask;
                ep[i][c] = is_signed ? bc6_sxt(sum, ep_bits) : int32_t(sum);
            }
        }
    } else if (is_signed) {
        for (uint32_t i = 1; i < n_ep; i++) {
            for (uint32_t c = 0; c < 3; c++) ep[i][c] = bc6_sxt(uint32_t(ep[i][c]), ep_bits);
        }
    }
    for (uint32_t i = 0; i < n_ep; i++) {
        for (uint32_t c = 0; c < 3; c++) ep[i][c] = bc6_unq(ep[i][c], ep_bits, is_signed);
    }
    if (two) {
        uint32_t part = bc6_bits(b, 77, 5), pos = 82;
        uint16_t p_mask = kPart2[part];
        uint8_t anchor1 = kAnchor2[part];
        for (uint32_t i = 0; i < 16; i++) {
            uint32_t sub = (p_mask >> i) & 1u;
            uint32_t ib = (i == 0 || i == anchor1) ? 2u : 3u;
            uint32_t idx = bc6_bits(b, pos, ib);
            pos += ib;
            int32_t w = kWeights3[idx];
            const int32_t* a = ep[sub * 2u];
            const int32_t* b2 = ep[sub * 2u + 1u];
            for (int c = 0; c < 3; c++) out[i][c] = bc6_finish((a[c] * (64 - w) + b2[c] * w + 32) >> 6, is_signed);
            out[i][3] = 0x3C00;
        }
    } else {
        uint32_t pos = 65;
        for (uint32_t i = 0; i < 16; i++) {
            uint32_t ib = i == 0 ? 3u : 4u;
            uint32_t idx = bc6_bits(b, pos, ib);
            pos += ib;
            int32_t w = kWeights4[idx];
            for (int c = 0; c < 3; c++)
                out[i][c] = bc6_finish((ep[0][c] * (64 - w) + ep[1][c] * w + 32) >> 6, is_signed);
            out[i][3] = 0x3C00;
        }
    }
}

enum Kind { kNone, kBc1, kBc1a, kBc2, kBc3, kBc4, kBc4s, kBc5, kBc5s, kBc7, kBc6s, kBc6u };

Kind kind(uint32_t nvn) {
    switch (nvn) {
        case 0x41: case 0x45: return kBc1;
        case 0x42: case 0x46: return kBc1a;
        case 0x43: case 0x47: return kBc2;
        case 0x44: case 0x48: return kBc3;
        case 0x49: return kBc4;
        case 0x4a: return kBc4s;
        case 0x4b: return kBc5;
        case 0x4c: return kBc5s;
        case 0x4d: case 0x4e: return kBc7;
        case 0x4f: return kBc6s;
        case 0x50: return kBc6u;
        default: return kNone;
    }
}

void decode_rows(Kind k, const uint8_t* blocks, uint32_t width, uint32_t height, uint32_t bw, uint32_t bh,
                 uint32_t bytes, uint32_t block_size, uint32_t row_begin, uint32_t row_end, uint8_t* out) {
    for (uint32_t row = row_begin; row < row_end; row++) {
        const uint32_t s = row / bh;
        const uint32_t by = row % bh;
        const bool full_y = (by * 4u + 4u <= height);
        for (uint32_t bx = 0; bx < bw; bx++) {
            const uint8_t* b = blocks + (uint64_t(row) * bw + bx) * block_size;
            const bool full_block = full_y && (bx * 4u + 4u <= width);
            if (k == kBc6s || k == kBc6u) {
                uint16_t px16[16][4];
                bc6h(b, k == kBc6s, px16);
                if (full_block) {
                    const uint64_t row_base = ((uint64_t(s) * height + by * 4u) * width + bx * 4u) * 8u;
                    const uint64_t stride = uint64_t(width) * 8u;
                    for (uint32_t y = 0; y < 4; y++) std::memcpy(out + row_base + y * stride, px16[y * 4], 32);
                } else {
                    for (uint32_t y = 0; y < 4; y++) {
                        uint32_t py = by * 4 + y;
                        if (py >= height) break;
                        for (uint32_t x = 0; x < 4; x++) {
                            uint32_t pxx = bx * 4 + x;
                            if (pxx >= width) break;
                            std::memcpy(out + ((uint64_t(s) * height + py) * width + pxx) * 8, px16[y * 4 + x], 8);
                        }
                    }
                }
                continue;
            }
            uint8_t px[16][4] = {};
            switch (k) {
                case kBc1: bc1_colors(b, px, false, false); break;
                case kBc1a: bc1_colors(b, px, false, true); break;
                case kBc2:
                    bc1_colors(b + 8, px, true, false);
                    for (int i = 0; i < 16; i++) px[i][3] = uint8_t(((b[i / 2] >> (4 * (i & 1))) & 15) * 17);
                    break;
                case kBc3: {
                    uint8_t a[16];
                    bc4(b, a, false);
                    bc1_colors(b + 8, px, true, false);
                    for (int i = 0; i < 16; i++) px[i][3] = a[i];
                    break;
                }
                case kBc4: case kBc4s: {
                    uint8_t r[16];
                    bc4(b, r, k == kBc4s);
                    for (int i = 0; i < 16; i++) px[i][0] = r[i];
                    break;
                }
                case kBc5: case kBc5s: {
                    uint8_t r[16], g[16];
                    bc4(b, r, k == kBc5s);
                    bc4(b + 8, g, k == kBc5s);
                    for (int i = 0; i < 16; i++) px[i][0] = r[i], px[i][1] = g[i];
                    break;
                }
                case kBc7: bc7(b, px); break;
                default: break;
            }
            if (full_block && bytes == 4) {
                const uint64_t row_base = ((uint64_t(s) * height + by * 4u) * width + bx * 4u) * 4u;
                const uint64_t stride = uint64_t(width) * 4u;
                for (uint32_t y = 0; y < 4; y++) std::memcpy(out + row_base + y * stride, px[y * 4], 16);
            } else {
                for (uint32_t y = 0; y < 4; y++) {
                    uint32_t py = by * 4 + y;
                    if (py >= height) break;
                    for (uint32_t x = 0; x < 4; x++) {
                        uint32_t pxx = bx * 4 + x;
                        if (pxx >= width) break;
                        std::memcpy(out + ((uint64_t(s) * height + py) * width + pxx) * bytes, px[y * 4 + x], bytes);
                    }
                }
            }
        }
    }
}

struct BcTask {
    Kind k;
    const uint8_t* blocks;
    uint32_t width, height, bw, bh, bytes, block_size, total_rows;
    uint8_t* out;
    std::atomic<uint32_t> next_row{0};
};

void drain_bc_task(BcTask& task) {
    for (;;) {
        const uint32_t r0 = task.next_row.fetch_add(4, std::memory_order_relaxed);
        if (r0 >= task.total_rows) break;
        const uint32_t r1 = std::min(r0 + 4u, task.total_rows);
        decode_rows(task.k, task.blocks, task.width, task.height, task.bw, task.bh, task.bytes, task.block_size, r0, r1,
                    task.out);
    }
}

class BcPool {
   public:
    static constexpr int kWorkers = 4;
    static BcPool& instance() {
        static BcPool p;
        return p;
    }
    void run(BcTask& task) {
        std::unique_lock<std::mutex> rlk(run_m_, std::try_to_lock);
        if (!rlk.owns_lock()) {
            drain_bc_task(task);
            return;
        }
        {
            std::lock_guard<std::mutex> lk(m_);
            cur_ = &task;
            pending_ = kWorkers;
            gen_++;
        }
        cv_.notify_all();
        drain_bc_task(task);
        std::unique_lock<std::mutex> lk(m_);
        done_cv_.wait(lk, [&] { return pending_ == 0; });
        cur_ = nullptr;
    }

   private:
    BcPool() {
        for (int i = 0; i < kWorkers; i++) {
            threads_.emplace_back([this] { worker_loop(); });
            threads_.back().detach();
        }
    }
    void worker_loop() {
        uint64_t seen = 0;
        for (;;) {
            BcTask* t = nullptr;
            {
                std::unique_lock<std::mutex> lk(m_);
                cv_.wait(lk, [&] { return gen_ != seen; });
                seen = gen_;
                t = cur_;
            }
            if (t) drain_bc_task(*t);
            bool last = false;
            {
                std::lock_guard<std::mutex> lk(m_);
                if (pending_ > 0 && --pending_ == 0) last = true;
            }
            if (last) done_cv_.notify_one();
        }
    }
    std::mutex run_m_, m_;
    std::condition_variable cv_, done_cv_;
    std::vector<std::thread> threads_;
    BcTask* cur_ = nullptr;
    uint64_t gen_ = 0;
    int pending_ = 0;
};

}  // namespace

bool bc_view_format(const GpuImage& img, uint32_t nvn, VkFormat& out) {
    if (!bc_host_decode(img.nvn_format)) return false;
    // A view of a decoded image: the decoded format of the view's BC format when it decodes to the same texel size
    // (BC7 UNORM seen as BC7 sRGB), else the image's own.
    bool same = nvn && bc_host_decode(nvn) && bc_decoded_bytes(nvn) == bc_decoded_bytes(img.nvn_format);
    out = same ? bc_decoded_format(nvn) : img.format;
    return true;
}

bool bc_host_decode(uint32_t nvn_format) {
    Kind k = kind(nvn_format);
    if (k == kNone) return false;
    static const bool force = std::getenv("NX_FORCE_BC_DECODE") != nullptr;
    if (force) return true;
    if (!vk_features().textureCompressionBC) return true;
    VkFormat vk = vk_format(nvn_format);
    if (vk != VK_FORMAT_UNDEFINED && vk_physical() && vk::p_vkGetPhysicalDeviceFormatProperties) {
        VkFormatProperties fp{};
        vk::p_vkGetPhysicalDeviceFormatProperties(vk_physical(), vk, &fp);
        if (!(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT)) return true;
    }
    return false;
}

VkFormat bc_decoded_format(uint32_t nvn_format) {
    bool srgb = nvn_format >= 0x45 && nvn_format <= 0x48 ? true : nvn_format == 0x4e;
    switch (kind(nvn_format)) {
        case kBc4: return VK_FORMAT_R8_UNORM;
        case kBc4s: return VK_FORMAT_R8_SNORM;
        case kBc5: return VK_FORMAT_R8G8_UNORM;
        case kBc5s: return VK_FORMAT_R8G8_SNORM;
        case kBc6s:
        case kBc6u: return VK_FORMAT_R16G16B16A16_SFLOAT;
        default: return srgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
    }
}

uint32_t bc_decoded_bytes(uint32_t nvn_format) {
    Kind k = kind(nvn_format);
    if (k == kBc6s || k == kBc6u) return 8;
    return k == kBc4 || k == kBc4s ? 1 : k == kBc5 || k == kBc5s ? 2 : 4;
}

void bc_decode(uint32_t nvn_format, const uint8_t* blocks, uint32_t width, uint32_t height, uint32_t slices,
               uint8_t* out) {
    Kind k = kind(nvn_format);
    uint32_t bytes = bc_decoded_bytes(nvn_format), bw = (width + 3) / 4, bh = (height + 3) / 4;
    uint32_t block_size = k == kBc1 || k == kBc1a || k == kBc4 || k == kBc4s ? 8 : 16;
    const uint32_t total_rows = slices * bh;
    static const bool single_thread = [] {
        const char* e = std::getenv("NX_BC_THREADS");
        return e && std::strcmp(e, "1") == 0;
    }();
    if (single_thread || uint64_t(total_rows) * bw < 2048 || total_rows < 8) {
        decode_rows(k, blocks, width, height, bw, bh, bytes, block_size, 0, total_rows, out);
        return;
    }
    BcTask task{k, blocks, width, height, bw, bh, bytes, block_size, total_rows, out};
    BcPool::instance().run(task);
}

}  // namespace nx::nvn
