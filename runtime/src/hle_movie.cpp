// nn::movie at the API level. The library is the game's subsdk0, which is not recompiled: its imports land here.
// P5R's CRI Mana Switch backend (CriVpdSwitch, main+1276000..127a000) drives its VP9 decoder like Android's
// MediaCodec in asynchronous mode, from three threads:
//   - setup: Decoder(type, format, mode, priority), CreateVp9DecoderConfigurationData(w, h, fps * 1000, md) (true
//     on success), Configure(md) (0), RegisterClientEvents(events), Start();
//   - an event thread waits on DecoderEvents {input available, output available, format changed, error} (four
//     nn::os::EventType*), then takes indices with AcquireInputBufferIndex(&index, &more) and
//     AcquireOutputBufferIndex(&index, &pts, &flags, &more) (flag 4: end of stream) until they fail;
//   - a feeder fills GetInputBuffer(index) and calls SendInputBufferForDecode(index, offset, size, pts, flags); a
//     consumer passes its own frame memory to GetOutputBuffer(index, buffer), which the decoder fills, reads the
//     NV12 layout with GetOutputBufferFormat (format 4) and gives the index back with ReleaseOutputBufferIndex.
// Objects: Decoder (0x10, the game's operator new) holds our vtable and an id; Buffer (0x40) +0 offset, +4 size,
// +0x18 capacity, +0x28 base; MediaData (8 bytes) points at a host-side map of values.
//
// Decoding: on Android the device's own VP9 decoder (AMediaCodec, ByteBuffer mode, polled from the calls above: the
// feeder's sends, the event thread's acquires and the consumer's releases, and every 2 ms by a guest thread of ours
// that signals the output event), its pictures converted to tight NV12.
// Elsewhere, or when the device has no VP9 decoder, each input frame comes out as a black frame (NV12 Y 16, UV 128)
// of the same timestamp, so a video plays black for its length and the game goes on.
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__ANDROID__)
#include <fcntl.h>
#include <media/NdkMediaCodec.h>
#include <media/NdkMediaExtractor.h>
#include <media/NdkMediaFormat.h>
#include <sys/syscall.h>
#include <unistd.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
#endif

#include <nx/runtime.h>

#include "diagnostics.h"
#include "game_image.h"
#include "media_image.h"
#include "platform.h"

namespace nx {

namespace nvn {
bool write_texture_rgba(uint64_t guest_texture, uint32_t w, uint32_t h, const uint8_t* rgba, size_t size);
void signal_video_sync(uint64_t guest_sync);
}  // namespace nvn

namespace {

// ---- MediaData ---------------------------------------------------------------------------------------------------

struct MediaValue {
    int32_t i = 0;
    int64_t i64 = 0;
    int32_t rect[4] = {};
    std::string s;
    bool is_rect = false;
    bool is_i64 = false;
    bool is_str = false;
};
using MediaMap = std::map<std::string, MediaValue>;

MediaMap* media(uint64_t md) { return md ? reinterpret_cast<MediaMap*>(ld<uint64_t>(md)) : nullptr; }

void md_set(uint64_t md, const char* key, int32_t v) {
    if (MediaMap* m = media(md)) {
        MediaValue& mv = (*m)[key];
        mv.i = v;
        mv.i64 = v;
        mv.is_rect = false;
        mv.is_str = false;
    }
}

void md_set64(uint64_t md, const char* key, int64_t v) {
    if (MediaMap* m = media(md)) {
        MediaValue& mv = (*m)[key];
        mv.i64 = v;
        mv.i = int32_t(v);
        mv.is_i64 = true;
        mv.is_rect = false;
        mv.is_str = false;
    }
}

void md_set_str(uint64_t md, const char* key, const std::string& s) {
    if (MediaMap* m = media(md)) {
        MediaValue& mv = (*m)[key];
        mv.s = s;
        mv.is_str = true;
        mv.is_rect = false;
    }
}

void md_set_rect(uint64_t md, const char* key, int32_t l, int32_t t, int32_t r, int32_t b) {
    if (MediaMap* m = media(md)) {
        MediaValue& v = (*m)[key];
        v.is_rect = true, v.rect[0] = l, v.rect[1] = t, v.rect[2] = r, v.rect[3] = b;
    }
}

bool md_get(uint64_t md, const char* key, int32_t& out) {
    MediaMap* m = media(md);
    auto it = m ? m->find(key) : MediaMap::iterator{};
    if (!m || it == m->end() || it->second.is_rect || it->second.is_str) return false;
    out = it->second.i;
    return true;
}

void MediaDataCtor(CPU& c) { st<uint64_t>(c.x[0], reinterpret_cast<uint64_t>(new MediaMap)); }
void MediaDataDtor(CPU& c) {
    delete media(c.x[0]);
    st<uint64_t>(c.x[0], 0);
}
void MediaDataClear(CPU& c) {
    if (MediaMap* m = media(c.x[0])) m->clear();
}
// bool FindInt32(const char* key, int32_t* out) const
void MediaDataFindInt32(CPU& c) {
    int32_t v = 0;
    bool found = md_get(c.x[0], reinterpret_cast<const char*>(c.x[1]), v);
    if (found && c.x[2]) st<int32_t>(c.x[2], v);
    c.x[0] = found;
}
// bool FindInt64(const char* key, int64_t* out) const
void MediaDataFindInt64(CPU& c) {
    MediaMap* m = media(c.x[0]);
    const char* key = reinterpret_cast<const char*>(c.x[1]);
    auto it = (m && key) ? m->find(key) : MediaMap::iterator{};
    bool found = m && key && it != m->end() && !it->second.is_rect && !it->second.is_str;
    if (found && c.x[2]) st<int64_t>(c.x[2], it->second.is_i64 ? it->second.i64 : int64_t(it->second.i));
    c.x[0] = found;
}
// bool FindString(const char* key, const char** out) const
void MediaDataFindString(CPU& c) {
    MediaMap* m = media(c.x[0]);
    const char* key = reinterpret_cast<const char*>(c.x[1]);
    auto it = (m && key) ? m->find(key) : MediaMap::iterator{};
    bool found = m && key && it != m->end() && (it->second.is_str || !it->second.s.empty());
    if (found && c.x[2]) st<uint64_t>(c.x[2], reinterpret_cast<uint64_t>(it->second.s.c_str()));
    c.x[0] = found;
}
// bool FindRect(const char* key, int32_t* left, int32_t* top, int32_t* right, int32_t* bottom) const
void MediaDataFindRect(CPU& c) {
    MediaMap* m = media(c.x[0]);
    auto it = m ? m->find(reinterpret_cast<const char*>(c.x[1])) : MediaMap::iterator{};
    bool found = m && it != m->end() && it->second.is_rect;
    for (int k = 0; k < 4 && found; k++)
        if (c.x[2 + k]) st<int32_t>(c.x[2 + k], it->second.rect[k]);
    c.x[0] = found;
}

// ---- Buffer ------------------------------------------------------------------------------------------------------
// Verified subsdk0 field layout: +0x00 offset, +0x04 size, +0x08 int32_data, +0x0c uint32_data, +0x10 owns_mem,
// +0x18 capacity, +0x20 int64_data, +0x28 base, +0x30 nvn_dev, +0x38 nvn_sync.
constexpr uint64_t kBufOffset = 0x0, kBufSize = 0x4, kBufInt32 = 0x8, kBufUint32 = 0xC, kBufOwns = 0x10;
constexpr uint64_t kBufCapacity = 0x18, kBufInt64 = 0x20, kBufBase = 0x28, kBufNvnDev = 0x30, kBufNvnSync = 0x38;

std::mutex g_buf_lock;
std::unordered_map<uint64_t, std::vector<uint8_t>> g_owned_bufs;

void buf_free_owned(uint64_t self) {
    if (ld<uint8_t>(self + kBufOwns)) {
        uint64_t base = ld<uint64_t>(self + kBufBase);
        std::lock_guard lk(g_buf_lock);
        g_owned_bufs.erase(base);
        st<uint8_t>(self + kBufOwns, 0);
    }
}

void BufferCtor(CPU& c) { std::memset(reinterpret_cast<void*>(c.x[0]), 0, 0x40); }
void BufferCtorCap(CPU& c) {  // Buffer(size_t capacity)
    uint64_t self = c.x[0], cap = c.x[1];
    std::memset(reinterpret_cast<void*>(self), 0, 0x40);
    if (cap > 0) {
        std::vector<uint8_t> mem(cap);
        uint64_t ptr = reinterpret_cast<uint64_t>(mem.data());
        {
            std::lock_guard lk(g_buf_lock);
            g_owned_bufs[ptr] = std::move(mem);
        }
        st<uint64_t>(self + kBufBase, ptr);
        st<uint64_t>(self + kBufCapacity, cap);
        st<uint8_t>(self + kBufOwns, 1);
    }
}
void BufferCtorWrap(CPU& c) {  // Buffer(void* data, size_t capacity)
    uint64_t self = c.x[0];
    std::memset(reinterpret_cast<void*>(self), 0, 0x40);
    st<uint64_t>(self + kBufBase, c.x[1]);
    st<uint64_t>(self + kBufCapacity, c.x[2]);
}
void BufferCtorCopy(CPU& c) {  // Buffer(const Buffer& other)
    uint64_t self = c.x[0], other = c.x[1];
    std::memcpy(reinterpret_cast<void*>(self), reinterpret_cast<const void*>(other), 0x40);
    if (ld<uint8_t>(other + kBufOwns)) {
        uint64_t cap = ld<uint64_t>(other + kBufCapacity);
        uint64_t obase = ld<uint64_t>(other + kBufBase);
        std::vector<uint8_t> mem(cap);
        if (obase && cap) std::memcpy(mem.data(), reinterpret_cast<const void*>(obase), cap);
        uint64_t ptr = reinterpret_cast<uint64_t>(mem.data());
        {
            std::lock_guard lk(g_buf_lock);
            g_owned_bufs[ptr] = std::move(mem);
        }
        st<uint64_t>(self + kBufBase, ptr);
        st<uint8_t>(self + kBufOwns, 1);
    }
}
void BufferDtor(CPU& c) { buf_free_owned(c.x[0]); }
void BufferSetDataAndCapacity(CPU& c) {  // (void* data, size_t capacity)
    buf_free_owned(c.x[0]);
    st<uint64_t>(c.x[0] + kBufBase, c.x[1]);
    st<uint64_t>(c.x[0] + kBufCapacity, c.x[2]);
    st<int32_t>(c.x[0] + kBufOffset, 0);
    st<int32_t>(c.x[0] + kBufSize, 0);
}
void BufferSetRange(CPU& c) {  // (int offset, int size)
    st<int32_t>(c.x[0] + kBufOffset, int32_t(c.x[1]));
    st<int32_t>(c.x[0] + kBufSize, int32_t(c.x[2]));
}
void BufferBase(CPU& c) { c.x[0] = ld<uint64_t>(c.x[0] + kBufBase); }
void BufferCapacity(CPU& c) { c.x[0] = ld<uint64_t>(c.x[0] + kBufCapacity); }
void BufferSize(CPU& c) { c.x[0] = uint64_t(int64_t(ld<int32_t>(c.x[0] + kBufSize))); }
void BufferOffset(CPU& c) { c.x[0] = uint64_t(int64_t(ld<int32_t>(c.x[0] + kBufOffset))); }
void BufferSetInt32Data(CPU& c) { st<int32_t>(c.x[0] + kBufInt32, int32_t(c.x[1])); }
void BufferSetInt64Data(CPU& c) { st<int64_t>(c.x[0] + kBufInt64, int64_t(c.x[1])); }
void BufferSetUint32Data(CPU& c) { st<uint32_t>(c.x[0] + kBufUint32, uint32_t(c.x[1])); }
void BufferGetInt64Data(CPU& c) { c.x[0] = uint64_t(ld<int64_t>(c.x[0] + kBufInt64)); }
void BufferGetUint32Data(CPU& c) { c.x[0] = ld<uint32_t>(c.x[0] + kBufUint32); }
void BufferSetNvnDev(CPU& c) { st<uint64_t>(c.x[0] + kBufNvnDev, c.x[1]); }
void BufferSetNvnSync(CPU& c) { st<uint64_t>(c.x[0] + kBufNvnSync, c.x[1]); }

// ---- Decoder -----------------------------------------------------------------------------------------------------

enum Event { kInputAvailable, kOutputAvailable, kFormatChanged, kError };
constexpr int kInputs = 4, kMaxOutputs = 32;
// Output slots offered to the game (GetOutputBufferSize). CriVpdSwitch keeps the pictures it took until it shows
// them, and feeds no more input while it holds nearly all: with 8, P5R's opening film (after New Game) played ~4 s
// and froze ~4 s over and over, 7 of the 8 held and no input sent. With 16 it plays through at 30 pictures a second.
// NX_MOVIE_OUTPUTS=<n> (2..32): another count.
int output_slots() {
    static const int n = [] {
        const char* e = std::getenv("NX_MOVIE_OUTPUTS");
        return e ? std::clamp(std::atoi(e), 2, kMaxOutputs) : 16;
    }();
    return n;
}
constexpr uint32_t kEndOfStream = 4;
constexpr uint32_t kTryAgain = 0x1001;  // any non-zero status: CriVpdSwitch waits for the next event

struct Decoder {
    uint32_t type = 0, format = 0, mode = 0;
    uint64_t events[4] = {};
    int32_t width = 0, height = 0, max_input = 0;
    std::vector<uint8_t> inputs[kInputs];
    std::deque<int> free_inputs;
    struct Output {
        int64_t pts = 0;
        uint32_t flags = 0;
        bool busy = false;
    } outputs[kMaxOutputs];
    std::deque<int> ready;                              // outputs to acquire
    std::deque<std::pair<int64_t, uint32_t>> pending;  // black frames made with every output slot busy
    std::vector<uint8_t> frames[kMaxOutputs];              // decoded picture (tight NV12 or RGBA8); empty: black
    std::vector<uint8_t> nv12_scratch;                     // reusable NV12 staging buffer when converting to RGBA8
    std::vector<uint8_t> black_rgba;                       // reusable opaque black RGBA8 fallback frame
    std::string mime = "video/x-vnd.on2.vp9";
    bool started = false, format_sent = false, need_input = false;
    uint64_t frames_in = 0, frames_out = 0, decoded = 0;
    // Per-window counts for the movie: line (pump thread, every 2 s while the decoder runs).
    struct Stats {
        uint64_t sent = 0, decoded = 0, acquired = 0, got = 0, released = 0, empty_acquires = 0;
        uint64_t convert_ns = 0, copy_ns = 0, send_wait_ns = 0, max_got_gap_ns = 0, max_pump_ns = 0;
    } win;
    uint64_t last_got_ns = 0, win_from_ns = 0, last_sent_ns = 0;
    int64_t last_sent_pts = 0, last_got_pts = 0;
    bool starve_logged = false;  // NX_MOVIE_DUMP: threads logged for this pause in the input
#if defined(__ANDROID__)
    AMediaCodec* codec = nullptr;
    struct Input {
        std::vector<uint8_t> data;
        int64_t pts;
        uint32_t flags;
    };
    std::deque<Input> to_codec;  // sent by the game, not yet taken by the codec
    int32_t color = 0, codec_stride = 0, codec_rows = 0, crop_left = 0, crop_top = 0;
    std::string codec_name;
    bool is_planar = false;
    bool has_planes = false;
    uint32_t plane_off[3] = {}, plane_col[3] = {}, plane_row[3] = {};
    bool eos_sent = false;       // the game's end of stream went to the codec
    bool codec_eos = false;      // the end-of-stream picture came out
    bool codec_running = false;  // AMediaCodec is in Running state (Flush+Start must flush, not start again)
    bool codec_stopped = false;  // AMediaCodec_stop was called: must recreate before next AMediaCodec_start
#endif

    // CriVpdSwitch puts its UV texture right after width x height bytes of Y, whatever offsets we report: rows of
    // exactly the width, no padding. Format 2 (ARGB) / 3 (ABGR -> RGBA8 in LE memory) is 4 bytes per pixel.
    bool is_rgba() const { return format == 2 || format == 3; }
    int32_t stride() const { return is_rgba() ? width * 4 : width; }
    int32_t rows() const { return height; }
    uint64_t nv12_size() const { return uint64_t(width) * height * 3 / 2; }
    uint64_t output_size() const { return is_rgba() ? uint64_t(width) * height * 4 : nv12_size(); }

    int free_slot() const {
        for (int k = 0; k < output_slots(); k++)
            if (!outputs[k].busy) return k;
        return -1;
    }
    // A black frame into a free output slot (or the pending queue when all are taken).
    void emit(int64_t pts, uint32_t flags) {
        if (int k = free_slot(); k >= 0) {
            outputs[k] = {pts, flags, true};
            frames[k].clear();
            ready.push_back(k);
            return;
        }
        pending.emplace_back(pts, flags);
    }
    void reset_queues() {
        free_inputs.clear();
        for (int k = 0; k < kInputs; k++) free_inputs.push_back(k);
        for (auto& o : outputs) o = {};
        ready.clear(), pending.clear();
        need_input = false;
        win = {}, win_from_ns = last_got_ns = last_sent_ns = 0;
#if defined(__ANDROID__)
        to_codec.clear();
        eos_sent = codec_eos = false;
#endif
    }
};

#if defined(__ANDROID__)
constexpr int32_t kColorI420 = 19, kColorNV12 = 21, kColorFlexible = 0x7F420888, kColorQcom32m = 0x7FA30C04;

// The codec's picture (color format `color`, rows of `stride` bytes, `rows` rows per plane) into tight NV12 of the
// video's size, from the crop origin. I420 is interleaved; every other format is taken as semi-planar (NV12 and the
// Qualcomm/MediaTek variants with aligned strides, which is what ByteBuffer decoders hand out).
void to_nv12(const Decoder& d, const uint8_t* src, size_t size, std::vector<uint8_t>& out) {
    const int32_t w = d.width, h = d.height;
    int32_t stride = d.codec_stride > 0 ? d.codec_stride : w, rows = d.codec_rows > 0 ? d.codec_rows : h;
    // Qualcomm (kColorQcom32m or c2.qti.*/OMX.qcom.*): stride aligns to 128, rows to 32. Apply each
    // alignment independently — the codec may report stride but omit slice-height, leaving rows wrong.
    // However, only apply 32-row alignment if the buffer size actually accommodates it; otherwise standard rows=h.
    {
        const bool is_qcom = d.color == kColorQcom32m ||
                             (!d.has_planes && (d.codec_name.find(".qti.") != std::string::npos ||
                                               d.codec_name.find(".qcom.") != std::string::npos));
        if (is_qcom) {
            if (d.codec_stride <= 0) {
                const int32_t aligned_stride = (w + 127) & ~127;
                if (size_t(aligned_stride) * h * 3 / 2 <= size) stride = aligned_stride;
                else stride = w;
            }
            if (d.codec_rows <= 0) {
                const int32_t aligned_rows = (h + 31) & ~31;
                if (size_t(stride) * aligned_rows + size_t(stride) * (h / 2) <= size) {
                    rows = aligned_rows;
                } else {
                    rows = h;
                }
            }
        }
    }
    // Safety check: if stride * rows + stride * (h / 2) > size, rows must be clamped down to h
    if (size_t(stride) * rows + size_t(stride) * (h / 2) > size && rows > h) {
        rows = h;
    }
    out.resize(size_t(w) * h * 3 / 2);
    diag::movie_frame_alloc(out.size(), out.capacity());
    if (d.has_planes && d.plane_row[0] > 0 && d.plane_row[1] > 0 && d.plane_row[2] > 0) {
        const size_t y_stride = d.plane_row[0];
        const size_t y_off = d.plane_off[0] + size_t(d.crop_top) * y_stride + size_t(d.crop_left);
        for (int32_t r = 0; r < h; r++) {
            size_t at = y_off + size_t(r) * y_stride;
            if (at + w > size) return;
            std::memcpy(out.data() + size_t(r) * w, src + at, size_t(w));
        }
        uint8_t* uv = out.data() + size_t(w) * h;
        const size_t u_stride = d.plane_row[1], v_stride = d.plane_row[2];
        const size_t u_step = d.plane_col[1] ? d.plane_col[1] : 1;
        const size_t v_step = d.plane_col[2] ? d.plane_col[2] : 1;
        const size_t u_off = d.plane_off[1] + size_t(d.crop_top / 2) * u_stride + size_t(d.crop_left / 2) * u_step;
        const size_t v_off = d.plane_off[2] + size_t(d.crop_top / 2) * v_stride + size_t(d.crop_left / 2) * v_step;
        if (u_step == 2 && v_step == 2 && u_stride == v_stride && v_off == u_off + 1) {
            for (int32_t r = 0; r < h / 2; r++) {
                size_t at = u_off + size_t(r) * u_stride;
                if (at + w > size) return;
                std::memcpy(uv + size_t(r) * w, src + at, size_t(w));
            }
            return;
        }
        for (int32_t r = 0; r < h / 2; r++) {
            size_t u_at = u_off + size_t(r) * u_stride;
            size_t v_at = v_off + size_t(r) * v_stride;
            for (int32_t x = 0; x < w / 2; x++) {
                size_t ui = u_at + size_t(x) * u_step, vi = v_at + size_t(x) * v_step;
                if (ui >= size || vi >= size) return;
                uv[size_t(r) * w + 2 * x] = src[ui];
                uv[size_t(r) * w + 2 * x + 1] = src[vi];
            }
        }
        return;
    }
    const size_t y0 = size_t(d.crop_top) * stride + d.crop_left;
    for (int32_t r = 0; r < h; r++) {
        size_t at = y0 + size_t(r) * stride;
        if (at + w > size) return;
        std::memcpy(out.data() + size_t(r) * w, src + at, size_t(w));
    }
    uint8_t* uv = out.data() + size_t(w) * h;
    const size_t chroma = size_t(stride) * rows;
    if (d.is_planar || d.color == kColorI420) {
        const size_t cs = size_t(stride) / 2, plane = cs * (size_t(rows) / 2);
        for (int32_t r = 0; r < h / 2; r++) {
            size_t at = chroma + size_t(d.crop_top / 2 + r) * cs + d.crop_left / 2;
            if (at + plane + w / 2 > size) return;
            for (int32_t x = 0; x < w / 2; x++)
                uv[size_t(r) * w + 2 * x] = src[at + x], uv[size_t(r) * w + 2 * x + 1] = src[at + plane + x];
        }
        return;
    }
    for (int32_t r = 0; r < h / 2; r++) {
        size_t at = chroma + size_t(d.crop_top / 2 + r) * stride + (d.crop_left & ~1);
        if (at + w > size) return;
        std::memcpy(uv + size_t(r) * w, src + at, size_t(w));
    }
}

// Tight NV12 or planar I420 -> RGBA8888 (OutputFormat_VideoColorAbgr: LE R,G,B,A) using BT.709 limited-range math.
#if defined(__aarch64__)
inline void yuv16_pair_to_rgba_neon(uint8x16_t y0, uint8x16_t y1, uint8x8_t u8, uint8x8_t v8, uint8_t* d0,
                                    uint8_t* d1) {
    const int16x8_t c128 = vdupq_n_s16(128);
    const int16x8_t u = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(u8)), c128);
    const int16x8_t v = vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(v8)), c128);
    const int16x8_t rv8 = vmulq_n_s16(v, 115);
    const int16x8_t guv8 = vmlsq_n_s16(vmulq_n_s16(u, -14), v, 34);
    const int16x8_t bu8 = vmulq_n_s16(u, 135);
    const int16x8_t rv_lo = vzip1q_s16(rv8, rv8), rv_hi = vzip2q_s16(rv8, rv8);
    const int16x8_t guv_lo = vzip1q_s16(guv8, guv8), guv_hi = vzip2q_s16(guv8, guv8);
    const int16x8_t bu_lo = vzip1q_s16(bu8, bu8), bu_hi = vzip2q_s16(bu8, bu8);
    const uint8x16_t y16 = vdupq_n_u8(16);
    const int16x8_t round32 = vdupq_n_s16(32);
    const uint8x16_t alpha = vdupq_n_u8(255);
    auto store_row = [&](uint8x16_t yv, uint8_t* dst) {
        const uint8x16_t ys = vqsubq_u8(yv, y16);
        const int16x8_t c_lo = vmlaq_n_s16(round32, vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(ys))), 75);
        const int16x8_t c_hi = vmlaq_n_s16(round32, vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(ys))), 75);
        uint8x16x4_t rgba;
        rgba.val[0] = vcombine_u8(vqmovun_s16(vshrq_n_s16(vqaddq_s16(c_lo, rv_lo), 6)),
                                  vqmovun_s16(vshrq_n_s16(vqaddq_s16(c_hi, rv_hi), 6)));
        rgba.val[1] = vcombine_u8(vqmovun_s16(vshrq_n_s16(vqaddq_s16(c_lo, guv_lo), 6)),
                                  vqmovun_s16(vshrq_n_s16(vqaddq_s16(c_hi, guv_hi), 6)));
        rgba.val[2] = vcombine_u8(vqmovun_s16(vshrq_n_s16(vqaddq_s16(c_lo, bu_lo), 6)),
                                  vqmovun_s16(vshrq_n_s16(vqaddq_s16(c_hi, bu_hi), 6)));
        rgba.val[3] = alpha;
        vst4q_u8(dst, rgba);
    };
    store_row(y0, d0);
    if (d1) store_row(y1, d1);
}
#endif

inline uint32_t pack_yuv_rgba(int c, int rv, int guv, int bu) {
    int r = c + rv, g = c + guv, b = c + bu;
    r = r < 0 ? 0 : (r > 255 ? 255 : r);
    g = g < 0 ? 0 : (g > 255 ? 255 : g);
    b = b < 0 ? 0 : (b > 255 ? 255 : b);
    return 0xFF000000u | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
}

// Fast single-pass planar I420 -> RGBA8 directly from AMediaCodec output buffer (skips NV12 staging copy).
bool codec_i420_to_rgba(const Decoder& d, const uint8_t* src, size_t size, std::vector<uint8_t>& out) {
    if (!d.has_planes || d.plane_col[0] != 1 || d.plane_col[1] != 1 || d.plane_col[2] != 1) return false;
    if (!d.plane_row[0] || !d.plane_row[1] || !d.plane_row[2] || d.width <= 0 || d.height <= 0) return false;
    const int32_t w = d.width, h = d.height;
    const size_t y_stride = d.plane_row[0], u_stride = d.plane_row[1], v_stride = d.plane_row[2];
    const size_t y_off = d.plane_off[0] + size_t(d.crop_top) * y_stride + size_t(d.crop_left);
    const size_t u_off = d.plane_off[1] + size_t(d.crop_top / 2) * u_stride + size_t(d.crop_left / 2);
    const size_t v_off = d.plane_off[2] + size_t(d.crop_top / 2) * v_stride + size_t(d.crop_left / 2);
    if (y_off + size_t(h - 1) * y_stride + size_t(w) > size) return false;
    if (u_off + size_t((h / 2) - 1) * u_stride + size_t(w / 2) > size) return false;
    if (v_off + size_t((h / 2) - 1) * v_stride + size_t(w / 2) > size) return false;
    out.resize(size_t(w) * h * 4);
    uint8_t* dst_bytes = out.data();
    for (int32_t y = 0; y < h; y += 2) {
        const uint8_t* yr0 = src + y_off + size_t(y) * y_stride;
        const uint8_t* yr1 = (y + 1 < h) ? (yr0 + y_stride) : yr0;
        const uint8_t* ur = src + u_off + size_t(y >> 1) * u_stride;
        const uint8_t* vr = src + v_off + size_t(y >> 1) * v_stride;
        uint8_t* d0 = dst_bytes + size_t(y) * w * 4;
        uint8_t* d1 = (y + 1 < h) ? (d0 + size_t(w) * 4) : nullptr;
        int32_t x = 0;
#if defined(__aarch64__)
        for (; x + 15 < w; x += 16) {
            yuv16_pair_to_rgba_neon(vld1q_u8(yr0 + x), vld1q_u8(yr1 + x), vld1_u8(ur + (x >> 1)),
                                    vld1_u8(vr + (x >> 1)), d0 + x * 4, d1 ? d1 + x * 4 : nullptr);
        }
#endif
        uint32_t* d0w = reinterpret_cast<uint32_t*>(d0);
        uint32_t* d1w = reinterpret_cast<uint32_t*>(d1);
        for (; x + 1 < w; x += 2) {
            int u = int(ur[x >> 1]) - 128, v = int(vr[x >> 1]) - 128;
            int rv = (459 * v) >> 8, guv = (-55 * u - 136 * v) >> 8, bu = (541 * u) >> 8;
            d0w[x] = pack_yuv_rgba((298 * (int(yr0[x]) - 16) + 128) >> 8, rv, guv, bu);
            d0w[x + 1] = pack_yuv_rgba((298 * (int(yr0[x + 1]) - 16) + 128) >> 8, rv, guv, bu);
            if (d1w) {
                d1w[x] = pack_yuv_rgba((298 * (int(yr1[x]) - 16) + 128) >> 8, rv, guv, bu);
                d1w[x + 1] = pack_yuv_rgba((298 * (int(yr1[x + 1]) - 16) + 128) >> 8, rv, guv, bu);
            }
        }
    }
    return true;
}

void nv12_to_rgba(const uint8_t* nv12, int32_t w, int32_t h, std::vector<uint8_t>& out) {
    if (!nv12 || w <= 0 || h <= 0) return;
    out.resize(size_t(w) * h * 4);
    const uint8_t* y_plane = nv12;
    const uint8_t* uv_plane = nv12 + size_t(w) * h;
    uint8_t* dst_bytes = out.data();
    for (int32_t y = 0; y < h; y += 2) {
        const uint8_t* yr0 = y_plane + size_t(y) * w;
        const uint8_t* yr1 = (y + 1 < h) ? (yr0 + w) : yr0;
        const uint8_t* uvr = uv_plane + size_t(y >> 1) * w;
        uint8_t* d0 = dst_bytes + size_t(y) * w * 4;
        uint8_t* d1 = (y + 1 < h) ? (d0 + size_t(w) * 4) : nullptr;
        int32_t x = 0;
#if defined(__aarch64__)
        for (; x + 15 < w; x += 16) {
            uint8x8x2_t uv = vld2_u8(uvr + x);
            yuv16_pair_to_rgba_neon(vld1q_u8(yr0 + x), vld1q_u8(yr1 + x), uv.val[0], uv.val[1], d0 + x * 4,
                                    d1 ? d1 + x * 4 : nullptr);
        }
#endif
        uint32_t* d0w = reinterpret_cast<uint32_t*>(d0);
        uint32_t* d1w = reinterpret_cast<uint32_t*>(d1);
        for (; x + 1 < w; x += 2) {
            int u = int(uvr[x]) - 128;
            int v = int(uvr[x + 1]) - 128;
            int rv = (459 * v) >> 8;
            int guv = (-55 * u - 136 * v) >> 8;
            int bu = (541 * u) >> 8;
            d0w[x] = pack_yuv_rgba((298 * (int(yr0[x]) - 16) + 128) >> 8, rv, guv, bu);
            d0w[x + 1] = pack_yuv_rgba((298 * (int(yr0[x + 1]) - 16) + 128) >> 8, rv, guv, bu);
            if (d1w) {
                d1w[x] = pack_yuv_rgba((298 * (int(yr1[x]) - 16) + 128) >> 8, rv, guv, bu);
                d1w[x + 1] = pack_yuv_rgba((298 * (int(yr1[x + 1]) - 16) + 128) >> 8, rv, guv, bu);
            }
        }
    }
}

void read_codec_format(Decoder& d) {
    AMediaFormat* f = AMediaCodec_getOutputFormat(d.codec);
    if (!f) return;
    AMediaFormat_getInt32(f, "color-format", &d.color);
    AMediaFormat_getInt32(f, "stride", &d.codec_stride);
    AMediaFormat_getInt32(f, "slice-height", &d.codec_rows);
    int32_t l = 0, t = 0, r = 0, b = 0;
    if (AMediaFormat_getRect(f, "crop", &l, &t, &r, &b)) d.crop_left = l, d.crop_top = t;

    d.has_planes = false;
    void* img_data = nullptr;
    size_t img_size = 0;
    media::Planes planes;
    if (AMediaFormat_getBuffer(f, "image-data", &img_data, &img_size) &&
        media::parse_image_data(img_data, img_size, planes)) {
        for (int p = 0; p < 3; p++) {
            d.plane_off[p] = planes.off[p];
            d.plane_col[p] = planes.col[p];
            d.plane_row[p] = planes.row[p];
        }
        d.has_planes = true;
        d.is_planar = (d.plane_col[1] == 1);
        if (d.plane_row[0] > 0) d.codec_stride = int32_t(d.plane_row[0]);
        log("movie: image-data planes: Y %u/%u/%u, U %u/%u/%u, V %u/%u/%u (offset/column/row)", d.plane_off[0],
            d.plane_col[0], d.plane_row[0], d.plane_off[1], d.plane_col[1], d.plane_row[1], d.plane_off[2],
            d.plane_col[2], d.plane_row[2]);
    }

    if (!d.has_planes) {
        if (d.color == kColorI420) {
            d.is_planar = true;
        } else if (d.color == kColorFlexible || d.color == 0) {
            // Google software decoders (c2.android.*, OMX.google.*) and legacy OMX.MTK.* output I420;
            // modern MediaTek Codec2 (c2.mtk.*) outputs semi-planar NV12 when image-data is absent.
            d.is_planar = (d.codec_name.find(".MTK.") != std::string::npos ||
                           d.codec_name.find(".android.") != std::string::npos ||
                           d.codec_name.find(".google.") != std::string::npos);
        } else {
            d.is_planar = false;
        }
    }

    log("movie: codec output: color format %#x (planar %d), stride %d, slice height %d, crop at %d,%d (%s)", d.color,
        d.is_planar, d.codec_stride, d.codec_rows, d.crop_left, d.crop_top, AMediaFormat_toString(f));
    diag::movie_buffers(uint64_t(output_slots()), d.nv12_size(),
                        uint64_t(d.max_input > 0 ? d.max_input : 0), uint64_t(d.codec_stride),
                        uint64_t(d.codec_rows));
    AMediaFormat_delete(f);
}

// Feeds the codec what it will take, then takes its pictures into free output slots (waiting up to `wait_us` for the
// first). True when an output became ready.
bool pump(Decoder& d, int64_t wait_us) {
    if (!d.codec) return false;
    const uint64_t pump_from = plat::monotonic_ns();
    while (!d.to_codec.empty()) {
        ssize_t i = AMediaCodec_dequeueInputBuffer(d.codec, 0);
        if (i < 0) break;
        size_t cap = 0;
        uint8_t* buf = AMediaCodec_getInputBuffer(d.codec, size_t(i), &cap);
        Decoder::Input& in = d.to_codec.front();
        size_t n = buf ? std::min(cap, in.data.size()) : 0;
        if (n) std::memcpy(buf, in.data.data(), n);
        AMediaCodec_queueInputBuffer(d.codec, size_t(i), 0, n, uint64_t(in.pts),
                                     (in.flags & kEndOfStream) ? AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM : 0);
        d.to_codec.pop_front();
    }
    bool any = false;
    for (int slot; (slot = d.free_slot()) >= 0;) {
        AMediaCodecBufferInfo info{};
        ssize_t o = AMediaCodec_dequeueOutputBuffer(d.codec, &info, any ? 0 : wait_us);
        if (o == AMEDIACODEC_INFO_OUTPUT_FORMAT_CHANGED) {
            read_codec_format(d);
            continue;
        }
        if (o == AMEDIACODEC_INFO_OUTPUT_BUFFERS_CHANGED) continue;
        if (o < 0) break;
        size_t size = 0;
        uint8_t* buf = AMediaCodec_getOutputBuffer(d.codec, size_t(o), &size);
        const uint64_t t0 = plat::monotonic_ns();
        if (buf && info.size > 0) {
            if (d.is_rgba()) {
                if (!codec_i420_to_rgba(d, buf + info.offset, size_t(info.size), d.frames[slot])) {
                    to_nv12(d, buf + info.offset, size_t(info.size), d.nv12_scratch);
                    nv12_to_rgba(d.nv12_scratch.data(), d.width, d.height, d.frames[slot]);
                }
            } else {
                to_nv12(d, buf + info.offset, size_t(info.size), d.frames[slot]);
            }
            const uint64_t t1 = plat::monotonic_ns();
            d.win.convert_ns += t1 - t0;
            diag::movie_convert(t1 - t0, t1);
        } else {
            d.frames[slot].clear();
            d.win.convert_ns += plat::monotonic_ns() - t0;
        }
        d.win.decoded++;
        AMediaCodec_releaseOutputBuffer(d.codec, size_t(o), false);
        const bool eos = info.flags & AMEDIACODEC_BUFFER_FLAG_END_OF_STREAM;
        d.codec_eos |= eos;
        d.outputs[slot] = {info.presentationTimeUs, eos ? kEndOfStream : 0u, true};
        d.ready.push_back(slot);
        d.decoded += info.size > 0;
        any = true;
    }
    const uint64_t pump_to = plat::monotonic_ns();
    const uint64_t pump_ns = pump_to - pump_from;
    d.win.max_pump_ns = std::max(d.win.max_pump_ns, pump_ns);
    diag::movie_pump(pump_ns, pump_to);
    return any;
}

// The device's video decoder, configured for the stream's MIME type (null: none, black frames).
AMediaCodec* make_codec(Decoder& d) {
    const char* mime = d.mime.empty() ? "video/x-vnd.on2.vp9" : d.mime.c_str();
    auto try_configure = [&](AMediaCodec* c) -> bool {
        if (!c) return false;
        AMediaFormat* f = AMediaFormat_new();
        AMediaFormat_setString(f, AMEDIAFORMAT_KEY_MIME, mime);
        AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_WIDTH, d.width);
        AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_HEIGHT, d.height);
        AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_MAX_INPUT_SIZE, d.max_input);
        AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_COLOR_FORMAT, 0x7F420888);  // COLOR_FormatYUV420Flexible
        media_status_t st = AMediaCodec_configure(c, f, nullptr, nullptr, 0);
        if (st != AMEDIA_OK) {
            AMediaFormat_delete(f);
            f = AMediaFormat_new();
            AMediaFormat_setString(f, AMEDIAFORMAT_KEY_MIME, mime);
            AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_WIDTH, d.width);
            AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_HEIGHT, d.height);
            AMediaFormat_setInt32(f, AMEDIAFORMAT_KEY_MAX_INPUT_SIZE, d.max_input);
            st = AMediaCodec_configure(c, f, nullptr, nullptr, 0);
        }
        AMediaFormat_delete(f);
        return st == AMEDIA_OK;
    };
    const bool force_sw = std::getenv("NX_MOVIE_SW") != nullptr;
    AMediaCodec* c = nullptr;
    if (!force_sw) {
        c = AMediaCodec_createDecoderByType(mime);
        if (c && !try_configure(c)) {
            AMediaCodec_delete(c);
            c = nullptr;
        }
    }
    if (!c && std::strcmp(mime, "video/x-vnd.on2.vp9") == 0) {
        for (const char* sw : {"c2.android.vp9.decoder", "OMX.google.vp9.decoder"}) {
            c = AMediaCodec_createCodecByName(sw);
            if (c && try_configure(c)) break;
            if (c) {
                AMediaCodec_delete(c);
                c = nullptr;
            }
        }
    }
    if (!c) {
        log("movie: no %s decoder accepted %dx%d: black frames", mime, d.width, d.height);
        return nullptr;
    }
    char* name = nullptr;
    if (AMediaCodec_getName(c, &name) == AMEDIA_OK && name) {
        log("movie: video decoder %s (%s)", name, mime);
        d.codec_name = name;
        d.is_planar = (d.codec_name.find(".MTK.") != std::string::npos ||
                       d.codec_name.find(".android.") != std::string::npos ||
                       d.codec_name.find(".google.") != std::string::npos);
        AMediaCodec_releaseName(c, name);
    }
    return c;
}
#endif

}  // namespace

// For the per-present film trace (NX_TRACE_VIDEO, nvn_queue.cpp): pictures the game took, and the last one's pts.
std::atomic<uint64_t> g_movie_gots{0};
std::atomic<int64_t> g_movie_last_pts{0};

namespace {

// NX_TRACE_VIDEO: each decoder event (acquired, taken, released) with its time, for the film's cadence.
void ev_trace(const char* what, int index, int extra = -1) {
    static long left = std::getenv("NX_TRACE_VIDEO") ? 3 * std::atol(std::getenv("NX_TRACE_VIDEO")) : 0;
    if (left <= 0) return;
    left--;
    log("movie trace: %s %d (%d) at %.1f ms, thread %lld", what, index, extra, double(plat::monotonic_ns() % 1000000000000ull) / 1e6,
        (long long)plat::current_thread_id());
}

std::mutex g_lock;
std::mutex g_signal_lock;  // serializes movie_pump_thread SignalEvent calls against DecoderStop/DecoderDtor
std::condition_variable g_pump_cv;
std::unordered_map<uint64_t, std::unique_ptr<Decoder>> g_decoders;
uint64_t g_next_id = 1;

Decoder* decoder(uint64_t self) {
    auto it = g_decoders.find(ld<uint64_t>(self + 8));
    return it == g_decoders.end() ? nullptr : it->second.get();
}

// The SDK's own nn::os::SignalEvent, outside our lock (the woken threads call back in).
void signal(CPU& c, uint64_t event) {
    static const uint64_t fn = find_symbol("_ZN2nn2os11SignalEventEPNS0_9EventTypeE");
    if (!event || !fn) return;
    c.x[0] = event;
    call(c, fn);
}

void signal_all(CPU& c, const std::vector<uint64_t>& events) {
    for (uint64_t e : events) signal(c, e);
}

void clear_event(CPU& c, uint64_t event) {
    static const uint64_t fn = find_symbol("_ZN2nn2os10ClearEventEPNS0_9EventTypeE");
    if (!event || !fn) return;
    c.x[0] = event;
    call(c, fn);
}

// D2 (+0x00) and D0 (+0x08): the deleting one frees the object with the game's operator delete.
void DecoderDtor(CPU& c) {
    uint64_t ev[4] = {};
    bool existed = false;
    {
        std::lock_guard slk(g_signal_lock);
        std::lock_guard lk(g_lock);
        auto it = g_decoders.find(ld<uint64_t>(c.x[0] + 8));
        if (it != g_decoders.end()) {
            existed = true;
            for (int k = 0; k < 4; k++) ev[k] = it->second->events[k];
#if defined(__ANDROID__)
            if (it->second->codec) AMediaCodec_delete(it->second->codec);
#endif
            g_decoders.erase(it);
        }
    }
    if (existed) {
        for (int k = 0; k < 4; k++) clear_event(c, ev[k]);
        plat::sleep_ns(2'000'000);
    }
}
void DecoderDeletingDtor(CPU& c) {
    const uint64_t self = c.x[0];
    DecoderDtor(c);
    static const uint64_t del = find_symbol("_ZdlPv");
    if (del) {
        c.x[0] = self;
        call(c, del);
    }
}

// Status Configure(MediaData*)
void DecoderConfigure(CPU& c) {
    std::lock_guard lk(g_lock);
    Decoder* d = decoder(c.x[0]);
    if (!d) return void(c.x[0] = kTryAgain);
    md_get(c.x[1], "width", d->width);
    md_get(c.x[1], "height", d->height);
    md_get(c.x[1], "max-input-size", d->max_input);
    if (d->max_input <= 0) d->max_input = 2 << 20;
    if (MediaMap* m = media(c.x[1])) {
        if (auto it = m->find("mime"); it != m->end() && !it->second.s.empty()) d->mime = it->second.s;
    }
    if (d->mime.empty()) {
        if (d->type == 1) d->mime = "video/avc";
        else if (d->type == 2) d->mime = "video/x-vnd.on2.vp8";
        else d->mime = "video/x-vnd.on2.vp9";
    }
    log("movie: decoder configured %dx%d (%s, fmt %u, mode %u), input buffers of %d bytes", d->width, d->height,
        d->mime.c_str(), d->format, d->mode, d->max_input);
    diag::movie_buffers(uint64_t(output_slots()), d->output_size(), uint64_t(d->max_input), uint64_t(d->stride()),
                        uint64_t(d->rows()));
#if defined(__ANDROID__)
    if (!d->codec && d->width > 0 && d->height > 0 && d->mime.rfind("video/", 0) == 0) d->codec = make_codec(*d);
#endif
    c.x[0] = 0;
}

// Status RegisterClientEvents(DecoderEvents*)
void DecoderRegisterClientEvents(CPU& c) {
    std::lock_guard lk(g_lock);
    if (Decoder* d = decoder(c.x[0]))
        for (int k = 0; k < 4; k++) d->events[k] = c.x[1] ? ld<uint64_t>(c.x[1] + 8ull * k) : 0;
    c.x[0] = 0;
}

void put_format(const Decoder& d, uint64_t md) {
    md_set(md, "width", d.width);
    md_set(md, "height", d.height);
    md_set(md, "stride", d.stride());
    md_set(md, "slice-height", d.rows());
    md_set_rect(md, "crop", 0, 0, d.width - 1, d.height - 1);
    md_set(md, "nv12-y-stride", d.stride());
    md_set(md, "nv12-y-offset", 0);
    md_set(md, "nv12-uv-stride", d.stride());
    md_set(md, "nv12-uv-offset", d.stride() * d.rows());
}

// Status GetOutputFormat(MediaData*)
void DecoderGetOutputFormat(CPU& c) {
    std::lock_guard lk(g_lock);
    Decoder* d = decoder(c.x[0]);
    if (d) put_format(*d, c.x[1]);
    c.x[0] = d ? 0 : kTryAgain;
}

#if defined(__ANDROID__)
// The codec's pictures come out on their own time, a few to tens of milliseconds after their input. The calls above
// only take them when the game makes one, and CriVpdSwitch waits for the output event before it feeds much more: a
// picture that came out after the last call stalled the video for good (the casino opening after New Game, 424
// frames in, at ~15 fps until then). This guest thread drains every started codec every 2 ms and signals the event.
void movie_pump_thread(CPU& c) {
    plat::pin_current_thread_to_little_cores("movie_pump");
    log("movie: pump thread running");
    for (;;) {
        std::vector<uint64_t> events;
        bool dump = false;
        {
            std::unique_lock lk(g_lock);
            g_pump_cv.wait_for(lk, std::chrono::milliseconds(16), [] {
                for (const auto& [id, d] : g_decoders)
                    if (d->started && !d->codec_eos) return true;
                return false;
            });
            bool any_active = false;
            for (const auto& [id, d] : g_decoders)
                if (d->started && !d->codec_eos) { any_active = true; break; }
            if (!any_active) {
                g_pump_cv.wait(lk, [] {
                    for (const auto& [id, d] : g_decoders)
                        if (d->started && !d->codec_eos) return true;
                    return false;
                });
            } else {
                lk.unlock();
                plat::sleep_ns(2'000'000);
            }
        }
        {
            std::lock_guard slk(g_signal_lock);
            {
                std::lock_guard lk(g_lock);
                const uint64_t now = plat::monotonic_ns();
                for (auto& [id, d] : g_decoders) {
                    if (d->started && d->codec && pump(*d, 0)) events.push_back(d->events[kOutputAvailable]);
                    if (d->started && d->need_input && !d->eos_sent && !d->free_inputs.empty() && d->to_codec.size() < 4) {
                        d->need_input = false;
                        events.push_back(d->events[kInputAvailable]);
                    }
                    if (!d->started) continue;
                    // NX_MOVIE_DUMP=<n>: the first n times a playing film gets no input for a second, where every guest
                    // thread is (what the game's reader waits for).
                    static int dumps_left = std::getenv("NX_MOVIE_DUMP") ? std::atoi(std::getenv("NX_MOVIE_DUMP")) : 0;
                    if (dumps_left > 0 && d->last_sent_ns && !d->eos_sent && !d->starve_logged &&
                        now - d->last_sent_ns > 1'000'000'000ull) {
                        int busy = 0;
                        std::string held;
                        for (const auto& o : d->outputs)
                            if (o.busy) busy++, held += " " + std::to_string(o.pts / 1000);
                        log("movie: no input for %.0f ms (%llu frames in, last pts %lld ms, last taken %lld ms; ready %zu, "
                            "slots busy %d, their pts in ms:%s): threads follow", double(now - d->last_sent_ns) / 1e6,
                            (unsigned long long)d->frames_in, (long long)(d->last_sent_pts / 1000),
                            (long long)(d->last_got_pts / 1000), d->ready.size(), busy, held.c_str());
                        d->starve_logged = true, dumps_left--, dump = true;
                    }
                    if (!d->win_from_ns) d->win_from_ns = now;
                    if (now - d->win_from_ns < 2'000'000'000ull) continue;
                    // What the video did over the window: pictures the game sent, the codec decoded, the game took
                    // (got = copied into its memory, about one per shown frame) and the longest wait between two.
                    const auto& w = d->win;
                    const double s = double(now - d->win_from_ns) / 1e9;
                    int busy = 0;
                    for (const auto& o : d->outputs) busy += o.busy;
                    diag::movie_queues(d->to_codec.size(), d->ready.size(), uint64_t(busy));
                    log("movie: per s: sent %.1f, decoded %.1f, acquired %.1f (+%.1f empty), got %.1f, released %.1f; "
                        "longest gap between gots %.0f ms; convert %.2f ms, copy %.2f ms, send wait %.2f ms per picture, "
                        "longest pump %.2f ms; queued for the codec %zu, ready %zu, slots busy %d",
                        w.sent / s, w.decoded / s, w.acquired / s, w.empty_acquires / s, w.got / s, w.released / s,
                        double(w.max_got_gap_ns) / 1e6, w.decoded ? double(w.convert_ns) / 1e6 / double(w.decoded) : 0.0,
                        w.got ? double(w.copy_ns) / 1e6 / double(w.got) : 0.0,
                        w.sent ? double(w.send_wait_ns) / 1e6 / double(w.sent) : 0.0, double(w.max_pump_ns) / 1e6,
                        d->to_codec.size(), d->ready.size(), busy);
                    d->win = {};
                    d->win_from_ns = now;
                }
            }
            if (dump) log_guest_threads();
            signal_all(c, events);
        }
    }
}

void start_pump_thread(CPU& c) {
    static bool started = false;
    if (started) return;
    started = true;
    uint64_t create = find_symbol("_ZN2nn2os12CreateThreadEPNS0_10ThreadTypeEPFvPvES3_S3_mi");
    uint64_t start = find_symbol("_ZN2nn2os11StartThreadEPNS0_10ThreadTypeE");
    if (!create || !start) return log("movie: nn::os::CreateThread/StartThread not found, no pump thread");
    uint64_t thread = kernel_alloc(0x1000), stack = kernel_alloc(0x20000);
    uint64_t saved[31];
    std::memcpy(saved, c.x, sizeof saved);
    c.x[0] = thread, c.x[1] = hle_address("__movie_pump_thread"), c.x[2] = 0, c.x[3] = stack, c.x[4] = 0x20000;
    c.x[5] = 16;  // high priority: it only polls
    call(c, create);
    uint64_t rc = c.x[0];
    c.x[0] = thread;
    if (rc == 0) call(c, start);
    std::memcpy(c.x, saved, sizeof saved);
    if (rc) log("movie: CreateThread failed (%llx)", (unsigned long long)rc);
}
#endif

// Status Start(), Stop(), Reset(), Flush()
void DecoderStart(CPU& c) {
    uint64_t input_event = 0;
    {
        std::lock_guard lk(g_lock);
        if (Decoder* d = decoder(c.x[0])) {
            d->reset_queues();
            d->started = true;
            input_event = d->events[kInputAvailable];
#if defined(__ANDROID__)
            if (d->codec && d->codec_stopped) {
                AMediaCodec_delete(d->codec);
                d->codec = make_codec(*d);
                d->codec_stopped = false;
                d->codec_running = false;
            }
            if (d->codec && d->codec_running) {
                // Dread loops videos with Flush() -> Start() without Stop(): AMediaCodec is already Running,
                // so calling AMediaCodec_start() again returns INVALID_STATE. Flush keeps it running cleanly.
                if (AMediaCodec_flush(d->codec) != AMEDIA_OK) {
                    AMediaCodec_delete(d->codec);
                    d->codec = make_codec(*d);
                    d->codec_running = false;
                }
            }
            if (d->codec && !d->codec_running) {
                if (AMediaCodec_start(d->codec) == AMEDIA_OK) {
                    d->codec_running = true;
                } else {
                    log("movie: the VP9 decoder did not start: black frames");
                    AMediaCodec_delete(d->codec);
                    d->codec = nullptr;
                }
            }
#endif
            g_pump_cv.notify_all();
        }
    }
#if defined(__ANDROID__)
    start_pump_thread(c);
#endif
    signal(c, input_event);
    c.x[0] = 0;
}
void DecoderStop(CPU& c) {
    uint64_t ev[4] = {};
    bool was_started = false;
    {
        std::lock_guard slk(g_signal_lock);
        std::lock_guard lk(g_lock);
        if (Decoder* d = decoder(c.x[0])) {
            log("movie: decoder stopped at present %llu, %.1f s, after %llu frames in, %llu out (%llu decoded)",
                (unsigned long long)g_frame.load(), double(plat::monotonic_ns()) / 1e9, (unsigned long long)d->frames_in,
                (unsigned long long)d->frames_out, (unsigned long long)d->decoded);
            was_started = d->started;
            for (int k = 0; k < 4; k++) ev[k] = d->events[k];
#if defined(__ANDROID__)
            if (d->codec && d->codec_running) {
                AMediaCodec_stop(d->codec);
                d->codec_running = false;
                d->codec_stopped = true;
            }
#endif
            d->started = false;
            d->reset_queues();
            g_pump_cv.notify_all();
        }
    }
    if (was_started) {
        for (int k = 0; k < 4; k++) clear_event(c, ev[k]);
        plat::sleep_ns(2'000'000);
    }
    c.x[0] = 0;
}
void DecoderFlush(CPU& c) {
    uint64_t input_event = 0;
    {
        std::lock_guard lk(g_lock);
        if (Decoder* d = decoder(c.x[0])) {
#if defined(__ANDROID__)
            if (d->codec && d->started) AMediaCodec_flush(d->codec);
#endif
            d->reset_queues();
            d->format_sent = false;
            input_event = d->started ? d->events[kInputAvailable] : 0;
        }
    }
    signal(c, input_event);
    c.x[0] = 0;
}

// Status AcquireInputBufferIndex(int* index, int* more)
void DecoderAcquireInputBufferIndex(CPU& c) {
    std::lock_guard lk(g_lock);
    Decoder* d = decoder(c.x[0]);
    if (!d || !d->started || d->free_inputs.empty()) return void(c.x[0] = kTryAgain);
#if defined(__ANDROID__)
    if (d->eos_sent) return void(c.x[0] = kTryAgain);
    if (d->codec && d->to_codec.size() >= 4) {
        d->need_input = true;
        return void(c.x[0] = kTryAgain);
    }
#endif
    int index = d->free_inputs.front();
    d->free_inputs.pop_front();
    if (c.x[1]) st<int32_t>(c.x[1], index);
    // Report 0 more so Dread's event thread queues one input buffer per wakeup instead of flooding MessageQueue with
    // 4 duplicate wrapper pointers (SendInputBufferForDecode re-signals kInputAvailable whenever to_codec < 4).
    if (c.x[2]) st<int32_t>(c.x[2], 0);
    c.x[0] = 0;
}

// Status GetInputBuffer(int index, Buffer*)
void DecoderGetInputBuffer(CPU& c) {
    std::lock_guard lk(g_lock);
    Decoder* d = decoder(c.x[0]);
    int index = int(c.x[1]);
    if (!d || index < 0 || index >= kInputs || !c.x[2]) return void(c.x[0] = kTryAgain);
    auto& mem = d->inputs[index];
    if (mem.size() < size_t(d->max_input)) mem.assign(size_t(d->max_input), 0);
    st<uint64_t>(c.x[2] + kBufBase, reinterpret_cast<uint64_t>(mem.data()));
    st<uint64_t>(c.x[2] + kBufCapacity, mem.size());
    st<int32_t>(c.x[2] + kBufOffset, 0);
    st<int32_t>(c.x[2] + kBufSize, 0);
    c.x[0] = 0;
}

// Status SendInputBufferForDecode(int index, int offset, int size, int64_t pts, uint32_t flags)
void DecoderSendInputBufferForDecode(CPU& c) {
    std::vector<uint64_t> events;
    {
        std::lock_guard lk(g_lock);
        Decoder* d = decoder(c.x[0]);
        int index = int(c.x[1]);
        if (!d || index < 0 || index >= kInputs) return void(c.x[0] = kTryAgain);
        const int64_t pts = int64_t(c.x[4]);
        const uint32_t flags = uint32_t(c.x[5]);
#if defined(__ANDROID__)
        if (d->eos_sent) {
            d->free_inputs.push_back(index);
            return void(c.x[0] = 0);
        }
#endif
        if (d->frames_in++ == 0)
            log("movie: first input frame, %d bytes, pts %lld, flags %#x", int(c.x[3]), (long long)pts, flags);
        bool output = true;
        bool want_input = (flags & kEndOfStream) == 0;
#if defined(__ANDROID__)
        d->eos_sent |= (flags & kEndOfStream) != 0;
        if (d->codec) {
            // The bytes are the game's (its input buffer goes back at once): copied for the codec. A picture usually
            // comes out a frame or two later; at the end of the stream everything still inside is waited for.
            const uint8_t* src = d->inputs[index].data() + std::max<int64_t>(0, int64_t(c.x[2]));
            size_t n = std::min<size_t>(size_t(std::max<int64_t>(0, int64_t(c.x[3]))),
                                        d->inputs[index].size() - std::min<size_t>(d->inputs[index].size(), size_t(c.x[2])));
            d->to_codec.push_back({std::vector<uint8_t>(src, src + n), pts, flags & kEndOfStream});
            const uint64_t t0 = plat::monotonic_ns();
            output = pump(*d, 0);
            for (int k = 0; (flags & kEndOfStream) && !d->codec_eos && k < 15 && d->free_slot() >= 0; k++)
                output |= pump(*d, 20000);
            const uint64_t t1 = plat::monotonic_ns();
            d->win.send_wait_ns += t1 - t0;
            diag::movie_send_wait(t1 - t0, t1);
            d->win.sent++;
            d->last_sent_ns = plat::monotonic_ns(), d->starve_logged = false, d->last_sent_pts = pts;
            if (d->eos_sent) {
                want_input = false;
                d->need_input = false;
            } else if (d->to_codec.size() >= 4) {
                want_input = false;
                d->need_input = true;
            }
            g_pump_cv.notify_all();
        } else {
            d->emit(pts, flags & kEndOfStream);
            if (d->eos_sent) {
                want_input = false;
                d->need_input = false;
            }
        }
#else
        d->emit(pts, flags & kEndOfStream);
#endif
        d->free_inputs.push_back(index);
        if (!d->format_sent) d->format_sent = true, events.push_back(d->events[kFormatChanged]);
        if (output) events.push_back(d->events[kOutputAvailable]);
        if (want_input) events.push_back(d->events[kInputAvailable]);
    }
    signal_all(c, events);
    c.x[0] = 0;
}

// Status AcquireOutputBufferIndex(int* index, int64_t* pts, uint32_t* flags, int* more)
void DecoderAcquireOutputBufferIndex(CPU& c) {
    std::lock_guard lk(g_lock);
    Decoder* d = decoder(c.x[0]);
#if defined(__ANDROID__)
    if (d && d->ready.empty()) pump(*d, 0);
#endif
    if (d && d->ready.empty()) d->win.empty_acquires++;
    if (!d || d->ready.empty()) return void(c.x[0] = kTryAgain);
    int index = d->ready.front();
    d->ready.pop_front();
    ev_trace("acquired", index, int(d->ready.size()));
    d->win.acquired++;
    if (c.x[1]) st<int32_t>(c.x[1], index);
    if (c.x[2]) st<int64_t>(c.x[2], d->outputs[index].pts);
    if (c.x[3]) st<uint32_t>(c.x[3], d->outputs[index].flags);
    if (c.x[4]) st<int32_t>(c.x[4], int32_t(d->ready.size()));
    c.x[0] = 0;
}

// The frame into the client's memory: black RGBA8 (format 2/3) or black NV12 (format 4).
void fill_black(const Decoder& d, uint64_t buffer) {
    uint8_t* base = reinterpret_cast<uint8_t*>(ld<uint64_t>(buffer + kBufBase));
    uint64_t cap = ld<uint64_t>(buffer + kBufCapacity);
    if (!base || !cap) return;
    const uint64_t want = std::min<uint64_t>(cap, d.output_size());
    if (d.is_rgba()) {
        std::memset(base, 0, want);
        for (uint64_t i = 3; i < want; i += 4) base[i] = 255;
    } else {
        uint64_t y = std::min<uint64_t>(cap, uint64_t(d.width) * d.height);
        std::memset(base, 16, y);
        std::memset(base + y, 128, std::min<uint64_t>(cap - y, d.nv12_size() - y));
    }
    st<int32_t>(buffer + kBufOffset, 0);
    st<int32_t>(buffer + kBufSize, int32_t(want));
}

// Status GetOutputBuffer(int index, Buffer*) and GetOutputBuffer(int index, Buffer*, MediaData*)
void DecoderGetOutputBuffer(CPU& c) {
    std::lock_guard lk(g_lock);
    Decoder* d = decoder(c.x[0]);
    int index = int(c.x[1]);
    if (!d || index < 0 || index >= output_slots() || !d->outputs[index].busy || !c.x[2]) return void(c.x[0] = kTryAgain);
    static bool said = false;
    if (!said)
        said = true, log("movie: output format %u (3: ABGR/RGBA, 4: NV12), mode %u, frame %dx%d", d->format, d->mode,
                         d->width, d->height);
    const std::vector<uint8_t>& frame = d->frames[index];
    uint64_t base_addr = ld<uint64_t>(c.x[2] + kBufBase);
    uint8_t* base = reinterpret_cast<uint8_t*>(base_addr);
    uint64_t cap = ld<uint64_t>(c.x[2] + kBufCapacity);
    const uint64_t t0 = plat::monotonic_ns();
    // DecoderMode_NativeTexture (mode == 2): Metroid Dread passes NVNtexture* directly as Buffer::Base() with cap == 8.
    // Try base_addr directly first; fall back to *base_addr if a game passes NVNtexture**. Never overwrite base_addr.
    if (d->mode == 2 || cap <= 16) {
        auto upload_to = [&](const uint8_t* pixels, size_t sz) {
            if (!base_addr) return;
            if (!nvn::write_texture_rgba(base_addr, uint32_t(d->width), uint32_t(d->height), pixels, sz) && cap >= 8) {
                uint64_t indirect = ld<uint64_t>(base_addr);
                if (indirect)
                    nvn::write_texture_rgba(indirect, uint32_t(d->width), uint32_t(d->height), pixels, sz);
            }
        };
        if (d->width > 0 && d->height > 0) {
            if (!frame.empty() && d->is_rgba()) {
                upload_to(frame.data(), frame.size());
            } else {
                const size_t want = size_t(d->width) * d->height * 4;
                if (d->black_rgba.size() != want) {
                    d->black_rgba.assign(want, 0);
                    for (size_t i = 3; i < want; i += 4) d->black_rgba[i] = 255;
                }
                upload_to(d->black_rgba.data(), d->black_rgba.size());
            }
        }
        if (uint64_t sync_ptr = ld<uint64_t>(c.x[2] + kBufNvnSync)) nvn::signal_video_sync(sync_ptr);
    } else {
        // The game's textures over this memory are write-watched (a fault per 4 KB page otherwise: 760 per 1080p frame).
        if (base && cap) plat::write_watch_will_write(base_addr, std::min<uint64_t>(cap, d->output_size()));
        if (!frame.empty() && base && cap) {  // a decoded picture
            std::memcpy(base, frame.data(), std::min<uint64_t>(cap, frame.size()));
            st<int32_t>(c.x[2] + kBufOffset, 0);
            st<int32_t>(c.x[2] + kBufSize, int32_t(std::min<uint64_t>(cap, frame.size())));
        } else {
            fill_black(*d, c.x[2]);
        }
    }
    const uint64_t t1 = plat::monotonic_ns();
    d->win.copy_ns += t1 - t0;
    diag::movie_copy(t1 - t0, t1);
    d->win.got++;
    d->last_got_pts = d->outputs[index].pts;
    g_movie_last_pts = d->last_got_pts, g_movie_gots++;
    ev_trace("taken", index, int(base_addr & 0xfffffff));
    if (d->last_got_ns) d->win.max_got_gap_ns = std::max(d->win.max_got_gap_ns, t1 - d->last_got_ns);
    d->last_got_ns = t1;
    d->frames_out++;
    c.x[0] = 0;
}
void DecoderGetOutputBufferWithFormat(CPU& c) {
    const uint64_t self = c.x[0], md = c.x[3];
    DecoderGetOutputBuffer(c);
    std::lock_guard lk(g_lock);
    if (Decoder* d = decoder(self); d && c.x[0] == 0) put_format(*d, md);
}

// Status GetOutputBufferSize(int* size, int* ?)
void DecoderGetOutputBufferSize(CPU& c) {
    std::lock_guard lk(g_lock);
    Decoder* d = decoder(c.x[0]);
    if (!d) return void(c.x[0] = kTryAgain);
    if (c.x[1]) st<int32_t>(c.x[1], int32_t(d->output_size()));
    if (c.x[2]) st<int32_t>(c.x[2], output_slots());
    diag::movie_buffers(uint64_t(output_slots()), d->output_size(), uint64_t(d->max_input > 0 ? d->max_input : 0),
                        uint64_t(d->stride()), uint64_t(d->rows()));
    c.x[0] = 0;
}

// Status ReleaseOutputBufferIndex(int index)
void DecoderReleaseOutputBufferIndex(CPU& c) {
    std::vector<uint64_t> events;
    {
        std::lock_guard lk(g_lock);
        Decoder* d = decoder(c.x[0]);
        int index = int(c.x[1]);
        if (d && index >= 0 && index < output_slots()) {
            d->outputs[index].busy = false;
            d->win.released++;
            ev_trace("released", index);
            if (!d->pending.empty()) {  // a slot is free again for a frame that waited
                auto [pts, flags] = d->pending.front();
                d->pending.pop_front();
                d->emit(pts, flags);
                events.push_back(d->events[kOutputAvailable]);
            }
#if defined(__ANDROID__)
            // A picture the codec held for want of a slot; after the end of stream, the last ones are waited for.
            if (d->started && d->codec) {
                for (int k = 0; k < (d->eos_sent && !d->codec_eos ? 15 : 1); k++)
                    if (pump(*d, d->eos_sent && !d->codec_eos ? 20000 : 0)) {
                        events.push_back(d->events[kOutputAvailable]);
                        break;
                    }
            }
            if (d->started && d->need_input && !d->eos_sent && !d->free_inputs.empty() && d->to_codec.size() < 4) {
                d->need_input = false;
                events.push_back(d->events[kInputAvailable]);
            }
#endif
        }
    }
    signal_all(c, events);
    c.x[0] = 0;
}

// Status GetOutputBufferFormat(int index, MediaData*)
void DecoderGetOutputBufferFormat(CPU& c) {
    std::lock_guard lk(g_lock);
    Decoder* d = decoder(c.x[0]);
    if (d) put_format(*d, c.x[2]);
    c.x[0] = d ? 0 : kTryAgain;
}

void DecoderGetLastError(CPU& c) { c.x[0] = 0; }

const std::array<const char*, 19> kVtable = {
    "movie::Decoder::~Decoder", "movie::Decoder::~Decoder (deleting)", "movie::Decoder::Configure",
    "movie::Decoder::RegisterClientEvents", "movie::Decoder::GetOutputFormat", "movie::Decoder::Start",
    "movie::Decoder::Stop", "movie::Decoder::Reset", "movie::Decoder::Flush",
    "movie::Decoder::AcquireInputBufferIndex", "movie::Decoder::GetInputBuffer",
    "movie::Decoder::SendInputBufferForDecode", "movie::Decoder::AcquireOutputBufferIndex",
    "movie::Decoder::GetOutputBuffer", "movie::Decoder::GetOutputBufferSize",
    "movie::Decoder::ReleaseOutputBufferIndex", "movie::Decoder::GetLastError",
    "movie::Decoder::GetOutputBuffer (format)", "movie::Decoder::GetOutputBufferFormat"};

// Decoder::Decoder(DecoderType, DecoderOutputFormat, DecoderMode, int priority)
void DecoderCtor(CPU& c) {
    static uint64_t vtable[kVtable.size()];
    static std::once_flag once;
    std::call_once(once, [] {
        for (size_t k = 0; k < kVtable.size(); k++) vtable[k] = hle_address(kVtable[k]);
    });
    const uint64_t self = c.x[0];
    auto d = std::make_unique<Decoder>();
    d->type = uint32_t(c.x[1]), d->format = uint32_t(c.x[2]), d->mode = uint32_t(c.x[3]);
    log("movie: Decoder(type %u, output format %u, mode %u, priority %d) at present %llu, %.1f s", d->type, d->format,
        d->mode, int(c.x[4]), (unsigned long long)g_frame.load(), double(plat::monotonic_ns()) / 1e9);
    std::lock_guard lk(g_lock);
    const uint64_t id = g_next_id++;
    g_decoders[id] = std::move(d);
    st<uint64_t>(self, reinterpret_cast<uint64_t>(vtable));
    st<uint64_t>(self + 8, id);
}

// bool CreateVp9DecoderConfigurationData(int width, int height, int frame rate x 1000, MediaData*)
void CreateVp9DecoderConfigurationData(CPU& c) {
    const int32_t w = int32_t(c.x[0]), h = int32_t(c.x[1]);
    const uint64_t md = c.x[3];
    if (MediaMap* m = media(md)) (*m)["mime"].s = "video/x-vnd.on2.vp9";
    md_set(md, "width", w);
    md_set(md, "height", h);
    md_set(md, "frame-rate", int32_t(c.x[2]));
    md_set(md, "max-input-size", std::max(1 << 20, w * h * 3 / 2));
    c.x[0] = 1;
}

int32_t g_priority = 16;
void SetAllocator(CPU& c) { c.x[0] = 0; }  // our buffers are host memory
void SetCoreMask(CPU& c) { c.x[0] = 0; }
void SetThreadPriority(CPU& c) {
    g_priority = int32_t(c.x[0]);
    c.x[0] = 0;
}
void GetThreadPriority(CPU& c) { c.x[0] = uint32_t(g_priority); }

// ---- Extractor (Android AMediaExtractor for WebM/MP4/MKV) --------------------------------------------------------
constexpr uint32_t kStatusEos = 0x2000;
constexpr uint32_t kStatusErr = 0x2001;

struct ExtractorObj {
    uint32_t container = 0;
    std::string path;
    int64_t duration_us = 0;
#if defined(__ANDROID__)
    AMediaExtractor* ex = nullptr;
    int fd = -1;
    ~ExtractorObj() {
        if (ex) AMediaExtractor_delete(ex);
        if (fd >= 0) close(fd);
    }
#endif
};

std::mutex g_ex_lock;
std::unordered_map<uint64_t, std::unique_ptr<ExtractorObj>> g_extractors;

ExtractorObj* get_extractor(uint64_t self) {
    auto it = g_extractors.find(self);
    return it == g_extractors.end() ? nullptr : it->second.get();
}

#if defined(__ANDROID__)
int make_mem_fd(const uint8_t* data, size_t size) {
    int fd = -1;
#ifdef __NR_memfd_create
    fd = int(syscall(__NR_memfd_create, "nx_movie", 1u /* MFD_CLOEXEC */));
#endif
    if (fd < 0) {
        static std::atomic<uint64_t> seq{0};
        std::string tmp = plat::data_root() + "/.movie_" + std::to_string(seq.fetch_add(1)) + ".tmp";
        fd = open(tmp.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
        if (fd >= 0) unlink(tmp.c_str());
    }
    if (fd < 0) return -1;
    size_t off = 0;
    while (off < size) {
        ssize_t w = write(fd, data + off, size - off);
        if (w <= 0) {
            close(fd);
            return -1;
        }
        off += size_t(w);
    }
    lseek(fd, 0, SEEK_SET);
    return fd;
}

bool extractor_load_bytes(ExtractorObj& e, const uint8_t* data, size_t size) {
    if (e.ex) AMediaExtractor_delete(e.ex), e.ex = nullptr;
    if (e.fd >= 0) close(e.fd), e.fd = -1;
    e.fd = make_mem_fd(data, size);
    if (e.fd < 0) return false;
    e.ex = AMediaExtractor_new();
    if (!e.ex) return false;
    media_status_t st = AMediaExtractor_setDataSourceFd(e.ex, e.fd, 0, off64_t(size));
    if (st != AMEDIA_OK) {
        log("movie: AMediaExtractor_setDataSourceFd failed (%d) for %s (%zu bytes)", int(st), e.path.c_str(), size);
        AMediaExtractor_delete(e.ex), e.ex = nullptr;
        return false;
    }
    size_t ntracks = AMediaExtractor_getTrackCount(e.ex);
    for (size_t i = 0; i < ntracks; i++) {
        AMediaFormat* fmt = AMediaExtractor_getTrackFormat(e.ex, i);
        if (!fmt) continue;
        int64_t dur = 0;
        if (AMediaFormat_getInt64(fmt, AMEDIAFORMAT_KEY_DURATION, &dur) && dur > e.duration_us) e.duration_us = dur;
        const char* mime = nullptr;
        int32_t w = 0, h = 0;
        AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &mime);
        AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, &w);
        AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, &h);
        log("movie: Extractor %s track %zu: mime=%s %dx%d dur=%.2fs", e.path.c_str(), i, mime ? mime : "?", w, h,
            double(dur) / 1e6);
        AMediaFormat_delete(fmt);
    }
    return true;
}
#endif

void ExtractorCtor(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    auto e = std::make_unique<ExtractorObj>();
    e->container = uint32_t(c.x[1]);
    log("movie: Extractor(container %u) this=%llx", e->container, (unsigned long long)c.x[0]);
    g_extractors[c.x[0]] = std::move(e);
}
void ExtractorDtor(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    g_extractors.erase(c.x[0]);
}
void ExtractorSetDataSourcePath(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
    const char* raw = reinterpret_cast<const char*>(c.x[1]);
    if (!e || !raw) return void(c.x[0] = kStatusErr);
    e->path = raw;
    std::string rel = raw;
    if (rel.rfind("rom:", 0) == 0) rel = rel.substr(4);
    rel = game_image_path(rel);
    std::vector<uint8_t> bytes;
    if (const nxi::GameImage* img = game_image()) {
        if (const nxi::GameImage::Entry* ent = img->find(rel)) {
            bytes.resize(size_t(ent->size));
            if (img->read(*ent, 0, bytes.data(), bytes.size()) != bytes.size()) bytes.clear();
        }
    }
    if (bytes.empty()) {
        // Try host extract/romfs fallback
        std::string host = plat::data_root() + "/extract/romfs/" + rel;
        if (FILE* f = plat::fopen_utf8(host, "rb")) {
            std::fseek(f, 0, SEEK_END);
            long sz = std::ftell(f);
            std::fseek(f, 0, SEEK_SET);
            if (sz > 0) {
                bytes.resize(size_t(sz));
                if (std::fread(bytes.data(), 1, bytes.size(), f) != bytes.size()) bytes.clear();
            }
            std::fclose(f);
        }
    }
    if (bytes.empty()) {
        log("movie: Extractor::SetDataSource(%s) -> file not found (rel=%s)", raw, rel.c_str());
        return void(c.x[0] = kStatusErr);
    }
#if defined(__ANDROID__)
    bool ok = extractor_load_bytes(*e, bytes.data(), bytes.size());
    c.x[0] = ok ? 0 : kStatusErr;
#else
    c.x[0] = kStatusErr;
#endif
}
void ExtractorSetDataSourceMem(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
    const uint8_t* data = reinterpret_cast<const uint8_t*>(c.x[1]);
    int32_t sz = int32_t(c.x[2]);
    if (!e || !data || sz <= 0) return void(c.x[0] = kStatusErr);
    e->path = "<memory>";
#if defined(__ANDROID__)
    c.x[0] = extractor_load_bytes(*e, data, size_t(sz)) ? 0 : kStatusErr;
#else
    c.x[0] = kStatusErr;
#endif
}
void ExtractorSetDataSourceStream(CPU& c) {
    log("movie: Extractor::SetDataSource(ClientStreamReader) not implemented");
    c.x[0] = kStatusErr;
}
void ExtractorGetTrackCount(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
#if defined(__ANDROID__)
    if (!e || !e->ex) {
        if (c.x[1]) st<int32_t>(c.x[1], 0);
        return void(c.x[0] = kStatusErr);
    }
    int32_t n = int32_t(AMediaExtractor_getTrackCount(e->ex));
    if (c.x[1]) st<int32_t>(c.x[1], n);
    c.x[0] = 0;
#else
    if (c.x[1]) st<int32_t>(c.x[1], 0);
    c.x[0] = kStatusErr;
#endif
}
void ExtractorGetTrackConfiguration(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
    int32_t idx = int32_t(c.x[1]);
    uint64_t md = c.x[2];
#if defined(__ANDROID__)
    if (!e || !e->ex || !md || idx < 0 || size_t(idx) >= AMediaExtractor_getTrackCount(e->ex))
        return void(c.x[0] = kStatusErr);
    AMediaFormat* fmt = AMediaExtractor_getTrackFormat(e->ex, size_t(idx));
    if (!fmt) return void(c.x[0] = kStatusErr);
    if (MediaMap* m = media(md)) m->clear();
    const char* mime = nullptr;
    if (AMediaFormat_getString(fmt, AMEDIAFORMAT_KEY_MIME, &mime) && mime) md_set_str(md, "mime", mime);
    int32_t iv = 0;
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_WIDTH, &iv)) md_set(md, "width", iv);
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_HEIGHT, &iv)) md_set(md, "height", iv);
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_MAX_INPUT_SIZE, &iv)) md_set(md, "max-input-size", iv);
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_FRAME_RATE, &iv)) md_set(md, "frame-rate", iv);
    else {
        float fv = 0.0f;
        if (AMediaFormat_getFloat(fmt, AMEDIAFORMAT_KEY_FRAME_RATE, &fv) && fv > 0) md_set(md, "frame-rate", int32_t(fv + 0.5f));
    }
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_CHANNEL_COUNT, &iv)) md_set(md, "channel-count", iv);
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_SAMPLE_RATE, &iv)) md_set(md, "sample-rate", iv);
    if (AMediaFormat_getInt32(fmt, AMEDIAFORMAT_KEY_BIT_RATE, &iv)) md_set(md, "bit-rate", iv);
    int64_t dur = 0;
    if (AMediaFormat_getInt64(fmt, AMEDIAFORMAT_KEY_DURATION, &dur)) md_set64(md, "durationUs", dur);
    AMediaFormat_delete(fmt);
    c.x[0] = 0;
#else
    c.x[0] = kStatusErr;
#endif
}
void ExtractorSelectTrack(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
#if defined(__ANDROID__)
    if (!e || !e->ex) return void(c.x[0] = kStatusErr);
    c.x[0] = AMediaExtractor_selectTrack(e->ex, size_t(int32_t(c.x[1]))) == AMEDIA_OK ? 0 : kStatusErr;
#else
    c.x[0] = kStatusErr;
#endif
}
void ExtractorUnselectTrack(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
#if defined(__ANDROID__)
    if (!e || !e->ex) return void(c.x[0] = kStatusErr);
    c.x[0] = AMediaExtractor_unselectTrack(e->ex, size_t(int32_t(c.x[1]))) == AMEDIA_OK ? 0 : kStatusErr;
#else
    c.x[0] = kStatusErr;
#endif
}
void ExtractorSeekTo(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
#if defined(__ANDROID__)
    if (!e || !e->ex) return void(c.x[0] = kStatusErr);
    c.x[0] = AMediaExtractor_seekTo(e->ex, int64_t(c.x[1]), AMEDIAEXTRACTOR_SEEK_PREVIOUS_SYNC) == AMEDIA_OK ? 0 : kStatusErr;
#else
    c.x[0] = kStatusErr;
#endif
}
void ExtractorGetTrackIndexForAvailableData(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
#if defined(__ANDROID__)
    if (!e || !e->ex) return void(c.x[0] = kStatusEos);
    int idx = AMediaExtractor_getSampleTrackIndex(e->ex);
    if (idx < 0) return void(c.x[0] = kStatusEos);
    if (c.x[1]) st<uint64_t>(c.x[1], uint64_t(idx));
    c.x[0] = 0;
#else
    c.x[0] = kStatusEos;
#endif
}
void ExtractorRead(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
    uint64_t buf = c.x[1];
#if defined(__ANDROID__)
    if (!e || !e->ex || !buf) return void(c.x[0] = kStatusEos);
    uint8_t* base = reinterpret_cast<uint8_t*>(ld<uint64_t>(buf + kBufBase));
    uint64_t cap = ld<uint64_t>(buf + kBufCapacity);
    if (!base || !cap) return void(c.x[0] = kStatusErr);
    ssize_t n = AMediaExtractor_readSampleData(e->ex, base, size_t(cap));
    if (n < 0) return void(c.x[0] = kStatusEos);
    st<int32_t>(buf + kBufOffset, 0);
    st<int32_t>(buf + kBufSize, int32_t(n));
    c.x[0] = 0;
#else
    c.x[0] = kStatusEos;
#endif
}
void ExtractorAdvance(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
#if defined(__ANDROID__)
    if (!e || !e->ex) return void(c.x[0] = kStatusEos);
    c.x[0] = AMediaExtractor_advance(e->ex) ? 0 : kStatusEos;
#else
    c.x[0] = kStatusEos;
#endif
}
void ExtractorGetSampleTime(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
#if defined(__ANDROID__)
    if (!e || !e->ex) return void(c.x[0] = kStatusEos);
    int64_t t = AMediaExtractor_getSampleTime(e->ex);
    if (t < 0) return void(c.x[0] = kStatusEos);
    if (c.x[1]) st<int64_t>(c.x[1], t);
    c.x[0] = 0;
#else
    c.x[0] = kStatusEos;
#endif
}
void ExtractorGetSampleMetaData(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
    uint64_t md = c.x[1];
#if defined(__ANDROID__)
    if (!e || !e->ex || !md) return void(c.x[0] = kStatusErr);
    uint32_t flags = AMediaExtractor_getSampleFlags(e->ex);
    int64_t t = AMediaExtractor_getSampleTime(e->ex);
    md_set(md, "is-sync-frame", (flags & AMEDIAEXTRACTOR_SAMPLE_FLAG_SYNC) ? 1 : 0);
    md_set64(md, "timeUs", t >= 0 ? t : 0);
    c.x[0] = 0;
#else
    c.x[0] = kStatusErr;
#endif
}
void ExtractorGetBufferedRange(CPU& c) {
    std::lock_guard lk(g_ex_lock);
    ExtractorObj* e = get_extractor(c.x[0]);
    if (c.x[1]) st<int64_t>(c.x[1], 0);
    if (c.x[2]) st<int64_t>(c.x[2], e ? e->duration_us : 0);
    c.x[0] = 0;
}

}  // namespace

NX_HLE("_ZN5movie12SetAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_", SetAllocator);
NX_HLE("_ZN5movie11SetCoreMaskEm", SetCoreMask);
NX_HLE("_ZN5movie17SetThreadPriorityEi", SetThreadPriority);
NX_HLE("_ZN5movie17GetThreadPriorityEv", GetThreadPriority);
NX_HLE("_ZN5movie33CreateVp9DecoderConfigurationDataEiiiPNS_9MediaDataE", CreateVp9DecoderConfigurationData);
NX_HLE("_ZN5movie9MediaDataC1Ev", MediaDataCtor);
NX_HLE("_ZN5movie9MediaDataD1Ev", MediaDataDtor);
NX_HLE("_ZN5movie9MediaData5ClearEv", MediaDataClear);
NX_HLE("_ZNK5movie9MediaData9FindInt32EPKcPi", MediaDataFindInt32);
NX_HLE("_ZNK5movie9MediaData9FindInt64EPKcPl", MediaDataFindInt64);
NX_HLE("_ZNK5movie9MediaData10FindStringEPKcPS2_", MediaDataFindString);
NX_HLE("_ZNK5movie9MediaData8FindRectEPKcPiS3_S3_S3_", MediaDataFindRect);
NX_HLE("_ZN5movie6BufferC1Ev", BufferCtor);
NX_HLE("_ZN5movie6BufferC1Em", BufferCtorCap);
NX_HLE("_ZN5movie6BufferC1EPvm", BufferCtorWrap);
NX_HLE("_ZN5movie6BufferC1ERKS0_", BufferCtorCopy);
NX_HLE("_ZN5movie6BufferD1Ev", BufferDtor);
NX_HLE("_ZN5movie6Buffer18SetDataAndCapacityEPvm", BufferSetDataAndCapacity);
NX_HLE("_ZN5movie6Buffer8SetRangeEii", BufferSetRange);
NX_HLE("_ZNK5movie6Buffer4BaseEv", BufferBase);
NX_HLE("_ZNK5movie6Buffer8CapacityEv", BufferCapacity);
NX_HLE("_ZNK5movie6Buffer4SizeEv", BufferSize);
NX_HLE("_ZNK5movie6Buffer6OffsetEv", BufferOffset);
NX_HLE("_ZN5movie6Buffer12SetInt32DataEi", BufferSetInt32Data);
NX_HLE("_ZN5movie6Buffer12SetInt64DataEl", BufferSetInt64Data);
NX_HLE("_ZN5movie6Buffer13SetUint32DataEj", BufferSetUint32Data);
NX_HLE("_ZNK5movie6Buffer12GetInt64DataEv", BufferGetInt64Data);
NX_HLE("_ZNK5movie6Buffer13GetUint32DataEv", BufferGetUint32Data);
NX_HLE("_ZN5movie6Buffer9SetNvnDevEPv", BufferSetNvnDev);
NX_HLE("_ZN5movie6Buffer10SetNvnSyncEPv", BufferSetNvnSync);
// movie::Extractor direct symbols (Metroid Dread subclasses movie::Extractor and calls base methods directly)
NX_HLE("_ZN5movie9ExtractorC1ENS_13ContainerTypeE", ExtractorCtor);
NX_HLE("_ZN5movie9ExtractorC2ENS_13ContainerTypeE", ExtractorCtor);
NX_HLE("_ZN5movie9ExtractorD1Ev", ExtractorDtor);
NX_HLE("_ZN5movie9ExtractorD2Ev", ExtractorDtor);
NX_HLE("_ZN5movie9Extractor13SetDataSourceEPKc", ExtractorSetDataSourcePath);
NX_HLE("_ZN5movie9Extractor13SetDataSourceEPKvi", ExtractorSetDataSourceMem);
NX_HLE("_ZN5movie9Extractor13SetDataSourceEPNS_18ClientStreamReaderEPKc", ExtractorSetDataSourceStream);
NX_HLE("_ZN5movie9Extractor13GetTrackCountEPi", ExtractorGetTrackCount);
NX_HLE("_ZN5movie9Extractor21GetTrackConfigurationEiPNS_9MediaDataE", ExtractorGetTrackConfiguration);
NX_HLE("_ZN5movie9Extractor11SelectTrackEi", ExtractorSelectTrack);
NX_HLE("_ZN5movie9Extractor13UnselectTrackEi", ExtractorUnselectTrack);
NX_HLE("_ZN5movie9Extractor6SeekToEl", ExtractorSeekTo);
NX_HLE("_ZN5movie9Extractor29GetTrackIndexForAvailableDataEPm", ExtractorGetTrackIndexForAvailableData);
NX_HLE("_ZN5movie9Extractor4ReadEPNS_6BufferE", ExtractorRead);
NX_HLE("_ZN5movie9Extractor7AdvanceEv", ExtractorAdvance);
NX_HLE("_ZN5movie9Extractor13GetSampleTimeEPl", ExtractorGetSampleTime);
NX_HLE("_ZN5movie9Extractor17GetSampleMetaDataEPNS_9MediaDataE", ExtractorGetSampleMetaData);
NX_HLE("_ZN5movie9Extractor16GetBufferedRangeEPlS1_", ExtractorGetBufferedRange);
// movie::Decoder direct symbols + vtable entries
NX_HLE("_ZN5movie7DecoderC1ENS_11DecoderTypeENS_19DecoderOutputFormatENS_11DecoderModeEi", DecoderCtor);
NX_HLE("_ZN5movie7DecoderD1Ev", DecoderDtor);
NX_HLE("_ZN5movie7Decoder9ConfigureEPNS_9MediaDataE", DecoderConfigure);
NX_HLE("_ZN5movie7Decoder20RegisterClientEventsEPNS_13DecoderEventsE", DecoderRegisterClientEvents);
NX_HLE("_ZN5movie7Decoder15GetOutputFormatEPNS_9MediaDataE", DecoderGetOutputFormat);
NX_HLE("_ZN5movie7Decoder5StartEv", DecoderStart);
NX_HLE("_ZN5movie7Decoder4StopEv", DecoderStop);
NX_HLE("_ZN5movie7Decoder5FlushEv", DecoderFlush);
NX_HLE("_ZN5movie7Decoder14GetInputBufferEiPNS_6BufferE", DecoderGetInputBuffer);
NX_HLE("_ZN5movie7Decoder24SendInputBufferForDecodeEiiilj", DecoderSendInputBufferForDecode);
NX_HLE("_ZN5movie7Decoder15GetOutputBufferEiPNS_6BufferE", DecoderGetOutputBuffer);
NX_HLE("_ZN5movie7Decoder24ReleaseOutputBufferIndexEi", DecoderReleaseOutputBufferIndex);
NX_HLE("_ZN5movie7Decoder21GetOutputBufferFormatEiPNS_9MediaDataE", DecoderGetOutputBufferFormat);
// The vtable's entries (kVtable), in its order: subsdk0's _ZTVN5movie7DecoderE.
NX_HLE("movie::Decoder::~Decoder", DecoderDtor);
NX_HLE("movie::Decoder::~Decoder (deleting)", DecoderDeletingDtor);
NX_HLE("movie::Decoder::Configure", DecoderConfigure);
NX_HLE("movie::Decoder::RegisterClientEvents", DecoderRegisterClientEvents);
NX_HLE("movie::Decoder::GetOutputFormat", DecoderGetOutputFormat);
NX_HLE("movie::Decoder::Start", DecoderStart);
NX_HLE("movie::Decoder::Stop", DecoderStop);
NX_HLE("movie::Decoder::Reset", DecoderStop);
NX_HLE("movie::Decoder::Flush", DecoderFlush);
NX_HLE("movie::Decoder::AcquireInputBufferIndex", DecoderAcquireInputBufferIndex);
NX_HLE("movie::Decoder::GetInputBuffer", DecoderGetInputBuffer);
NX_HLE("movie::Decoder::SendInputBufferForDecode", DecoderSendInputBufferForDecode);
NX_HLE("movie::Decoder::AcquireOutputBufferIndex", DecoderAcquireOutputBufferIndex);
NX_HLE("movie::Decoder::GetOutputBuffer", DecoderGetOutputBuffer);
NX_HLE("movie::Decoder::GetOutputBufferSize", DecoderGetOutputBufferSize);
NX_HLE("movie::Decoder::ReleaseOutputBufferIndex", DecoderReleaseOutputBufferIndex);
NX_HLE("movie::Decoder::GetLastError", DecoderGetLastError);
NX_HLE("movie::Decoder::GetOutputBuffer (format)", DecoderGetOutputBufferWithFormat);
NX_HLE("movie::Decoder::GetOutputBufferFormat", DecoderGetOutputBufferFormat);
#if defined(__ANDROID__)
NX_HLE("__movie_pump_thread", movie_pump_thread);
#endif

}  // namespace nx
