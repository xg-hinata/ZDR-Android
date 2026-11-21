// Copies of guest buffers kept across submits (games/p5r/OPTIMIZACION.md, session 7). Every submit copied each vertex
// buffer, index buffer and uniform/storage block it binds into the upload ring again, though on P5R's 3D title 72% of
// the vertex bytes, 97% of the index bytes and 88% of the block bytes were the same as their last copy
// (NX_UPLOAD_STATS): ~25 MB a frame through memcpy into uncached memory, a fifth of the render thread on the tablet.
//
// Here a copy outlives its submit, in blocks of host-visible memory of their own, keyed by the guest address with the
// size and a hash of the bytes: a later submit that binds the same bytes (the hash, which reads them once, is several
// times cheaper than the copy) binds that copy. Changed bytes get a new copy; the old one is freed once every submit
// that read it has retired, and copies no submit asked for in a while go the same way. Within a submit, nvn_dedup.cpp
// still answers repeats first.
// Only bytes seen unchanged by kStableSeen submits in a row get a copy here: the first version cached everything, and
// the blocks rewritten every frame (~900 a frame on the title) churned a first-fit allocator over thousands of free
// ranges: 10.6 fps against 12.6 without the cache. Copies come from power-of-two size classes now (O(1)). NX_BUFFER_CACHE=1: on (off by default until measured).
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <nx/runtime.h>

#include "nvn_draw_state.h"
#include "platform.h"

namespace nx::nvn {
uint64_t hash_memory(const uint8_t* p, uint64_t size);  // nvn_sampler.cpp
uint64_t frame_recording_serial();                     // nvn_frames.cpp
}  // namespace nx::nvn

namespace nx::nvn::replay {

using namespace nx::vk;

namespace {

constexpr uint64_t kBlock = 16ull << 20;  // bytes per memory block
constexpr uint64_t kAlign = 256;          // every copy: the largest buffer offset alignment Vulkan allows
constexpr uint64_t kMaxCopy = 4ull << 20; // larger ranges stay in the ring (one copy per submit)
constexpr uint64_t kStaleSubmits = 1800;  // keep resident across camera turns (~600 frames) without swap bloat
constexpr uint32_t kStableSeen = 3;       // 3 consecutive matching submits before entering g_entries
constexpr uint64_t kSeenGap = 24;         // tolerate temporary frustum-cull intervals (~8 frames)
constexpr uint32_t kClasses = 15;         // copy sizes 256 B << class, up to 4 MB

struct Block {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint8_t* ptr = nullptr;
    uint64_t used = 0;  // carved so far (bump), in size-class slots
};
std::vector<Block> g_blocks;
struct Slot {
    uint32_t block;
    uint64_t offset;
};
std::vector<Slot> g_free[kClasses];  // free slots per size class

// Bytes not cached yet: fixed flat slot table in BSS + per-64KB-page churn tracker (no per-address heap nodes).
struct SeenSlot {
    const uint8_t* p = nullptr;
    uint64_t size = 0, hash = 0;
    uint32_t count = 0;
    uint32_t unstable_streak = 0;
    uint64_t skip_until = 0;
    uint64_t submit = 0;  // g_submits when last seen
};
constexpr uint32_t kSeenSlots = 32768;
constexpr uint32_t kSeenProbe = 4;
SeenSlot g_seen_slots[kSeenSlots];

inline uint32_t seen_slot_index(const uint8_t* p) {
    uint64_t v = reinterpret_cast<uintptr_t>(p);
    return uint32_t(((v >> 4) ^ (v >> 13) ^ (v >> 22) ^ (v * 0x9E3779B97F4A7C15ull >> 48)) & (kSeenSlots - 1));
}

struct PageChurn {
    uint16_t cached_count = 0;
    uint16_t unstable_count = 0;
    uint32_t last_window = 0;
    uint64_t skip_until = 0;
    uint64_t cached_bloom = 0;
};
constexpr uint32_t kPageSlots = 65536;
PageChurn g_page_churn[kPageSlots];

inline uint32_t page_churn_index(const uint8_t* p) {
    return uint32_t((reinterpret_cast<uintptr_t>(p) >> 16) & (kPageSlots - 1));
}

inline uint64_t page_bloom_bit(const uint8_t* p) {
    const uintptr_t v = reinterpret_cast<uintptr_t>(p);
    return 1ull << (((v >> 4) ^ (v >> 10)) & 63);
}

struct UnstableSlot {
    const uint8_t* p = nullptr;
    uint64_t skip_until = 0;
};
constexpr uint32_t kUnstableSlots = 16384;
UnstableSlot g_unstable_front[kUnstableSlots];

inline uint32_t unstable_slot_index(const uint8_t* p) {
    uint64_t v = reinterpret_cast<uintptr_t>(p);
    return uint32_t(((v >> 4) ^ (v >> 16)) & (kUnstableSlots - 1));
}

struct Entry {
    uint64_t size = 0, padded = 0, hash = 0, sample_hash = 0;
    uint32_t block = 0;
    uint64_t offset = 0;
    uint32_t size_class = 0;  // the allocation: one slot of that class
    uint32_t verified_streak = 0;
    uint64_t serial = 0;             // the last submit that bound it (frees wait for it to retire)
    uint64_t seen = 0;               // g_submits then (copies unbound for kStaleSubmits are freed)
    uint64_t last_verified_submit = 0;  // the last submit that hashed guest memory
    uint64_t last_verified_frame = 0;   // the last frame (g_frame + 1) that hashed guest memory
};
std::unordered_map<const uint8_t*, Entry> g_entries;

struct EntryFront {
    const uint8_t* p = nullptr;
    Entry* entry = nullptr;
    uint64_t gen = 0;
};
constexpr uint32_t kEntryFrontSlots = 32768;
constexpr uint32_t kEntryFrontProbe = 4;
EntryFront g_entry_front[kEntryFrontSlots];
uint64_t g_entry_front_gen = 1;

inline uint32_t entry_front_index(const uint8_t* p) {
    const uintptr_t v = reinterpret_cast<uintptr_t>(p);
    return uint32_t(((v >> 4) ^ (v >> 13) ^ (v >> 21) ^ (v * 0x9E3779B97F4A7C15ull >> 48)) & (kEntryFrontSlots - 1));
}

inline Entry* find_entry_front(const uint8_t* p, EntryFront*& slot_out) {
    const uint32_t base = entry_front_index(p);
    EntryFront* empty_slot = nullptr;
    for (uint32_t step = 0; step < kEntryFrontProbe; step++) {
        EntryFront& ef = g_entry_front[(base + step) & (kEntryFrontSlots - 1)];
        if (ef.gen == g_entry_front_gen && ef.p == p) {
            slot_out = &ef;
            return ef.entry;
        }
        if (!empty_slot && (ef.gen != g_entry_front_gen || ef.p == nullptr)) {
            empty_slot = &ef;
        }
    }
    slot_out = empty_slot ? empty_slot : &g_entry_front[base];
    return nullptr;
}

inline void clear_entry_front(const uint8_t* p) {
    const uint32_t base = entry_front_index(p);
    for (uint32_t step = 0; step < kEntryFrontProbe; step++) {
        EntryFront& ef = g_entry_front[(base + step) & (kEntryFrontSlots - 1)];
        if (ef.gen == g_entry_front_gen && ef.p == p) {
            ef.p = nullptr;
            ef.entry = nullptr;
        }
    }
}

uint64_t full_hash_memory(const uint8_t* p, uint64_t size) {
    uint64_t h = hash_memory(p, size);  // whole 8-byte words: the last size % 8 bytes go in here
    for (uint64_t k = size & ~7ull; k < size; k++) h = (h ^ p[k]) * 0x100000001B3ull;
    return h;
}

uint64_t fast_sample_hash(const uint8_t* p, uint64_t size) {
    if (size <= 192) return full_hash_memory(p, size);
    constexpr uint64_t kChunk = 64;
    const uint64_t mid = (size / 2) & ~7ull;
    const uint64_t tail = (size - kChunk) & ~7ull;
    uint64_t h = hash_memory(p, kChunk);
    h ^= hash_memory(p + mid, kChunk) * 0x9E3779B97F4A7C15ull;
    h ^= hash_memory(p + tail, size - tail) * 0xC2B2AE3D27D4EB4Full;
    return h;
}

static inline uint64_t fast_ticks() {
#if defined(__aarch64__)
    uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return plat::monotonic_ns();
#endif
}

static inline double ticks_to_ms(uint64_t ticks) {
#if defined(__aarch64__)
    uint64_t f;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f ? (double(ticks) * 1000.0 / double(f)) : 0.0;
#else
    return double(ticks) / 1e6;
#endif
}

struct Retired {
    uint64_t serial;
    Slot slot;
    uint32_t size_class;
};
std::vector<Retired> g_retired;
uint64_t g_used = 0;     // bytes allocated
uint64_t g_submits = 0;  // cache_begin_submit calls
uint64_t g_budget = 0;
uint64_t g_evict_epoch = 1;

inline void record_page_churn(PageChurn& pc) {
    const uint32_t win = uint32_t(g_submits >> 4);
    if (pc.last_window != win) {
        pc.last_window = win;
        pc.unstable_count = 0;
    }
    if (++pc.unstable_count >= 12) {
        pc.skip_until = g_submits + 90;
        pc.unstable_count = 0;
    }
}

struct {
    uint64_t hits = 0, hit_bytes = 0, misses = 0, miss_bytes = 0, hash_ns = 0, full = 0, evicted = 0, wrong = 0;
    uint64_t unstable = 0;  // asked for, not stable yet: copied into the ring
} g_stats_cache;

bool on() {
    static const bool v = [] {
        const char* e = std::getenv("NX_BUFFER_CACHE");
        return !e || e[0] != '0';
    }();
    return v;
}

uint64_t budget() {
    if (!g_budget) {
        long mb = std::getenv("NX_BUFFER_CACHE_MB") ? std::atol(std::getenv("NX_BUFFER_CACHE_MB")) : 64;
        g_budget = uint64_t(std::clamp(mb, 16l, 1024l)) << 20;
    }
    return g_budget;
}

bool add_block() {
    if ((g_blocks.size() + 1) * kBlock > budget()) return false;
    VkDevice d = vk_device();
    Block b;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = kBlock;
    bi.usage = kUploadUsage;
    if (p_vkCreateBuffer(d, &bi, nullptr, &b.buffer) != VK_SUCCESS) return false;
    VkMemoryRequirements req{};
    p_vkGetBufferMemoryRequirements(d, b.buffer, &req);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = upload_memory_type(req.memoryTypeBits);  // as the upload ring
    void* p = nullptr;
    if (p_vkAllocateMemory(d, &ai, nullptr, &b.memory) != VK_SUCCESS || p_vkBindBufferMemory(d, b.buffer, b.memory, 0) ||
        p_vkMapMemory(d, b.memory, 0, VK_WHOLE_SIZE, 0, &p) != VK_SUCCESS) {
        p_vkDestroyBuffer(d, b.buffer, nullptr);
        if (b.memory) p_vkFreeMemory(d, b.memory, nullptr);
        log("nvn: buffer cache: no memory for another %llu MB block", (unsigned long long)(kBlock >> 20));
        g_budget = g_blocks.size() * kBlock;  // no more tries
        return false;
    }
    b.ptr = static_cast<uint8_t*>(p);
    g_blocks.push_back(std::move(b));
    return true;
}

uint32_t size_class(uint64_t bytes) {
    uint32_t c = 0;
    while ((kAlign << c) < bytes) c++;
    return c;
}

uint64_t class_bytes(uint32_t c) { return kAlign << c; }

void retire(const Entry& e) {
    g_retired.push_back({e.serial, {e.block, e.offset}, e.size_class});
}

// Copies no submit bound lately, retired (their memory comes back once those submits are done).
void evict_stale() {
    const uint64_t stale_limit = (g_used < budget() / 2) ? (kStaleSubmits * 3) : kStaleSubmits;
    for (auto it = g_entries.begin(); it != g_entries.end();)
        if (it->second.seen + stale_limit < g_submits) {
            const uint8_t* key_p = it->first;
            clear_entry_front(key_p);
            retire(it->second);
            g_evict_epoch++;
            g_stats_cache.evicted++;
            PageChurn& pc = g_page_churn[page_churn_index(key_p)];
            if (pc.cached_count > 0) {
                if (--pc.cached_count == 0) pc.cached_bloom = 0;
            }
            it = g_entries.erase(it);
        } else {
            ++it;
        }
}

// A slot of class c: a freed one, else carved from the last block (a new block when it is full).
bool alloc(uint32_t c, Slot& out) {
    if (!g_free[c].empty()) {
        out = g_free[c].back();
        g_free[c].pop_back();
        return true;
    }
    const uint64_t bytes = class_bytes(c);
    if (g_blocks.empty() || g_blocks.back().used + bytes > kBlock)
        if (!add_block()) return false;
    Block& b = g_blocks.back();
    out = {uint32_t(g_blocks.size() - 1), b.used};
    b.used += bytes;
    return true;
}

}  // namespace

bool is_buffer_cache_buffer(VkBuffer buf) {
    if (buf == VK_NULL_HANDLE) return false;
    for (const Block& b : g_blocks)
        if (b.buffer == buf) return true;
    return false;
}

uint64_t cache_evict_epoch() { return g_evict_epoch; }

void cache_evict_address(const uint8_t* p) {
    if (!on() || !p) return;
    if (auto it = g_entries.find(p); it != g_entries.end()) {
        clear_entry_front(p);
        retire(it->second);
        g_evict_epoch++;
        g_stats_cache.evicted++;
        PageChurn& pc = g_page_churn[page_churn_index(p)];
        if (pc.cached_count > 0) {
            if (--pc.cached_count == 0) pc.cached_bloom = 0;
        }
        g_entries.erase(it);
    }
    const uint32_t s_base = seen_slot_index(p);
    for (uint32_t step = 0; step < kSeenProbe; step++) {
        SeenSlot& cand = g_seen_slots[(s_base + step) & (kSeenSlots - 1)];
        if (cand.p == p) cand = {};
    }
}

void cache_begin_submit() {
    if (!on()) return;
    const uint64_t done = async_gpu() ? frame_completed() : UINT64_MAX;  // synchronous: every submit was waited for
    size_t kept = 0;
    for (const Retired& r : g_retired)
        if (r.serial <= done) {
            g_free[r.size_class].push_back(r.slot);
            g_used -= class_bytes(r.size_class);
        } else {
            g_retired[kept++] = r;
        }
    g_retired.resize(kept);
    if (++g_submits % 64 == 0) evict_stale();
}

bool cache_upload(const uint8_t* p, uint64_t size, uint64_t padded, uint64_t align, Upload& out) {
    if (!on() || !p || !size || padded > kMaxCopy || align > kAlign) return false;
    // Never cache guest pages written by GPU compute shaders (skinning/morph outputs): when NX_EXP_NO_WB=1 skips
    // write-back to guest RAM and written_reset() clears g_regions at present, guest RAM at `p` appears unchanged
    // and would otherwise be promoted to a static BufferCache entry after kStableSeen submits.
    if (may_overlap_ever_written_pages(p, size)) return false;
    // Small uniform/storage blocks are sub-allocated from volatile per-frame guest rings (intra-submit repeats are
    // already answered in O(1) by dedup_upload / g_front); bypass <= 4 KB blocks unless NX_CACHE_BLOCKS=1, while
    // allowing larger (> 4 KB) static blocks to cache unless their 64 KB guest page exhibits ring-buffer churn.
    static const bool cache_blocks = std::getenv("NX_CACHE_BLOCKS") != nullptr;
    if (g_upload_kind == kUploadBlock && size <= 4096 && !cache_blocks) return false;

    const uint32_t u_idx = unstable_slot_index(p);
    if (g_unstable_front[u_idx].p == p) {
        if (g_unstable_front[u_idx].skip_until > g_submits) {
            g_stats_cache.unstable++;
            return false;
        }
        g_unstable_front[u_idx].p = nullptr;
    }

    static const bool time_hash = std::getenv("NX_BUFFER_CACHE_STATS") != nullptr;
    static const bool verify = std::getenv("NX_BUFFER_CACHE_VERIFY") != nullptr;
    const uint64_t cur_frame = g_frame.load(std::memory_order_relaxed) + 1;

    PageChurn& pc = g_page_churn[page_churn_index(p)];

    // Check if this buffer is already cached (via O(1) 4-probe front cache before page-churn & g_entries)
    EntryFront* ef_slot = nullptr;
    Entry* found_entry = find_entry_front(p, ef_slot);
    if (!found_entry) {
        const uint64_t b_bit = page_bloom_bit(p);
        if ((pc.cached_bloom & b_bit) == 0 && pc.skip_until > g_submits) {
            g_stats_cache.unstable++;
            return false;
        }
        if ((pc.cached_bloom & b_bit) != 0 && pc.cached_count > 0) {
            if (auto it = g_entries.find(p); it != g_entries.end()) {
                found_entry = &it->second;
                *ef_slot = {p, found_entry, g_entry_front_gen};
            }
        }
        if (!found_entry && pc.skip_until > g_submits) {
            g_stats_cache.unstable++;
            return false;
        }
    }

    if (found_entry) {
        Entry& e = *found_entry;
        const bool is_3d_geom = g_is_3d_scene_draw && (g_upload_kind == kUploadVertex || g_upload_kind == kUploadIndex);
        const uint64_t addr_phase = (reinterpret_cast<uintptr_t>(p) >> 6) & 7ull;
        const bool frame_verified =
            (cur_frame == e.last_verified_frame) ||
            (e.verified_streak >= 8 && cur_frame > e.last_verified_frame &&
             (cur_frame - e.last_verified_frame) < 8 && ((cur_frame ^ addr_phase) & 7ull) != 0);
        if (!verify && e.size == size && e.padded == padded && e.block < g_blocks.size() &&
            g_blocks[e.block].buffer != VK_NULL_HANDLE &&
            (e.last_verified_submit == g_submits ||
             (is_3d_geom && e.verified_streak >= 3 && frame_verified))) {
            e.serial = std::max(e.serial, frame_recording_serial());
            e.seen = g_submits;
            g_stats_cache.hits++;
            g_stats_cache.hit_bytes += padded;
            out = {g_blocks[e.block].buffer, e.offset, g_blocks[e.block].ptr + e.offset};
            return true;
        }

        const uint64_t t0 = fast_ticks();
        const bool can_sample = (is_3d_geom && size > 512 && e.verified_streak >= 3);
        bool match = false;
        if (can_sample) {
            uint64_t sh = fast_sample_hash(p, size);
            match = (e.size == size && e.padded == padded && e.sample_hash == sh);
        } else {
            uint64_t h = full_hash_memory(p, size);
            match = (e.size == size && e.padded == padded && e.hash == h);
        }
        g_stats_cache.hash_ns += fast_ticks() - t0;

        if (match) {
            e.last_verified_submit = g_submits;
            e.last_verified_frame = cur_frame;
            if (e.verified_streak < 1000) e.verified_streak++;
            // NX_BUFFER_CACHE_VERIFY=1: every hit compared with the guest bytes (slow: the copy is uncached memory).
            if (verify && std::memcmp(g_blocks[e.block].ptr + e.offset, p, size) != 0) {
                static int said = 0;
                if (said++ < 20) log("nvn: buffer cache: a hit differs from the guest bytes (%llu bytes at %p)",
                                     (unsigned long long)size, (const void*)p);
                g_stats_cache.wrong++;
            }
            e.serial = std::max(e.serial, frame_recording_serial()), e.seen = g_submits;
            g_stats_cache.hits++, g_stats_cache.hit_bytes += padded;
            out = {g_blocks[e.block].buffer, e.offset, g_blocks[e.block].ptr + e.offset};
            return true;
        }
        retire(e);  // read by the submits up to e.serial: freed after them
        g_evict_epoch++;
        if (pc.cached_count > 0) {
            if (--pc.cached_count == 0) pc.cached_bloom = 0;
        }
        clear_entry_front(p);
        g_entries.erase(p);
    }

    // Not cached: track candidate stability in a 4-way set-associative zero-allocation slot table until promoted.
    const uint32_t s_base = seen_slot_index(p);
    SeenSlot* seen_ptr = nullptr;
    SeenSlot* oldest_slot = &g_seen_slots[s_base];
    for (uint32_t step = 0; step < kSeenProbe; step++) {
        SeenSlot& cand = g_seen_slots[(s_base + step) & (kSeenSlots - 1)];
        if (cand.p == p) {
            seen_ptr = &cand;
            break;
        }
        if (cand.p == nullptr) {
            oldest_slot = &cand;
            break;
        }
        if (cand.submit < oldest_slot->submit) {
            oldest_slot = &cand;
        }
    }
    SeenSlot& seen = seen_ptr ? *seen_ptr : *oldest_slot;
    if (seen.p == p && seen.skip_until > g_submits) {
        seen.submit = g_submits;
        g_stats_cache.unstable++;
        return false;
    }

    const uint64_t t0 = fast_ticks();
    const bool use_sample_cand = (g_upload_kind == kUploadVertex || g_upload_kind == kUploadIndex) && size > 512;
    const uint64_t sh = use_sample_cand ? fast_sample_hash(p, size) : full_hash_memory(p, size);
    g_stats_cache.hash_ns += fast_ticks() - t0;

    const uint64_t serial = frame_recording_serial();
    if (seen.p != p) {
        seen = {p, size, sh, 1, 0, 0, g_submits};
    } else if (seen.size == size && seen.hash == sh && seen.submit + kSeenGap >= g_submits) {
        seen.count++;
        seen.unstable_streak = 0;
        seen.size = size;
        seen.hash = sh;
        seen.submit = g_submits;
    } else {
        seen.count = 1;
        if (seen.size != size || seen.hash != sh) {
            seen.unstable_streak++;
            // Progressive backoff up to 480 submits (~160 frames) on first content mismatch
            const uint32_t shift = std::min<uint32_t>(seen.unstable_streak - 1, 4);
            seen.skip_until = g_submits + (30ull << shift);
            g_unstable_front[u_idx] = {p, seen.skip_until};
            record_page_churn(pc);
        }
        seen.size = size;
        seen.hash = sh;
        seen.submit = g_submits;
    }
    if (seen.count < kStableSeen) {
        g_stats_cache.unstable++;
        return false;  // the ring
    }
    const uint32_t c = size_class(padded);
    Slot slot;
    if (c >= kClasses || !alloc(c, slot)) {
        g_stats_cache.full++;
        return false;  // the ring then, as before
    }
    seen = {};
    if (g_unstable_front[u_idx].p == p) g_unstable_front[u_idx].p = nullptr;
    const uint64_t t1 = use_sample_cand ? fast_ticks() : 0;
    const uint64_t h = use_sample_cand ? full_hash_memory(p, size) : sh;
    if (use_sample_cand) g_stats_cache.hash_ns += fast_ticks() - t1;
    const uint64_t sample_h = use_sample_cand ? sh : ((size > 512) ? fast_sample_hash(p, size) : h);
    g_used += class_bytes(c);
    uint8_t* dst = g_blocks[slot.block].ptr + slot.offset;
    frame_guest_read(p, size);
    plat::fast_memcpy(dst, p, size);
    if (padded > size) std::memset(dst + size, 0, padded - size);
    auto [ins_it, inserted] = g_entries.insert_or_assign(
        p, Entry{size, padded, h, sample_h, slot.block, slot.offset, c, 1, serial, g_submits, g_submits, cur_frame});
    if (inserted && pc.cached_count < UINT16_MAX) {
        pc.cached_count++;
        pc.cached_bloom |= page_bloom_bit(p);
        pc.skip_until = 0;
        pc.unstable_count = 0;
    }
    *ef_slot = {p, &ins_it->second, g_entry_front_gen};
    g_stats_cache.misses++, g_stats_cache.miss_bytes += padded;
    out = {g_blocks[slot.block].buffer, slot.offset, dst};
    return true;
}

void cache_log() {
    auto& s = g_stats_cache;
    if (s.hits + s.misses + s.full + s.unstable)
        log("nvn: buffer cache: %llu hits (%.1f MB not copied), %llu copies (%.1f MB), %llu not stable, %llu full, "
            "%llu evicted; hashing %.1f ms; %zu copies kept in %.1f of %zu MB%s",
            (unsigned long long)s.hits, double(s.hit_bytes) / 1e6, (unsigned long long)s.misses,
            double(s.miss_bytes) / 1e6, (unsigned long long)s.unstable, (unsigned long long)s.full,
            (unsigned long long)s.evicted,
            ticks_to_ms(s.hash_ns), g_entries.size(), double(g_used) / 1e6, size_t(g_blocks.size() * kBlock >> 20),
            s.wrong ? "; HITS THAT DIFFER (NX_BUFFER_CACHE_VERIFY)" : "");
    s = {};
}

}  // namespace nx::nvn::replay
