// The replay of recorded NVN command lists into a Vulkan command buffer (split from nvn_vk.cpp): render targets,
// clears, calls into other lists, copies; draws and state go to nvn_draw.cpp.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#include "nvn_draw_state.h"
#include "nvn_vk.h"

namespace nx::nvn {

using namespace nx::vk;

namespace {
// The lists called during one submit, by handle (PHASE4.md C5): command_list() takes a mutex and walks a map, and a
// course frame calls hundreds of lists, each looked up twice (the dispatch count, then the replay). The first lookup
// of a handle in a submit is kept for the rest of it: a list re-recorded meanwhile on the game's recording thread
// is not mixed into a submit already being replayed. NX_CALL_TABLE=0: every call looks up.
std::unordered_map<uint64_t, std::shared_ptr<CmdList>> g_called;
thread_local uint64_t s_mru_handle[2] = {0, 0};
thread_local std::shared_ptr<CmdList> s_mru_list[2] = {};
}  // namespace

void submit_lists_reset() {
    g_called.clear();
    s_mru_handle[0] = s_mru_handle[1] = 0;
    s_mru_list[0].reset();
    s_mru_list[1].reset();
}

std::shared_ptr<CmdList> submit_command_list(uint64_t handle) {
    static const bool off = [] {
        const char* e = std::getenv("NX_CALL_TABLE");
        return e && e[0] == '0';
    }();
    if (off) return command_list(handle);
    if (handle == s_mru_handle[0] && s_mru_list[0]) return s_mru_list[0];
    if (handle == s_mru_handle[1] && s_mru_list[1]) {
        std::swap(s_mru_handle[0], s_mru_handle[1]);
        std::swap(s_mru_list[0], s_mru_list[1]);
        return s_mru_list[0];
    }
    auto it = g_called.find(handle);
    std::shared_ptr<CmdList> l;
    if (it != g_called.end()) {
        l = it->second;
    } else {
        l = command_list(handle);
        if (l) g_called.emplace(handle, l);
    }
    if (l) {
        s_mru_handle[1] = s_mru_handle[0];
        s_mru_list[1] = std::move(s_mru_list[0]);
        s_mru_handle[0] = handle;
        s_mru_list[0] = l;
    }
    return l;
}

static bool targets_equal_cmd(const Targets& st, const Cmd& nc, const CmdList& list) {
    if (nc.n != st.color_count || nc.a[8] != st.depth) return false;
    for (uint32_t k = 0; k < st.color_count && k < 8; k++)
        if (nc.a[k] != st.colors[k]) return false;
    TargetViews nv{};
    if (nc.blob < list.blobs.size() && list.blob_size(nc.blob) == sizeof(TargetViews))
        std::memcpy(&nv, list.blob_ptr(nc.blob, sizeof(TargetViews)), sizeof nv);
    return std::memcmp(&nv, &st.views, sizeof(TargetViews)) == 0;
}

struct DsHintState {
    bool reads_depth = false;
    bool writes_depth = false;
    bool uses_stencil = false;
    bool exited_pass = false;
    int budget = 4096;
};

static void scan_list_ds_hint(const CmdList& list, size_t start_idx, int call_depth, const Targets& st,
                              DsHintState& s) {
    if (call_depth >= 4) {
        s.reads_depth = s.writes_depth = s.uses_stencil = true;
        s.exited_pass = true;
        return;
    }
    for (size_t j = start_idx; j < list.cmds.size() && (!s.writes_depth || !s.uses_stencil) && !s.exited_pass; j++) {
        if (--s.budget <= 0) {
            s.reads_depth = s.writes_depth = s.uses_stencil = true;
            s.exited_pass = true;
            break;
        }
        const Cmd& nc = list.cmds[j];
        if (nc.op == Cmd::BindDepthStencilState) {
            if (const auto* nd = reinterpret_cast<const DepthStencilInfo*>(list.blob_ptr(nc.blob, sizeof(DepthStencilInfo)))) {
                s.reads_depth |= (nd->depth_test != 0);
                s.writes_depth |= (nd->depth_write != 0);
                s.uses_stencil |= (nd->stencil_test != 0);
            }
        } else if (nc.op == Cmd::SetRenderTargets) {
            if (!targets_equal_cmd(st, nc, list)) {
                s.exited_pass = true;
                break;
            }
        } else if (nc.op == Cmd::ClearDepthStencil) {
            s.reads_depth = true;
            s.writes_depth = true;
            s.uses_stencil = true;
            s.exited_pass = true;
            break;
        } else if (nc.op == Cmd::CallList) {
            if (call_depth < 3) {
                if (auto sub = submit_command_list(nc.a[0])) {
                    scan_list_ds_hint(*sub, 0, call_depth + 1, st, s);
                    continue;
                }
            }
            s.reads_depth = true;
            s.writes_depth = true;
            s.uses_stencil = true;
            s.exited_pass = true;
            break;
        } else if (nc.op == Cmd::CopyTextureToTexture || nc.op == Cmd::CopyBufferToTexture ||
                   nc.op == Cmd::ClearTexture || nc.op == Cmd::DispatchCompute) {
            s.exited_pass = true;
            break;
        }
    }
}

static void scan_ds_store_hint(const CmdList& list, size_t idx, int depth, bool more_lists, const Targets& st) {
    DsHintState s;
    s.reads_depth = replay::g_state.depth.depth_test != 0;
    s.writes_depth = replay::g_state.depth.depth_write != 0;
    s.uses_stencil = replay::g_state.depth.stencil_test != 0;
    scan_list_ds_hint(list, idx + 1, 0, st, s);
    if (!s.exited_pass && (depth > 0 || more_lists)) {
        s.reads_depth = true;
        s.writes_depth = true;
        s.uses_stencil = true;
    }
    pass_set_ds_store_hint(s.reads_depth, s.writes_depth, s.uses_stencil);
}

void replay_list(VkCommandBuffer cb, Targets& st, const CmdList& list, int depth, bool more_lists) {
    if (skin_lists_on()) skin_list_replayed(list, depth);
    for (size_t idx = 0; idx < list.cmds.size(); idx++) {
        const Cmd& c = list.cmds[idx];
        cb = frame_chunk(cb, replay::g_stats.done, pass_open());  // part of the submit queued early (PLAN.md R1)
        // NX_TRACE_CMDS=<first draw>: the ops replayed from that draw on (20000), with n, mask and a[0].
        static const char* tc = std::getenv("NX_TRACE_CMDS");
        static const uint64_t tc_from = tc ? std::strtoull(tc, nullptr, 10) : 0;
        static int tc_n = 0;
        if (tc && replay::g_stats.done >= tc_from && tc_n++ < 20000)
            log("cmd after draw %llu: op %u n %u mask %x a0 %llx a1 %llx i %d %d %d %d f %g %g %g %g depth %d",
                (unsigned long long)replay::g_stats.done, unsigned(c.op), c.n, c.mask, (unsigned long long)c.a[0],
                (unsigned long long)c.a[1], c.i[0], c.i[1], c.i[2], c.i[3], c.f[0], c.f[1], c.f[2], c.f[3], depth);
        switch (c.op) {
            case Cmd::SetRenderTargets: {  // the pass ends only when a draw or clear needs other targets
                static const bool always = std::getenv("NX_TARGETS_BREAK") != nullptr;  // bisecting: the old way
                if (always) draw_end_rendering(cb, 7);
                st.color_count = c.n;
                for (uint32_t k = 0; k < 8; k++) st.colors[k] = k < c.n ? c.a[k] : 0;
                st.depth = c.a[8];
                st.views = {};
                if (c.blob < list.blobs.size() && list.blob_size(c.blob) == sizeof(TargetViews))
                    std::memcpy(&st.views, list.blob_ptr(c.blob, sizeof(TargetViews)), sizeof st.views);
                replay::g_targets_version++;
                break;
            }
            case Cmd::ClearColor:
            case Cmd::ClearDepthStencil: draw_clear(cb, st, c); break;
            case Cmd::CallList:
                if (depth < 8) {
                    if (auto l = submit_command_list(c.a[0])) {
                        replay_list(cb, st, *l, depth + 1, more_lists || (idx + 1 < list.cmds.size()));
                        cb = frame_cb(cb);  // NX_PIPELINED: the called list may have split the submit
                    } else {
                        static uint64_t missing = 0;
                        if (++missing <= 20 || missing % 1000 == 0)
                            log("nvn: called handle %llx is unknown (%llu calls of unknown handles): %s",
                                (unsigned long long)c.a[0], (unsigned long long)missing,
                                explain_handle(c.a[0]).c_str());
                    }
                }
                break;
            case Cmd::CopyTextureToTexture:
                if (clears_keep_pending()) {
                    pass_end_only(cb, 5);
                    clears_flush_reason(5);
                    clears_flush_image(cb, c.a[0]);
                    clears_flush_image(cb, c.a[1]);
                } else {
                    draw_end_rendering(cb, 5);
                }
                if (gpu_profile_on()) gpu_profile_mark(cb, "texture copies");
                replay_copy(cb, c, list);
                if (gpu_profile_on()) gpu_profile_mark(cb, "outside passes (clears, uploads, copies)");
                break;
            case Cmd::CopyBufferToTexture:
                if (clears_keep_pending()) {
                    pass_end_only(cb, 5);
                    clears_flush_reason(5);
                    clears_flush_image(cb, c.a[1]);
                } else {
                    draw_end_rendering(cb, 5);
                }
                replay_buffer_copy(cb, c, list);
                break;
            case Cmd::ClearTexture:
                if ((c.mask & 15) != 15) break;
                if (clears_keep_pending()) {
                    pass_end_only(cb, 5);
                    clears_flush_reason(5);
                    clears_flush_image(cb, c.a[0]);
                } else {
                    draw_end_rendering(cb, 5);
                }
                replay_clear_texture(cb, c, list);
                break;
            default: {
                const bool hinted = (c.op == Cmd::DrawArrays || c.op == Cmd::DrawElements) && !pass_open_on(st) &&
                                    st.depth != 0 && vk_load_store_op_none();
                if (hinted) scan_ds_store_hint(list, idx, depth, more_lists, st);
                draw_command(cb, c, list, st);
                if (hinted) pass_clear_ds_store_hint();
                // NX_PIPELINED: after the submit's last dispatch the rest goes into a second command buffer.
                if (c.op == Cmd::DispatchCompute) cb = frame_after_dispatch(cb);
                break;
            }
        }
    }
}

}  // namespace nx::nvn
