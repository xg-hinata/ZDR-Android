// The replay's state commands (split from nvn_draw.cpp): binds, viewports, uniform updates, counters; draws and
// dispatches go back to nvn_draw.cpp.
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "nvn_draw_state.h"
#include "platform.h"

namespace nx::nvn {

using namespace replay;

namespace replay {
uint32_t g_pipeline_state_version = 1;
uint32_t g_dynamic_state_version = 1;
uint32_t g_attrib_state_version = 1;
uint32_t g_tex_state_version = 1;
uint32_t g_targets_version = 1;
uint32_t g_uniform_state_version[kStages] = {1, 1, 1, 1, 1, 1};
uint32_t g_storage_state_version[kStages] = {1, 1, 1, 1, 1, 1};
}

static inline uint64_t state_fast_ticks() {
#if defined(__aarch64__)
    uint64_t v;
    asm volatile("mrs %0, cntvct_el0" : "=r"(v));
    return v;
#else
    return plat::monotonic_ns();
#endif
}

void draw_command(VkCommandBuffer cb, const Cmd& c, const CmdList& list, const Targets& targets) {
    State& s = g_state;
    auto blob = [&](size_t size) -> const uint8_t* {
        return list.blob_ptr(c.blob, size);
    };
    switch (c.op) {
        case Cmd::SetViewport:
            if (!s.has_viewport || std::memcmp(s.viewport, c.i, sizeof s.viewport) != 0) {
                std::memcpy(s.viewport, c.i, sizeof s.viewport);
                s.has_viewport = true;
                g_dynamic_state_version++;
            }
            break;
        case Cmd::SetRasterizerDiscard: {
            bool rd = c.n != 0;
            if (s.rasterizer_discard != rd) {
                s.rasterizer_discard = rd;
                g_dynamic_state_version++;
            }
            break;
        }
        case Cmd::SetPrimitiveRestart: {
            bool rst = c.n != 0;
            uint32_t idx = uint32_t(c.a[0]);
            if (s.restart != rst || s.restart_index != idx) {
                s.restart = rst, s.restart_index = idx;
                g_dynamic_state_version++;
                if (!vk_dynamic_caps().primitive_restart_enable) g_pipeline_state_version++;
            }
            break;
        }
        case Cmd::BindImage:
            if (c.i[0] >= 0 && c.i[0] < kStages && c.i[1] >= 0 && c.i[1] < 16) {
                if (s.images[c.i[0]][c.i[1]] != c.a[0]) {
                    s.images[c.i[0]][c.i[1]] = c.a[0];
                    g_tex_state_version++;
                }
            }
            break;
        case Cmd::DispatchCompute:
            dispatch_call(cb, c);
            break;
        case Cmd::SetRenderEnable:
            s.render_enable = c.n != 0;
            s.cond_on = false;
            occlusion_render_enable();
            break;
        case Cmd::SetScissor:
            if (!s.has_scissor || std::memcmp(s.scissor, c.i, sizeof s.scissor) != 0) {
                std::memcpy(s.scissor, c.i, sizeof s.scissor);
                s.has_scissor = true;
                g_dynamic_state_version++;
            }
            break;
        case Cmd::SetDepthRange:
            if (s.depth_near != c.f[0] || s.depth_far != c.f[1]) {
                s.depth_near = c.f[0], s.depth_far = c.f[1];
                g_dynamic_state_version++;
            }
            break;
        case Cmd::BindProgram: {
            s.program = c.a[0];
            const uint32_t bits = c.mask ? c.mask : (1u << kStages) - 1;  // NVN_SHADER_STAGE_*_BIT, 1 << stage
            for (int k = 0; k < kStages; k++)
                if (bits & (1u << k)) s.stage_programs[k] = c.a[0];
            break;
        }
        case Cmd::BindUniformBuffer:
            if (c.i[0] >= 0 && c.i[0] < kStages && c.i[1] >= 0 && c.i[1] < 18) {
                Range& dst = s.uniforms[c.i[0]][c.i[1]];
                if (dst.address != c.a[0] || dst.size != c.a[1]) {
                    dst = {c.a[0], c.a[1]};
                    g_uniform_state_version[c.i[0]]++;
                }
            }
            break;
        case Cmd::BindStorageBuffer:
            if (c.i[0] >= 0 && c.i[0] < kStages && c.i[1] >= 0 && c.i[1] < 16) {
                Range& dst = s.storage[c.i[0]][c.i[1]];
                if (dst.address != c.a[0] || dst.size != c.a[1]) {
                    dst = {c.a[0], c.a[1]};
                    g_storage_state_version[c.i[0]]++;
                }
            }
            break;
        case Cmd::BindDeferred: {
            const DeferredValues* dv = current_deferred();
            auto it = dv ? dv->find(c.a[0]) : DeferredValues::const_iterator{};
            uint32_t words = c.mask >= 2 ? 2 : 1;
            int st = c.i[0];
            if (st < 0 || st >= kStages) break;
            for (uint32_t k = 0; k < c.n; k++) {
                auto value = [&](uint32_t w) {  // the submit's snapshot, else the memory now
                    size_t at = size_t(k) * words + w;
                    return dv && it != dv->end() && at < it->second.size() ? it->second[at]
                                                                            : ld<uint64_t>(c.a[0] + 8 * at);
                };
                int slot = c.i[1] + int(k);
                if (c.mask == 0 && slot >= 0 && slot < 32) {
                    uint64_t v0 = value(0);
                    if (s.textures[st][slot] != v0) s.textures[st][slot] = v0, g_tex_state_version++;
                }
                if (c.mask == 1 && slot >= 0 && slot < 16) {
                    uint64_t v0 = value(0);
                    if (s.images[st][slot] != v0) s.images[st][slot] = v0, g_tex_state_version++;
                }
                if (c.mask == 2 && slot >= 0 && slot < 18) {
                    uint64_t a0 = value(0), a1 = value(1);
                    Range& dst = s.uniforms[st][slot];
                    if (dst.address != a0 || dst.size != a1) {
                        dst = {a0, a1};
                        g_uniform_state_version[st]++;
                    }
                }
                if (c.mask == 3 && slot >= 0 && slot < 16) {
                    uint64_t a0 = value(0), a1 = value(1);
                    Range& dst = s.storage[st][slot];
                    if (dst.address != a0 || dst.size != a1) {
                        dst = {a0, a1};
                        g_storage_state_version[st]++;
                    }
                }
            }
            break;
        }
        case Cmd::BindTexture:
            if (c.i[0] >= 0 && c.i[0] < kStages && c.i[1] >= 0 && c.i[1] < 32) {
                if (s.textures[c.i[0]][c.i[1]] != c.a[0]) {
                    s.textures[c.i[0]][c.i[1]] = c.a[0];
                    g_tex_state_version++;
                }
            }
            break;
        case Cmd::BindVertexBuffer:
            if (c.i[0] >= 0 && c.i[0] < 16) {
                s.vertex_buffers[c.i[0]] = {c.a[0], c.a[1]};
            }
            break;
        case Cmd::SetTexturePool:
            if (s.texture_pool != c.a[0]) s.texture_pool = c.a[0], g_tex_state_version++;
            break;
        case Cmd::SetSamplerPool:
            if (s.sampler_pool != c.a[0]) s.sampler_pool = c.a[0], g_tex_state_version++;
            break;
        case Cmd::BindVertexAttribs:
            if (auto* p = blob(c.n * sizeof(VertexAttrib))) {
                if (s.attribs.size() != c.n || std::memcmp(s.attribs.data(), p, c.n * sizeof(VertexAttrib)) != 0) {
                    s.attribs.assign(reinterpret_cast<const VertexAttrib*>(p), reinterpret_cast<const VertexAttrib*>(p) + c.n);
                    g_attrib_state_version++;
                    g_pipeline_state_version++;
                }
            }
            break;
        case Cmd::BindVertexStreams:
            if (auto* p = blob(c.n * sizeof(VertexStream))) {
                if (s.streams.size() != c.n || std::memcmp(s.streams.data(), p, c.n * sizeof(VertexStream)) != 0) {
                    const auto* vs = reinterpret_cast<const VertexStream*>(p);
                    bool need_pipe_bump = !vk_dynamic_caps().vertex_stride || s.streams.size() != c.n;
                    if (!need_pipe_bump) {
                        for (uint32_t i = 0; i < c.n; i++) {
                            if (s.streams[i].divisor != vs[i].divisor) {
                                need_pipe_bump = true;
                                break;
                            }
                        }
                    }
                    s.streams.assign(vs, vs + c.n);
                    if (need_pipe_bump) g_pipeline_state_version++;
                }
            }
            break;
        case Cmd::BindBlendState:
            if (auto* p = blob(sizeof(BlendTarget))) {
                BlendTarget b;
                std::memcpy(&b, p, sizeof b);
                if (b.target < 8 && std::memcmp(&s.blend[b.target], &b, sizeof b) != 0) {
                    s.blend[b.target] = b;
                    g_dynamic_state_version++;
                    if (!vk_dynamic_caps().color_blend_equation) g_pipeline_state_version++;
                }
            }
            break;
        case Cmd::BindColorState:
            if (auto* p = blob(sizeof(ColorStateInfo))) {
                if (std::memcmp(&s.color, p, sizeof s.color) != 0) {
                    std::memcpy(&s.color, p, sizeof s.color);
                    g_dynamic_state_version++;
                    if (!vk_dynamic_caps().color_blend_enable) g_pipeline_state_version++;
                }
            }
            break;
        case Cmd::BindChannelMaskState:
            if (auto* p = blob(sizeof(ChannelMaskInfo))) {
                if (std::memcmp(&s.mask, p, sizeof s.mask) != 0) {
                    std::memcpy(&s.mask, p, sizeof s.mask);
                    g_dynamic_state_version++;
                    if (!vk_dynamic_caps().color_write_mask) g_pipeline_state_version++;
                }
            }
            break;
        case Cmd::BindDepthStencilState:
            if (auto* p = blob(sizeof(DepthStencilInfo))) {
                if (std::memcmp(&s.depth, p, sizeof s.depth) != 0) {
                    const auto* nd = reinterpret_cast<const DepthStencilInfo*>(p);
                    const DynamicStateCaps& dc = vk_dynamic_caps();
                    const bool need_pipe_bump =
                        (!dc.depth_test_enable && s.depth.depth_test != nd->depth_test) ||
                        (!dc.depth_write_enable && s.depth.depth_write != nd->depth_write) ||
                        (!dc.depth_compare_op && (s.depth.depth_func != nd->depth_func || s.depth.depth_write != nd->depth_write)) ||
                        (!dc.depth_bias_enable && (s.depth.depth_test != nd->depth_test || s.depth.depth_write != nd->depth_write || s.depth.depth_func != nd->depth_func)) ||
                        (!dc.stencil_test_enable && s.depth.stencil_test != nd->stencil_test) ||
                        (!dc.stencil_op && (std::memcmp(s.depth.stencil_func, nd->stencil_func, sizeof s.depth.stencil_func) != 0 ||
                                            std::memcmp(s.depth.stencil_fail, nd->stencil_fail, sizeof s.depth.stencil_fail) != 0 ||
                                            std::memcmp(s.depth.depth_fail, nd->depth_fail, sizeof s.depth.depth_fail) != 0 ||
                                            std::memcmp(s.depth.depth_pass, nd->depth_pass, sizeof s.depth.depth_pass) != 0));
                    std::memcpy(&s.depth, p, sizeof s.depth);
                    g_dynamic_state_version++;
                    if (need_pipe_bump) g_pipeline_state_version++;
                }
            }
            break;
        case Cmd::BindPolygonState:
            if (auto* p = blob(sizeof(PolygonInfo))) {
                if (std::memcmp(&s.polygon, p, sizeof s.polygon) != 0) {
                    const auto* np = reinterpret_cast<const PolygonInfo*>(p);
                    const DynamicStateCaps& dc = vk_dynamic_caps();
                    const bool need_pipe_bump =
                        (!dc.cull_mode && s.polygon.cull != np->cull) ||
                        (!dc.front_face && s.polygon.front != np->front) ||
                        (!dc.polygon_mode && s.polygon.mode != np->mode) ||
                        (!dc.depth_bias_enable && (s.polygon.offset_enables != np->offset_enables || s.polygon.mode != np->mode)) ||
                        (!dc.stencil_op && (s.polygon.cull == 0) != (np->cull == 0));
                    std::memcpy(&s.polygon, p, sizeof s.polygon);
                    g_dynamic_state_version++;
                    if (need_pipe_bump) g_pipeline_state_version++;
                }
            }
            break;
        case Cmd::BindMultisampleState: {
            bool a2c = c.n != 0;
            if (s.alpha_to_coverage != a2c) {
                s.alpha_to_coverage = a2c;
                g_dynamic_state_version++;
                if (!vk_dynamic_caps().alpha_to_coverage_enable) g_pipeline_state_version++;
            }
            break;
        }
        case Cmd::SetBlendColor:
            if (std::memcmp(s.blend_color, c.f, sizeof s.blend_color) != 0) {
                std::memcpy(s.blend_color, c.f, sizeof s.blend_color);
                g_dynamic_state_version++;
            }
            break;
        case Cmd::UpdateUniformBuffer: {  // (buffer address, size, offset, data): written in order, like the GPU
            uint64_t off = c.a[2];
            // NX_TRACE_UUB=<first draw>: logs 40 updates from that draw on.
            static const char* uub = std::getenv("NX_TRACE_UUB");
            static const uint64_t uub_from = uub ? std::strtoull(uub, nullptr, 10) : 0;
            static int uub_n = 0;
            if (uub && g_stats.done >= uub_from && uub_n++ < 1500 && c.blob < list.blobs.size()) {
                const uint8_t* b_ptr = list.blob_ptr(c.blob);
                const size_t b_sz = list.blob_size(c.blob);
                float f[8] = {};
                if (b_ptr) std::memcpy(f, b_ptr, std::min<size_t>(b_sz, sizeof f));
                log("uub before draw %llu: buffer %llx size %llu offset %llu data %zu bytes: %g %g %g %g | %g %g %g %g",
                    (unsigned long long)g_stats.done, (unsigned long long)c.a[0], (unsigned long long)c.a[1],
                    (unsigned long long)off, b_sz, f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
            }
            if (c.blob < list.blobs.size()) {
                static uint64_t s_last_uub_gpu = 0;
                static uint8_t* s_last_uub_base = nullptr;
                static uint64_t s_last_uub_serial = 0;
                uint8_t* base = nullptr;
                if (c.a[0] && c.a[0] == s_last_uub_gpu && t_replay_serial == s_last_uub_serial && s_last_uub_base) {
                    base = s_last_uub_base;
                } else if (c.a[0]) {
                    base = const_cast<uint8_t*>(guest_bytes(c.a[0]));
                    s_last_uub_gpu = c.a[0];
                    s_last_uub_base = base;
                    s_last_uub_serial = t_replay_serial;
                }
                if (uint8_t* p = base ? (base + off) : const_cast<uint8_t*>(guest_bytes(c.a[0] + off))) {
                    const uint8_t* b_ptr = list.blob_ptr(c.blob);
                    const size_t b_sz = list.blob_size(c.blob);
                    if (b_ptr && b_sz) {
                        if (std::memcmp(p, b_ptr, b_sz) != 0) {
                            const uint64_t t_uub0 = state_fast_ticks();
                            std::memcpy(p, b_ptr, b_sz);
                            dedup_invalidate(p, b_sz);
                            written_cpu_write(p, b_sz);
                            note_uub_ticks(state_fast_ticks() - t_uub0);
                        }
                    }
                }
            }
            break;
        }
        case Cmd::ResetCounter: occlusion_reset(cb, c, targets); break;
        case Cmd::RenderConditional: occlusion_conditional(cb, c, targets); break;
        case Cmd::ReportCounter: {  // {value, timestamp}: the work before it is done when the replay gets here
            occlusion_report(cb, c, targets);
            static uint64_t reports = 0, types_seen = 0;
            if (c.n < 64 && !(types_seen >> c.n & 1)) types_seen |= 1ull << c.n, log("nvn: first counter report of type %u", c.n);
            if (uint8_t* p = const_cast<uint8_t*>(guest_bytes(c.a[0]))) {
                // NX_FAKE_GPU_TIME=1: every report of a submit carries the submit's start (a GPU with no cost):
                // the game picks its present interval from the timings it measures (bisecting its 30 fps mode).
                static const bool fake = std::getenv("NX_FAKE_GPU_TIME") != nullptr;
                const uint64_t ts = (fake || c.n == 1) ? submit_start_ns() : plat::monotonic_ns();
                uint64_t v[2] = {++reports, ts};
                std::memcpy(p, v, sizeof v);
                if (c.n != 1) {
                    dedup_invalidate(p, sizeof v);
                    written_cpu_write(p, sizeof v);
                }
            }
            break;
        }
        case Cmd::DrawArrays: draw_call(cb, targets, c.n, false, c); break;
        case Cmd::DrawElements: draw_call(cb, targets, c.n, true, c); break;
        default: raster_command(c); break;
    }
}

}  // namespace nx::nvn
