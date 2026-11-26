// Metroid Dread (010093801237C000, Base 1.0.0, BID 49161D9CCBC15DF9):
// Game-specific metadata, HLE stubs (nn::nfp Amiibo, nn::prepo telemetry), and mobile NVN runtime defaults.
// Engine: MercurySteam Mercury Engine on NVN (NintendoSDK 11.4.3), 801 unique VS/FS shaders in /system/shd/*.bshdat,
// audio via AudioRenderer (.bfstm/.bfgrp), video via nn::movie (subsdk0, .webm).
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

#include <nx/game.h>
#include <nx/runtime.h>

#include "platform.h"

namespace nx {

const game::Info game::kInfo = {
    "dread",
    "Metroid Dread",
    0x010093801237C000ull,
    "io.github.dreadrecomp",
    "private/dread",
    0 /* main.npdm META +0x14: 0 MiB system resource (SDK 11.4.3) */,
    "49161D9CCBC15DF9" /* Base 1.0.0 */
};

static constexpr const char* kPerfBuildTag = "v0.1.0-beta.2";

namespace {

// Dread Base 1.0.0 (BID 49161D9CCBC15DF9) video teardown synchronization:
// 1. In VideoEventMgr::EnqueueLinkOrUnlink (main+0x222838), Dread signals Event 4 (unlink, +0x2b0) at main+0x2228a0
//    and Event 5 (exit WaitAny loop, +0x228) 18 instructions later at main+0x2228e8. Because Event 5's holder was
//    linked BEFORE Event 4's holder (main+0x22254c vs 0x22259c), on Linux CFS the main thread on a big core signals
//    Event 5 before the event thread on a little core exits WaitAny; WaitAny then returns Event 5 and exits without
//    unlinking the 4 decoder MultiWaitHolders, crashing WaitAnyImpl (sdk+34ab8c) on the next video.
// 2. When tearing down a VideoPlayer, the main thread frees VideoPlayer/VideoDecoderWrapper before the demuxer thread
//    drains VideoMgr->msg_queue (crashing LockMutex at main+0x224ea8 with a reused heap pointer).
// Waiting for Event 4 to be consumed before signaling Event 5, revoking queued VideoDecoderWrapper* tokens when
// ~VideoDecoderWrapper runs (without waiting on VideoMgr->msg_queue while holding player->mutex_0x960, which
// deadlocked when DemuxerThread was blocked on that mutex), and waiting for DemuxerThread to exit
// ProcessInputBuffer before ~VideoPlayer destroys player->mutex_0x960 eliminates both UAF races deterministically.
std::atomic<uint64_t> g_dread_main_base{0};
std::atomic<uint64_t> g_dread_event_mgr{0};
std::atomic<uint64_t> g_dread_msg_queue{0};
std::mutex g_video_mq_lock;
uint64_t g_video_next_token = 1;
std::unordered_map<uint64_t, uint64_t> g_video_tokens;  // monotonic token -> VideoDecoderWrapper*
uint64_t g_demuxer_active_decoder = 0;  // last decoder returned to DemuxerThread
uint64_t g_demuxer_active_player = 0;   // player of that decoder while DemuxerThread runs ProcessInputBuffer

inline uint64_t dread_main_base(uint64_t lr) {
    uint64_t base = g_dread_main_base.load(std::memory_order_relaxed);
    if (base) return base;
    if (const LoadedModule* m = module_at(lr); m && m->info && m->info->name && std::strcmp(m->info->name, "main") == 0) {
        // Verify Base 1.0.0 instruction at main+0x2228e8 (bl 0x11f3350 = SignalEvent).
        if (m->size > 0x222900 && ld<uint32_t>(m->base + 0x2228e8) == 0x943f429au) {
            g_dread_main_base.store(m->base, std::memory_order_release);
            return m->base;
        }
    }
    return 0;
}

void dread_signal_event(CPU& c) {
    static const uint64_t real = find_symbol("_ZN2nn2os11SignalEventEPNS0_9EventTypeE");
    const uint64_t lr = c.x[30];
    const uint64_t base = dread_main_base(lr);
    if (base) {
        const uint64_t off = lr - base;
        if (off == 0x2228a4) {
            g_dread_event_mgr.store(c.x[0] - 0x2b0, std::memory_order_release);
        } else if (off == 0x2228ec) {
            const uint64_t mgr = c.x[0] - 0x228;
            g_dread_event_mgr.store(mgr, std::memory_order_release);
            const uint64_t event4 = mgr + 0x2b0;
            // Wait until EventThread consumes Event 4 (ClearEvent sets EventType+0x10 = 0 BEFORE locking mgr mutex).
            const uint64_t t0 = plat::monotonic_ns();
            while (ld<uint8_t>(event4 + 0x10) != 0 && plat::monotonic_ns() - t0 < 20'000'000ull) {
                plat::sleep_ns(50'000);
            }
        }
    }
    if (real) call(c, real);
}

void dread_send_message_queue(CPU& c) {
    static const uint64_t real = find_symbol("_ZN2nn2os16SendMessageQueueEPNS0_16MessageQueueTypeEm");
    const uint64_t lr = c.x[30];
    const uint64_t base = dread_main_base(lr);
    if (base && lr >= base + 0x222a80 && lr <= base + 0x222b80) {
        g_dread_msg_queue.store(c.x[0], std::memory_order_release);
        if (c.x[1] != 0) {
            std::lock_guard<std::mutex> lk(g_video_mq_lock);
            uint64_t token = g_video_next_token++;
            g_video_tokens[token] = c.x[1];
            c.x[1] = token;
        }
    }
    if (real) call(c, real);
}

void dread_timed_receive_message_queue(CPU& c) {
    static const uint64_t real = find_symbol("_ZN2nn2os24TimedReceiveMessageQueueEPmPNS0_16MessageQueueTypeENS_8TimeSpanE");
    const uint64_t lr = c.x[30];
    const uint64_t base = dread_main_base(lr);
    if (base && (lr - base) == 0x223770) {
        const uint64_t out_ptr = c.x[0];
        {
            std::lock_guard<std::mutex> lk(g_video_mq_lock);
            g_demuxer_active_decoder = 0;
            g_demuxer_active_player = 0;
        }
        if (real) call(c, real);
        if ((c.x[0] & 1u) != 0 && out_ptr != 0) {
            const uint64_t token = ld<uint64_t>(out_ptr);
            std::lock_guard<std::mutex> lk(g_video_mq_lock);
            auto it = g_video_tokens.find(token);
            if (it != g_video_tokens.end()) {
                const uint64_t decoder = it->second;
                g_video_tokens.erase(it);
                g_demuxer_active_decoder = decoder;
                g_demuxer_active_player = decoder ? ld<uint64_t>(decoder + 0x30) : 0;
                st<uint64_t>(out_ptr, decoder);
            } else {
                st<uint64_t>(out_ptr, 0ull);
            }
        }
        return;
    }
    if (real) call(c, real);
}

void dread_finalize_mutex(CPU& c) {
    static const uint64_t real = find_symbol("_ZN2nn2os13FinalizeMutexEPNS0_9MutexTypeE");
    const uint64_t lr = c.x[30];
    const uint64_t base = dread_main_base(lr);
    if (base) {
        const uint64_t off = lr - base;
        if (off == 0x225964) {
            const uint64_t decoder = c.x[0] - 0x2a0;
            const uint64_t mgr = g_dread_event_mgr.load(std::memory_order_acquire);
            uint32_t scrubbed = 0;
            bool in_flight = false;
            {
                std::lock_guard<std::mutex> lk(g_video_mq_lock);
                for (auto it = g_video_tokens.begin(); it != g_video_tokens.end();) {
                    if (it->second == decoder) {
                        it = g_video_tokens.erase(it);
                        scrubbed++;
                    } else {
                        ++it;
                    }
                }
                in_flight = (g_demuxer_active_decoder == decoder);
            }
            const uint64_t t0 = plat::monotonic_ns();
            while (plat::monotonic_ns() - t0 < 15'000'000ull) {
                const bool unlink_done = !mgr || ld<uint64_t>(mgr + 0x370) == 0;
                if (unlink_done) break;
                plat::sleep_ns(100'000);
            }
            if (in_flight) {
                // Let DemuxerThread finish loading player = [decoder + 0x30] before ~VideoDecoderWrapper frees it.
                plat::sleep_ns(500'000);
            }
            const uint64_t dt_us = (plat::monotonic_ns() - t0) / 1000;
            log("dread: video teardown sync at main+0x225964: %llu us (scrubbed=%u, in_flight=%d, pending_unlink=%llu)",
                (unsigned long long)dt_us, scrubbed, in_flight ? 1 : 0,
                (unsigned long long)(mgr ? ld<uint64_t>(mgr + 0x370) : 0));
        } else if (off == 0x21e420) {
            const uint64_t player = c.x[0] - 0x960;
            const uint64_t mgr = g_dread_event_mgr.load(std::memory_order_acquire);
            const uint64_t t0 = plat::monotonic_ns();
            bool busy_player = false;
            while (plat::monotonic_ns() - t0 < 40'000'000ull) {
                const bool unlink_done = !mgr || ld<uint64_t>(mgr + 0x370) == 0;
                {
                    std::lock_guard<std::mutex> lk(g_video_mq_lock);
                    busy_player = (g_demuxer_active_player == player);
                }
                if (unlink_done && !busy_player) break;
                plat::sleep_ns(100'000);
            }
            plat::sleep_ns(2'000'000);
            const uint64_t dt_us = (plat::monotonic_ns() - t0) / 1000;
            log("dread: video teardown sync at main+0x21e420: %llu us (pending_unlink=%llu, busy_player=%d)",
                (unsigned long long)dt_us,
                (unsigned long long)(mgr ? ld<uint64_t>(mgr + 0x370) : 0),
                busy_player ? 1 : 0);
        }
    }
    if (real) call(c, real);
}

// Forward guest stderr assertion messages to nx.log before calling the real fputs.
void fputs_logged(CPU& c) {
    static const uint64_t real = find_symbol("fputs");
    const char* s = reinterpret_cast<const char*>(c.x[0]);
    if (s) {
        std::string line(s, strnlen(s, 1024));
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
        log("[guest fputs] %s", line.c_str());
    }
    if (real) call(c, real);
    else c.x[0] = 0;
}

// ---- nn::nfp (Amiibo NFC) HLE stubs ----
// Metroid Dread imports 26 nn::nfp functions for optional Amiibo scanning.
// Report initialized state with zero NFC devices so the game never stalls on nfp:user IPC.
std::atomic<int> g_nfp_state{0};  // 0 = NonInitialized, 1 = Initialized

void nfp_initialize(CPU& c) {
    g_nfp_state.store(1, std::memory_order_relaxed);
    c.x[0] = 0;
}
void nfp_finalize(CPU& c) {
    g_nfp_state.store(0, std::memory_order_relaxed);
    c.x[0] = 0;
}
void nfp_get_state(CPU& c) {
    c.x[0] = uint64_t(g_nfp_state.load(std::memory_order_relaxed));
}
void nfp_list_devices(CPU& c) {
    // Result ListDevices(DeviceHandle* outBuffer, int* outCount, int maxCount)
    if (c.x[1]) st<int32_t>(c.x[1], 0);
    c.x[0] = 0;  // ResultSuccess
}
void nfp_get_device_state(CPU& c) {
    c.x[0] = 0;  // DeviceState_Initialized
}
void nfp_attach_event_ok(CPU& c) {
    // Initialize an unsignaled SystemEventType if non-null so WaitSystemEvent doesn't see garbage,
    // and return ResultSuccess (0).
    if (c.x[0]) std::memset(reinterpret_cast<void*>(c.x[0]), 0, 0x30);
    c.x[0] = 0;
}
void nfp_ok(CPU& c) { c.x[0] = 0; }
// Result code: Module 115 (NFP), Desc 128 (DeviceNotFound / NotMounted) -> (115 | (128 << 9)) = 0x10073
void nfp_not_found(CPU& c) { c.x[0] = 0x10073u; }

// ---- nn::prepo (PlayReport telemetry) HLE stubs ----
void prepo_ok(CPU& c) { c.x[0] = 0; }

}  // namespace

NX_HLE("fputs", fputs_logged);
NX_HLE("_ZN2nn2os11SignalEventEPNS0_9EventTypeE", dread_signal_event);
NX_HLE("_ZN2nn2os16SendMessageQueueEPNS0_16MessageQueueTypeEm", dread_send_message_queue);
NX_HLE("_ZN2nn2os24TimedReceiveMessageQueueEPmPNS0_16MessageQueueTypeENS_8TimeSpanE", dread_timed_receive_message_queue);
NX_HLE("_ZN2nn2os13FinalizeMutexEPNS0_9MutexTypeE", dread_finalize_mutex);

// nn::nfp stubs
NX_HLE("_ZN2nn3nfp10InitializeEv", nfp_initialize);
NX_HLE("_ZN2nn3nfp8FinalizeEv", nfp_finalize);
NX_HLE("_ZN2nn3nfp8GetStateEv", nfp_get_state);
NX_HLE("_ZN2nn3nfp11ListDevicesEPNS0_12DeviceHandleEPii", nfp_list_devices);
NX_HLE("_ZN2nn3nfp14GetDeviceStateERKNS0_12DeviceHandleE", nfp_get_device_state);
NX_HLE("_ZN2nn3nfp19AttachActivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE", nfp_attach_event_ok);
NX_HLE("_ZN2nn3nfp21AttachDeactivateEventEPNS_2os15SystemEventTypeERKNS0_12DeviceHandleE", nfp_attach_event_ok);
NX_HLE("_ZN2nn3nfp29AttachAvailabilityChangeEventEPNS_2os15SystemEventTypeE", nfp_attach_event_ok);
NX_HLE("_ZN2nn3nfp14StartDetectionERKNS0_12DeviceHandleE", nfp_ok);
NX_HLE("_ZN2nn3nfp13StopDetectionERKNS0_12DeviceHandleE", nfp_ok);
NX_HLE("_ZN2nn3nfp5MountERKNS0_12DeviceHandleENS0_9ModelTypeENS0_11MountTargetE", nfp_not_found);
NX_HLE("_ZN2nn3nfp7UnmountERKNS0_12DeviceHandleE", nfp_ok);
NX_HLE("_ZN2nn3nfp10GetTagInfoEPNS0_7TagInfoERKNS0_12DeviceHandleE", nfp_not_found);
NX_HLE("_ZN2nn3nfp12GetModelInfoEPNS0_9ModelInfoERKNS0_12DeviceHandleE", nfp_not_found);
NX_HLE("_ZN2nn3nfp13GetCommonInfoEPNS0_10CommonInfoERKNS0_12DeviceHandleE", nfp_not_found);
NX_HLE("_ZN2nn3nfp15GetRegisterInfoEPNS0_12RegisterInfoERKNS0_12DeviceHandleE", nfp_not_found);
NX_HLE("_ZN2nn3nfp19OpenApplicationAreaERKNS0_12DeviceHandleEj", nfp_not_found);
NX_HLE("_ZN2nn3nfp18GetApplicationAreaEPvPmRKNS0_12DeviceHandleEm", nfp_not_found);
NX_HLE("_ZN2nn3nfp18SetApplicationAreaERKNS0_12DeviceHandleEPKvm", nfp_not_found);
NX_HLE("_ZN2nn3nfp21CreateApplicationAreaERKNS0_12DeviceHandleERKNS0_25ApplicationAreaCreateInfoE", nfp_not_found);
NX_HLE("_ZN2nn3nfp5FlushERKNS0_12DeviceHandleE", nfp_ok);
NX_HLE("_ZN2nn3nfp7RestoreERKNS0_12DeviceHandleE", nfp_not_found);
NX_HLE("_ZN2nn3nfp9GetNpadIdEPjRKNS0_12DeviceHandleE", nfp_not_found);
NX_HLE("_ZN2nn3nfp13StartRestorerEPNS0_12DeviceHandleERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE", nfp_not_found);
NX_HLE("_ZN2nn3nfp19StartGameDataEraserEPNS0_12DeviceHandleERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE", nfp_not_found);
NX_HLE("_ZN2nn3nfp29StartNicknameAndOwnerSettingsEPNS0_12DeviceHandleEPbPNS0_12RegisterInfoERKNS0_24AmiiboSettingsStartParamERKNS0_7TagInfoE", nfp_not_found);

// nn::prepo stubs
NX_HLE("_ZN2nn5prepo10PlayReportC1EPKc", prepo_ok);
NX_HLE("_ZN2nn5prepo10PlayReport9SetBufferEPvm", prepo_ok);
NX_HLE("_ZN2nn5prepo10PlayReport3AddEPKcS3_", prepo_ok);
NX_HLE("_ZN2nn5prepo10PlayReport3AddEPKcf", prepo_ok);
NX_HLE("_ZN2nn5prepo10PlayReport3AddEPKcl", prepo_ok);
NX_HLE("_ZN2nn5prepo10PlayReport4SaveERKNS_7account3UidE", prepo_ok);

void game::after_init() {
    game::renderer_start();
#if defined(_WIN32)
    if (!std::getenv("NX_IMAGE_STORAGE")) _putenv_s("NX_IMAGE_STORAGE", "0");
    if (!std::getenv("NX_GAME_LANGUAGES")) _putenv_s("NX_GAME_LANGUAGES", "ja en-us fr de it es nl ru ko zh-hans zh-hant");
#else
    setenv("NX_IMAGE_STORAGE", "0", 0);
    setenv("NX_GAME_LANGUAGES", "ja en-us fr de it es nl ru ko zh-hans zh-hant", 0);
#endif
}

void game::renderer_start() {
    static std::atomic<bool> started{false};
    if (started.exchange(true)) return;
    log("dread build: %s (universal NVN opt stack: PipeMru 1024, DescBank, EDS1-3, BufferCache)", kPerfBuildTag);
#if defined(__ANDROID__)
    setenv("NX_CLEARS_V2", "1", 0);
    setenv("NX_LOAD_CLEAR", "0", 0);
    setenv("NX_D24S8", "1", 0);
    setenv("NX_R11G11B10_NATIVE", "1", 0);
    setenv("NX_BOOT_DIAG", "1", 0);
#endif
}

void game::first_frame() {
    log("dread: first frame presented!");
}

}  // namespace nx
