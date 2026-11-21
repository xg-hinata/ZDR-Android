# ZDR-Android

Native ARM64 Android runtime for **Metroid Dread** (Nintendo Switch, v1.0.0), built on the Android port runtime architecture developed by [Brandon04Gonzalez](https://github.com/Brandon04Gonzalez) (`p5r-android`, `nier-android`, `p4au-android`). Runs the game's AArch64 binaries directly on the phone's CPU using Native Code Execution (NCE) and translates NVN and Horizon OS calls to Android Vulkan, AAudio, and MediaCodec.

> [!IMPORTANT]
> **Engine only.** No game files, executables, shaders, firmware, or keys are included in this repo or in any APK. You need your own legally dumped copy of **Metroid Dread 1.0.0** (`.nsp` or `.xci`, Title ID `010093801237C000`) and your console's `prod.keys`.

*Metroid Dread is copyright Nintendo / MercurySteam. This is an unofficial fan project and is not affiliated with Nintendo or MercurySteam.*

## How it works

- **Native CPU (`runtime/src/nce/`)**: Loads `main` and `sdk` straight from your `.nsp`/`.xci` into memory and runs them natively on ARM64, patching only `svc`, thread-pointer registers, and system counters.
- **In-place RomFS (`runtime/src/game_image.cpp`)**: Reads and decrypts game data directly from the dump on demand without extracting the full 4 GB image.
- **On-device shader prep (`games/dread/import/dread_shaders.cpp`)**: Scans the 478 uncompressed `.bshdat` files in `/system/shd/` (801 unique vertex/fragment shaders, no compute or geometry stages) and translates them once to SPIR-V on the phone via `libspirvharness.so`.
- **NVN-over-Vulkan (`runtime/src/nvn/`)**: Uses the mobile NVN optimization stack (`PipeMru` 1,024-slot cache, persistent `DescBank`, `EDS1-3` dynamic state, multi-submit `BufferCache`, dual-big-core async replay `NX_REPLAY_THREAD=1`, single-pass Mobile FSR, and Mali/Adreno/PowerVR/Xclipse fixes).

## Setup

1. Copy your **Metroid Dread 1.0.0** `.nsp`/`.xci` and `prod.keys` onto the phone.
2. Install the APK and allow All Files Access (stores saves, shaders, config, and logs in `Internal storage/Metroid Dread/`).
3. Pick your game file and `prod.keys` in the launcher, tap **Prepare**, and wait ~10-20 seconds for the 801 shaders to finish.
4. Tap **Play**.

## Requirements

- 64-bit ARM (`arm64-v8a`) Android device, **Android 10+ (API 29+)**, **Vulkan 1.1+**
- **6 GB RAM** minimum (8 GB recommended)

## Building from source

Prerequisites: **Android NDK 28** (`28.2.13676358`), **Android Build Tools 35**, **Android Platform 35**, **Java 17**, **CMake**, **Ninja**, and **Vulkan Headers** (`VULKAN_SDK`).

1. Place `libspirvharness.so` (arm64-v8a) at `private/build/spirv_harness_android/libspirvharness.so`.
2. Export `ANDROID_HOME` and `VULKAN_SDK`, then run:
   ```bash
   bash android/build_apk.sh --game dread --nce --release
   ```
3. The signed APK is written to `private/dread/apk_nce/dread.apk`.

## Development Note

AI tools were used to help write and refactor parts of this codebase. Hardware testing, profiling, log analysis, and direction are done by a human on real Android devices.

## Credits & License

- **BelmanteGu** (`SwitchGamesRecomp` / `SuperMarioWonderRecomp`, MIT): base Switch Horizon HLE runtime and NVN-over-Vulkan implementation.
- **[Brandon04Gonzalez](https://github.com/Brandon04Gonzalez)** (`p5r-android`, `nier-android`, `p4au-android`, MIT): AArch64 NCE loader, on-device NSP/XCI reader, shader import flow, and Android launcher base.
- **Ryujinx / Ryubing Contributors** & **.NET Foundation (NativeAOT)** (MIT): Maxwell-to-SPIR-V shader translator (`libspirvharness.so`, see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)).
- **libadrenotools**: optional custom Vulkan driver loading (`third_party/libadrenotools/LICENSE`).
- **xg-hinata** (MIT): ZDR-Android / Metroid Dread port (`games/dread/`).

MIT License. See [LICENSE](LICENSE).
