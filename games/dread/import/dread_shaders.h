// Metroid Dread shader extraction on the phone (games/dread/import):
// Reads all 478 uncompressed .bshdat ("MSHD") archives from system/shd/ inside the player's NSP/XCI,
// extracts every NVN control (0x98761234) + Maxwell code (0x12345678) stage pair into the batch layout
// expected by libspirvharness.so: <out>/<group>/manifest.tsv, code_<sha1>.bin, ctrl_<sha1>.bin.
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "nxgame.h"

namespace dread {

// Returns number of unique shader code blocks written, or -1 with `error` set.
long extract_shaders(const nxi::GameImage& game, const std::string& out_dir,
                     const std::function<void(int, int)>& progress, std::string& error);

}  // namespace dread
