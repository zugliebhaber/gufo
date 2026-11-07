#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEBUG_DUMP_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEBUG_DUMP_HPP_

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace gufo::models::qwen38_flash_next::rocm::debug {

/// Debug-only residual dump used by the prefill probe. No effect unless
/// GUFO_PREFILL_DEBUG_DIR and GUFO_PREFILL_DEBUG_TAG are set.
inline void DumpPrefillLayer(const char* tag, std::uint32_t layer,
                             const float* device, std::size_t count) {
  const char* directory = std::getenv("GUFO_PREFILL_DEBUG_DIR");
  if (directory == nullptr || *directory == '\0' || device == nullptr ||
      count == 0)
    return;
  try {
    std::filesystem::create_directories(directory);
    std::vector<float> host(count);
    if (hipDeviceSynchronize() != hipSuccess ||
        hipMemcpy(host.data(), device, count * sizeof(float),
                  hipMemcpyDeviceToHost) != hipSuccess)
      return;
    const std::string path = std::string(directory) + "/" + tag + "_" +
                             (layer < 10 ? "0" : "") + std::to_string(layer) +
                             ".bin";
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(host.data()),
                 static_cast<std::streamsize>(host.size() * sizeof(float)));
  } catch (...) {
    // Debug output must never change model behavior.
  }
}

}  // namespace gufo::models::qwen38_flash_next::rocm::debug

#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_KERNELS_ROCM_DEBUG_DUMP_HPP_
