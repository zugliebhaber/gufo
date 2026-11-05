// Decode-gather bandwidth probe.
//
// The batch decode step reads a sparse subset of expert blocks: 238 of 512 at
// 32 sessions, 2.68 MB each. Measured end to end that step achieves ~128 GB/s
// while a prefill pass, which reads every expert in order, reaches 232 GB/s of
// the 240.76 GB/s ceiling. This probe isolates the access pattern from the
// kernel: a plain read of N sparse blocks and a read of the same byte count
// contiguously, both with the same grid shape.
//
// Build inside the container (no hipcc there; the HIP clang from the store and
// the include set of artifacts/serving/gxx-wrapper.sh are enough):
//   CXX=/nix/store/65dxwr554b4ij9f4aviz07f9v8jgai0a-hipClang/bin/clang++
//   FLAGS=$(grep -o -- "-isystem [^ ]*" artifacts/serving/gxx-wrapper.sh | tr "\n" " ")
//   $CXX -O3 --offload-arch=gfx1151 -x hip -std=c++20 $FLAGS -I. -c tools/bench/gather_bw.hip -o /tmp/gather_bw.o
//   $CXX --offload-arch=gfx1151 /tmp/gather_bw.o -o /tmp/gather_bw \
//     -L/nix/store/al4qwi7jfyrdkyb25xsri8b00b1agj1r-clr-10.1/lib -lamdhip64 \
//     -L/nix/store/pqyiidz5mi73wv5f91d48s9plcqyj5bz-glibc-2.44-57/lib \
//     -L/nix/store/7a2cjhcgzyym5zaz6m4fksw47qvm150w-gcc-16.2.0-lib/lib \
//     -lstdc++ -lm -lc -lgcc_s -Wl,-rpath,<the same three lib dirs>
// Result on gfx1151: 231-232 GB/s for both the contiguous and the sparse read,
// so the expert gather pattern is not the decode limit (see
// artifacts/serving/batch-pp-tg-levers.md section 17).

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#define HIP_CHECK(call)                                                     \
  do {                                                                      \
    hipError_t err__ = (call);                                              \
    if (err__ != hipSuccess) {                                              \
      std::fprintf(stderr, "%s:%d %s -> %s\n", __FILE__, __LINE__, #call,   \
                   hipGetErrorString(err__));                               \
      std::exit(1);                                                         \
    }                                                                       \
  } while (0)

namespace {

/// Reduces `bytes` of one block to a single float so the compiler cannot drop
/// the load. One block per lane-set keeps the access pattern identical to the
/// expert GEMM: each block reads its expert's rows sequentially.
__global__ void ReduceBlocks(const char* __restrict__ data,
                             const std::size_t* __restrict__ offsets,
                             std::size_t bytes, float* __restrict__ out) {
  // One block per expert block, all in one launch: the real expert GEMM shape.
  const std::size_t base = offsets[blockIdx.x];
  const std::size_t threads = blockDim.x;
  float sum = 0.0F;
  for (std::size_t offset = threadIdx.x * 16; offset + 16 <= bytes;
       offset += threads * 16) {
    const float4 value =
        *reinterpret_cast<const float4*>(data + base + offset);
    sum += value.x + value.y + value.z + value.w;
  }
  out[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

struct Timer {
  hipEvent_t start{};
  hipEvent_t stop{};
  Timer() {
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
  }
  ~Timer() {
    hipEventDestroy(start);
    hipEventDestroy(stop);
  }
  void Begin() { HIP_CHECK(hipEventRecord(start)); }
  double EndMs() {
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0F;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    return ms;
  }
};

double Run(const char* data, const std::vector<std::size_t>& offsets,
           std::size_t block_bytes, int block_threads, int iterations,
           std::size_t* device_offsets, float* scratch) {
  HIP_CHECK(hipMemcpy(device_offsets, offsets.data(),
                      offsets.size() * sizeof(std::size_t),
                      hipMemcpyHostToDevice));
  Timer timer;
  const auto launch = [&] {
    ReduceBlocks<<<offsets.size(), block_threads>>>(
        data, device_offsets, block_bytes, scratch);
  };
  launch();
  HIP_CHECK(hipDeviceSynchronize());
  timer.Begin();
  for (int iteration = 0; iteration < iterations; ++iteration) launch();
  const double ms = timer.EndMs();
  const double bytes =
      static_cast<double>(offsets.size()) * block_bytes * iterations;
  return bytes / (ms / 1000.0) / 1e9;
}

}  // namespace

int main() {
  // One layer's expert array: 512 blocks of 2.68 MB.
  constexpr std::size_t kBlocks = 512;
  constexpr std::size_t kBlockBytes = 2'684'354;  // 66 GB / 48 / 512, rounded
  constexpr int kThreads = 256;
  constexpr int kIterations = 20;
  const std::size_t total = kBlocks * kBlockBytes;

  char* data = nullptr;
  HIP_CHECK(hipMalloc(&data, total));
  HIP_CHECK(hipMemset(data, 1, total));
  float* scratch = nullptr;
  HIP_CHECK(hipMalloc(&scratch, 2048 * 2048 * sizeof(float) / 8));
  std::size_t* device_offsets = nullptr;
  HIP_CHECK(hipMalloc(&device_offsets, kBlocks * sizeof(std::size_t)));

  std::vector<std::size_t> all(kBlocks);
  for (std::size_t block = 0; block < kBlocks; ++block)
    all[block] = block * kBlockBytes;

  // Contiguous baseline: the first 238 blocks in order.
  std::vector<std::size_t> contiguous(all.begin(), all.begin() + 238);
  // Sparse: 238 blocks drawn from the whole array, as a 32-session step does.
  std::mt19937 generator(17);
  std::vector<std::size_t> indices(kBlocks);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), generator);
  std::vector<std::size_t> sparse;
  for (std::size_t i = 0; i < 238; ++i) sparse.push_back(indices[i] * kBlockBytes);
  std::sort(sparse.begin(), sparse.end());

  for (int threads : {256, 512, 1024}) {
    const double contiguous_gbps =
        Run(data, contiguous, kBlockBytes, threads, kIterations,
            device_offsets, scratch);
    const double sparse_gbps =
        Run(data, sparse, kBlockBytes, threads, kIterations, device_offsets,
            scratch);
    std::printf("threads=%4d  contiguous238=%6.1f GB/s   sparse238=%6.1f GB/s\n",
                threads, contiguous_gbps, sparse_gbps);
  }

  HIP_CHECK(hipFree(device_offsets));
  HIP_CHECK(hipFree(scratch));
  HIP_CHECK(hipFree(data));
  return 0;
}
