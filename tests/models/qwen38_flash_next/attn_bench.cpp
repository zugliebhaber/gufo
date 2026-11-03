// Times the prefill WMMA attention on synthetic caches and selections
// (diagnostic): one prefill chunk at several depths, sparse selections of
// 512 four-token blocks per query that neighbouring queries mostly share.
//
// attn_bench [tokens] [reps] [start positions...]
// attn_bench --masks reps FILE...
//
// A start position of 0 runs the dense causal window. A mask file replays a
// selection dumped from the model (three words: start position, tokens and
// mask words, then the rows' mask words) at its own geometry. Each line prints
// an FNV-1a hash of the output after the first call, so kernel rewrites can be
// checked for identical results.
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen38_flash_next/kernels/rocm/kernels.hpp"

namespace {

namespace q = gufo::models::qwen38_flash_next::rocm;

constexpr std::uint32_t kHeads = 24;
constexpr std::uint32_t kKvHeads = 2;
constexpr std::uint32_t kDim = 256;
constexpr std::uint32_t kKvWidth = kKvHeads * kDim;
constexpr std::uint32_t kRatio = 4;
constexpr std::uint32_t kSelected = 512;

void CheckHip(hipError_t error, const char* operation) {
  if (error != hipSuccess)
    throw std::runtime_error(std::string(operation) + ": " +
                             hipGetErrorString(error));
}

std::uint32_t Next(std::uint32_t* state) {
  *state ^= *state << 13;
  *state ^= *state >> 17;
  *state ^= *state << 5;
  return *state;
}

template<typename T>
T* Upload(const std::vector<T>& host) {
  void* p = nullptr;
  CheckHip(hipMalloc(&p, host.size() * sizeof(T)), "hipMalloc");
  CheckHip(
      hipMemcpy(p, host.data(), host.size() * sizeof(T), hipMemcpyHostToDevice),
      "hipMemcpy");
  return static_cast<T*>(p);
}

std::vector<float> Values(std::size_t count, std::uint32_t seed, float scale) {
  std::vector<float> host(count);
  for (auto& v : host)
    v = scale * (static_cast<float>(Next(&seed) % 2001) / 1000.0F - 1.0F);
  return host;
}

std::uint64_t Hash(const float* device, std::size_t count) {
  std::vector<std::uint8_t> host(count * sizeof(float));
  CheckHip(hipMemcpy(host.data(), device, host.size(), hipMemcpyDeviceToHost),
           "hipMemcpy");
  std::uint64_t h = 1469598103934665603ULL;
  for (std::uint8_t c : host)
    h = (h ^ c) * 1099511628211ULL;
  return h;
}

}  // namespace

std::vector<std::uint32_t> ReadWords(const char* path) {
  std::FILE* file = std::fopen(path, "rb");
  if (file == nullptr)
    throw std::runtime_error(std::string("cannot open ") + path);
  std::vector<std::uint32_t> words;
  std::uint32_t word = 0;
  while (std::fread(&word, sizeof(word), 1, file) == 1)
    words.push_back(word);
  std::fclose(file);
  if (words.size() < 3 ||
      words.size() != 3 + std::size_t{words[1]} * words[2] ||
      words[2] < (words[0] + words[1] + kRatio * 32 - 1) / (kRatio * 32))
    throw std::runtime_error(std::string("malformed mask file ") + path);
  return words;
}

int main(int argc, char** argv) try {
  bool paged = false;
  bool shuffle = true;
  std::vector<std::string> args;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--paged")
      paged = true;
    else if (std::string(argv[i]) == "--identity")
      shuffle = false;
    else
      args.emplace_back(argv[i]);
  }
  const bool replay = !args.empty() && args[0] == "--masks";
  std::uint32_t tokens =
      replay || args.empty()
          ? 4096
          : static_cast<std::uint32_t>(std::atoi(args[0].c_str()));
  const int reps = args.size() > 1 ? std::atoi(args[1].c_str()) : 5;
  std::vector<std::uint32_t> starts;
  std::vector<std::vector<std::uint32_t>> replays;
  for (std::size_t i = 2; i < args.size(); ++i) {
    if (replay) {
      replays.push_back(ReadWords(args[i].c_str()));
      starts.push_back(replays.back()[0]);
    } else {
      starts.push_back(static_cast<std::uint32_t>(std::atoi(args[i].c_str())));
    }
  }
  if (starts.empty())
    starts = {0, 8192, 32768, 65536, 118000};
  hipEvent_t start;
  hipEvent_t stop;
  CheckHip(hipEventCreate(&start), "hipEventCreate");
  CheckHip(hipEventCreate(&stop), "hipEventCreate");
  for (std::size_t run_index = 0; run_index < starts.size(); ++run_index) {
    const std::uint32_t start_pos = starts[run_index];
    if (replay)
      tokens = replays[run_index][1];
    const std::uint32_t n_kv = start_pos + tokens;
    const std::size_t q_count = std::size_t{tokens} * kHeads * kDim;
    const std::size_t kv_count = std::size_t{n_kv} * kKvHeads * kDim;
    float* d_q = Upload(Values(q_count, 0x1234ABCDU, 4.0F));
    float* d_gate = Upload(Values(q_count, 0x0BADF00DU, 3.0F));
    std::vector<__half> kh(kv_count);
    std::vector<__half> vh(kv_count);
    {
      std::uint32_t s = 0xBADC0FFEU;
      for (std::size_t i = 0; i < kv_count; ++i) {
        kh[i] =
            __float2half(static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F);
        vh[i] =
            __float2half(static_cast<float>(Next(&s) % 2001) / 1000.0F - 1.0F);
      }
    }
    __half* d_k = Upload(kh);
    __half* d_v = Upload(vh);
    // Sixteen-query groups share a base selection; each query swaps a fifth
    // of it, so four packed queries select a union of roughly 1.6x 512.
    const std::uint32_t blocks = (n_kv + kRatio - 1) / kRatio;
    const std::uint32_t mask_words =
        replay ? replays[run_index][2] : (blocks + 31) / 32;
    std::uint32_t* d_mask = nullptr;
    if (replay) {
      d_mask = Upload(std::vector<std::uint32_t>(replays[run_index].begin() + 3,
                                                 replays[run_index].end()));
    } else if (start_pos != 0) {
      std::vector<std::uint32_t> mask(std::size_t{tokens} * mask_words);
      std::vector<std::uint32_t> base;
      std::uint32_t s = 0xDEADBEEFU;
      for (std::uint32_t t = 0; t < tokens; ++t) {
        const std::uint32_t complete = (start_pos + t + 1) / kRatio;
        if (t % 16 == 0) {
          base.clear();
          for (std::uint32_t j = 0; j < kSelected; ++j)
            base.push_back(Next(&s) % complete);
        }
        std::uint32_t* row = mask.data() + std::size_t{t} * mask_words;
        for (std::uint32_t b : base) {
          if (Next(&s) % 5 == 0)
            b = Next(&s) % complete;
          row[b / 32] |= 1U << (b % 32);
        }
      }
      d_mask = Upload(mask);
    }
    float* d_out = Upload(std::vector<float>(q_count));
    // Optional paged store: the same rows, scattered into a block pool by a
    // shuffled table. The paged run must reproduce the contiguous hash, and the
    // printed delta is the price the paged addressing adds to one call.
    constexpr std::uint32_t kBlockTokens = 128;
    constexpr std::uint32_t kBlockShift = 7;  // log2(kBlockTokens)
    const std::uint32_t n_blocks = (n_kv + kBlockTokens - 1) / kBlockTokens;
    const std::size_t block_stride = std::size_t{kBlockTokens} * kKvWidth;
    std::vector<std::uint32_t> table(n_blocks);
    std::vector<__half> pool_k(paged ? n_blocks * block_stride : 0);
    std::vector<__half> pool_v(paged ? n_blocks * block_stride : 0);
    if (paged) {
      std::uint32_t s = 0x5EED5EEDU;
      for (std::uint32_t b = 0; b < n_blocks; ++b)
        table[b] = b;
      for (std::uint32_t b = n_blocks; shuffle && b > 1; --b) {
        const std::uint32_t j = Next(&s) % b;
        std::swap(table[b - 1], table[j]);
      }
      for (std::uint32_t b = 0; b < n_blocks; ++b) {
        const std::uint32_t dst = table[b];
        for (std::uint32_t r = 0; r < kBlockTokens; ++r) {
          const std::size_t src_row = std::size_t{b} * kBlockTokens + r;
          if (src_row >= n_kv)
            break;
          const auto row = static_cast<std::ptrdiff_t>(kKvWidth);
          const auto src = static_cast<std::ptrdiff_t>(src_row) * row;
          const auto to = static_cast<std::ptrdiff_t>(
              (std::size_t{dst} * kBlockTokens + r) * kKvWidth);
          std::copy_n(kh.data() + src, kKvWidth, pool_k.begin() + to);
          std::copy_n(vh.data() + src, kKvWidth, pool_v.begin() + to);
        }
      }
    }
    __half* d_pk = paged ? Upload(pool_k) : nullptr;
    __half* d_pv = paged ? Upload(pool_v) : nullptr;
    std::uint32_t* d_table = paged ? Upload(table) : nullptr;
    gufo::hip::KvPageView view{};
    if (paged) {
      view.table = d_table;
      view.block_stride = block_stride;
      view.layer_stride = block_stride;  // one layer in this bench
      view.context_stride = block_stride * n_blocks;
      view.block_shift = kBlockShift;
    }
    const auto run = [&](const gufo::hip::KvPageView* page) {
      if (!q::WmmaCausalAttention(d_q, d_gate, page ? d_pk : d_k,
                                  page ? d_pv : d_v, d_mask, mask_words, d_out,
                                  tokens, start_pos, kHeads, kKvHeads, kDim,
                                  kRatio, nullptr, false, page, 0))
        throw std::runtime_error("attention rejected the geometry");
    };
    run(nullptr);
    CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
    const std::uint64_t hash = Hash(d_out, q_count);
    std::uint64_t paged_hash = hash;
    run(nullptr);
    std::vector<float> ms(reps);
    for (int i = 0; i < reps; ++i) {
      CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
      run(nullptr);
      CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
      CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
      CheckHip(hipEventElapsedTime(&ms[i], start, stop), "hipEventElapsedTime");
    }
    std::sort(ms.begin(), ms.end());
    std::printf("start %6u tokens %u %s  %8.3f ms  hash %016llx\n", start_pos,
                tokens, d_mask ? "sparse" : "dense ", ms[reps / 2],
                static_cast<unsigned long long>(hash));
    if (paged) {
      run(&view);
      CheckHip(hipDeviceSynchronize(), "hipDeviceSynchronize");
      paged_hash = Hash(d_out, q_count);
      std::vector<float> pms(reps);
      for (int i = 0; i < reps; ++i) {
        CheckHip(hipEventRecord(start, nullptr), "hipEventRecord");
        run(&view);
        CheckHip(hipEventRecord(stop, nullptr), "hipEventRecord");
        CheckHip(hipEventSynchronize(stop), "hipEventSynchronize");
        CheckHip(hipEventElapsedTime(&pms[i], start, stop),
                 "hipEventElapsedTime");
      }
      std::sort(pms.begin(), pms.end());
      std::printf(
          "start %6u tokens %u %s  %8.3f ms paged (%+5.1f%%)  hash %016llx "
          "%s\n",
          start_pos, tokens, d_mask ? "sparse" : "dense ", pms[reps / 2],
          100.0F * (pms[reps / 2] - ms[reps / 2]) / ms[reps / 2],
          static_cast<unsigned long long>(paged_hash),
          paged_hash == hash ? "identical" : "DIFFERENT");
      if (paged_hash != hash)
        return 1;
    }
    for (void* p : {static_cast<void*>(d_q), static_cast<void*>(d_gate),
                    static_cast<void*>(d_k), static_cast<void*>(d_v),
                    static_cast<void*>(d_mask), static_cast<void*>(d_out),
                    static_cast<void*>(d_pk), static_cast<void*>(d_pv),
                    static_cast<void*>(d_table)})
      if (p != nullptr)
        CheckHip(hipFree(p), "hipFree");
  }
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
