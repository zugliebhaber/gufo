// Compare serial prefill against Model::PrefillBatch for one or more prompts.
// This isolates batched-prefill arithmetic from HTTP and scheduling. Every
// prompt is prefilled serially first and then all prompts are prefilled in one
// batch; each item's frontier logits must match its serial reference. With a
// single distinct prompt the per-layer residual rows are compared as well.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/models/qwen/chat_template.hpp"
#include "src/models/qwen38_flash_next/engine.hpp"

namespace qfn = gufo::models::qwen38_flash_next;

namespace {

void Require(bool ok, const std::string& error) {
  if (!ok)
    throw std::runtime_error(error);
}

std::size_t Argmax(std::span<const float> values) {
  return static_cast<std::size_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

double MaxAbsDiff(std::span<const float> a, std::span<const float> b) {
  double diff = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    diff = std::max(diff, std::abs(static_cast<double>(a[i]) - b[i]));
  return diff;
}

}  // namespace

int main(int argc, char** argv) try {
  std::string model_path;
  std::uint32_t context = 8192;
  std::string prompt =
      "Deterministic plan check 0. Reply with one short sentence containing "
      "the number.";
  std::vector<std::string> prompts;
  bool chat = false;
  std::size_t split = 0;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg = argv[i];
    if (i + 1 >= argc)
      throw std::invalid_argument("missing option value");
    const std::string value = argv[++i];
    if (arg == "--model")
      model_path = value;
    else if (arg == "--context")
      context = static_cast<std::uint32_t>(std::stoul(value));
    else if (arg == "--prompt") {
      prompt = value;
      prompts.push_back(value);
    } else if (arg == "--chat")
      chat = value != "0";
    else if (arg == "--split")
      split = static_cast<std::size_t>(std::stoul(value));
    else {
      throw std::invalid_argument("unknown option");
    }
  }
  Require(!model_path.empty(), "requires --model PATH [--context N]");
  if (prompts.empty())
    prompts.push_back(prompt);
  Require(prompts.size() <= 32, "at most 32 prompts");

  const std::string debug_dir = "/tmp/gufo_prefill_debug";
  std::filesystem::remove_all(debug_dir);
  setenv("GUFO_PREFILL_DEBUG_DIR", debug_dir.c_str(), 1);
  std::string error;
  auto model = qfn::Model::Load(
      model_path,
      {.max_context = context,
       .decode_concurrency = 1,
       .pool_sessions = static_cast<std::uint32_t>(2 * prompts.size() + 1)},
      &error);
  Require(bool(model), error);

  std::vector<std::vector<std::int32_t>> token_sets;
  token_sets.reserve(prompts.size());
  bool same_tokens = true;
  for (const auto& text : prompts) {
    std::vector<std::int32_t> tokens;
    if (chat) {
      const gufo::tokenization::ChatMessage message(
          gufo::tokenization::ChatRole::kUser, text);
      gufo::tokenization::ChatTemplateOptions options;
      options.enable_thinking = false;
      const auto rendered =
          gufo::tokenization::QwenChatTemplate::RenderAndTokenize(
              model->tokenizer(), std::span(&message, 1), options, &error);
      Require(bool(rendered), error);
      tokens.assign(rendered->begin(), rendered->end());
    } else {
      const auto host_tokens = model->tokenizer().Encode(text);
      tokens.reserve(host_tokens.size());
      for (const auto token : host_tokens)
        tokens.push_back(static_cast<std::int32_t>(token));
    }
    Require(!tokens.empty(), "tokenizer returned no tokens");
    if (!token_sets.empty() && token_sets.front() != tokens)
      same_tokens = false;
    token_sets.push_back(std::move(tokens));
  }

  std::vector<std::unique_ptr<qfn::Session>> serial;
  std::vector<std::vector<float>> serial_logits;
  serial.reserve(prompts.size());
  serial_logits.reserve(prompts.size());
  for (const auto& tokens : token_sets) {
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, context, &error);
    Require(bool(session), error);
    if (split != 0 && split < tokens.size())
      Require(session->Sync(std::span(tokens).first(split), &error), error);
    Require(session->Sync(tokens, &error), error);
    const auto logits = session->Logits();
    serial_logits.emplace_back(logits.begin(), logits.end());
    serial.push_back(std::move(session));
  }

  std::vector<std::unique_ptr<qfn::Session>> batched;
  batched.reserve(prompts.size());
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    auto session = model->CreateSession(
        gufo::core::SessionMode::kAutoregressive, context, &error);
    Require(bool(session), error);
    batched.push_back(std::move(session));
  }
  std::vector<qfn::Model::PrefillBatchItem> batch_items;
  batch_items.reserve(prompts.size());
  for (std::size_t i = 0; i < prompts.size(); ++i)
    batch_items.push_back({batched[i].get(), token_sets[i]});
  Require(model->PrefillBatch(batch_items, &error), error);

  std::size_t bad_items = 0;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    const auto logits = batched[i]->Logits();
    Require(!logits.empty() && logits.size() == serial_logits[i].size(),
            "logit size mismatch");
    const double diff = MaxAbsDiff(serial_logits[i], logits);
    if (diff > 1e-6) {
      ++bad_items;
      std::cout << "item=" << i << " tokens=" << token_sets[i].size()
                << " max_abs_diff=" << diff
                << " serial_argmax=" << Argmax(serial_logits[i])
                << " batched_argmax=" << Argmax(logits) << '\n';
    }
  }

  std::size_t first_layer_diff = model->config().num_layers + 1;
  if (same_tokens) {
    const auto read_layer = [&](const std::string& tag, std::uint32_t layer) {
      const std::string path = debug_dir + "/" + tag + "_" +
                               (layer < 10 ? "0" : "") + std::to_string(layer) +
                               ".bin";
      std::ifstream input(path, std::ios::binary);
      if (!input)
        return std::vector<float>{};
      input.seekg(0, std::ios::end);
      const auto bytes = static_cast<std::size_t>(input.tellg());
      input.seekg(0, std::ios::beg);
      std::vector<float> values(bytes / sizeof(float));
      input.read(reinterpret_cast<char*>(values.data()),
                 static_cast<std::streamsize>(bytes));
      return values;
    };
    const auto rows = static_cast<std::size_t>(token_sets.front().size());
    const auto stride = rows * model->config().HcDim();
    for (std::uint32_t layer = 0; layer <= model->config().num_layers;
         ++layer) {
      const auto reference = read_layer("serial", layer);
      const auto candidate = read_layer("batch", layer);
      if (reference.size() != stride ||
          candidate.size() != stride * prompts.size())
        continue;
      for (std::size_t i = 0; i < prompts.size(); ++i) {
        const double diff = MaxAbsDiff(
            reference, std::span(candidate).subspan(i * stride, stride));
        if (diff > 1e-4) {
          first_layer_diff = layer;
          std::cout << "first_layer_diff=" << layer << " item=" << i
                    << " layer_max_abs_diff=" << diff << '\n';
          break;
        }
      }
      if (first_layer_diff <= model->config().num_layers)
        break;
    }
    if (first_layer_diff > model->config().num_layers)
      std::cout << "no_layer_diff_above_1e-4\n";
  }

  const auto serial_token = static_cast<std::int32_t>(Argmax(serial_logits[0]));
  const auto batched_token =
      static_cast<std::int32_t>(Argmax(batched[0]->Logits()));
  std::cout << "items=" << prompts.size()
            << " vocab=" << serial_logits[0].size()
            << " serial_argmax=" << serial_token
            << " serial_text=" << model->TokenText(serial_token)
            << " batched_argmax=" << batched_token
            << " batched_text=" << model->TokenText(batched_token)
            << " bad_items=" << bad_items << '\n';
  return bad_items == 0 ? 0 : 2;
} catch (const std::exception& error) {
  std::cerr << "prefill_probe: " << error.what() << '\n';
  return 1;
}
