#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/engram.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/text_generate.hpp"
#include "dsv41/weights.hpp"

#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace fs = std::filesystem;

namespace {
struct FileIdentity {
  off_t size = 0;
  timespec mtime{};
  timespec ctime{};
};

FileIdentity stat_file(const fs::path& path) {
  struct stat st {};
  if (lstat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) {
    throw std::runtime_error("missing or non-regular file: " + path.string());
  }
  return {st.st_size, st.st_mtimespec, st.st_ctimespec};
}

bool same_identity(const FileIdentity& a, const FileIdentity& b) {
  return a.size == b.size &&
         a.mtime.tv_sec == b.mtime.tv_sec && a.mtime.tv_nsec == b.mtime.tv_nsec &&
         a.ctime.tv_sec == b.ctime.tv_sec && a.ctime.tv_nsec == b.ctime.tv_nsec;
}

fs::path env_path(const char* name) {
  if (const char* value = std::getenv(name); value && *value) return fs::path(value);
  return {};
}

struct Args {
  fs::path checkpoint = env_path("DSV41_CHECKPOINT");
  fs::path m1_summary = env_path("DSV41_M1_SUMMARY");
  fs::path engram_metadata = env_path("DSV41_ENGRAM_METADATA");
  std::vector<std::uint32_t> prompt{0, 3};
};

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    auto need_value = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (key == "--checkpoint") args.checkpoint = need_value("--checkpoint");
    else if (key == "--m1-summary") args.m1_summary = need_value("--m1-summary");
    else if (key == "--engram-metadata") args.engram_metadata = need_value("--engram-metadata");
    else if (key == "--prompt-token") args.prompt.push_back(static_cast<std::uint32_t>(std::stoul(need_value("--prompt-token"))));
    else if (key == "--clear-default-prompt") args.prompt.clear();
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (args.checkpoint.empty()) throw std::runtime_error("--checkpoint or DSV41_CHECKPOINT is required");
  if (args.m1_summary.empty()) throw std::runtime_error("--m1-summary or DSV41_M1_SUMMARY is required");
  if (args.engram_metadata.empty()) throw std::runtime_error("--engram-metadata or DSV41_ENGRAM_METADATA is required");
  if (args.prompt.empty()) throw std::runtime_error("prompt must be nonempty");
  for (auto token : args.prompt) {
    if (token >= 129280) throw std::runtime_error("prompt token outside vocabulary");
  }
  return args;
}

std::map<fs::path, FileIdentity> checkpoint_identities(const fs::path& checkpoint) {
  std::map<fs::path, FileIdentity> ids;
  const auto index_path = checkpoint / "model.safetensors.index.json";
  ids[index_path] = stat_file(index_path);
  const auto index = dsv41::read_json_file(index_path).at("weight_map");
  std::set<std::string> shards;
  for (auto it = index.begin(); it != index.end(); ++it) shards.insert(it.value().get<std::string>());
  for (const auto& shard : shards) {
    if (fs::path(shard).filename() != shard) throw std::runtime_error("unsafe shard in index: " + shard);
    ids[checkpoint / shard] = stat_file(checkpoint / shard);
  }
  return ids;
}

void verify_unchanged(const std::map<fs::path, FileIdentity>& before) {
  for (const auto& [path, id] : before) {
    if (!same_identity(id, stat_file(path))) {
      throw std::runtime_error("checkpoint file changed during smoke: " + path.string());
    }
  }
}

void print_selectors() {
  std::cout << "production selectors:" << '\n'
            << "  packed_expert_bank=" << dsv41::runtime_packed_expert_bank_enabled() << '\n'
            << "  resident_expert_atlas=" << dsv41::runtime_resident_expert_atlas_enabled() << '\n'
            << "  compact_expert_bank=" << dsv41::runtime_compact_expert_bank_enabled() << '\n'
            << "  group_selected_experts=" << dsv41::runtime_group_selected_experts_enabled() << '\n'
            << "  grouped_expert_pipeline=" << dsv41::runtime_grouped_expert_pipeline_enabled() << '\n'
            << "  fixed_tile_attention=" << dsv41::runtime_fixed_tile_attention_enabled() << '\n'
            << "  ragged_tail_qk=" << dsv41::runtime_ragged_tail_qk_enabled() << '\n'
            << "  ragged_tail_av=" << dsv41::runtime_ragged_tail_av_enabled() << '\n'
            << "  layer_sweep=" << dsv41::runtime_layer_sweep_enabled() << '\n'
            << "  deferred_decoder=" << dsv41::runtime_deferred_decoder_enabled() << '\n'
            << "  decode_stack_graph=" << dsv41::runtime_decode_stack_graph_enabled() << '\n'
            << "  fused_mhc=" << dsv41::runtime_fused_mhc_enabled() << '\n'
            << "  chunk_attention=" << dsv41::runtime_chunk_attention_enabled() << '\n'
            << "  packed_chunk_attention=" << dsv41::runtime_packed_chunk_attention_enabled() << '\n'
            << "  wide_attention=" << dsv41::runtime_wide_attention_enabled() << '\n';
}
}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    const auto checkpoint = fs::canonical(args.checkpoint);
    const auto m1_summary = fs::canonical(args.m1_summary);
    const auto engram_metadata_path = fs::canonical(args.engram_metadata);

    std::cout << "resolved paths:" << '\n'
              << "  checkpoint=" << checkpoint << '\n'
              << "  m1_summary=" << m1_summary << '\n'
              << "  engram_metadata=" << engram_metadata_path << '\n';
    std::cout << "prompt_tokens:";
    for (auto token : args.prompt) std::cout << ' ' << token;
    std::cout << '\n';
    print_selectors();

    const auto before = checkpoint_identities(checkpoint);
    dsv41::WeightCatalog catalog(checkpoint, m1_summary);
    std::cout << "checkpoint_revision=" << catalog.revision() << '\n';
    auto metadata = dsv41::EngramMetadata::load(engram_metadata_path);
    std::cout << "engram_revision=" << metadata->revision << '\n'
              << "engram_identity=" << metadata->identity << '\n';

    dsv41::TextGenerationReference generator(catalog, metadata);
    std::cout << "model_construction=PASS" << '\n';
    const dsv41::SamplingConfig sampling{.temperature = 0.0f, .seed = 0};
    const auto result = generator.generate(args.prompt, 1, sampling);
    std::cout << "generation_finished=PASS" << '\n';
    std::cout << "output_token_count=" << result.tokens.size() << '\n';
    if (result.tokens.size() != 1) throw std::runtime_error("expected exactly one output token");
    const auto token = result.tokens.front();
    if (token >= 129280) throw std::runtime_error("output token outside vocabulary");
    if (result.next_position != args.prompt.size() + result.tokens.size()) {
      throw std::runtime_error("unexpected final position");
    }
    verify_unchanged(before);
    std::cout << "output_token_id=" << token << '\n'
              << "prompt_token_count=" << args.prompt.size() << '\n'
              << "final_model_position=" << result.next_position << '\n'
              << "checkpoint_files_unchanged=PASS" << '\n'
              << "full_checkpoint_smoke=PASS" << '\n';
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "full_checkpoint_smoke=FAIL" << '\n'
              << "error=" << e.what() << '\n';
    return 1;
  }
}
