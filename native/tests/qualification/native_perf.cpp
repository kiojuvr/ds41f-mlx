#include "dsv41/checkpoint_atlas.hpp"
#include "dsv41/engram.hpp"
#include "dsv41/execution_policy.hpp"
#include "dsv41/sampling.hpp"
#include "dsv41/text_backbone.hpp"
#include "dsv41/weights.hpp"
#include "nlohmann/json.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <mach/mach.h>
#include <mlx/mlx.h>
#include <sys/mount.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
namespace mx = mlx::core;
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;

namespace {
constexpr std::uint32_t kVocabSize = 129280;

struct Args {
  fs::path checkpoint = std::getenv("DSV41_CHECKPOINT") ? std::getenv("DSV41_CHECKPOINT") : "";
  fs::path m1_summary = std::getenv("DSV41_M1_SUMMARY") ? std::getenv("DSV41_M1_SUMMARY") : "";
  fs::path engram_metadata = std::getenv("DSV41_ENGRAM_METADATA") ? std::getenv("DSV41_ENGRAM_METADATA") : "";
  fs::path output = "artifacts/performance/native-short-context-baseline.json";
  std::vector<std::size_t> prefill_lengths{128, 512, 1024, 2048, 4096, 8192};
  std::vector<std::size_t> decode_counts{32, 128};
  std::size_t decode_prompt = 1024;
  std::size_t sanity_prompt = 128;
  std::size_t sanity_generate = 8;
  std::size_t short_warmups = 1;
  std::size_t short_repeats = 3;
  std::size_t expensive_repeats = 1;
  bool skip_sanity = false;
};

std::string need_value(int& i, int argc, char** argv, const char* name) {
  if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
  return argv[++i];
}

std::vector<std::size_t> parse_sizes(const std::string& csv) {
  std::vector<std::size_t> out;
  std::stringstream ss(csv);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) out.push_back(static_cast<std::size_t>(std::stoull(item)));
  }
  if (out.empty()) throw std::runtime_error("empty size list");
  return out;
}

Args parse_args(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--checkpoint") args.checkpoint = need_value(i, argc, argv, "--checkpoint");
    else if (key == "--m1-summary") args.m1_summary = need_value(i, argc, argv, "--m1-summary");
    else if (key == "--engram-metadata") args.engram_metadata = need_value(i, argc, argv, "--engram-metadata");
    else if (key == "--output") args.output = need_value(i, argc, argv, "--output");
    else if (key == "--prefill-lengths") args.prefill_lengths = parse_sizes(need_value(i, argc, argv, "--prefill-lengths"));
    else if (key == "--decode-counts") args.decode_counts = parse_sizes(need_value(i, argc, argv, "--decode-counts"));
    else if (key == "--decode-prompt") args.decode_prompt = static_cast<std::size_t>(std::stoull(need_value(i, argc, argv, "--decode-prompt")));
    else if (key == "--sanity-prompt") args.sanity_prompt = static_cast<std::size_t>(std::stoull(need_value(i, argc, argv, "--sanity-prompt")));
    else if (key == "--sanity-generate") args.sanity_generate = static_cast<std::size_t>(std::stoull(need_value(i, argc, argv, "--sanity-generate")));
    else if (key == "--short-warmups") args.short_warmups = static_cast<std::size_t>(std::stoull(need_value(i, argc, argv, "--short-warmups")));
    else if (key == "--short-repeats") args.short_repeats = static_cast<std::size_t>(std::stoull(need_value(i, argc, argv, "--short-repeats")));
    else if (key == "--expensive-repeats") args.expensive_repeats = static_cast<std::size_t>(std::stoull(need_value(i, argc, argv, "--expensive-repeats")));
    else if (key == "--skip-sanity") args.skip_sanity = true;
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (args.checkpoint.empty()) throw std::runtime_error("--checkpoint or DSV41_CHECKPOINT is required");
  if (args.m1_summary.empty()) throw std::runtime_error("--m1-summary or DSV41_M1_SUMMARY is required");
  if (args.engram_metadata.empty()) throw std::runtime_error("--engram-metadata or DSV41_ENGRAM_METADATA is required");
  return args;
}

std::string run_capture(const char* cmd) {
  std::array<char, 256> buf{};
  std::string out;
  FILE* pipe = popen(cmd, "r");
  if (!pipe) return "unavailable";
  while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) out += buf.data();
  pclose(pipe);
  while (!out.empty() && (out.back() == '\n' || out.back() == ' ' || out.back() == '\t')) out.pop_back();
  return out.empty() ? "unavailable" : out;
}

std::string sysctl_string(const char* name) {
  size_t size = 0;
  if (sysctlbyname(name, nullptr, &size, nullptr, 0) != 0 || size == 0) return "unavailable";
  std::string value(size, '\0');
  if (sysctlbyname(name, value.data(), &size, nullptr, 0) != 0) return "unavailable";
  while (!value.empty() && value.back() == '\0') value.pop_back();
  return value.empty() ? "unavailable" : value;
}

std::uint64_t sysctl_u64(const char* name) {
  std::uint64_t value = 0;
  size_t size = sizeof(value);
  if (sysctlbyname(name, &value, &size, nullptr, 0) != 0) return 0;
  return value;
}

json memory_snapshot(const std::string& label) {
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  json j{{"label", label}};
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
    j["process_rss_bytes"] = static_cast<std::uint64_t>(info.phys_footprint);
    j["resident_size_bytes"] = static_cast<std::uint64_t>(info.resident_size);
  } else {
    j["process_rss_bytes"] = nullptr;
    j["resident_size_bytes"] = nullptr;
  }
  j["mlx_active_bytes"] = static_cast<std::uint64_t>(mx::get_active_memory());
  j["mlx_cache_bytes"] = static_cast<std::uint64_t>(mx::get_cache_memory());
  j["mlx_peak_bytes"] = static_cast<std::uint64_t>(mx::get_peak_memory());
  j["swap"] = run_capture("/usr/sbin/sysctl -n vm.swapusage 2>/dev/null");
  return j;
}

std::string sha256_file(const fs::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open for sha256: " + path.string());
  CC_SHA256_CTX ctx;
  CC_SHA256_Init(&ctx);
  std::array<char, 1 << 20> buf{};
  while (in) {
    in.read(buf.data(), buf.size());
    const auto n = in.gcount();
    if (n > 0) CC_SHA256_Update(&ctx, buf.data(), static_cast<CC_LONG>(n));
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &ctx);
  std::ostringstream os;
  for (auto b : digest) os << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(b);
  return os.str();
}

struct FileIdentity { off_t size = 0; timespec mtime{}; timespec ctime{}; };
FileIdentity stat_file(const fs::path& path) {
  struct stat st {};
  if (lstat(path.c_str(), &st) != 0 || !S_ISREG(st.st_mode)) throw std::runtime_error("missing checkpoint file: " + path.string());
  return {st.st_size, st.st_mtimespec, st.st_ctimespec};
}
bool same_identity(const FileIdentity& a, const FileIdentity& b) {
  return a.size == b.size && a.mtime.tv_sec == b.mtime.tv_sec && a.mtime.tv_nsec == b.mtime.tv_nsec && a.ctime.tv_sec == b.ctime.tv_sec && a.ctime.tv_nsec == b.ctime.tv_nsec;
}
std::map<fs::path, FileIdentity> checkpoint_identities(const fs::path& checkpoint) {
  std::map<fs::path, FileIdentity> ids;
  const auto index_path = checkpoint / "model.safetensors.index.json";
  ids[index_path] = stat_file(index_path);
  const auto index = dsv41::read_json_file(index_path).at("weight_map");
  std::set<std::string> shards;
  for (auto it = index.begin(); it != index.end(); ++it) shards.insert(it.value().get<std::string>());
  for (const auto& shard : shards) ids[checkpoint / shard] = stat_file(checkpoint / shard);
  return ids;
}
void verify_unchanged(const std::map<fs::path, FileIdentity>& before) {
  for (const auto& [path, id] : before) if (!same_identity(id, stat_file(path))) throw std::runtime_error("checkpoint changed: " + path.string());
}

std::vector<std::uint32_t> make_prompt(std::size_t n) {
  std::vector<std::uint32_t> ids;
  ids.reserve(n);
  for (std::size_t i = 0; i < n; ++i) ids.push_back((i % 2 == 0) ? 0u : 3u);
  return ids;
}

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

json selectors_json() {
  return {
    {"packed_expert_bank", dsv41::runtime_packed_expert_bank_enabled()},
    {"resident_expert_atlas", dsv41::runtime_resident_expert_atlas_enabled()},
    {"compact_expert_bank", dsv41::runtime_compact_expert_bank_enabled()},
    {"group_selected_experts", dsv41::runtime_group_selected_experts_enabled()},
    {"grouped_expert_pipeline", dsv41::runtime_grouped_expert_pipeline_enabled()},
    {"fixed_tile_attention", dsv41::runtime_fixed_tile_attention_enabled()},
    {"ragged_tail_qk", dsv41::runtime_ragged_tail_qk_enabled()},
    {"ragged_tail_av", dsv41::runtime_ragged_tail_av_enabled()},
    {"layer_sweep", dsv41::runtime_layer_sweep_enabled()},
    {"deferred_decoder", dsv41::runtime_deferred_decoder_enabled()},
    {"decode_stack_graph", dsv41::runtime_decode_stack_graph_enabled()},
    {"fused_mhc", dsv41::runtime_fused_mhc_enabled()},
    {"chunk_attention", dsv41::runtime_chunk_attention_enabled()},
    {"packed_chunk_attention", dsv41::runtime_packed_chunk_attention_enabled()},
    {"wide_attention", dsv41::runtime_wide_attention_enabled()}
  };
}

struct BenchRuntime {
  std::shared_ptr<const dsv41::EngramMetadata> metadata;
  dsv41::TextBackboneReference model;
  BenchRuntime(dsv41::WeightCatalog& catalog, std::shared_ptr<const dsv41::EngramMetadata> m)
      : metadata(std::move(m)), model(catalog, metadata) {}

  dsv41::BlockResult prefill(std::span<const std::uint32_t> prompt, dsv41::TextBackboneState& state) const {
    std::optional<dsv41::BlockResult> result;
    const bool sweep = dsv41::runtime_layer_sweep_enabled();
    const bool deferred = sweep && dsv41::runtime_deferred_decoder_enabled();
    std::optional<dsv41::DeferredDecoderTransaction> pending_decoder;
    for (std::size_t offset = 0; offset < prompt.size();) {
      const auto step = dsv41::runtime_prefill_step(prompt.size() - offset, sweep, deferred, pending_decoder.has_value());
      auto input = prompt.subspan(offset, step.tokens);
      if (step.action == dsv41::RuntimePrefillStep::Action::BeginDeferredDecoder) {
        pending_decoder.emplace(model.begin_deferred_decoder(input, state, offset));
      } else if (step.action == dsv41::RuntimePrefillStep::Action::FinishDeferredDecoder) {
        result.emplace(model.finish_deferred_decoder(std::move(*pending_decoder), input, state));
        pending_decoder.reset();
      } else {
        result.emplace(sweep ? model.forward_packed_sweep(input, state, offset) : model.forward(input, state, offset));
      }
      offset += step.tokens;
    }
    if (pending_decoder) throw std::runtime_error("prefill ended with pending deferred decoder");
    if (!result) throw std::runtime_error("prefill produced no result");
    return std::move(*result);
  }

  mlx::core::array logits_last(const dsv41::BlockResult& result) const {
    auto logits = model.logits(result);
    const int n = int(result.hidden.shape(0));
    auto last = mx::slice(logits, {n - 1, 0}, {n, int(kVocabSize)});
    mx::eval(last);
    return last;
  }

  mlx::core::array decode_one(std::uint32_t token, dsv41::TextBackboneState& state, std::uint64_t position) const {
    const std::array<std::uint32_t, 1> one{token};
    const bool sweep = dsv41::runtime_layer_sweep_enabled();
    auto out = sweep ? model.forward_packed_sweep(one, state, position) : model.forward(std::span<const std::uint32_t>(one), state, position);
    auto step_logits = model.logits(out);
    auto last = mx::slice(step_logits, {0, 0}, {1, int(kVocabSize)});
    mx::eval(last);
    return last;
  }
};

json summarize(const std::vector<double>& xs) {
  if (xs.empty()) return json{{"count", 0}};
  auto sorted = xs;
  std::sort(sorted.begin(), sorted.end());
  return {{"count", xs.size()}, {"values", xs}, {"median", sorted[sorted.size() / 2]}, {"min", sorted.front()}, {"max", sorted.back()}};
}

json run_prefill_case(const BenchRuntime& rt, std::size_t prompt_len, std::size_t repeats, std::size_t warmups) {
  json raw = json::array();
  std::vector<double> times;
  for (std::size_t r = 0; r < warmups + repeats; ++r) {
    dsv41::TextBackboneState state(rt.metadata);
    auto prompt = make_prompt(prompt_len);
    mx::reset_peak_memory();
    const auto start = Clock::now();
    auto result = rt.prefill(prompt, state);
    auto last = rt.logits_last(result);
    (void)last;
    const double secs = seconds_since(start);
    if (state.revision() == 0) throw std::runtime_error("state did not advance in prefill");
    const bool measured = r >= warmups;
    if (measured) times.push_back(secs);
    raw.push_back({{"run", r}, {"warmup", !measured}, {"seconds", secs}, {"tokens_per_second", double(prompt_len) / secs}, {"memory", memory_snapshot("prefill_" + std::to_string(prompt_len))}});
  }
  auto summary = summarize(times);
  if (!times.empty()) summary["median_tokens_per_second"] = double(prompt_len) / summary.at("median").get<double>();
  return {{"prompt_tokens", prompt_len}, {"warmups", warmups}, {"measured_repeats", repeats}, {"raw", raw}, {"summary", summary}};
}

json run_decode_case(const BenchRuntime& rt, std::size_t prompt_len, std::size_t gen_count, std::size_t repeats, std::size_t warmups) {
  json raw = json::array();
  std::vector<double> total_times;
  std::vector<double> first_times;
  std::vector<double> steady_tps;
  for (std::size_t r = 0; r < warmups + repeats; ++r) {
    dsv41::TextBackboneState state(rt.metadata);
    auto prompt = make_prompt(prompt_len);
    auto result = rt.prefill(prompt, state);
    auto last = rt.logits_last(result);
    std::uint64_t rng = 0;
    std::vector<std::uint32_t> generated;
    std::vector<double> per_token;
    generated.reserve(gen_count);
    mx::reset_peak_memory();
    const auto total_start = Clock::now();
    for (std::size_t i = 0; i < gen_count; ++i) {
      const auto tok_start = Clock::now();
      const auto token = dsv41::sample_reference(last, 0.0f, rng);
      if (token >= kVocabSize) throw std::runtime_error("sampled token outside vocabulary");
      generated.push_back(token);
      last = rt.decode_one(token, state, prompt_len + i);
      per_token.push_back(seconds_since(tok_start));
    }
    const double total_secs = seconds_since(total_start);
    if (generated.size() != gen_count) throw std::runtime_error("wrong generated token count");
    const bool measured = r >= warmups;
    if (measured) {
      total_times.push_back(total_secs);
      first_times.push_back(per_token.empty() ? 0.0 : per_token.front());
      if (per_token.size() > 1) {
        const double rest = std::accumulate(per_token.begin() + 1, per_token.end(), 0.0);
        steady_tps.push_back(double(per_token.size() - 1) / rest);
      }
    }
    raw.push_back({{"run", r}, {"warmup", !measured}, {"total_seconds", total_secs}, {"tokens_per_second", double(gen_count) / total_secs}, {"first_token_seconds", per_token.empty() ? 0.0 : per_token.front()}, {"per_token_seconds", per_token}, {"memory", memory_snapshot("decode_" + std::to_string(gen_count))}});
  }
  auto total = summarize(total_times);
  if (!total_times.empty()) total["median_tokens_per_second"] = double(gen_count) / total.at("median").get<double>();
  return {{"prompt_tokens", prompt_len}, {"generated_tokens", gen_count}, {"temperature", 0}, {"warmups", warmups}, {"measured_repeats", repeats}, {"raw", raw}, {"summary", total}, {"first_token_seconds", summarize(first_times)}, {"steady_state_tokens_per_second", summarize(steady_tps)}};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto args = parse_args(argc, argv);
    fs::create_directories(args.output.parent_path());
    const auto checkpoint = fs::canonical(args.checkpoint);
    const auto m1_summary = fs::canonical(args.m1_summary);
    const auto engram_metadata_path = fs::canonical(args.engram_metadata);
    const auto before = checkpoint_identities(checkpoint);

    json artifact;
    artifact["schema"] = "ds41f.native-short-context-baseline.v1";
    artifact["identity"] = {
      {"git_commit", run_capture("git rev-parse HEAD 2>/dev/null")},
      {"checkpoint_path", checkpoint.string()},
      {"checkpoint_index_sha256", sha256_file(checkpoint / "model.safetensors.index.json")},
      {"mlx_version", mx::version()},
      {"macos_version", run_capture("/usr/bin/sw_vers -productVersion 2>/dev/null")},
      {"hardware_identity", sysctl_string("hw.model")},
      {"cpu_brand", sysctl_string("machdep.cpu.brand_string")},
      {"physical_unified_memory_bytes", sysctl_u64("hw.memsize")},
      {"run_temperature", "cold process for first construction; subsequent cases are warm process in same executable unless stated"}
    };
    artifact["runtime_selectors"] = selectors_json();
    artifact["fixture_generation"] = {
      {"method", "deterministic alternating existing valid text-token IDs"},
      {"pattern", json::array({0, 3})},
      {"image_tokens", false},
      {"large_token_json_allocated", false}
    };
    artifact["timer_methodology"] = {
      {"clock", "std::chrono::steady_clock"},
      {"synchronization", "runtime forward methods already call mlx::core::eval on hidden/pre_mix boundaries; this harness additionally calls mlx::core::eval on sliced logits at prefill and every decode step before stopping timers"},
      {"prefill_boundary", "starts immediately before production prefill schedule; stops after final logits slice is evaluated; excludes checkpoint/catalog/model construction"},
      {"decode_boundary", "starts after prefill and first logits are ready; per token includes greedy sampling, one-token model forward, logits slice, and mlx eval; excludes construction and prefill"}
    };

    const auto load_start = Clock::now();
    dsv41::WeightCatalog catalog(checkpoint, m1_summary);
    artifact["identity"]["checkpoint_revision"] = catalog.revision();
    artifact["memory"]["after_catalog"] = memory_snapshot("after_catalog");
    auto metadata = dsv41::EngramMetadata::load(engram_metadata_path);
    artifact["identity"]["engram_revision"] = metadata->revision;
    artifact["identity"]["engram_identity"] = metadata->identity;
    artifact["identity"]["engram_metadata_path"] = engram_metadata_path.string();
    artifact["memory"]["after_metadata"] = memory_snapshot("after_metadata");
    BenchRuntime rt(catalog, metadata);
    const double load_secs = seconds_since(load_start);
    artifact["load"] = {{"cold_process", true}, {"seconds_until_ready_for_first_request", load_secs}, {"memory_after_model_construction", memory_snapshot("after_model_construction")}, {"wired_limit_bytes", rt.model.wired_limit_bytes()}, {"expert_backing_file_backed", rt.model.expert_backing_file_backed()}};

    if (!args.skip_sanity) {
      auto sanity = run_decode_case(rt, args.sanity_prompt, args.sanity_generate, 1, 0);
      sanity["classification"] = "timer_harness_sanity";
      artifact["sanity"] = sanity;
    }

    artifact["prefill"] = json::array();
    for (auto len : args.prefill_lengths) {
      const auto repeats = len >= 4096 ? args.expensive_repeats : args.short_repeats;
      const auto warmups = len >= 4096 ? 0 : args.short_warmups;
      artifact["prefill"].push_back(run_prefill_case(rt, len, repeats, warmups));
      std::ofstream(args.output) << std::setw(2) << artifact << '\n';
    }

    artifact["decode"] = json::array();
    for (auto count : args.decode_counts) {
      const auto repeats = count >= 128 ? args.expensive_repeats : args.short_repeats;
      const auto warmups = count >= 128 ? 0 : args.short_warmups;
      artifact["decode"].push_back(run_decode_case(rt, args.decode_prompt, count, repeats, warmups));
      std::ofstream(args.output) << std::setw(2) << artifact << '\n';
    }

    verify_unchanged(before);
    artifact["correctness_sanity"] = {{"status", "PASS"}, {"generated_token_count_checked", true}, {"token_ids_in_vocabulary", true}, {"checkpoint_files_unchanged", true}, {"normal_state_progression_checked", true}};
    artifact["engram_observations"] = {{"mode", "SSD-backed Engram metadata/store path used by production runtime"}, {"counters_available", false}, {"note", "No non-invasive Engram read/page counters are currently exposed; first baseline uses synchronized wall-clock and memory telemetry."}};
    artifact["dominant_observed_bottleneck"] = "UNKNOWN_PENDING_REVIEW";
    artifact["qualification_status"] = "MEASURED_BASELINE_PENDING_CLASSIFICATION";
    artifact["memory"]["final"] = memory_snapshot("final");
    std::ofstream(args.output) << std::setw(2) << artifact << '\n';
    std::cout << "native_perf=PASS\noutput=" << args.output.string() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "native_perf=FAIL\nerror=" << e.what() << '\n';
    return 1;
  }
}
