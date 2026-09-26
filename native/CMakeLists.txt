cmake_minimum_required(VERSION 3.24)
project(ds41f_native_core LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT APPLE)
  message(FATAL_ERROR "Native checkpoint/storage foundation currently uses Apple CommonCrypto.")
endif()

add_library(dsv41_checkpoint src/model/checkpoint_atlas.cpp)
target_include_directories(dsv41_checkpoint PUBLIC include)
target_include_directories(dsv41_checkpoint SYSTEM PUBLIC third_party)
target_compile_options(dsv41_checkpoint PRIVATE -Wall -Wextra -Wpedantic -Wno-deprecated-declarations)

add_library(dsv41_storage
  src/model/weights.cpp
  src/engram/index.cpp
  src/engram/store.cpp
  src/attention/swa_state.cpp)
target_link_libraries(dsv41_storage PUBLIC dsv41_checkpoint)
target_compile_options(dsv41_storage PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

add_library(dsv41_runtime_bridge src/runtime/bridge_stub.cpp)
target_include_directories(dsv41_runtime_bridge PUBLIC include)
target_compile_options(dsv41_runtime_bridge PRIVATE -Wall -Wextra -Wpedantic)

add_executable(dsv41-test-swa tests/attention/test_swa.cpp)
target_link_libraries(dsv41-test-swa PRIVATE dsv41_storage)
add_executable(dsv41-test-engram tests/engram/test_engram.cpp)
target_link_libraries(dsv41-test-engram PRIVATE dsv41_storage)
add_executable(dsv41-execution-policy-test tests/runtime/test_execution_policy.cpp)
target_include_directories(dsv41-execution-policy-test PRIVATE include)
add_executable(dsv41-bridge-lifecycle-test tests/runtime/test_bridge.cpp)
target_include_directories(dsv41-bridge-lifecycle-test PRIVATE src/runtime)
target_link_libraries(dsv41-bridge-lifecycle-test PRIVATE dsv41_runtime_bridge)
add_executable(dsv41-generation-loop-test tests/attention/test_generation_loop.cpp)
target_include_directories(dsv41-generation-loop-test PRIVATE include)

enable_testing()
add_test(NAME dsv41-test-swa COMMAND dsv41-test-swa)
add_test(NAME dsv41-test-engram COMMAND dsv41-test-engram)
add_test(NAME execution_policy COMMAND dsv41-execution-policy-test)
add_test(NAME bridge_lifecycle COMMAND dsv41-bridge-lifecycle-test)
add_test(NAME generation_loop COMMAND dsv41-generation-loop-test)

option(DSV41_ENABLE_MLX "Build imported native MLX/Metal attention/session components" OFF)
set(DSV41_MLX_ROOT "" CACHE PATH "MLX install root or Python package root (contains include/, lib/, share/cmake/MLX)")
if(DSV41_ENABLE_MLX)
  set(DSV41_MLX_HINTS)
  if(DSV41_MLX_ROOT)
    list(APPEND DSV41_MLX_HINTS
      "${DSV41_MLX_ROOT}"
      "${DSV41_MLX_ROOT}/share/cmake/MLX")
  endif()
  if(DEFINED ENV{DSV41_MLX_ROOT})
    list(APPEND DSV41_MLX_HINTS
      "$ENV{DSV41_MLX_ROOT}"
      "$ENV{DSV41_MLX_ROOT}/share/cmake/MLX")
  endif()
  if(NOT MLX_DIR)
    file(GLOB DSV41_LOCAL_MLX_PACKAGES CONFIGURE_DEPENDS
      "${CMAKE_CURRENT_SOURCE_DIR}/../.venv/lib/python*/site-packages/mlx")
    foreach(DSV41_LOCAL_MLX_PACKAGE IN LISTS DSV41_LOCAL_MLX_PACKAGES)
      list(APPEND DSV41_MLX_HINTS
        "${DSV41_LOCAL_MLX_PACKAGE}"
        "${DSV41_LOCAL_MLX_PACKAGE}/share/cmake/MLX")
    endforeach()
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
      execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import importlib, pathlib; m=importlib.import_module('mlx.core'); print(pathlib.Path(m.__file__).resolve().parent)"
        RESULT_VARIABLE DSV41_PYTHON_MLX_RESULT
        OUTPUT_VARIABLE DSV41_PYTHON_MLX_ROOT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
      if(DSV41_PYTHON_MLX_RESULT EQUAL 0 AND EXISTS "${DSV41_PYTHON_MLX_ROOT}/share/cmake/MLX/MLXConfig.cmake")
        list(APPEND DSV41_MLX_HINTS
          "${DSV41_PYTHON_MLX_ROOT}"
          "${DSV41_PYTHON_MLX_ROOT}/share/cmake/MLX")
      endif()
    endif()
  endif()
  if(DSV41_MLX_HINTS)
    list(REMOVE_DUPLICATES DSV41_MLX_HINTS)
  endif()
  find_package(MLX 0.32.2 EXACT CONFIG REQUIRED HINTS ${DSV41_MLX_HINTS})
  message(STATUS "DSV41 MLX include dirs: ${MLX_INCLUDE_DIRS}")
  message(STATUS "DSV41 MLX library: ${MLX_LIBRARY}")
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/engram/dequantize.metal DSV41_ENGRAM_KERNEL_SOURCE)
  configure_file(src/engram/kernel_source.hpp.in generated/engram_kernel_source.hpp @ONLY)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/engram/fp8_header.metal DSV41_FP8_HEADER)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/engram/activation_quant.metal DSV41_ACTIVATION_QUANT)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/engram/projection.metal DSV41_PROJECTION)
  configure_file(src/engram/layer_kernels.hpp.in generated/layer_kernels.hpp @ONLY)

  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/kv_quant.metal DSV41_KV_QUANT)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/batched_splitk_qk.metal DSV41_BATCHED_SPLITK_QK)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/batched_splitk_accum.metal DSV41_BATCHED_SPLITK_ACCUM)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/ragged_tail_qk.metal DSV41_RAGGED_TAIL_QK)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/ragged_tail_accum.metal DSV41_RAGGED_TAIL_ACCUM)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/ragged_width_one_qk.metal DSV41_RAGGED_WIDTH_ONE_QK)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/gemv_header.metal DSV41_GEMV_HEADER)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/ragged_tail_av.metal DSV41_RAGGED_TAIL_AV)
  list(GET MLX_INCLUDE_DIRS 0 DSV41_MLX_INCLUDE_DIR)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/metal-module-cache)
  execute_process(COMMAND xcrun -sdk macosx metal -E -P
    -fmodules-cache-path=${CMAKE_CURRENT_BINARY_DIR}/metal-module-cache
    -I ${DSV41_MLX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/steel_gemm_header.metal
    -o ${CMAKE_CURRENT_BINARY_DIR}/generated/steel_gemm_header.metal
    RESULT_VARIABLE DSV41_STEEL_PREPROCESS_RESULT ERROR_VARIABLE DSV41_STEEL_PREPROCESS_ERROR)
  if(NOT DSV41_STEEL_PREPROCESS_RESULT EQUAL 0)
    message(FATAL_ERROR "cannot preprocess MLX Steel headers: ${DSV41_STEEL_PREPROCESS_ERROR}")
  endif()
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/generated/steel_gemm_header.metal DSV41_STEEL_HEADER)
  string(REGEX REPLACE "#pragma clang module import[^\n]*\n" "" DSV41_STEEL_HEADER "${DSV41_STEEL_HEADER}")
  execute_process(COMMAND xcrun -sdk macosx metal -E -P
    -fmodules-cache-path=${CMAKE_CURRENT_BINARY_DIR}/metal-module-cache
    -I ${DSV41_MLX_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/steel_attention_header.metal
    -o ${CMAKE_CURRENT_BINARY_DIR}/generated/steel_attention_header.metal
    RESULT_VARIABLE DSV41_STEEL_ATTN_PREPROCESS_RESULT ERROR_VARIABLE DSV41_STEEL_ATTN_PREPROCESS_ERROR)
  if(NOT DSV41_STEEL_ATTN_PREPROCESS_RESULT EQUAL 0)
    message(FATAL_ERROR "cannot preprocess MLX Steel attention headers: ${DSV41_STEEL_ATTN_PREPROCESS_ERROR}")
  endif()
  file(READ ${CMAKE_CURRENT_BINARY_DIR}/generated/steel_attention_header.metal DSV41_STEEL_ATTENTION_HEADER)
  string(REGEX REPLACE "#pragma clang module import[^\n]*\n" "" DSV41_STEEL_ATTENTION_HEADER "${DSV41_STEEL_ATTENTION_HEADER}")
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/packed_chunk_attention.metal DSV41_PACKED_CHUNK_ATTENTION)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/wide_chunk_attention.metal DSV41_WIDE_CHUNK_ATTENTION)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/attention/packed_attention_worklist.metal DSV41_PACKED_ATTENTION_WORKLIST)
  configure_file(src/attention/kv_quant.hpp.in generated/kv_quant.hpp @ONLY)
  configure_file(src/attention/batched_splitk_qk.hpp.in generated/batched_splitk_qk.hpp @ONLY)
  configure_file(src/attention/ragged_tail_qk.hpp.in generated/ragged_tail_qk.hpp @ONLY)
  configure_file(src/attention/ragged_width_one_qk.hpp.in generated/ragged_width_one_qk.hpp @ONLY)
  configure_file(src/attention/ragged_tail_av.hpp.in generated/ragged_tail_av.hpp @ONLY)
  configure_file(src/attention/packed_attention_worklist.hpp.in generated/packed_attention_worklist.hpp @ONLY)
  configure_file(src/attention/packed_chunk_attention.hpp.in generated/packed_chunk_attention.hpp @ONLY)
  configure_file(src/attention/wide_chunk_attention.hpp.in generated/wide_chunk_attention.hpp @ONLY)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/moe/route_select.metal DSV41_ROUTE_SELECT)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/moe/route_reduce.metal DSV41_ROUTE_REDUCE)
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/metal/mhc/split_sinkhorn.metal DSV41_MHC_SPLIT_SINKHORN)
  configure_file(src/moe/route_select.hpp.in generated/route_select.hpp @ONLY)
  configure_file(src/moe/route_reduce.hpp.in generated/route_reduce.hpp @ONLY)
  configure_file(src/mhc/split_sinkhorn.hpp.in generated/split_sinkhorn.hpp @ONLY)

  add_library(dsv41_engram_mlx src/engram/mlx.cpp src/engram/layer.cpp)
  target_include_directories(dsv41_engram_mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
  target_link_libraries(dsv41_engram_mlx PUBLIC dsv41_storage mlx)
  target_compile_options(dsv41_engram_mlx PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

  add_library(dsv41_attention_session
    src/model/linear.cpp
    src/model/entry.cpp
    src/model/trace.cpp
    src/attention/swa_projection.cpp
    src/attention/swa_layer.cpp
    src/attention/swa_attention.cpp
    src/attention/compressor.cpp
    src/attention/compressed_rope.cpp
    src/attention/index_key.cpp
    src/attention/kv_quant.cpp
    src/attention/index_query.cpp
    src/attention/compressed_layer.cpp
    src/attention/shared_attention.cpp
    src/cache/global_kv.cpp)
  target_include_directories(dsv41_attention_session PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
  target_link_libraries(dsv41_attention_session PUBLIC dsv41_storage mlx)
  target_compile_options(dsv41_attention_session PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

  add_library(dsv41_model_mlx
    src/model/entry.cpp
    src/model/linear.cpp
    src/model/block.cpp
    src/model/compressed_block.cpp
    src/model/reused_block.cpp
    src/model/text_front.cpp
    src/model/trace.cpp
    src/mhc/reference.cpp
    src/moe/reference.cpp
    src/moe/expert_bank.cpp
    src/moe/expert_backing.cpp
    src/moe/grouped_expert_pipeline.cpp
    src/runtime/residency.cpp)
  target_include_directories(dsv41_model_mlx PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)
  target_link_libraries(dsv41_model_mlx PUBLIC dsv41_attention_session mlx)
  target_compile_options(dsv41_model_mlx PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

  add_library(dsv41_text_pair_mlx
    src/model/text_pair.cpp
    src/model/text_triple.cpp
    src/model/text_quad.cpp
    src/model/text_octet.cpp
    src/model/text_encoder.cpp
    src/model/text_decoder.cpp
    src/model/text_backbone.cpp
    src/model/sampling.cpp
    src/model/text_generate.cpp)
  target_link_libraries(dsv41_text_pair_mlx PUBLIC dsv41_model_mlx dsv41_engram_mlx)
  target_compile_options(dsv41_text_pair_mlx PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

  add_library(dsv41_runtime_bridge_mlx SHARED src/runtime/bridge_stub.cpp src/runtime/bridge_mlx.cpp)
  target_compile_definitions(dsv41_runtime_bridge_mlx PRIVATE DSV41_BRIDGE_MLX=1)
  target_link_libraries(dsv41_runtime_bridge_mlx PRIVATE dsv41_text_pair_mlx)
  target_include_directories(dsv41_runtime_bridge_mlx PUBLIC include)
  target_compile_options(dsv41_runtime_bridge_mlx PRIVATE -Wall -Wextra -Wpedantic)

  add_executable(dsv41-swa-attention-test tests/attention/test_attention.cpp)
  target_link_libraries(dsv41-swa-attention-test PRIVATE dsv41_model_mlx)
  add_executable(dsv41-producer-chunk-test tests/attention/test_producer_chunk.cpp)
  target_link_libraries(dsv41-producer-chunk-test PRIVATE dsv41_model_mlx)
  add_executable(dsv41-block-test tests/attention/test_block.cpp)
  target_link_libraries(dsv41-block-test PRIVATE dsv41_model_mlx)
  add_executable(dsv41-route-policy-test tests/attention/test_route_policy.cpp)
  target_link_libraries(dsv41-route-policy-test PRIVATE dsv41_model_mlx)
  add_executable(dsv41-residency-test tests/runtime/test_residency.cpp)
  target_link_libraries(dsv41-residency-test PRIVATE dsv41_model_mlx)
  add_executable(dsv41-text-front-test tests/attention/test_text_front.cpp)
  target_link_libraries(dsv41-text-front-test PRIVATE dsv41_model_mlx)
  add_executable(dsv41-text-pair-test tests/attention/test_text_pair.cpp)
  target_link_libraries(dsv41-text-pair-test PRIVATE dsv41_text_pair_mlx)
  add_executable(dsv41-text-triple-test tests/attention/test_text_triple.cpp)
  target_link_libraries(dsv41-text-triple-test PRIVATE dsv41_text_pair_mlx)
  add_executable(dsv41-text-quad-test tests/attention/test_text_quad.cpp)
  target_link_libraries(dsv41-text-quad-test PRIVATE dsv41_text_pair_mlx)
  add_executable(dsv41-text-octet-test tests/attention/test_text_octet.cpp)
  target_link_libraries(dsv41-text-octet-test PRIVATE dsv41_text_pair_mlx)
  add_executable(dsv41-text-encoder-test tests/attention/test_text_encoder.cpp)
  target_link_libraries(dsv41-text-encoder-test PRIVATE dsv41_text_pair_mlx)
  add_executable(dsv41-text-decoder-test tests/attention/test_text_decoder.cpp)
  target_link_libraries(dsv41-text-decoder-test PRIVATE dsv41_text_pair_mlx)
  add_executable(dsv41-text-backbone-test tests/attention/test_text_backbone.cpp)
  target_link_libraries(dsv41-text-backbone-test PRIVATE dsv41_text_pair_mlx)
  add_executable(dsv41-sampling-test tests/attention/test_sampling.cpp)
  target_link_libraries(dsv41-sampling-test PRIVATE dsv41_text_pair_mlx)
  add_executable(dsv41-full-checkpoint-smoke tests/qualification/full_checkpoint_smoke.cpp)
  target_link_libraries(dsv41-full-checkpoint-smoke PRIVATE dsv41_text_pair_mlx)
  target_compile_options(dsv41-full-checkpoint-smoke PRIVATE -Wall -Wextra -Wpedantic)

  add_test(NAME route_policy COMMAND dsv41-route-policy-test)
  add_test(NAME runtime_residency COMMAND dsv41-residency-test)
  add_test(NAME sampling_reference COMMAND dsv41-sampling-test)
endif()
