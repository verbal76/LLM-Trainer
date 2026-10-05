cmake_minimum_required(VERSION 3.14)
project(hag_engine LANGUAGES C CXX)

# HAG engine: C++17 wrapper around llama.cpp/ggml (CPU backend only) exposing include/hag_engine.h.
#
#   native/fetch-llama.sh                       # fetch + patch the pinned llama.cpp into native/.deps/llama.cpp
#   cmake -S native/engine -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j
#
# Android (NDK r28):  cmake -S native/engine -B build-android -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#                       -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-26 -DHAG_BUILD_CLI=OFF -DHAG_BUILD_TESTS=OFF
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(HAG_BUILD_SHARED "Build libhag_engine.so (what the JNI layer loads)" ON)
option(HAG_BUILD_CLI    "Build hag_cli"                                      ON)
option(HAG_NATIVE       "Compile for the host CPU (GGML_NATIVE); OFF for portable/CI/Android builds" ON)
set(LLAMA_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../.deps/llama.cpp" CACHE PATH "patched llama.cpp tree (native/fetch-llama.sh)")

if(NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "llama.cpp not found at ${LLAMA_CPP_DIR}. Run native/fetch-llama.sh (see native/llama.cpp.pin).")
endif()

# pinned commit -> version string
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/../llama.cpp.pin" _pin_lines REGEX "^[0-9a-f]+$")
list(GET _pin_lines 0 HAG_LLAMA_COMMIT_FULL)
string(SUBSTRING "${HAG_LLAMA_COMMIT_FULL}" 0 7 HAG_LLAMA_COMMIT_SHORT)

# ---- llama.cpp / ggml: static, CPU only, no extras ------------------------------------------------------------------
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON   OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL        OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP          OFF CACHE BOOL "" FORCE)   # no libomp dependency: ggml's own thread pool
set(GGML_BACKEND_DL      OFF CACHE BOOL "" FORCE)
set(GGML_CPU_ALL_VARIANTS OFF CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_RPC OFF CACHE BOOL "" FORCE)
if(HAG_NATIVE AND NOT ANDROID AND NOT CMAKE_CROSSCOMPILING)
  set(GGML_NATIVE ON CACHE BOOL "" FORCE)
else()
  set(GGML_NATIVE OFF CACHE BOOL "" FORCE)
endif()
add_subdirectory("${LLAMA_CPP_DIR}" llama.cpp EXCLUDE_FROM_ALL)

# ---- engine ----------------------------------------------------------------------------------------------------------
set(HAG_SRC src/engine.cpp src/train.cpp src/hag_util.cpp)

function(hag_configure tgt)
  target_include_directories(${tgt} PUBLIC include)
  target_include_directories(${tgt} PRIVATE src "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/ggml/include")
  target_compile_definitions(${tgt} PRIVATE HAG_LLAMA_COMMIT="${HAG_LLAMA_COMMIT_SHORT}" _FILE_OFFSET_BITS=64)
  set_target_properties(${tgt} PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  if(NOT MSVC)
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wno-unused-parameter -Wno-missing-field-initializers)
  endif()
  target_link_libraries(${tgt} PUBLIC llama ggml)
endfunction()

add_library(hag_engine_static STATIC ${HAG_SRC})
hag_configure(hag_engine_static)

if(HAG_BUILD_SHARED)
  add_library(hag_engine SHARED ${HAG_SRC})
  hag_configure(hag_engine)
  if(ANDROID)
    target_link_libraries(hag_engine PRIVATE log)
  endif()
  # export only hag_* (llama/ggml stay private inside the .so)
  if(NOT APPLE AND NOT MSVC)
    target_link_options(hag_engine PRIVATE "-Wl,--exclude-libs,ALL")
  endif()
endif()

if(HAG_BUILD_CLI)
  add_executable(hag_cli tools/hag_cli.cpp)
  target_include_directories(hag_cli PRIVATE include)
  target_compile_definitions(hag_cli PRIVATE _FILE_OFFSET_BITS=64)
  target_link_libraries(hag_cli PRIVATE hag_engine_static)
  if(NOT MSVC)
    target_compile_options(hag_cli PRIVATE -Wall -Wextra -Wno-unused-parameter)
  endif()
endif()
