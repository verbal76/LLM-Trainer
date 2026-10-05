# HAG runtime native build: libhagrt.so (JNI + CPU gate, baseline ISA) and libhagengine.so (native/engine + llama.cpp).
#
# Inputs (passed by runtime/build.gradle.kts):
#   HAG_ENGINE_DIR  absolute path of native/engine         (C API header in include/, sources in src/)
#   LLAMA_CPP_DIR   absolute path of a fetched llama.cpp checkout (native/fetch-llama.sh or $LLAMA_CPP_SRC)
cmake_minimum_required(VERSION 3.22.1)
project(hag_runtime LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ggml is unusably slow unoptimised (an emulator training run would take hours): ALWAYS build the native code in
# Release, also for the Gradle "debug" variant. Debuggability of the Kotlin side is unaffected.
set(CMAKE_BUILD_TYPE Release CACHE STRING "HAG native build type is always Release" FORCE)

if(NOT HAG_ENGINE_DIR OR NOT EXISTS "${HAG_ENGINE_DIR}/include/hag_engine.h")
  message(FATAL_ERROR "HAG_ENGINE_DIR='${HAG_ENGINE_DIR}' does not contain include/hag_engine.h")
endif()
if(NOT LLAMA_CPP_DIR OR NOT EXISTS "${LLAMA_CPP_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "LLAMA_CPP_DIR='${LLAMA_CPP_DIR}' is not a llama.cpp checkout (run native/fetch-llama.sh or set LLAMA_CPP_SRC)")
endif()

# ---- 16 KB page-size compatible ELF output (Android 15+ / API 36 16 KB devices), no text relocations --------------
set(HAG_LINK_FLAGS
  -Wl,-z,max-page-size=16384
  -Wl,-z,common-page-size=16384
  -Wl,-z,text              # error on text relocations
  -Wl,-z,noexecstack
  -Wl,-z,relro -Wl,-z,now
  -Wl,--no-undefined
  -Wl,--gc-sections
  -Wl,--build-id=sha1)
string(REPLACE ";" " " HAG_LINK_FLAGS_STR "${HAG_LINK_FLAGS}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} ${HAG_LINK_FLAGS_STR}")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} ${HAG_LINK_FLAGS_STR}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")

# ---- llama.cpp / ggml: CPU backend only, static, ISA baseline chosen deliberately ----------------------------------
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
set(LLAMA_OPENSSL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_CCACHE OFF CACHE BOOL "" FORCE)
set(GGML_NATIVE OFF CACHE BOOL "never tune for the build machine" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "ggml's own thread pool; avoids shipping libomp" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "single static CPU backend (see docs/architecture/android-runtime.md)" FORCE)
set(GGML_CPU_KLEIDIAI OFF CACHE BOOL "no build-time network fetches" FORCE)
foreach(b CUDA HIP VULKAN OPENCL METAL SYCL MUSA RPC BLAS)
  set(GGML_${b} OFF CACHE BOOL "CPU backend only" FORCE)
endforeach()

if(ANDROID_ABI STREQUAL "arm64-v8a")
  # Baseline ARMv8.2-A + dot-product + half-precision. Runtime-gated by NativeBridge.nativeCpuCheck() (same set).
  # i8mm/SVE are NOT in the baseline: GGML_CPU_ALL_VARIANTS needs backend-DL (dlopen'd per-ISA .so files), which
  # is unverified here; ggml's runtime-dispatched ARM repack kernels still use i8mm where compiled in.
  set(GGML_CPU_ARM_ARCH "armv8.2-a+dotprod+fp16" CACHE STRING "" FORCE)
elseif(ANDROID_ABI STREQUAL "x86_64")
  # Emulator / x86 devices: SSE4.2 only so the engine runs on every emulator CPU model (AVX2 is not guaranteed).
  foreach(o AVX AVX2 AVX_VNNI AVX512 AVX512_VBMI AVX512_VNNI AVX512_BF16 FMA F16C BMI2 AMX_TILE AMX_INT8 AMX_BF16)
    set(GGML_${o} OFF CACHE BOOL "" FORCE)
  endforeach()
  set(GGML_SSE42 ON CACHE BOOL "" FORCE)
endif()

add_subdirectory("${LLAMA_CPP_DIR}" llama.cpp EXCLUDE_FROM_ALL)

# ---- libhagengine.so: engine + llama.cpp (never loaded before the CPU gate passes) -------------------------------
file(GLOB_RECURSE HAG_ENGINE_SOURCES CONFIGURE_DEPENDS
  "${HAG_ENGINE_DIR}/src/*.cpp" "${HAG_ENGINE_DIR}/src/*.cc" "${HAG_ENGINE_DIR}/src/*.c")
# Host-side test/tool/bench sources that live under src/ are not part of the shipped library.
list(FILTER HAG_ENGINE_SOURCES EXCLUDE REGEX "/(tests?|tools?|bench|benchmarks?|examples?)/|(_test|_tests|_main)\\.(c|cc|cpp)$")
if(NOT HAG_ENGINE_SOURCES)
  message(FATAL_ERROR "no engine sources under ${HAG_ENGINE_DIR}/src")
endif()
message(STATUS "HAG engine sources: ${HAG_ENGINE_SOURCES}")
add_library(hagengine SHARED ${HAG_ENGINE_SOURCES})
target_include_directories(hagengine PRIVATE
  "${HAG_ENGINE_DIR}/include" "${HAG_ENGINE_DIR}/src"
  "${LLAMA_CPP_DIR}/include" "${LLAMA_CPP_DIR}/src" "${LLAMA_CPP_DIR}/ggml/include" "${LLAMA_CPP_DIR}/ggml/src")
target_link_libraries(hagengine PRIVATE llama ggml)
if(ANDROID)
  target_link_libraries(hagengine PRIVATE log android)
endif()
target_link_options(hagengine PRIVATE -Wl,--exclude-libs,ALL)  # only the hag_* C API is exported

# ---- libhagrt.so: JNI bridge + CPU gate, plain baseline ISA, no link dependency on the engine -----------------------
add_library(hagrt SHARED hag_jni.cpp)
target_include_directories(hagrt PRIVATE "${HAG_ENGINE_DIR}/include")
target_link_libraries(hagrt PRIVATE ${CMAKE_DL_LIBS})
if(ANDROID)
  target_link_libraries(hagrt PRIVATE log)
endif()
