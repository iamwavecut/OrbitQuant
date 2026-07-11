cmake_minimum_required(VERSION 3.19)
project(orbitquant_vulkan LANGUAGES C CXX)

set(
  ORBITQUANT_EXECUTORCH_ROOT
  ""
  CACHE PATH "Path to the ExecuTorch source checkout used to build Vulkan"
)
if(NOT ORBITQUANT_EXECUTORCH_ROOT)
  message(
    FATAL_ERROR
      "Set ORBITQUANT_EXECUTORCH_ROOT to a post-2026-07-01 ExecuTorch source checkout."
  )
endif()

find_package(executorch CONFIG REQUIRED COMPONENTS vulkan_backend)

# ExecuTorch's shader helper reads this source-tree variable internally.
set(EXECUTORCH_ROOT ${ORBITQUANT_EXECUTORCH_ROOT})
if(NOT PYTHON_EXECUTABLE)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
endif()

include(${ORBITQUANT_EXECUTORCH_ROOT}/tools/cmake/Utils.cmake)
include(
  ${ORBITQUANT_EXECUTORCH_ROOT}/backends/vulkan/cmake/ShaderLibrary.cmake
)

set(
  VULKAN_THIRD_PARTY_PATH
  ${ORBITQUANT_EXECUTORCH_ROOT}/backends/vulkan/third-party
)
set(VULKAN_HEADERS_PATH ${VULKAN_THIRD_PARTY_PATH}/Vulkan-Headers/include)
set(VOLK_PATH ${VULKAN_THIRD_PARTY_PATH}/volk)
set(VMA_PATH ${VULKAN_THIRD_PARTY_PATH}/VulkanMemoryAllocator)

set(VULKAN_CXX_FLAGS "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-fexceptions>")
list(APPEND VULKAN_CXX_FLAGS "$<$<CXX_COMPILER_ID:MSVC>:/EHsc>")
set(
  VULKAN_COMPILE_DEFINITIONS
  USE_VULKAN_WRAPPER
  USE_VULKAN_VOLK
  "$<$<PLATFORM_ID:Windows>:NOMINMAX>"
  "$<$<PLATFORM_ID:Windows>:WIN32_LEAN_AND_MEAN>"
)

gen_vulkan_shader_lib_cpp(${CMAKE_CURRENT_SOURCE_DIR}/glsl)
vulkan_shader_lib(orbitquant_vulkan_shaderlib ${generated_spv_cpp})
target_compile_definitions(
  orbitquant_vulkan_shaderlib PRIVATE ${VULKAN_COMPILE_DEFINITIONS}
)
target_include_directories(
  orbitquant_vulkan_shaderlib PRIVATE ${ORBITQUANT_EXECUTORCH_ROOT}/src
)

add_library(orbitquant_vulkan_ops STATIC OrbitLinear.cpp)
target_compile_features(orbitquant_vulkan_ops PRIVATE cxx_std_17)
target_compile_options(orbitquant_vulkan_ops PRIVATE ${VULKAN_CXX_FLAGS})
target_compile_definitions(
  orbitquant_vulkan_ops PRIVATE ${VULKAN_COMPILE_DEFINITIONS}
)
target_include_directories(
  orbitquant_vulkan_ops
  PRIVATE
    ${ORBITQUANT_EXECUTORCH_ROOT}/src
    ${VULKAN_HEADERS_PATH}
    ${VOLK_PATH}
    ${VMA_PATH}
)
target_link_libraries(
  orbitquant_vulkan_ops
  PRIVATE executorch_core vulkan_backend orbitquant_vulkan_shaderlib
)
executorch_target_link_options_shared_lib(orbitquant_vulkan_ops)

install(
  TARGETS orbitquant_vulkan_ops orbitquant_vulkan_shaderlib
  ARCHIVE DESTINATION lib
)

option(ORBITQUANT_VULKAN_BUILD_TESTS "Build the Vulkan hardware test" OFF)
if(ORBITQUANT_VULKAN_BUILD_TESTS)
  add_executable(
    orbitquant_vulkan_test
    tests/OrbitLinearTest.cpp
    ${ORBITQUANT_EXECUTORCH_ROOT}/backends/vulkan/test/custom_ops/utils.cpp
    ${ORBITQUANT_EXECUTORCH_ROOT}/backends/vulkan/test/custom_ops/conv2d_utils.cpp
    ${ORBITQUANT_EXECUTORCH_ROOT}/backends/vulkan/test/custom_ops/cm_utils.cpp
  )
  target_compile_features(orbitquant_vulkan_test PRIVATE cxx_std_17)
  target_compile_options(orbitquant_vulkan_test PRIVATE ${VULKAN_CXX_FLAGS})
  target_compile_definitions(
    orbitquant_vulkan_test PRIVATE ${VULKAN_COMPILE_DEFINITIONS}
  )
  target_include_directories(
    orbitquant_vulkan_test
    PRIVATE
      ${ORBITQUANT_EXECUTORCH_ROOT}/src
      ${ORBITQUANT_EXECUTORCH_ROOT}/backends/vulkan/test/custom_ops
      ${VULKAN_HEADERS_PATH}
      ${VOLK_PATH}
      ${VMA_PATH}
  )
  target_link_libraries(
    orbitquant_vulkan_test
    PRIVATE
      executorch_core
      vulkan_backend
      orbitquant_vulkan_shaderlib
      orbitquant_vulkan_ops
  )
endif()

option(
  ORBITQUANT_VULKAN_BUILD_PTE_RUNNER
  "Build an ExecuTorch PTE runner with the OrbitQuant Vulkan op registered"
  OFF
)
if(ORBITQUANT_VULKAN_BUILD_PTE_RUNNER)
  set(GFLAGS_BUILD_TESTING OFF)
  set(GFLAGS_BUILD_PACKAGING OFF)
  add_subdirectory(
    ${ORBITQUANT_EXECUTORCH_ROOT}/third-party/gflags
    ${CMAKE_CURRENT_BINARY_DIR}/third-party/gflags
    EXCLUDE_FROM_ALL
  )

  add_executable(
    orbitquant_vulkan_pte_runner
    ${ORBITQUANT_EXECUTORCH_ROOT}/examples/portable/executor_runner/executor_runner.cpp
  )
  target_compile_features(orbitquant_vulkan_pte_runner PRIVATE cxx_std_17)
  target_compile_options(
    orbitquant_vulkan_pte_runner PRIVATE ${VULKAN_CXX_FLAGS}
  )
  target_compile_definitions(
    orbitquant_vulkan_pte_runner PRIVATE ${VULKAN_COMPILE_DEFINITIONS}
  )
  target_include_directories(
    orbitquant_vulkan_pte_runner PRIVATE ${ORBITQUANT_EXECUTORCH_ROOT}/..
  )
  target_link_libraries(
    orbitquant_vulkan_pte_runner
    PRIVATE
      executorch
      extension_data_loader
      extension_evalue_util
      extension_flat_tensor
      extension_runner_util
      gflags
      portable_ops_lib
      vulkan_backend
      orbitquant_vulkan_shaderlib
      orbitquant_vulkan_ops
  )
endif()
