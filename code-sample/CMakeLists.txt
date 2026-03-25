cmake_minimum_required(VERSION 3.21)

project(KokkosExamples LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")

include(FetchContent)

# --- Fetch Kokkos ---
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        5.0.2   # pick a stable release
)
FetchContent_MakeAvailable(kokkos)


#CUDA only flags
if (Kokkos_ENABLE_CUDA)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lineinfo -Xptxas -v")
endif()
# --- Collect all cpp files in src/ dirty.. https://cmake.org/cmake/help/latest/command/file.html#filesystem---
file(GLOB EXAMPLES "*.cpp")

foreach(source_file ${EXAMPLES})
  # Extract file name without extension
  get_filename_component(exe_name ${source_file} NAME_WE)

  add_executable(${exe_name} ${source_file})

  target_compile_features(${exe_name} PRIVATE cxx_std_20) 
  
  target_link_libraries(${exe_name} PRIVATE Kokkos::kokkos)

  message(STATUS "Added executable: ${exe_name}")
endforeach()