cmake_minimum_required(VERSION 3.18)
project(KokkosExamples LANGUAGES CXX CUDA)

include(FetchContent)

# --- Fetch Kokkos ---
FetchContent_Declare(
  kokkos
  GIT_REPOSITORY https://github.com/kokkos/kokkos.git
  GIT_TAG        4.5.01   # pick a stable release
)
FetchContent_MakeAvailable(kokkos)

# --- Collect all cpp files in src/ (adjust folder if needed) ---
file(GLOB EXAMPLES "*.cpp")

#Flages:
#lineinfo to have source in ncu files
#-Xptxas -v to see local memory usage, register counts
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -lineinfo -Xptxas -v")
foreach(source_file ${EXAMPLES})
  # Extract file name without extension
  get_filename_component(exe_name ${source_file} NAME_WE)

  add_executable(${exe_name} ${source_file})

  # Link Kokkos
  target_link_libraries(${exe_name} PRIVATE Kokkos::kokkoscore)

  message(STATUS "Added executable: ${exe_name}")
endforeach()