cmake_minimum_required(VERSION 3.15)

# https://stackoverflow.com/a/53635241
#[[
CC=$RISCV/bin/riscv64-unknown-elf-gcc \
CXX=$RISCV/bin/riscv64-unknown-elf-g++ \
cmake .. \
(other options...)
]]
set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")

project(
  DLR
  VERSION 1.0.0
  DESCRIPTION ""
  HOMEPAGE_URL ""
  LANGUAGES C CXX)

add_library(DLR STATIC)

target_include_directories(DLR PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(DLR PUBLIC cxx_std_11)

add_executable(DumpKernel)

target_link_libraries(DumpKernel PRIVATE DLR)

add_subdirectory(src)
add_subdirectory(include)
add_subdirectory(3rdparty)

install(TARGETS DLR ARCHIVE DESTINATION lib)
install(TARGETS DumpKernel RUNTIME DESTINATION bin)
