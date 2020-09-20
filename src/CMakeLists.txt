target_sources(
  DLR
  PUBLIC dlr.cc
          runtime/module_util.cc
          runtime/workspace_pool.cc
          runtime/module.cc
          runtime/graph/graph_runtime.cc
          runtime/file_util.cc
          runtime/cpu_device_api.cc
          runtime/threading_backend.cc
          runtime/ndarray.cc
          runtime/system_lib_module.cc
          runtime/registry.cc
          runtime/thread_pool.cc
          runtime/c_runtime_api.cc
          contrib/sort/sort.cc)

target_include_directories(DLR PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/runtime")

target_sources(DumpKernel PUBLIC dump-kernel.cc)

include(GNUInstallDirs)
install(FILES runtime/graph/graph_runtime.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/graph)
