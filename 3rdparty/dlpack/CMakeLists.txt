target_include_directories(DLR PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

include(GNUInstallDirs)
install(FILES include/dlpack/dlpack.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dlpack)
