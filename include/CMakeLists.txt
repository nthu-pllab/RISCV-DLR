include(GNUInstallDirs)
install(FILES dlr.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES tvm/runtime/c_backend_api.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/c_runtime_api.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/device_api.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/module.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/ndarray.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/node_base.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/packed_func.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/registry.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/serializer.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/threading_backend.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
install(FILES tvm/runtime/util.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tvm/runtime)
