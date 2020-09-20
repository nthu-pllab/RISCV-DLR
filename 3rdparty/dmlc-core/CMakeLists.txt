target_include_directories(DLR PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

include(GNUInstallDirs)
install(FILES include/dmlc/any.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/base.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/endian.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/io.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/json.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/logging.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/memory_io.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/serializer.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/thread_local.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
install(FILES include/dmlc/type_traits.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dmlc)
