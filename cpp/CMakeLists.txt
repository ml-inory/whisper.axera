cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(whisper)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
    add_definitions("-D__DEBUG__")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2")
endif()

# 设置安装路径
set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Installation prefix")
set(CMAKE_INSTALL_LIBDIR "lib")
set(CMAKE_INSTALL_INCLUDEDIR "include")
set(CMAKE_INSTALL_BINDIR ".")

include(cmake/msp_dependencies.cmake)
add_definitions(-DENV_HAS_STD_FILESYSTEM)
add_definitions(-DENV_HAS_POSIX_FILE_STAT)

# OpenCC for Traditional Chinese to Simplified Chinese
include_directories(opencc/include/opencc)
link_directories(opencc/lib)

# Axera BSP
include_directories(${MSP_INC_DIR})
link_directories(${MSP_LIB_DIR})

# Project sources
aux_source_directory(src SRC)
aux_source_directory(src/utils SRC)
aux_source_directory(src/ax_model_runner SRC)

# libax_whisper.so - 共享库
add_library(ax_whisper SHARED src/api/ax_whisper_api.cpp ${SRC})

# 设置库属性
set_target_properties(ax_whisper PROPERTIES
    OUTPUT_NAME "ax_whisper"
    DEBUG_POSTFIX "d"
    PUBLIC_HEADER "src/api/ax_whisper_api.h"
)

# 链接依赖库到 ax_whisper 库
target_link_libraries(ax_whisper PRIVATE ${MSP_LIBS} opencc marisa)

# 设置包含目录
target_include_directories(ax_whisper
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/api>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/utils
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ax_model_runner
        ${MSP_INC_DIR}
        opencc/include/opencc
)

# whisper_cli executable - 依赖于 ax_whisper 库
add_executable(whisper_cli whisper_cli.cpp)
target_include_directories(whisper_cli PRIVATE src)
target_link_libraries(whisper_cli PUBLIC ax_whisper PRIVATE ${MSP_LIBS})

# whisper_srv executable - 依赖于 ax_whisper 库
add_executable(whisper_srv whisper_srv.cpp)
target_include_directories(whisper_srv PRIVATE src src/utils/)
target_link_libraries(whisper_srv PUBLIC ax_whisper PRIVATE ${MSP_LIBS} pthread)

# 安装 ax_whisper 库和头文件
install(TARGETS ax_whisper
        EXPORT ax_whisper-targets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 安装头文件（额外的安装方式，确保头文件被安装）
install(FILES src/api/ax_whisper_api.h
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 安装可执行文件
install(TARGETS whisper_cli whisper_srv
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# 创建配置文件供 find_package 使用
install(EXPORT ax_whisper-targets
        FILE ax_whisper-config.cmake
        NAMESPACE ax::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ax_whisper
)

# 可选：为开发人员提供导出目标
export(EXPORT ax_whisper-targets
       FILE ${CMAKE_CURRENT_BINARY_DIR}/ax_whisper-targets.cmake
       NAMESPACE ax::
)

# 在构建时自动复制OpenCC文件到构建目录（便于测试）
file(COPY ${CMAKE_SOURCE_DIR}/opencc/share/opencc/t2s.json DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY ${CMAKE_SOURCE_DIR}/opencc/share/opencc/TSPhrases.ocd2 DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY ${CMAKE_SOURCE_DIR}/opencc/share/opencc/TSCharacters.ocd2 DESTINATION ${CMAKE_INSTALL_PREFIX})

# 设置RPATH
set_target_properties(whisper_cli whisper_srv PROPERTIES
    INSTALL_RPATH "$ORIGIN/lib"
    BUILD_WITH_INSTALL_RPATH TRUE  # 构建时也使用install的RPATH
    SKIP_BUILD_RPATH FALSE
)
