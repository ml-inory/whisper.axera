cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(whisper)

set(CMAKE_CXX_STANDARD 17)

if (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -g -O0")
elseif (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_CXX_FLAGS "-fvisibility=hidden -O2")
endif()

include(cmake/msp_dependencies.cmake)
add_definitions(-DENV_HAS_STD_FILESYSTEM)
add_definitions(-DENV_HAS_POSIX_FILE_STAT)

# OpenCC for 简繁转换
include_directories(opencc/include/opencc)
link_directories(opencc/lib)

include_directories(${MSP_INC_DIR})
link_directories(${MSP_LIB_DIR})

include_directories(src)
aux_source_directory(src SRC)
aux_source_directory(src/utils SRC)

add_executable(whisper whisper.cpp ${SRC})
target_link_libraries(whisper ${MSP_LIBS} opencc marisa)

add_executable(whisper_srv whisper_srv.cpp ${SRC})
target_link_libraries(whisper_srv ${MSP_LIBS} opencc marisa pthread)

file(COPY ${CMAKE_SOURCE_DIR}/opencc/share/opencc/t2s.json DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY ${CMAKE_SOURCE_DIR}/opencc/share/opencc/TSPhrases.ocd2 DESTINATION ${CMAKE_INSTALL_PREFIX})
file(COPY ${CMAKE_SOURCE_DIR}/opencc/share/opencc/TSCharacters.ocd2 DESTINATION ${CMAKE_INSTALL_PREFIX})

install(TARGETS whisper whisper_srv
        RUNTIME
            DESTINATION .)