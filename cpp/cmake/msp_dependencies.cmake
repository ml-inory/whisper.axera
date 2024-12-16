# bsp
if(NOT BSP_MSP_DIR)
    if (CHIP_AX650)
        add_definitions(-DCHIP_AX650)
        set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/ax650n_bsp_sdk/msp/out)
    else()
        add_definitions(-DCHIP_AX630C)
        set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/ax620e_bsp_sdk/msp/out/arm64_glibc)
    endif()
endif()
message(STATUS "BSP_MSP_DIR = ${BSP_MSP_DIR}")

# check bsp exist
if(NOT EXISTS ${BSP_MSP_DIR})
    message(FATAL_ERROR "FATAL: BSP_MSP_DIR ${BSP_MSP_DIR} not exist")
endif()

set(MSP_INC_DIR ${BSP_MSP_DIR}/include)
set(MSP_LIB_DIR ${BSP_MSP_DIR}/lib)

list(APPEND MSP_LIBS
        ax_sys
        ax_engine
        ax_interpreter)