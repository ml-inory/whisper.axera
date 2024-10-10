# bsp
if(NOT BSP_MSP_DIR)
    set(BSP_MSP_DIR ${CMAKE_SOURCE_DIR}/ax650n_bsp_sdk/msp/out)
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
        ax_ivps
        ax_venc
        ax_engine
        ax_interpreter)