/**************************************************************************************************
 *
 * Copyright (c) 2019-2025 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/

#include "ax_model_runner/ax_model_runner.hpp"
#include "utils/logger.h"
#include "utils/memory_utils.hpp"

#include <ax_sys_api.h>

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <vector>
#include <cstdint>

#define IO_CMM_ALIGN_SIZE   128

AxModelRunner::AxModelRunner():
    m_handle(nullptr),
    m_pIOinfo(nullptr),
    m_input_num(0),
    m_output_num(0),
    m_loaded(false) {

    memset(&m_io, 0, sizeof(AX_ENGINE_IO_T));
}

AxModelRunner::~AxModelRunner() {
    unload_model();
}

int AxModelRunner::load_model(const char* model_path, IO_BUFFER_STRATEGY_T strategy) {
    AX_CHAR *pModelBufferVirAddr = nullptr;
    AX_U32 nModelBufferSize = 0;
        
    MMap model_buffer(model_path);
    pModelBufferVirAddr = (char*)model_buffer.data();
    nModelBufferSize = model_buffer.size();

    auto freeModelBuffer = [&]() {
        model_buffer.close_file();
        return;
    };

    int ret = AX_ENGINE_CreateHandle(&m_handle, pModelBufferVirAddr, nModelBufferSize);
    if (0 != ret) {
        ALOGE("AX_ENGINE_CreateHandle failed! ret=0x%x", ret);
        freeModelBuffer();
        return ret;
    }
        
    ret = AX_ENGINE_CreateContext(m_handle);
    if (0 != ret) {
        ALOGE("AX_ENGINE_CreateContext failed! ret=0x%x", ret);
        freeModelBuffer();
        return ret;
    }

    m_strategy = strategy;
    ret = _prepare_io();
    if (0 != ret) {
        ALOGE("_prepare_io failed! ret=0x%x", ret);
        freeModelBuffer();
        _free_io();
        return ret;
    }

    freeModelBuffer();
    m_loaded = (ret == 0);

    return ret;
}

int AxModelRunner::unload_model(void) {
    int ret = 0;
    if (m_handle != 0) {
        ret = AX_ENGINE_DestroyHandle(m_handle);
        if (0 == ret)   
            m_handle = 0;
    }

    _free_io();

    return ret;
}

int AxModelRunner::run(void) {
    if (m_strategy == IO_BUFFER_STRATEGY_CACHED) {
        for (int index = 0; index < m_input_num; index++) {
            _cache_io_flush(m_io.pInputs[index]);
        }
    }

    int ret = AX_ENGINE_RunSync(m_handle, &m_io);
    if (0 != ret) {
        ALOGE("AX_ENGINE_RunSync failed! ret=0x%x", ret);
        return ret;
    }
    return ret;
}

int AxModelRunner::set_input(int index, void* data) {
    if (index < 0)  index += m_input_num;
    if (index > m_input_num - 1) {
        ALOGE("index(%d) exceed input_num(%d)", index, m_input_num);
        return -1;
    }

    if (!data) {
        ALOGE("data is null");
        return -1;
    }

    memcpy(m_io.pInputs[index].pVirAddr, data, m_io.pInputs[index].nSize);

    return 0;
}

int AxModelRunner::set_inputs(const std::vector<void*>& datas) {
    for (int index = 0; index < m_input_num; index++) {
        void* data = datas[index];
        if (!data) {
            ALOGE("index %d data is null", index);
            return -1;
        }

        memcpy(m_io.pInputs[index].pVirAddr, data, m_io.pInputs[index].nSize);
    }

    return 0;
}

int AxModelRunner::get_output(int index, void* data) {
    if (m_strategy == IO_BUFFER_STRATEGY_CACHED)
        _cache_io_flush(m_io.pOutputs[index]);

    memcpy(data, m_io.pOutputs[index].pVirAddr, m_io.pOutputs[index].nSize);

    return 0;
}

int AxModelRunner::get_outputs(const std::vector<void*>& datas) {
    for (int index = 0; index < m_output_num; index++) {
        void* data = datas[index];
        if (!data) {
            ALOGE("index %d data is null", index);
            return -1;
        }

        if (m_strategy == IO_BUFFER_STRATEGY_CACHED)
            _cache_io_flush(m_io.pOutputs[index]);

        memcpy(data, m_io.pOutputs[index].pVirAddr, m_io.pOutputs[index].nSize);
    }
    
    return 0;
}

void* AxModelRunner::get_input_ptr(int index) {
    return m_io.pInputs[index].pVirAddr;
}

void* AxModelRunner::get_output_ptr(int index) {
    if (m_strategy == IO_BUFFER_STRATEGY_CACHED)
        _cache_io_flush(m_io.pOutputs[index]);

    return m_io.pOutputs[index].pVirAddr;
}

AX_U64 AxModelRunner::get_input_phy_addr(int index) {
    return m_io.pInputs[index].phyAddr;
}

AX_U64 AxModelRunner::get_output_phy_addr(int index) {
    return m_io.pOutputs[index].phyAddr;
}

const char* AxModelRunner::get_input_name(int index) {
    return m_input_names[index].c_str();
}

const char* AxModelRunner::get_output_name(int index) {
    return m_output_names[index].c_str();
}

int AxModelRunner::get_input_size(int index) {
    return m_pIOinfo->pInputs[index].nSize;
}

int AxModelRunner::get_output_size(int index) {
    return m_pIOinfo->pOutputs[index].nSize;
}

std::vector<int> AxModelRunner::get_input_shape(int index) {
    std::vector<int> shape;

    shape.resize(m_pIOinfo->pInputs[index].nShapeSize);
    for (int i = 0; i < shape.size(); i++) {
        shape[i] = m_pIOinfo->pInputs[index].pShape[i];
    }
    return shape;
}

std::vector<int> AxModelRunner::get_output_shape(int index) {
    std::vector<int> shape;

    shape.resize(m_pIOinfo->pOutputs[index].nShapeSize);
    for (int i = 0; i < shape.size(); i++) {
        shape[i] = m_pIOinfo->pOutputs[index].pShape[i];
    }
    return shape;
}

// ================ PRIVATE ================
int AxModelRunner::_prepare_io() {
    int ret = AX_ENGINE_GetIOInfo(m_handle, &m_pIOinfo);
    if (0 != ret) {
        ALOGE("AX_ENGINE_GetIOInfo failed! ret=0x%x", ret);
        return ret;
    }

    m_input_num = m_pIOinfo->nInputSize;
    m_output_num = m_pIOinfo->nOutputSize;

    m_io.nInputSize = m_pIOinfo->nInputSize;
    m_io.nOutputSize = m_pIOinfo->nOutputSize;

    m_io.pInputs = new AX_ENGINE_IO_BUFFER_T[m_pIOinfo->nInputSize];
    m_io.pOutputs = new AX_ENGINE_IO_BUFFER_T[m_pIOinfo->nOutputSize];

    for (int i = 0; i < m_pIOinfo->nInputSize; i++) {
        const char* layer_name = m_pIOinfo->pInputs[i].pName;
        m_input_names.push_back(std::string(layer_name));

        ret = _alloc_io_buffer(m_io.pInputs[i], m_pIOinfo->pInputs[i], m_strategy);
        if (0 != ret) {
            ALOGE("_alloc_io_buffer for input[%d] failed! ret=0x%x", i, ret);
            return ret;
        }
    }

    for (int i = 0; i < m_pIOinfo->nOutputSize; i++) {
        const char* layer_name = m_pIOinfo->pOutputs[i].pName;
        m_output_names.push_back(std::string(layer_name));

        ret = _alloc_io_buffer(m_io.pOutputs[i], m_pIOinfo->pOutputs[i], m_strategy);
        if (0 != ret) {
            ALOGE("_alloc_io_buffer for output[%d] failed! ret=0x%x", i, ret);
            return ret;
        }
    }

    return ret;
}

void AxModelRunner::_free_io() {
    for (size_t i = 0; i < m_io.nInputSize; i++) {
        if (0 != m_io.pInputs[i].phyAddr)
            AX_SYS_MemFree(m_io.pInputs[i].phyAddr, m_io.pInputs[i].pVirAddr);
    }

    for (size_t i = 0; i < m_io.nOutputSize; i++) {
        if (0 != m_io.pOutputs[i].phyAddr)
            AX_SYS_MemFree(m_io.pOutputs[i].phyAddr, m_io.pOutputs[i].pVirAddr);
    }
    
    delete[] m_io.pInputs;
    delete[] m_io.pOutputs;
    memset(&m_io, 0, sizeof(AX_ENGINE_IO_T));
}

int AxModelRunner::_alloc_io_buffer(AX_ENGINE_IO_BUFFER_T& buffer, 
        const AX_ENGINE_IOMETA_T &meta, IO_BUFFER_STRATEGY_T strategy) {
    int ret = 0;
    
    memset(&buffer, 0, sizeof(AX_ENGINE_IO_BUFFER_T));
    buffer.nSize = meta.nSize;
    
    if (IO_BUFFER_STRATEGY_DEFAULT == strategy) {
        AX_SYS_MemAlloc((AX_U64*)&buffer.phyAddr, 
            (AX_VOID**)&buffer.pVirAddr, 
            meta.nSize, IO_CMM_ALIGN_SIZE, (const AX_S8*)meta.pName);
    } else {
        AX_SYS_MemAllocCached((AX_U64*)&buffer.phyAddr, 
            (AX_VOID**)&buffer.pVirAddr, 
            meta.nSize, IO_CMM_ALIGN_SIZE, (const AX_S8*)meta.pName);
    }

    return ret;
}

void AxModelRunner::_cache_io_flush(AX_ENGINE_IO_BUFFER_T &buffer) {
    if (buffer.phyAddr != 0) {
        AX_SYS_MflushCache(buffer.phyAddr, buffer.pVirAddr, buffer.nSize);
    }
}