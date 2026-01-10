/**************************************************************************************************
 *
 * Copyright (c) 2019-2025 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/

#pragma once

#include "ax_engine_api.h"

#include <vector>
#include <string>

typedef enum _IO_BUFFER_STRATEGY_T {
    IO_BUFFER_STRATEGY_DEFAULT = 0,
    IO_BUFFER_STRATEGY_CACHED
} IO_BUFFER_STRATEGY_T;

class AxModelRunner {
public:
    AxModelRunner();

    ~AxModelRunner();

    int load_model(const char* model_path, IO_BUFFER_STRATEGY_T strategy = IO_BUFFER_STRATEGY_CACHED);

    int unload_model(void);

    int run(void);

    int set_input(int index, void* data);
    int set_inputs(const std::vector<void*>& datas);

    int get_output(int index, void* data);
    int get_outputs(const std::vector<void*>& datas);

    inline int get_input_num(void) {
        return m_input_num;
    }
    inline int get_output_num(void) {
        return m_output_num;
    }

    void* get_input_ptr(int index);
    void* get_output_ptr(int index);

    AX_U64 get_input_phy_addr(int index);
    AX_U64 get_output_phy_addr(int index);

    const char* get_input_name(int index);
    const char* get_output_name(int index);

    int get_input_size(int index);
    int get_output_size(int index);

    std::vector<int> get_input_shape(int index);
    std::vector<int> get_output_shape(int index);

private:
    int _prepare_io();
    void _free_io();
    int _alloc_io_buffer(AX_ENGINE_IO_BUFFER_T &buffer, 
            const AX_ENGINE_IOMETA_T &meta, IO_BUFFER_STRATEGY_T strategy);
    void _cache_io_flush(AX_ENGINE_IO_BUFFER_T &buffer);        
    
private:
    AX_ENGINE_HANDLE m_handle;
    AX_ENGINE_IO_T m_io;
    AX_ENGINE_IO_INFO_T* m_pIOinfo;
    int m_input_num;
    int m_output_num;
    IO_BUFFER_STRATEGY_T m_strategy;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    bool m_loaded;
};
