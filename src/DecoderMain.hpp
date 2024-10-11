/**************************************************************************************************
 *
 * Copyright (c) 2019-2023 Axera Semiconductor (Ningbo) Co., Ltd. All Rights Reserved.
 *
 * This source file is the property of Axera Semiconductor (Ningbo) Co., Ltd. and
 * may not be copied or distributed in any isomorphic form without the prior
 * written consent of Axera Semiconductor (Ningbo) Co., Ltd.
 *
 **************************************************************************************************/

#pragma once

#include "EngineWrapper.hpp"

struct DecoderMainData {
    // Input
    int32_t* tokens;
    float* n_layer_cross_k;
    float* n_layer_cross_v;

    // Output
    float* logits;
    float* out_n_layer_self_k_cache;
    float* out_n_layer_self_v_cache;
};

class DecoderMain : public EngineWrapper {
public:
    DecoderMain() = default;
    ~DecoderMain() = default;

    int PrepareData(DecoderMainData* data) {
        data->tokens = new int32_t[GetInputSize(0) / sizeof(int32_t)];
        data->n_layer_cross_k = new float[GetInputSize(1) / sizeof(float)];
        data->n_layer_cross_v = new float[GetInputSize(2) / sizeof(float)];

        data->logits = new float[GetOutputSize(0) / sizeof(float)];
        data->out_n_layer_self_k_cache = new float[GetOutputSize(1) / sizeof(float)];
        data->out_n_layer_self_v_cache = new float[GetOutputSize(2) / sizeof(float)];
        return 0;
    }

    int Run(DecoderMainData* data) {
        int ret = 0;
        ret = SetInput((uint8_t*)data->tokens, 0);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data->n_layer_cross_k, 1);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data->n_layer_cross_v, 2);
        if (ret) {
            return ret;
        }

        ret = this->RunSync();
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data->logits, 0);
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data->out_n_layer_self_k_cache, 1);
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data->out_n_layer_self_v_cache, 2);
        if (ret) {
            return ret;
        }

        return ret;
    }

    void DestroyData(DecoderMainData* data) {
        delete[] data->tokens;
        delete[] data->n_layer_cross_k;
        delete[] data->n_layer_cross_v;

        delete[] data->logits;
        delete[] data->out_n_layer_self_k_cache;
        delete[] data->out_n_layer_self_v_cache;
    }
};