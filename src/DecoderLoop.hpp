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

struct DecoderLoopData {
    // Input
    int32_t* tokens;
    float* in_n_layer_self_k_cache;
    float* in_n_layer_self_v_cache;
    float* n_layer_cross_k;
    float* n_layer_cross_v;
    float* positional_embedding;
    float* mask;

    // Output
    float* logits;
    float* out_n_layer_self_k_cache;
    float* out_n_layer_self_v_cache;
};

class DecoderMain : public EngineWrapper {
public:
    DecoderMain() = default;
    ~DecoderMain() = default;

    int PrepareData(DecoderLoopData* data) {
        data->tokens = new int32_t[GetInputSize(0) / sizeof(int32_t)];
        data->in_n_layer_self_k_cache = new float[GetInputSize(1) / sizeof(float)];
        data->in_n_layer_self_v_cache = new float[GetInputSize(2) / sizeof(float)];
        data->n_layer_cross_k = new float[GetInputSize(3) / sizeof(float)];
        data->n_layer_cross_v = new float[GetInputSize(4) / sizeof(float)];
        data->positional_embedding = new float[GetInputSize(5) / sizeof(float)];
        data->mask = new float[GetInputSize(6) / sizeof(float)];

        data->logits = new float[GetOutputSize(0) / sizeof(float)];
        data->out_n_layer_self_k_cache = new float[GetOutputSize(1) / sizeof(float)];
        data->out_n_layer_self_v_cache = new float[GetOutputSize(2) / sizeof(float)];
        return 0;
    }

    int Run(DecoderLoopData* data) {
        int ret = 0;
        ret = SetInput((uint8_t*)data->tokens, 0);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data->in_n_layer_self_k_cache, 1);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data->in_n_layer_self_v_cache, 2);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data->n_layer_cross_k, 3);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data->n_layer_cross_v, 4);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data->positional_embedding, 5);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data->mask, 6);
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

    void DestroyData(DecoderLoopData* data) {
        delete[] data->tokens;
        delete[] data->in_n_layer_self_k_cache;
        delete[] data->in_n_layer_self_v_cache;
        delete[] data->n_layer_cross_k;
        delete[] data->n_layer_cross_v;
        delete[] data->positional_embedding;
        delete[] data->mask;

        delete[] data->logits;
        delete[] data->out_n_layer_self_k_cache;
        delete[] data->out_n_layer_self_v_cache;
    }
};