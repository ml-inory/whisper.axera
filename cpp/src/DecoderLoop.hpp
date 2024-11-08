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
    std::vector<int32_t> tokens;
    std::vector<float> in_n_layer_self_k_cache;
    std::vector<float> in_n_layer_self_v_cache;
    std::vector<float> n_layer_cross_k;
    std::vector<float> n_layer_cross_v;
    std::vector<float> positional_embedding;
    std::vector<float> mask;

    // Output
    std::vector<float> logits;
    std::vector<float> out_n_layer_self_k_cache;
    std::vector<float> out_n_layer_self_v_cache;
};

class DecoderLoop : public EngineWrapper {
public:
    DecoderLoop() = default;
    ~DecoderLoop() = default;

    int PrepareData(DecoderLoopData& data) {
        data.tokens.resize(GetInputSize(0) / sizeof(int32_t));
        data.in_n_layer_self_k_cache.resize(GetInputSize(1) / sizeof(float));
        data.in_n_layer_self_v_cache.resize(GetInputSize(2) / sizeof(float));
        data.n_layer_cross_k.resize(GetInputSize(3) / sizeof(float));
        data.n_layer_cross_v.resize(GetInputSize(4) / sizeof(float));
        data.positional_embedding.resize(GetInputSize(5) / sizeof(float));
        data.mask.resize(GetInputSize(6) / sizeof(float));

        data.logits.resize(GetOutputSize(0) / sizeof(float));
        data.out_n_layer_self_k_cache.resize(GetOutputSize(1) / sizeof(float));
        data.out_n_layer_self_v_cache.resize(GetOutputSize(2) / sizeof(float));
        return 0;
    }

    int Run(DecoderLoopData& data) {
        int ret = 0;
        ret = SetInput((uint8_t*)data.tokens.data(), 0);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data.in_n_layer_self_k_cache.data(), 1);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data.in_n_layer_self_v_cache.data(), 2);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data.n_layer_cross_k.data(), 3);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data.n_layer_cross_v.data(), 4);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data.positional_embedding.data(), 5);
        if (ret) {
            return ret;
        }

        ret = SetInput((uint8_t*)data.mask.data(), 6);
        if (ret) {
            return ret;
        }

        ret = this->RunSync();
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data.logits.data(), 0);
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data.out_n_layer_self_k_cache.data(), 1);
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data.out_n_layer_self_v_cache.data(), 2);
        if (ret) {
            return ret;
        }

        return ret;
    }
};