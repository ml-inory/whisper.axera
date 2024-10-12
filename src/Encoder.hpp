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

#include <vector>
#include "EngineWrapper.hpp"

struct EncoderData {
    // Input
    std::vector<float> mel;
    
    // Output
    std::vector<float> n_layer_cross_k;
    std::vector<float> n_layer_cross_v;
};

class Encoder : public EngineWrapper {
public:
    Encoder() = default;
    ~Encoder() = default;

    int PrepareData(EncoderData& data) {
        data.mel.resize(GetInputSize(0) / sizeof(float));
        data.n_layer_cross_k.resize(GetOutputSize(0) / sizeof(float));
        data.n_layer_cross_v.resize(GetOutputSize(1) / sizeof(float));
        return 0;
    }

    int Run(EncoderData& data) {
        int ret = 0;
        ret = SetInput((uint8_t*)data.mel.data(), 0);
        if (ret) {
            return ret;
        }

        ret = this->RunSync();
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data.n_layer_cross_k.data(), 0);
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data.n_layer_cross_v.data(), 1);
        if (ret) {
            return ret;
        }

        return ret;
    }
};