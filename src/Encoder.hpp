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

struct EncoderData {
    // Input
    float* mel;
    
    // Output
    float* n_layer_cross_k;
    float* n_layer_cross_v;
};

class Encoder : public EngineWrapper {
public:
    Encoder() = default;
    ~Encoder() = default;

    int PrepareData(EncoderData* data) {
        data->mel = new float[GetInputSize(0) / sizeof(float)];
        data->n_layer_cross_k = new float[GetOutputSize(0) / sizeof(float)];
        data->n_layer_cross_v = new float[GetOutputSize(1) / sizeof(float)];
        return 0;
    }

    int Run(EncoderData* data) {
        int ret = 0;
        ret = SetInput((uint8_t*)data->mel, 0);
        if (ret) {
            return ret;
        }

        ret = this->RunSync();
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data->n_layer_cross_k, 0);
        if (ret) {
            return ret;
        }

        ret = GetOutput((uint8_t*)data->n_layer_cross_v, 1);
        if (ret) {
            return ret;
        }

        return ret;
    }

    void DestroyData(EncoderData* data) {
        delete[] data->mel;
        delete[] data->n_layer_cross_k;
        delete[] data->n_layer_cross_v;
    }
};