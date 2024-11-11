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

class DecoderLoop : public EngineWrapper {
public:
    DecoderLoop() = default;
    ~DecoderLoop() = default;

    // int Run() {
    //     int ret = 0;
    //     ret = this->Run();
    //     if (ret) {
    //         return ret;
    //     }

    //     // ret = SetInput(GetOutputPtr(1), 1);
    //     // if (ret) {
    //     //     return ret;
    //     // }

    //     // ret = SetInput(GetOutputPtr(2), 2);
    //     // if (ret) {
    //     //     return ret;
    //     // }

    //     return ret;
    // }
};