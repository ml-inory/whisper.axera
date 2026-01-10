#include <stdio.h>
#include <vector>
#include <fstream>
#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include "cmdline.hpp"
#include "WhisperHTTPServer.hpp"

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<int>("port", 0, "http port", false, 8080);
    cmd.add<std::string>("model_type", 't', "tiny, base, small, turbo, large", false, "turbo");
#if defined(CHIP_AX650)    
    cmd.add<std::string>("model_path", 'p', "model path which contains tiny/ base/ small/ turbo/", false, "../models-ax650");
#else
    cmd.add<std::string>("model_path", 'p', "model path which contains tiny/ base/ small/ turbo/", false, "../models-ax630c");
#endif
    cmd.add<std::string>("language", 'l', "en, zh", false, "zh");
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto port = cmd.get<int>("port");
    auto model_path = cmd.get<std::string>("model_path");
    auto model_type = cmd.get<std::string>("model_type");
    auto language = cmd.get<std::string>("language");

    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

#if defined(CHIP_AX650)
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }
#else
    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = AX_ENGINE_VIRTUAL_NPU_DISABLE;
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }
#endif    

    printf("port: %d\n", port);
    printf("model_path: %s\n", model_path.c_str());
    printf("model_type: %s\n", model_type.c_str());
    printf("language: %s\n", language.c_str());

    WhisperHTTPServer server;
    ALOGI("Initializing server...");
    if (!server.init(model_path, model_type, language)) {
        printf("init server failed!\n");
        return -1;
    }
    ALOGI("Init server success");

    server.start(port);

    return 0;
}
