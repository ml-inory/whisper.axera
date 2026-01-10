#include <stdio.h>
#include <vector>
#include <fstream>
#include <ax_sys_api.h>
#include <ax_engine_api.h>

#include "cmdline.hpp"
#include "utils/timer.hpp"
#include "AudioFile.h"

#ifdef __cplusplus
extern "C" {
#endif
#include "ax_whisper_api.h"
#ifdef __cplusplus
}
#endif

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("wav", 'w', "wav file", true, "");
    cmd.add<std::string>("model_type", 't', "tiny, base, small, turbo, large", false, "turbo");
#if defined(CHIP_AX650)    
    cmd.add<std::string>("model_path", 'p', "model path which contains tiny/ base/ small/ turbo/", false, "../models-ax650");
#else
    cmd.add<std::string>("model_path", 'p', "model path which contains tiny/ base/ small/ turbo/", false, "../models-ax630c");
#endif
    cmd.add<std::string>("language", 0, "en, zh", false, "zh");
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto wav_file = cmd.get<std::string>("wav");
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

    printf("wav_file: %s\n", wav_file.c_str());
    printf("model_path: %s\n", model_path.c_str());
    printf("model_type: %s\n", model_type.c_str());
    printf("language: %s\n", language.c_str());

    AudioFile<float> audio_file;
    if (!audio_file.load(wav_file)) {
        printf("load wav failed!\n");
        return -1;
    }

    auto& samples = audio_file.samples[0];
    int n_samples = samples.size();
    float duration = n_samples * 1.f / 16000;

    Timer timer;

    // Init 
    timer.start();
    AX_WHISPER_HANDLE handle = AX_WHISPER_Init(model_type.c_str(), model_path.c_str(), language.c_str());
    timer.stop();

    if (!handle) {
        printf("AX_WHISPER_Init failed!\n");
        return -1;
    }

    printf("Init whisper success, take %.4fseconds\n", timer.elapsed<std::chrono::seconds>());

    // Run
    timer.start();
    char* result;
    if (0 != AX_WHISPER_RunFile(handle, wav_file.c_str(), &result)) {
        printf("AX_WHISPER_Run failed!\n");
        AX_WHISPER_Uninit(handle);
        return -1;
    }
    timer.stop();

    printf("Result: %s\n", result);
    printf("RTF: %.4f\n", timer.elapsed<std::chrono::seconds>() / duration);

    // Uninit
    free(result);
    AX_WHISPER_Uninit(handle);

    return 0;
}
