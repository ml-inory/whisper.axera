#include <stdio.h>
#include <vector>
#include <fstream>
#include <ax_sys_api.h>
#include <ctime>
#include <sys/time.h>

#include "cmdline.hpp"
#include "AudioFile.h"
#include "Whisper.hpp"
#include "utils/timer.hpp"


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

    // convert to mono
    if (audio_file.isStereo()) {
        for (int i = 0; i < n_samples; i++) {
            samples[i] = (samples[i] + audio_file.samples[1][i]) / 2;
        }
    }

    Whisper whisper(model_type, language);
    if (!whisper.load_models(model_path)) {
        printf("load models failed!\n");
        return -1;
    }

    printf("Init whisper success\n");

    timer.start();
    std::string result;
    if (!whisper.run(samples, result)) {
        printf("run whisper failed!\n");
        return -1;
    }
    timer.stop();

    printf("Result: %s\n", result.c_str());
    printf("RTF: %.4f\n", timer.elapsed<std::chrono::seconds>() / duration);
    return 0;
}
