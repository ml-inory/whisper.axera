#include <stdio.h>
#include <librosa/librosa.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <fstream>
#include <ax_sys_api.h>

#include "cmdline.hpp"
#include "AudioFile.h"
#include "Encoder.hpp"
#include "DecoderMain.hpp"
#include "DecoderLoop.hpp"
#include "base64.h"

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30
#define WHISPER_N_MELS      80

#define WHISPER_SOT         50258
#define WHISPER_EOT         50257
#define WHISPER_BLANK       220
#define WHISPER_NO_TIMESTAMPS   50363
#define WHISPER_NO_SPEECH   50362
#define WHISPER_TRANSLATE   50358
#define WHISPER_TRANSCRIBE  50359
#define WHISPER_VOCAB_SIZE  51865
#define WHISPER_N_TEXT_CTX  448
#define WHISPER_N_TEXT_STATE 384

#define NEG_INF             -std::numeric_limits<float>::infinity()

static std::vector<int32_t> SOT_SEQUENCE{WHISPER_SOT,50260,WHISPER_TRANSCRIBE,WHISPER_NO_TIMESTAMPS};


static void supress_tokens(std::vector<float>& logits, bool is_initial) {
    if (is_initial) {
        logits[WHISPER_EOT] = NEG_INF;
        logits[WHISPER_BLANK] = NEG_INF;
    }

    logits[WHISPER_NO_TIMESTAMPS] = NEG_INF;
    logits[WHISPER_SOT] = NEG_INF;
    logits[WHISPER_NO_SPEECH] = NEG_INF;
    logits[WHISPER_TRANSLATE] = NEG_INF;
}

static int argmax(const std::vector<float>& logits) {
    auto max_iter = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_iter); // absolute index of max
}


int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("encoder", 'e', "encoder axmodel", true, "");
    cmd.add<std::string>("decoder_main", 'm', "decoder_main axmodel", true, "");
    cmd.add<std::string>("decoder_loop", 'l', "decoder_loop axmodel", true, "");
    cmd.add<std::string>("position_embedding", 'p', "position_embedding.bin", true, "");
    cmd.add<std::string>("token", 't', "tokens txt", true, "");
    cmd.add<std::string>("wav", 'w', "wav file", true, "");
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto encoder_file = cmd.get<std::string>("encoder");
    auto decoder_main_file = cmd.get<std::string>("decoder_main");
    auto decoder_loop_file = cmd.get<std::string>("decoder_loop");
    auto pe_file = cmd.get<std::string>("position_embedding");
    auto token_file = cmd.get<std::string>("token");
    auto wav_file = cmd.get<std::string>("wav");

    int ret = AX_SYS_Init();
    if (0 != ret) {
        fprintf(stderr, "AX_SYS_Init failed! ret = 0x%x\n", ret);
        return -1;
    }

    AX_ENGINE_NPU_ATTR_T npu_attr;
    memset(&npu_attr, 0, sizeof(npu_attr));
    npu_attr.eHardMode = static_cast<AX_ENGINE_NPU_MODE_T>(0);
    ret = AX_ENGINE_Init(&npu_attr);
    if (0 != ret) {
        fprintf(stderr, "Init ax-engine failed{0x%8x}.\n", ret);
        return -1;
    }

    printf("encoder: %s\ndecoder_main: %s\ndecoder_loop: %s\nwav_file: %s\n", encoder_file.c_str(), decoder_main_file.c_str(), decoder_loop_file.c_str(), wav_file.c_str());

    AudioFile<float> audio_file;
    if (!audio_file.load(wav_file)) {
        printf("load wav failed!\n");
        return -1;
    }

    auto& samples = audio_file.samples[0];
    int n_samples = samples.size();

    std::vector<float> positional_embedding(WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE);
    FILE* fp = fopen(pe_file.c_str(), "rb");
    fread(positional_embedding.data(), sizeof(float), WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE, fp);
    fclose(fp);

    std::vector<std::string> token_tables;
    std::ifstream ifs(token_file);
    std::string line;
    while (std::getline(ifs, line)) {
        size_t i = line.find(' ');
        token_tables.push_back(line.substr(0, i));
    }

    auto mel = librosa::Feature::melspectrogram(samples, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, "hann", true, "reflect", 2.0f, WHISPER_N_MELS, 0.0f, WHISPER_SAMPLE_RATE / 2.0f);
    int n_mel = mel.size();
    int n_len = mel[0].size();

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < WHISPER_N_MELS; i++) {
        for (int n = 0; n < n_len; n++) {
            mel[i][n] = std::log10(std::max(mel[i][n], 1e-10f));

            if (mel[i][n] > mmax) {
                mmax = mel[i][n] ;
            }
        }
    }

    for (int i = 0; i < WHISPER_N_MELS; i++) {
        for (int n = 0; n < n_len; n++) {
            mel[i][n] = (std::max(mel[i][n], (float)(mmax - 8.0)) + 4.0)/4.0;
            mel[i].resize(3000);
        }
    }

    n_len = 3000;

    // fp = fopen("../mel.bin", "rb");
    // for (size_t i = 0; i < mel.size(); i++) {
    //     fread(mel[i].data(), sizeof(float), mel[i].size(), fp);
    // }
    // fclose(fp);

    EncoderData encoder_data;
    Encoder encoder;

    DecoderMainData decoder_main_data;
    DecoderMain decoder_main;

    DecoderLoopData decoder_loop_data;
    DecoderLoop decoder_loop;

    ret = encoder.Init(encoder_file.c_str());
    if (ret) {
        printf("encoder init failed!\n");
        return ret;
    }

    ret = decoder_main.Init(decoder_main_file.c_str());
    if (ret) {
        printf("decoder_main init failed!\n");
        return ret;
    }

    ret = decoder_loop.Init(decoder_loop_file.c_str());
    if (ret) {
        printf("decoder_loop init failed!\n");
        return ret;
    }

    ret = encoder.PrepareData(encoder_data);
    if (ret) {
        printf("encoder prepare data failed!\n");
        return ret;
    }

    ret = decoder_main.PrepareData(decoder_main_data);
    if (ret) {
        printf("decoder_main prepare data failed!\n");
        return ret;
    }

    ret = decoder_loop.PrepareData(decoder_loop_data);
    if (ret) {
        printf("decoder_loop prepare data failed!\n");
        return ret;
    }

    int offset = 0;
    std::vector<float> logits(WHISPER_VOCAB_SIZE);
    int max_token_id = -1;
    std::vector<int> results;
    std::vector<int> tokens(1);
    bool is_broke = false;
    std::vector<float> n_layer_self_k_cache = decoder_loop_data.out_n_layer_self_k_cache;
    std::vector<float> n_layer_self_v_cache = decoder_loop_data.out_n_layer_self_v_cache;

    // encoder
    for (int i = 0; i < n_mel; i++) {
        memcpy(encoder_data.mel.data() + i * n_len, mel[i].data(), sizeof(float) * n_len);
    }

    ret = encoder.Run(encoder_data);
    if (ret) {
        printf("encoder run failed!\n");
        return ret;
    }

    // fp = fopen("n_layer_cross_k.bin", "wb");
    // fwrite(encoder_data.n_layer_cross_k.data(), sizeof(float), encoder_data.n_layer_cross_k.size(), fp);
    // fclose(fp);

    // fp = fopen("n_layer_cross_v.bin", "wb");
    // fwrite(encoder_data.n_layer_cross_v.data(), sizeof(float), encoder_data.n_layer_cross_v.size(), fp);
    // fclose(fp);

    // decoder_main
    decoder_main_data.tokens = SOT_SEQUENCE;
    decoder_main_data.n_layer_cross_k = encoder_data.n_layer_cross_k;
    decoder_main_data.n_layer_cross_v = encoder_data.n_layer_cross_v;

    ret = decoder_main.Run(decoder_main_data);
    if (ret) {
        printf("decoder_main run failed!\n");
        return ret;
    }

    n_layer_self_k_cache = decoder_main_data.out_n_layer_self_k_cache;
    n_layer_self_v_cache = decoder_main_data.out_n_layer_self_v_cache;

    offset += SOT_SEQUENCE.size();
    // logits = logits[0, -1]
    std::copy(decoder_main_data.logits.begin() + 3 * WHISPER_VOCAB_SIZE, decoder_main_data.logits.end(), logits.begin());
    supress_tokens(logits, true);
    max_token_id = argmax(logits);

    // fp = fopen("logits.bin", "wb");
    // fwrite(logits.data(), sizeof(float), logits.size(), fp);
    // fclose(fp);

    for (int i = 0; i < WHISPER_N_TEXT_CTX - SOT_SEQUENCE.size(); i++) {
        if (max_token_id == WHISPER_EOT) {
            is_broke = true;
            break;
        }

        results.push_back(max_token_id);
        tokens[0] = results.back();
        // mask[:model.n_text_ctx - offset[0] - 1] = -torch.inf
        std::vector<float> mask(WHISPER_N_TEXT_CTX);
        for (int n = 0; n < WHISPER_N_TEXT_CTX - offset - 1; n++) {
            mask[n] = NEG_INF;
        }
        
        // inference
        decoder_loop_data.tokens = tokens;
        decoder_loop_data.in_n_layer_self_k_cache = n_layer_self_k_cache;
        decoder_loop_data.in_n_layer_self_v_cache = n_layer_self_v_cache;
        decoder_loop_data.n_layer_cross_k = encoder_data.n_layer_cross_k;
        decoder_loop_data.n_layer_cross_v = encoder_data.n_layer_cross_v;
        // positional_embedding=positional_embedding[offset[0] : offset[0] + tokens.shape[-1]],
        std::copy(positional_embedding.begin() + offset * WHISPER_N_TEXT_STATE, \
                    positional_embedding.begin() + (offset + 1) * WHISPER_N_TEXT_STATE, \
                    decoder_loop_data.positional_embedding.begin());
        decoder_loop_data.mask = mask;

        ret = decoder_loop.Run(decoder_loop_data);
        if (ret) {
            printf("decoder_loop run failed!\n");
            return ret;
        }       

        logits = decoder_loop_data.logits;
        n_layer_self_k_cache = decoder_loop_data.out_n_layer_self_k_cache; 
        n_layer_self_v_cache = decoder_loop_data.out_n_layer_self_v_cache;             

        offset += 1;
        supress_tokens(logits, false);
        max_token_id = argmax(logits);  

        printf("Next Token: %d\n", results.back());
    }

    // fp = fopen("n_layer_cross_k.bin", "wb");
    // fwrite(encoder_data.n_layer_cross_k.data(), sizeof(float), encoder_data.n_layer_cross_k.size(), fp);
    // fclose(fp);

    // fp = fopen("n_layer_cross_v.bin", "wb");
    // fwrite(encoder_data.n_layer_cross_v.data(), sizeof(float), encoder_data.n_layer_cross_v.size(), fp);
    // fclose(fp);

    std::string s;
    for (const auto i : results) {
        char str[1024];
        base64_decode((const uint8*)token_tables[i].c_str(), (uint32)token_tables[i].size(), str);
        s += str;
    }
    printf("Result: %s\n", s.c_str());

    return 0;
}
    