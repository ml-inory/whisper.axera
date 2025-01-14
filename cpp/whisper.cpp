#include <stdio.h>
#include <librosa/librosa.h>
#include <vector>
#include <limits>
#include <algorithm>
#include <fstream>
#include <ax_sys_api.h>
#include <unordered_map>
#include <ctime>
#include <sys/time.h>

#include "cmdline.hpp"
#include "AudioFile.h"
#include "Encoder.hpp"
#include "DecoderMain.hpp"
#include "DecoderLoop.hpp"
#include "base64.h"
#include "opencc.h"
#include "utilities/file.hpp"

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
#define NEG_INF             -std::numeric_limits<float>::infinity()

static std::vector<int> WHISPER_LANG_CODES{
    50273,50303,50288,50261,50342,50299,50330,50302,50336,50267,50287,50292,50294,50323,50348,50291,50317,
    50326,50289,50356,50290,50282,50347,50331,50354,50264,50333,50296,50339,50318,50305,50293,50280,50322,
    50312,50306,50353,50285,50275,50340,50278,50268,50337,50316,50266,50307,50310,50338,50334,50313,50351,
    50260,50344,50283,50327,50272,50324,50276,50281,50301,50332,50300,50309,50343,50349,50335,50320,50259,
    50284,50304,50277,50311,50319,50314,50352,50328,50286,50274,50329,50270,50269,50350,50263,50345,50298,
    50279,50297,50262,50315,50321,50308,50355,50265,50346,50295,50271,50357,50341,50325
};

static std::vector<std::string> WHISPER_LANG_NAMES{
    "sv","sr","no","de","nn","te", "be","bn","lo","pt","ta","bg","la","km","tl","hr","sq","so","th","jw","ur","ms","bo",
    "tg","ha","ko","gu","ml","ht", "sw","sl","lt","uk","si","hy","kn","ln","da","id","ps","vi","tr","uz","kk","ja","et",
    "eu","fo","am","ne","tt","zh", "sa","cs","af","ar","sn","hi","el","lv","sd","fa","br","mt","mg","yi","mr","en","ro",
    "az","fi","is","gl","mn","haw","oc","hu","it","ka","ca","pl","as","ru","lb","sk","he","cy","es","bs","pa","mk","ba",
    "fr","my","mi","nl","su","tk", "yo"
};

static std::unordered_map<std::string, int> WHISPER_N_TEXT_STATE_MAP{
    {"tiny",    384},
    {"small",   768}
};

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

static int detect_language(const std::string& language) {
    int i = 51; // zh
    for (int n = 0; n < WHISPER_LANG_CODES.size(); n++) {
        if (language == WHISPER_LANG_NAMES[n]) {
            i = n;
            break;
        }
    }
    
    return WHISPER_LANG_CODES[i];
}

static double get_current_time()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);

    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

static bool load_models(const std::string& model_path, const std::string& model_type, 
            Encoder& encoder, DecoderMain& decoder_main, DecoderLoop& decoder_loop, std::vector<float>& positional_embedding, std::vector<std::string>& token_tables) {
    std::string model_path_ = model_path;
    if (model_path_[model_path_.size() - 1] != '/') {
        model_path_ += "/";
    }
    std::string encoder_path = model_path_ + model_type + "-encoder.axmodel";
    std::string decoder_main_path = model_path_ + model_type + "-decoder-main.axmodel";
    std::string decoder_loop_path = model_path_ + model_type + "-decoder-loop.axmodel";
    std::string pe_path = model_path_ + model_type + "-positional_embedding.bin";
    std::string token_path = model_path_ + model_type + "-tokens.txt";
    double start, end;

    int ret = 0;
    int WHISPER_N_TEXT_STATE = WHISPER_N_TEXT_STATE_MAP[model_type];

    if (!utilities::exists(encoder_path)) {
        printf("model (%s) NOT exist!\n", encoder_path.c_str());
        return false;
    }
    if (!utilities::exists(decoder_main_path)) {
        printf("model (%s) NOT exist!\n", decoder_main_path.c_str());
        return false;
    }
    if (!utilities::exists(decoder_loop_path)) {
        printf("model (%s) NOT exist!\n", decoder_loop_path.c_str());
        return false;
    }
    if (!utilities::exists(pe_path)) {
        printf("positional_embedding (%s) NOT exist!\n", pe_path.c_str());
        return false;
    }
    if (!utilities::exists(token_path)) {
        printf("token (%s) NOT exist!\n", token_path.c_str());
        return false;
    }

    start = get_current_time();
    ret = encoder.Init(encoder_path.c_str());
    if (ret) {
        printf("encoder init failed!\n");
        return false;
    }
    end = get_current_time();
    printf("Load encoder take %.2fms\n", end - start);

    start = get_current_time();
    ret = decoder_main.Init(decoder_main_path.c_str());
    if (ret) {
        printf("decoder_main init failed!\n");
        return false;
    }
    end = get_current_time();
    printf("Load decoder_main take %.2fms\n", end - start);

    start = get_current_time();
    ret = decoder_loop.Init(decoder_loop_path.c_str());
    if (ret) {
        printf("decoder_loop init failed!\n");
        return false;
    }
    end = get_current_time();
    printf("Load decoder_loop take %.2fms\n", end - start);

    positional_embedding.resize(WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE);
    FILE* fp = fopen(pe_path.c_str(), "rb");
    if (!fp) {
        fprintf(stderr, "Can NOT open %s\n", pe_path.c_str());
        return false;
    }
    fread(positional_embedding.data(), sizeof(float), WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE, fp);
    fclose(fp);

    std::ifstream ifs(token_path);
    if (!ifs.is_open()) {
        fprintf(stderr, "Can NOT open %s\n", token_path.c_str());
        return false;
    }
    std::string line;
    while (std::getline(ifs, line)) {
        size_t i = line.find(' ');
        token_tables.push_back(line.substr(0, i));
    }

    return true;
}

int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("wav", 'w', "wav file", true, "");
    cmd.add<std::string>("model_type", 0, "tiny, small, large", false, "small");
    cmd.add<std::string>("model_path", 'p', "model path for *.axmodel, tokens.txt, positional_embedding.bin", false, "../models");
    cmd.add<std::string>("language", 0, "en, zh", false, "zh");
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto wav_file = cmd.get<std::string>("wav");
    auto model_path = cmd.get<std::string>("model_path");
    auto model_type = cmd.get<std::string>("model_type");
    auto language = cmd.get<std::string>("language");

    if (WHISPER_N_TEXT_STATE_MAP.find(model_type) == WHISPER_N_TEXT_STATE_MAP.end()) {
        fprintf(stderr, "Can NOT find n_text_state for model_type: %s\n", model_type.c_str());
        return -1;
    }

    int WHISPER_N_TEXT_STATE = WHISPER_N_TEXT_STATE_MAP[model_type];

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

    // Load models
    Encoder encoder;
    DecoderMain decoder_main;
    DecoderLoop decoder_loop;
    std::vector<float> positional_embedding;
    std::vector<std::string> token_tables;

    double start_load, end_load;
    start_load = get_current_time();
    if (!load_models(model_path, model_type, encoder, decoder_main, decoder_loop, positional_embedding, token_tables)) {
        printf("load models failed!\n");
        return -1;
    }
    end_load = get_current_time();
    printf("load models take %.2f ms\n", end_load - start_load);

    AudioFile<float> audio_file;
    if (!audio_file.load(wav_file)) {
        printf("load wav failed!\n");
        return -1;
    }

    auto& samples = audio_file.samples[0];
    int n_samples = samples.size();

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

    n_len = mel[0].size();

    // fp = fopen("../mel.bin", "rb");
    // for (size_t i = 0; i < mel.size(); i++) {
    //     fread(mel[i].data(), sizeof(float), mel[i].size(), fp);
    // }
    // fclose(fp);

    
    double start, end;
    double start_all, end_all;

    int offset = 0;
    std::vector<float> logits(WHISPER_VOCAB_SIZE);
    int max_token_id = -1;
    std::vector<int> results;
    std::vector<int> tokens(1);
    bool is_broke = false;
    
    std::vector<float> n_layer_cross_k(encoder.GetOutputSize(0) / sizeof(float));
    std::vector<float> n_layer_cross_v(encoder.GetOutputSize(1) / sizeof(float));

    std::vector<float> decoder_main_logits(4 * WHISPER_VOCAB_SIZE);
    std::vector<float> n_layer_self_k_cache(decoder_main.GetOutputSize(1) / sizeof(float));
    std::vector<float> n_layer_self_v_cache(decoder_main.GetOutputSize(2) / sizeof(float));

    // encoder
    std::vector<float> continous_mel(WHISPER_N_MELS * n_len);
    for (int i = 0; i < n_mel; i++) {
        memcpy(continous_mel.data() + i * n_len, mel[i].data(), sizeof(float) * n_len);
    }

    // fp = fopen("mel.bin", "wb");
    // fwrite(continous_mel.data(), sizeof(float), continous_mel.size(), fp);
    // fclose(fp);

    start = get_current_time();
    start_all = get_current_time();
    encoder.SetInput(continous_mel.data(), 0);
    ret = encoder.Run();
    if (ret) {
        printf("encoder run failed!\n");
        return ret;
    }
	end = get_current_time();
    printf("Encoder run take %.2f ms\n", (end - start));
    // encoder.GetOutput(n_layer_cross_k.data(), 0);
    // encoder.GetOutput(n_layer_cross_v.data(), 1);

    // fp = fopen("n_layer_cross_k.bin", "wb");
    // fwrite(n_layer_cross_k.data(), sizeof(float), n_layer_cross_k.size(), fp);
    // fclose(fp);

    // fp = fopen("n_layer_cross_v.bin", "wb");
    // fwrite(n_layer_cross_v.data(), sizeof(float), n_layer_cross_v.size(), fp);
    // fclose(fp);

    // detect language
    SOT_SEQUENCE[1] = detect_language(language);

    // decoder_main
    start = get_current_time();
    decoder_main.SetInput(SOT_SEQUENCE.data(), 0);
    decoder_main.SetInput(encoder.GetOutputPtr(0), 1);
    decoder_main.SetInput(encoder.GetOutputPtr(1), 2);
    ret = decoder_main.Run();
    if (ret) {
        printf("decoder_main run failed!\n");
        return ret;
    }
    decoder_main.GetOutput(decoder_main_logits.data(), 0);
    // decoder_main.GetOutput(n_layer_self_k_cache.data(), 1);
    // decoder_main.GetOutput(n_layer_self_v_cache.data(), 2);
    end = get_current_time();

    offset += SOT_SEQUENCE.size();
    // logits = logits[0, -1]
    std::copy(decoder_main_logits.begin() + 3 * WHISPER_VOCAB_SIZE, decoder_main_logits.end(), logits.begin());
    supress_tokens(logits, true);
    max_token_id = argmax(logits);

    // fp = fopen("logits.bin", "wb");
    // fwrite(logits.data(), sizeof(float),logits.size(), fp);
    // fclose(fp);

    printf("First token: %d \t take %.2fms\n", max_token_id, (end - start));

    std::vector<float> mask(WHISPER_N_TEXT_CTX);
    for (int n = 0; n < WHISPER_N_TEXT_CTX - offset - 1; n++) {
        mask[n] = NEG_INF;
    }

    // fp = fopen("logits.bin", "wb");
    // fwrite(logits.data(), sizeof(float), logits.size(), fp);
    // fclose(fp);
    decoder_loop.SetInput(decoder_main.GetOutputPtr(1), 1);
    decoder_loop.SetInput(decoder_main.GetOutputPtr(2), 2);
    decoder_loop.SetInput(encoder.GetOutputPtr(0), 3);
    decoder_loop.SetInput(encoder.GetOutputPtr(1), 4);

    for (int i = 0; i < WHISPER_N_TEXT_CTX - SOT_SEQUENCE.size(); i++) {
        if (max_token_id == WHISPER_EOT) {
            is_broke = true;
            break;
        }

        results.push_back(max_token_id);
        tokens[0] = results.back();
        // mask[:model.n_text_ctx - offset[0] - 1] = -torch.inf
       
        // inference
        start = get_current_time();
        decoder_loop.SetInput(tokens.data(), 0);
        // decoder_loop.SetInput(n_layer_cross_k.data(), 3);
        // decoder_loop.SetInput(n_layer_cross_v.data(), 4);
        decoder_loop.SetInput(positional_embedding.data() + offset * WHISPER_N_TEXT_STATE, 5);
        decoder_loop.SetInput(mask.data(), 6);

        // start = clock();
        ret = decoder_loop.Run();
        if (ret) {
            printf("decoder_loop run failed!\n");
            return ret;
        } 

        decoder_loop.SetInput(decoder_loop.GetOutputPtr(1), 1);
        decoder_loop.SetInput(decoder_loop.GetOutputPtr(2), 2);
        decoder_loop.GetOutput(logits.data(), 0);

        offset += 1;
        mask[WHISPER_N_TEXT_CTX - offset - 1] = 0;

        supress_tokens(logits, false);
        max_token_id = argmax(logits);  
        end = get_current_time();  

        printf("Next Token: %d \t take %.2fms\n", max_token_id, (end - start));
    }
	
    end_all = get_current_time(); 
    printf("All take %.2f ms\n", (end_all - start_all));
	

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

    if (language == "en")
        printf("Result: %s\n", s.c_str());
    else {
        const opencc::SimpleConverter converter("t2s.json");
        std::string simple_str = converter.Convert(s);
        printf("Result: %s\n", simple_str.c_str());
    }

    return 0;
}
