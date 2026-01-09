#include "Whisper.hpp"

#include "utils/logger.h"
#include <librosa/librosa.h>
#include "base64.h"
#include "opencc.h"

#include <fstream>
#include <limits>
#include <vector>
#include <string>
#include <sstream>
#include <type_traits>
#include <stdexcept>

#include "ax_dmadim_api.h"

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30

#define NEG_INF             -std::numeric_limits<float>::infinity()

static int argmax(const std::vector<float>& logits) {
    auto max_iter = std::max_element(logits.begin(), logits.end());
    return std::distance(logits.begin(), max_iter); // absolute index of max
}

template<typename T>
std::vector<T> stringToVector(const std::string& str) {
    std::vector<T> result;
    std::istringstream ss(str);
    std::string token;
    
    while (std::getline(ss, token, ',')) {
        std::istringstream tokenStream(token);
        T value;
        
        // 尝试将字符串转换为类型T
        if (!(tokenStream >> value)) {
            throw std::runtime_error("Failed to convert token: " + token);
        }
        
        result.push_back(value);
    }
    
    return result;
}

Whisper::Whisper(const std::string& model_type, const std::string& language):
    m_model_type(model_type),
    m_lang(language) {

}

Whisper::~Whisper() {
    m_encoder.Release();
    m_decoder.Release();
}

bool Whisper::init(const std::string& model_type, const std::string& language) {
    m_model_type = model_type;
    m_lang = language;
    
    return true;
}

bool Whisper::load_models(const std::string& model_root) {
    std::string encoder_path = model_root + "/" + m_model_type + "/" + m_model_type + "-encoder.axmodel";
    std::string decoder_path = model_root + "/" + m_model_type + "/" + m_model_type + "-decoder.axmodel";
    std::string token_path = model_root + "/" + m_model_type + "/" + m_model_type + "-tokens.txt";
    std::string config_path = model_root + "/" + m_model_type + "/" + m_model_type + "_config.json";

    int ret = 0;
    std::ifstream fs(config_path);
    m_config = json::parse(fs);
    m_config["all_language_tokens"] = json(stringToVector<int>(m_config["all_language_tokens"]));
    m_config["all_language_codes"] = json(stringToVector<std::string>(m_config["all_language_codes"]));
    fs.close();

    ret = m_encoder.Init(encoder_path.c_str());
    if (ret) {
        ALOGE("encoder init failed! ret=0x%x\n", ret);
        return false;
    }

    ret = m_decoder.Init(decoder_path.c_str());
    if (ret) {
        ALOGE("decoder init failed! ret=0x%x\n", ret);
        return false;
    }

    {
        m_token_tables.clear();
        fs.open(token_path);
        if (!fs.is_open()) {
            ALOGE("Can NOT open %s", token_path.c_str());
            return false;
        }
        std::string line;
        while (std::getline(fs, line)) {
            size_t i = line.find(' ');
            m_token_tables.push_back(line.substr(0, i));
        }
    }

    {
        int n_mels = m_config["n_mels"];
        int sot_token = m_config["sot"];
        int lang_token = get_lang_token(m_lang);
        int task_token = m_config["transcribe"];
        int no_ts_token = m_config["no_timestamps"];
        int vocab_size = m_config["n_vocab"];
        int n_text_state = m_config["n_text_state"];
        int n_text_ctx = m_config["n_text_ctx"];
        int n_text_layer = m_config["n_text_layer"];
        
        m_feature.mel.resize(n_mels * 3000); // 30s mel band
        m_feature.sot_seq = {sot_token, lang_token, task_token, no_ts_token};
        m_feature.mask.resize(n_text_ctx);
        m_feature.self_k_cache.resize(n_text_layer * n_text_ctx * n_text_state);
        m_feature.self_v_cache.resize(n_text_layer * n_text_ctx * n_text_state);
        m_feature.this_self_kv.resize(n_text_state);
        m_feature.logits.resize(vocab_size);
    }
    
    return true;
}

std::vector<float> Whisper::preprocess(std::vector<float>& audio_data, int n_mels) {
    int n_samples = audio_data.size();
    auto mel = librosa::Feature::melspectrogram(audio_data, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, "hann", true, "reflect", 2.0f, n_mels, 0.0f, WHISPER_SAMPLE_RATE / 2.0f);
    int n_mel = mel.size();
    int n_len = mel[0].size();

    // clamping and normalization
    double mmax = -1e20;
    for (int i = 0; i < n_mels; i++) {
        for (int n = 0; n < n_len; n++) {
            mel[i][n] = std::log10(std::max(mel[i][n], 1e-10f));

            if (mel[i][n] > mmax) {
                mmax = mel[i][n] ;
            }
        }
    }

    for (int i = 0; i < n_mels; i++) {
        for (int n = 0; n < n_len; n++) {
            mel[i][n] = (std::max(mel[i][n], (float)(mmax - 8.0)) + 4.0)/4.0;
            mel[i].resize(3000);
        }
    }

    n_len = mel[0].size();

    std::vector<float> continous_mel(n_mels * n_len);
    for (int i = 0; i < n_mel; i++) {
        memcpy(continous_mel.data() + i * n_len, mel[i].data(), sizeof(float) * n_len);
    }

    return continous_mel;
}

bool Whisper::run(std::vector<float>& audio_data, std::string& result) {
    std::vector<float> mel = preprocess(audio_data, m_config["n_mels"]);

    m_encoder.SetInput(mel.data(), 0);
    int ret = m_encoder.Run();
    if (ret) {
        ALOGE("encoder run failed! ret=0x%x", ret);
        return false;
    }

    // init mask
    std::fill(m_feature.mask.begin(), m_feature.mask.end(), 1);

    const int n_text_ctx = m_config["n_text_ctx"];
    const int WHISPER_EOT = m_config["eot"];

    int offset = 0;
    int idx = 0;
    std::vector<int> ans;

    for (auto token : m_feature.sot_seq) {
        idx = run_decoder(token, offset++);
        printf("token: %d\n", idx);
    }

    printf("eot: %d\n", WHISPER_EOT);
    printf("token: %d\n", idx);
    printf("offset: %d\n", offset);

    while (idx != WHISPER_EOT && offset < n_text_ctx) {
        ans.emplace_back(idx);
        printf("token: %d\n", idx);
        idx = run_decoder(idx, offset++);
    }

    std::string s;
    for (const auto i : ans) {
        char str[32];
        base64_decode((const uint8*)m_token_tables[i].c_str(), (uint32)m_token_tables[i].size(), str);
        s += str;
    }

    if (m_lang == "zh") {
        const opencc::SimpleConverter converter("t2s.json");
        result = converter.Convert(s);
    } else {
        result = s;
    }

    return true;
}

int Whisper::get_lang_token(const std::string& lang) {
    auto all_language_codes = m_config["all_language_codes"].get<std::vector<std::string>>();
    auto iter = std::find(all_language_codes.begin(), all_language_codes.end(), lang);
    if (iter == all_language_codes.end()) {
        // lang not found, use default lang
        iter = std::find(all_language_codes.begin(), all_language_codes.end(), DEFAULT_LANG);
        m_lang = DEFAULT_LANG;
    }

    return m_config["all_language_tokens"][iter - all_language_codes.begin()];
}

void Whisper::causal_mask_1d(int n) {
    if (n > 0) {
        // std::fill(m_feature.mask.begin(), m_feature.mask.begin() + n, 0);
        m_feature.mask[n - 1] = 0;
    }
}

void Whisper::dma_cross_kv() {
    // encoder output:
    // cross_k_0, cross_v_0, cross_k_1, cross_v_1, ...

    // decoder input:
    // tokens, self_k_0, self_v_0, ..., cross_k_0, cross_v_0, ..., offset, mask

    const int n_text_layer = m_config["n_text_layer"];
    int cross_kv_num = 2 * n_text_layer;
    int decoder_start_index = 1 + 2 * n_text_layer;

    for (int i = 0; i < cross_kv_num; i++) {
        AX_U64 phySrc = m_encoder.GetOutputPhyAddr(i);
        AX_U64 phyDst = m_decoder.GetInputPhyAddr(decoder_start_index + i);
        void* virSrc = m_encoder.GetOutputVirtAddr(i);
        void* virDst = m_decoder.GetInputVirtAddr(decoder_start_index + i);
        int size = m_encoder.GetOutputSize(i);
        // AX_DMA_MemCopy(phyDst, phySrc, (AX_U64)size);
        // memcpy(virDst, virSrc, size);
        m_decoder.SetInput(virSrc, decoder_start_index + i);
    }
}

int Whisper::run_decoder(int token, int offset) {
    // decoder input
    // token + self_kv + cross_kv + offset + mask

    // decoder output
    // logits + this_self_kv
    const int n_text_layer = m_config["n_text_layer"];
    const int n_text_ctx = m_config["n_text_ctx"];
    const int n_text_state = m_config["n_text_state"];
    const int self_kv_index = 1;
    const int offset_index = 1 + 2 * n_text_layer + 2 * n_text_layer;
    const int mask_index = offset_index + 1;
    const int this_kv_index = 1;

    causal_mask_1d(offset);

    m_decoder.SetInput(&token, 0);
    for (int i = 0; i < n_text_layer; i++) {
        m_decoder.SetInput(m_feature.self_k_cache.data() + i * n_text_ctx * n_text_state, 
            self_kv_index + i * 2);
        m_decoder.SetInput(m_feature.self_v_cache.data() + i * n_text_ctx * n_text_state, 
            self_kv_index + i * 2 + 1);
    }
    dma_cross_kv();
    m_decoder.SetInput(&offset, offset_index);
    m_decoder.SetInput(m_feature.mask.data(), mask_index);

    int ret = m_decoder.Run();
    if (ret) {
        ALOGE("decoder run failed! ret=0x%x", ret);
        return false;
    }

    for (int i = 0; i < n_text_layer; i++) {
        // k_cache
        m_decoder.GetOutput(m_feature.this_self_kv.data(), this_kv_index + i * 2);
        memcpy(m_feature.self_k_cache.data() + i * n_text_ctx * n_text_state + offset * n_text_state,
            m_feature.this_self_kv.data(), sizeof(float) * n_text_state);

        // v_cache
        m_decoder.GetOutput(m_feature.this_self_kv.data(), this_kv_index + i * 2 + 1);
        memcpy(m_feature.self_v_cache.data() + i * n_text_ctx * n_text_state + offset * n_text_state,
            m_feature.this_self_kv.data(), sizeof(float) * n_text_state);
    }

    m_decoder.GetOutput(m_feature.logits.data(), 0);
    return argmax(m_feature.logits);
}