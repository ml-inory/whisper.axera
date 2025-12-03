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
    m_decoder_main.Release();
    m_decoder_loop.Release();
}

bool Whisper::init(const std::string& model_type, const std::string& language) {
    m_model_type = model_type;
    m_lang = language;
    
    return true;
}

bool Whisper::load_models(const std::string& model_root) {
    std::string encoder_path = model_root + "/" + m_model_type + "/" + m_model_type + "-encoder.axmodel";
    std::string decoder_main_path = model_root + "/" + m_model_type + "/" + m_model_type + "-decoder-main.axmodel";
    std::string decoder_loop_path = model_root + "/" + m_model_type + "/" + m_model_type + "-decoder-loop.axmodel";
    std::string pe_path = model_root + "/" + m_model_type + "/" + m_model_type + "-positional_embedding.bin";
    std::string token_path = model_root + "/" + m_model_type + "/" + m_model_type + "-tokens.txt";
    std::string config_path = model_root + "/" + m_model_type + "/" + m_model_type + "_config.json";

    int ret = 0;
    std::ifstream fs(config_path);
    m_config = json::parse(fs);
    m_config["all_language_tokens"] = json(stringToVector<int>(m_config["all_language_tokens"]));
    m_config["all_language_codes"] = json(stringToVector<std::string>(m_config["all_language_codes"]));
    fs.close();

    int WHISPER_N_TEXT_STATE = m_config["n_text_state"];
    int WHISPER_N_TEXT_CTX = m_config["n_text_ctx"];

    ret = m_encoder.Init(encoder_path.c_str());
    if (ret) {
        ALOGE("encoder init failed! ret=0x%x\n", ret);
        return false;
    }

    ret = m_decoder_main.Init(decoder_main_path.c_str());
    if (ret) {
        ALOGE("decoder_main init failed! ret=0x%x\n", ret);
        return false;
    }

    ret = m_decoder_loop.Init(decoder_loop_path.c_str());
    if (ret) {
        ALOGE("decoder_loop init failed! ret=0x%x\n", ret);
        return false;
    }

    m_pe.resize(WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE);
    FILE* fp = fopen(pe_path.c_str(), "rb");
    if (!fp) {
        ALOGE("Can NOT open %s\n", pe_path.c_str());
        return false;
    }
    fread(m_pe.data(), sizeof(float), WHISPER_N_TEXT_CTX * WHISPER_N_TEXT_STATE, fp);
    fclose(fp);

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
        
        m_feature.mel.resize(n_mels * 3000); // 30s mel band
        m_feature.sot_seq = {sot_token, lang_token, task_token, no_ts_token};
        m_feature.n_layer_cross_k.resize(m_encoder.GetOutputSize(0) / sizeof(float));
        m_feature.n_layer_cross_v.resize(m_encoder.GetOutputSize(1) / sizeof(float));
        m_feature.decoder_main_logits.resize(m_feature.sot_seq.size() * vocab_size);
        m_feature.n_layer_self_k_cache.resize(m_decoder_main.GetOutputSize(1) / sizeof(float));
        m_feature.n_layer_self_v_cache.resize(m_decoder_main.GetOutputSize(2) / sizeof(float));
        m_feature.mask.resize(WHISPER_N_TEXT_CTX);
        m_feature.logits.resize(vocab_size);
    }
    
    return true;
}

std::vector<float> Whisper::preprocess(std::vector<float>& audio_data) {
    int n_samples = audio_data.size();
    int WHISPER_N_MELS = m_config["n_mels"];
    auto mel = librosa::Feature::melspectrogram(audio_data, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, "hann", true, "reflect", 2.0f, WHISPER_N_MELS, 0.0f, WHISPER_SAMPLE_RATE / 2.0f);
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

    std::vector<float> continous_mel(WHISPER_N_MELS * n_len);
    for (int i = 0; i < n_mel; i++) {
        memcpy(continous_mel.data() + i * n_len, mel[i].data(), sizeof(float) * n_len);
    }

    return continous_mel;
}

bool Whisper::run(std::vector<float>& audio_data, std::string& result) {
    std::vector<float> mel = preprocess(audio_data);

    m_encoder.SetInput(mel.data(), 0);
    int ret = m_encoder.Run();
    if (ret) {
        ALOGE("encoder run failed! ret=0x%x", ret);
        return false;
    }

    m_decoder_main.SetInput(m_feature.sot_seq.data(), 0);
    m_decoder_main.SetInput(m_encoder.GetOutputPtr(0), 1);
    m_decoder_main.SetInput(m_encoder.GetOutputPtr(1), 2);
    ret = m_decoder_main.Run();
    if (ret) {
        ALOGE("decoder_main run failed! ret=0x%x", ret);
        return false;
    }
    m_decoder_main.GetOutput(m_feature.decoder_main_logits.data(), 0);

    int offset = m_feature.sot_seq.size();
    int vocab_size = m_config["n_vocab"];
    int max_token_id;

    // logits = logits[0, -1]
    std::copy(m_feature.decoder_main_logits.begin() + (m_feature.sot_seq.size() - 1) * vocab_size,
         m_feature.decoder_main_logits.end(), m_feature.logits.begin());
    supress_tokens(m_feature.logits, true);
    max_token_id = argmax(m_feature.logits);

    int WHISPER_N_TEXT_CTX = m_config["n_text_ctx"];
    m_feature.mask.assign(WHISPER_N_TEXT_CTX - offset - 1, NEG_INF);

    m_decoder_loop.SetInput(m_decoder_main.GetOutputPtr(1), 1);
    m_decoder_loop.SetInput(m_decoder_main.GetOutputPtr(2), 2);
    m_decoder_loop.SetInput(m_encoder.GetOutputPtr(0), 3);
    m_decoder_loop.SetInput(m_encoder.GetOutputPtr(1), 4);

    bool is_broke = false;
    int WHISPER_EOT = m_config["eot"];
    int WHISPER_N_TEXT_STATE = m_config["n_text_state"];
    std::vector<int> token_list;
    int latest_token;
    for (int i = 0; i < WHISPER_N_TEXT_CTX - m_feature.sot_seq.size(); i++) {
        if (max_token_id == WHISPER_EOT) {
            is_broke = true;
            break;
        }

        token_list.push_back(max_token_id);
        latest_token = token_list.back();
        // mask[:model.n_text_ctx - offset[0] - 1] = -torch.inf
       
        // inference
        m_decoder_loop.SetInput(&latest_token, 0);
        // decoder_loop.SetInput(n_layer_cross_k.data(), 3);
        // decoder_loop.SetInput(n_layer_cross_v.data(), 4);
        m_decoder_loop.SetInput(m_pe.data() + offset * WHISPER_N_TEXT_STATE, 5);
        m_decoder_loop.SetInput(m_feature.mask.data(), 6);

        ret = m_decoder_loop.Run();
        if (ret) {
            ALOGE("decoder_loop run failed! ret=0x%x", ret);
            return false;
        } 

        m_decoder_loop.SetInput(m_decoder_loop.GetOutputPtr(1), 1);
        m_decoder_loop.SetInput(m_decoder_loop.GetOutputPtr(2), 2);
        m_decoder_loop.GetOutput(m_feature.logits.data(), 0);

        offset += 1;
        m_feature.mask[WHISPER_N_TEXT_CTX - offset - 1] = 0;

        supress_tokens(m_feature.logits, false);
        max_token_id = argmax(m_feature.logits);  
    }

    std::string s;
    for (const auto i : token_list) {
        char str[1024];
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

void Whisper::supress_tokens(std::vector<float>& logits, bool is_initial) {
    int eot_token = m_config["eot"];
    int blank_token = m_config["blank_id"];
    int no_ts_token = m_config["no_timestamps"];
    int sot_token = m_config["sot"];
    int no_speech_token = m_config["no_speech"];
    int translate_token = m_config["translate"];

    if (is_initial) {
        logits[eot_token] = NEG_INF;
        logits[blank_token] = NEG_INF;
    }

    logits[no_ts_token] = NEG_INF;
    logits[sot_token] = NEG_INF;
    logits[no_speech_token] = NEG_INF;
    logits[translate_token] = NEG_INF;
}