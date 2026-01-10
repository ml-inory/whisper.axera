#pragma once

// #include "EngineWrapper.hpp"
#include "utils/nlohmann/json.hpp"
#include "ax_model_runner/ax_model_runner.hpp"

#include <array>

// for convenience
using json = nlohmann::json;

#define DEFAULT_LANG    "zh"


typedef struct _WhisperFeature {
    std::array<int, 4> sot_seq;

    std::vector<int> mask;              // [n_text_ctx,]
    std::vector<float> self_k_cache;    // [n_text_layer, n_text_ctx, n_text_state]
    std::vector<float> self_v_cache;    // [n_text_layer, n_text_ctx, n_text_state]
    std::vector<float> this_self_k;     // [n_text_layer, 1, n_text_state]
    std::vector<float> this_self_v;     // [n_text_layer, 1, n_text_state]

    std::vector<float> logits;
} WhisperFeature;


class Whisper {
public:
    Whisper() = default;

    Whisper(const std::string& model_type, const std::string& language);

    ~Whisper();

    bool init(const std::string& model_type, const std::string& language);

    bool load_models(const std::string& model_root);

    std::vector<float> preprocess(std::vector<float>& audio_data, int n_mels);

    bool run(std::vector<float>& audio_data, std::string& result);

private:
    int get_lang_token(const std::string& lang);
    void causal_mask_1d(int offset);
    void dma_cross_kv();
    int run_decoder(int token, int offset);

private:
    std::string m_model_type;
    std::string m_lang;    

    AxModelRunner m_encoder;
    AxModelRunner m_decoder;
    std::vector<float> m_pe;
    std::vector<std::string> m_token_tables;
    json m_config;
    WhisperFeature m_feature;
};