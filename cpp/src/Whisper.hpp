#pragma once

#include "Encoder.hpp"
#include "DecoderMain.hpp"
#include "DecoderLoop.hpp"
#include "utils/nlohmann/json.hpp"

#include <array>

// for convenience
using json = nlohmann::json;

#define DEFAULT_LANG    "zh"


typedef struct _WhisperFeature {
    std::vector<float> mel;
    std::array<int, 4> sot_seq;

    std::vector<float> n_layer_cross_k;
    std::vector<float> n_layer_cross_v;

    std::vector<float> decoder_main_logits;
    std::vector<float> mask;
    std::vector<float> n_layer_self_k_cache;
    std::vector<float> n_layer_self_v_cache;

    std::vector<float> logits;
} WhisperFeature;


class Whisper {
public:
    Whisper() = default;

    Whisper(const std::string& model_type, const std::string& language);

    ~Whisper();

    bool init(const std::string& model_type, const std::string& language);

    bool load_models(const std::string& model_root);

    std::vector<float> preprocess(std::vector<float>& audio_data);

    bool run(std::vector<float>& audio_data, std::string& result);

private:
    int get_lang_token(const std::string& lang);
    void supress_tokens(std::vector<float>& logits, bool is_initial);

private:
    std::string m_model_type;
    std::string m_lang;    

    Encoder m_encoder;
    DecoderMain m_decoder_main;
    DecoderLoop m_decoder_loop;
    std::vector<float> m_pe;
    std::vector<std::string> m_token_tables;
    json m_config;
    WhisperFeature m_feature;
};