#include <stdio.h>
#include <librosa/librosa.h>
#include "cmdline.hpp"
#include "AudioFile.h"
#include "Encoder.hpp"

#define WHISPER_SAMPLE_RATE 16000
#define WHISPER_N_FFT       400
#define WHISPER_HOP_LENGTH  160
#define WHISPER_CHUNK_SIZE  30
#define WHISPER_N_MELS      80


int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("encoder", 'e', "encoder axmodel", true, "");
    cmd.add<std::string>("decoder_main", 'm', "decoder_main axmodel", true, "");
    cmd.add<std::string>("decoder_loop", 'l', "decoder_loop axmodel", true, "");
    cmd.add<std::string>("wav", 'w', "wav file", true, "");
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto encoder_file = cmd.get<std::string>("encoder");
    auto decoder_main_file = cmd.get<std::string>("decoder_main");
    auto decoder_loop_file = cmd.get<std::string>("decoder_loop");
    auto wav_file = cmd.get<std::string>("wav");

    printf("encoder: %s\ndecoder_main: %s\ndecoder_loop: %s\nwav_file: %s\n", encoder_file.c_str(), decoder_main_file.c_str(), decoder_loop_file.c_str(), wav_file.c_str());

    AudioFile<float> audio_file;
    if (!audio_file.load(wav_file)) {
        printf("load wav failed!\n");
        return -1;
    }

    auto& samples = audio_file.samples[0];
    auto mel = librosa::Feature::melspectrogram(samples, WHISPER_SAMPLE_RATE, WHISPER_N_FFT, WHISPER_HOP_LENGTH, "hann", true, "reflect", 2.0f, WHISPER_N_MELS, 0.0f, WHISPER_SAMPLE_RATE / 2.0f);

    FILE* fp = fopen("mel.bin", "wb");
    for (size_t i = 0; i < mel.size(); i++) {
        fwrite(mel[i].data(), sizeof(float), mel[i].size(), fp);
    }
    fclose(fp);

    // int ret = 0;
    // EncoderData encoder_data;
    // Encoder encoder;
    // ret = encoder.Init(encoder_file.c_str());
    // if (ret) {
    //     printf("encoder init failed!\n");
    //     return ret;
    // }

    // ret = encoder.PrepareData(&encoder_data);
    // if (ret) {
    //     printf("encoder prepare data failed!\n");
    //     return ret;
    // }

    // encoder.DestroyData(&encoder_data);

    return 0;
}
    