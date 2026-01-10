#include "Whisper.hpp"
#include "utils/logger.h"
#include "AudioFile.h"
#include "api/ax_whisper_api.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Opaque handle type for Whisper ASR context
 * 
 * This handle encapsulates all internal state of the Whisper ASR system.
 * The actual implementation is hidden from C callers to maintain ABI stability.
 */
typedef void* AX_WHISPER_HANDLE;

/**
 * @brief Initialize the Whisper ASR system with specific configuration
 * 
 * Creates and initializes a new Whisper ASR context with the specified
 * model type, model path, and language. This function loads the appropriate
 * models, configures the recognizer, and prepares it for speech recognition.
 * 
 * @param model_type Type of Whisper model to use (e.g., "tiny", "base", "small", "medium", "large")
 *                   or custom model identifier
 * @param model_path Directory path where model files are stored
 *                   Model files are expected to be in the format:
 *                   - {model_path}/{model_type}/{model_type}-encoder.axmodel
 *                   - {model_path}/{model_type}/{model_type}-decoder.axmodel
 *                   - {model_path}/{model_type}/{model_type}-tokens.txt
 *                   - {model_path}/{model_type}/{model_type}_config.json
 * @param language Language code for recognition (e.g., "en", "zh", "ja", "ko")
 *                 Use "auto" for automatic language detection if supported
 * 
 * @return AX_WHISPER_HANDLE Opaque handle to the initialized Whisper context,
 *         or NULL if initialization fails
 * 
 * @note The caller is responsible for calling AX_WHISPER_Uninit() to free
 *       resources when the handle is no longer needed.
 * @note If language is not supported by the model, the function may fall back
 *       to a default language or return NULL.
 * @example
 *   // Initialize English recognition with base model
 *   AX_WHISPER_HANDLE handle = AX_WHISPER_Init("base", "../models-ax650", "en");
 *   
 */
AX_WHISPER_API AX_WHISPER_HANDLE AX_WHISPER_Init(const char* model_type, const char* model_path, const char* language) {
    Whisper* handle = new Whisper(std::string(model_type), std::string(language));
    if (!handle->load_models(std::string(model_path))) {
        ALOGE("load models failed!");
        return nullptr;
    }

    return static_cast<AX_WHISPER_HANDLE>(handle);
}

/**
 * @brief Deinitialize and release Whisper ASR resources
 * 
 * Cleans up all resources associated with the Whisper context, including
 * unloading models, freeing memory, and releasing hardware resources.
 * 
 * @param handle Whisper context handle obtained from AX_WHISPER_Init()
 * 
 * @warning After calling this function, the handle becomes invalid and
 *          should not be used in any subsequent API calls.
 */
AX_WHISPER_API void AX_WHISPER_Uninit(AX_WHISPER_HANDLE handle) {
    if (handle) {
        auto wrapper = static_cast<Whisper*>(handle);
        delete wrapper;
    }
}

/**
 * @brief Perform speech recognition and return dynamically allocated string
 * 
 * @param handle Whisper context handle
 * @param wav_file Path to the input WAV audio file
 * @param result Pointer to receive the allocated result string
 * 
 * @return int Status code (0 = success, <0 = error)
 * 
 * @note The returned string is allocated with malloc() and must be freed
 *       by the caller using free() when no longer needed.
 */
AX_WHISPER_API int AX_WHISPER_RunFile(AX_WHISPER_HANDLE handle, 
                   const char* wav_file, 
                   char** result) {
    if (!handle || !wav_file || !result) {
        return -1;
    }

    AudioFile<float> audio_file;
    Whisper* whisper = (Whisper*)handle;

    *result = nullptr;

    if (!audio_file.load(wav_file)) {
        ALOGE("load wav failed!\n");
        return -1;
    }

    auto& samples = audio_file.samples[0];
    int n_samples = samples.size();
    
    // convert to mono
    if (audio_file.isStereo()) {
        for (int i = 0; i < n_samples; i++) {
            samples[i] = (samples[i] + audio_file.samples[1][i]) / 2;
        }
    }
    
    std::string text;
    if (!whisper->run(samples, text)) {
        printf("run whisper failed!\n");
        return -1;
    }

    *result = strdup(text.c_str());

    return 0;
}

/**
 * @brief Perform speech recognition and return dynamically allocated string
 * 
 * @param handle Whisper context handle
 * @param pcm_data 16k Mono PCM f32 data, range from -1.0 to 1.0
 * @param num_samples Sample num of PCM data
 * @param result Pointer to receive the allocated result string
 * 
 * @return int Status code (0 = success, <0 = error)
 * 
 * @note The returned string is allocated with malloc() and must be freed
 *       by the caller using free() when no longer needed.
 */
AX_WHISPER_API int AX_WHISPER_RunPCM(AX_WHISPER_HANDLE handle, 
                   float* pcm_data, 
                   int num_samples,
                   char** result) {
    if (!handle || !pcm_data || !result) {
        return -1;
    }

    Whisper* whisper = (Whisper*)handle;

    *result = nullptr;

    std::vector<float> samples(num_samples);
    memcpy(samples.data(), pcm_data, sizeof(float) * num_samples);
    
    std::string text;
    if (!whisper->run(samples, text)) {
        printf("run whisper failed!\n");
        return -1;
    }

    *result = strdup(text.c_str());

    return 0;                
}

#ifdef __cplusplus
}
#endif