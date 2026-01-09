#pragma once

#include "httplib.h"
#include "utils/logger.h"
#include "nlohmann/json.hpp"
#include "api/ax_whisper_api.h"

class WhisperHTTPServer {
private:
    httplib::Server m_srv;
    AX_WHISPER_HANDLE m_model;

public:
    WhisperHTTPServer() = default;

    ~WhisperHTTPServer() {
        AX_WHISPER_Uninit(m_model);
    }

    bool init(const std::string& model_path, const std::string& model_type, const std::string& language) {
        m_model = AX_WHISPER_Init(model_type.c_str(), model_path.c_str(), language.c_str());
        if (!m_model) {
            ALOGE("whisper load models failed!");
            return false;
        }
        return true;
    }

    void start(int port = 8080) {
        _setup_routes();
        
        ALOGI("Start server at port %d, POST binary stream to IP:%d/asr", port, port);
        m_srv.listen("0.0.0.0", port);
    }

private:
    void _setup_routes() {
        // 主要端点：接收二进制流
        m_srv.Post("/asr", [this](const httplib::Request& req, httplib::Response& res) {
            _run_model(req, res);
        });
    }
    
    void _run_model(const httplib::Request& req, httplib::Response& res) {
        // 设置CORS头
        setCORSHeaders(res);
        
        try {
            // 1. 检查Content-Type
            if (!req.has_header("Content-Type") ||
                req.get_header_value("Content-Type").find("application/octet-stream") == std::string::npos) {
                res.status = 400;
                res.set_content(R"({"error": "Content-Type must be application/octet-stream"})", "application/json");
                return;
            }
            
            // 2. 检查数据大小
            if (req.body.empty()) {
                res.status = 400;
                res.set_content(R"({"error": "Request body is empty"})", "application/json");
                return;
            }
            
            // 3. 验证数据大小必须是4的倍数
            if (req.body.size() % sizeof(float) != 0) {
                res.status = 400;
                std::string error = "Data size must be multiple of 4 bytes, received " + std::to_string(req.body.size()) + "bytes";

                res.set_content(R"({"error": "Data size must be multiple of 4 bytes"})", "application/json");
                return;
            }
            
            // 4. 解析二进制数据为float数组
            std::vector<float> audio_data = parseBinaryToFloats(req.body);

            // 5. 跑模型
            char* text;
            if (0 != AX_WHISPER_RunPCM(m_model, audio_data.data(), audio_data.size(), &text)) {
                ALOGE("run whisper failed!\n");
                res.status = 400;
                res.set_content(R"({"error": "Run model failed!"})", "application/json");
                return;
            }
            
            // 6. 构建响应
            nlohmann::json response;
            response["success"] = true;
            response["text"] = text;
            
            res.set_content(response.dump(2), "application/json");
            
        } catch (const std::exception& e) {
            res.status = 500;
            nlohmann::json error;
            error["error"] = "Internal server error";
            error["message"] = e.what();
            res.set_content(error.dump(2), "application/json");
            ALOGE("Error: %s", e.what());
        }
    }
    
    // 解析二进制数据为float数组
    std::vector<float> parseBinaryToFloats(const std::string& binary_data) {
        std::vector<float> result;
        
        size_t float_count = binary_data.size() / sizeof(float);
        result.resize(float_count);
        
        // 直接内存拷贝
        std::memcpy(result.data(), binary_data.data(), binary_data.size());
        
        return result;
    }

    // 设置CORS头
    void setCORSHeaders(httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", 
                      "Content-Type, X-Array-Name, X-Array-Description, X-Array-Size");
    }
};