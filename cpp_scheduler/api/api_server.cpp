#include "api_server.h"
#include "../core/resource_isolation_scheduler.h"
#include "../workers/cpu_tts_worker.h"
#include "../workers/gpu_llm_worker.h"
#include "../workers/gpu_image_worker.h"
#include <iostream>
#include <sstream>
#include <chrono>
#include <regex>

// Helper macro for logging to avoid direct std::cout usage in high-perf code
#define LOG_INFO(msg) do { std::cout << "[INFO] " << msg << std::endl; } while(0)
#define LOG_ERROR(msg) do { std::cerr << "[ERROR] " << msg << std::endl; } while(0)

namespace ai_scheduler::api {

APIServer::APIServer(int port)
    : port_(port),
      running_(false),
      server_handle_(nullptr),
      enable_auth_(false) {
    LOG_INFO("[API Server] Creating API server on port: " << port);
    registerRoutes();
}

APIServer::~APIServer() {
    stop();
    LOG_INFO("[API Server] Destroyed");
}

void APIServer::setScheduler(std::shared_ptr<ResourceIsolationScheduler> scheduler) {
    scheduler_ = scheduler;
    LOG_INFO("[API Server] Scheduler set");
}

void APIServer::setTTSWorker(std::shared_ptr<CPUTTSWorker> tts_worker) {
    tts_worker_ = tts_worker;
    LOG_INFO("[API Server] TTS worker set");
}

void APIServer::setLLMWorker(std::shared_ptr<GPULLMWorker> llm_worker) {
    llm_worker_ = llm_worker;
    LOG_INFO("[API Server] LLM worker set");
}

void APIServer::setImageWorker(std::shared_ptr<GPUImageWorker> image_worker) {
    image_worker_ = image_worker;
    LOG_INFO("[API Server] Image worker set");
}

bool APIServer::start() {
    if (running_) {
        LOG_ERROR("[API Server] Server already running");
        return false;
    }
    
    // 验证必要的组件是否设置
    if (!scheduler_) {
        LOG_ERROR("[API Server] Error: Scheduler not set");
        return false;
    }
    
    running_ = true;
    server_thread_ = std::thread(&APIServer::serverThread, this);
    
    LOG_INFO("[API Server] Started on port " << port_);
    return true;
}

void APIServer::stop() {
    if (running_) {
        running_ = false;
        
        // 这里应该关闭libuv服务器
        if (server_handle_) {
            // 实际实现中应该释放服务器资源
            server_handle_ = nullptr;
        }
        
        if (server_thread_.joinable()) {
            server_thread_.join();
        }
        
        LOG_INFO("[API Server] Stopped");
    }
}

bool APIServer::isRunning() const {
    return running_;
}

int APIServer::getPort() const {
    return port_;
}

void APIServer::registerRoutes() {
    // 健康检查
    routes_["GET /health"] = std::bind(&APIServer::handleHealth, this, std::placeholders::_1);
    
    // LLM相关API
    routes_["POST /api/v1/llm/generate"] = std::bind(&APIServer::handleLLMRequest, this, std::placeholders::_1);
    
    // TTS相关API
    routes_["POST /api/v1/tts/synthesize"] = std::bind(&APIServer::handleTTSRequest, this, std::placeholders::_1);
    
    // 图像生成相关API
    routes_["POST /api/v1/image/generate"] = std::bind(&APIServer::handleImageRequest, this, std::placeholders::_1);
    
    // 状态查询API
    routes_["GET /api/v1/status"] = std::bind(&APIServer::handleStatusRequest, this, std::placeholders::_1);
    
    // 任务取消API
    routes_["DELETE /api/v1/tasks/:id"] = std::bind(&APIServer::handleCancelTask, this, std::placeholders::_1);
    
    std::cout << "[API Server] Routes registered: " << routes_.size() << std::endl;
}

void APIServer::serverThread() {
    std::cout << "[API Server] Server thread started" << std::endl;
    
    // 模拟服务器运行
    // 实际实现中，这里应该使用libuv或其他库来实现HTTP服务器
    while (running_) {
        // 模拟服务器处理循环
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    std::cout << "[API Server] Server thread stopped" << std::endl;
}

APIResponse APIServer::handleHealth(const APIRequest& request) {
    (void)request;
    std::cout << "[API Server] Health check request received" << std::endl;
    
    // 检查所有必要组件的状态
    bool scheduler_ready = scheduler_ != nullptr; // ResourceIsolationScheduler has no isRunning? It has initialize.
    bool tts_ready = tts_worker_ == nullptr || tts_worker_->isReady();
    bool llm_ready = llm_worker_ == nullptr || llm_worker_->isReady();
    bool image_ready = image_worker_ == nullptr || image_worker_->isReady();
    
    bool all_ready = scheduler_ready && tts_ready && llm_ready && image_ready;
    
    std::stringstream ss;
    ss << "{" 
       << "\"status\": \"" << (all_ready ? "ok" : "degraded") << "\"," 
       << "\"components\": {" 
       << "\"scheduler\": \"" << (scheduler_ready ? "ready" : "not_ready") << "\"," 
       << "\"tts\": \"" << (tts_ready ? "ready" : "not_ready") << "\"," 
       << "\"llm\": \"" << (llm_ready ? "ready" : "not_ready") << "\"," 
       << "\"image\": \"" << (image_ready ? "ready" : "not_ready") << "\"" 
       << "}" 
       << "}";
    
    return APIResponse(APIStatus::SUCCESS, ss.str());
}

APIResponse APIServer::handleLLMRequest(const APIRequest& request) {
    std::cout << "[API Server] LLM request received" << std::endl;
    
    if (!scheduler_ || !llm_worker_) {
        return createErrorResponse(APIStatus::SERVICE_UNAVAILABLE, "LLM service not available");
    }
    
    std::string prompt = "";
    // 简化的Prompt提取
    size_t prompt_pos = request.body.find("\"prompt\":\"");
    if (prompt_pos != std::string::npos) {
        size_t start = prompt_pos + 10;
        size_t end = request.body.find("\"", start);
        if (end != std::string::npos) {
            prompt = request.body.substr(start, end - start);
        }
    }
    
    if (prompt.empty()) {
        return createErrorResponse(APIStatus::BAD_REQUEST, "Missing required field: prompt");
    }
    
    // 提交LLM任务
    auto submission = scheduler_->submitTask(TaskType::LLM_INFERENCE, TaskPriority::HIGH, 
        [this, prompt]() -> LLMInferenceResponse {
            LLMInferenceRequest req;
            req.prompt = prompt;
            req.maxTokens = 256;
            req.temperature = 0.7f;
            
            if (llm_worker_) {
                return llm_worker_->executeInference(req);
            }
            return LLMInferenceResponse{ "", 0, 0.0f, false, "No worker" };
        });
        
    std::string taskId = submission.first;
    
    std::stringstream ss;
    ss << "{" 
       << "\"success\": true," 
       << "\"task_id\": \"" << taskId << "\"," 
       << "\"status\": \"processing\"," 
       << "\"message\": \"LLM request submitted successfully\"" 
       << "}";
    
    return APIResponse(APIStatus::SUCCESS, ss.str());
}

APIResponse APIServer::handleTTSRequest(const APIRequest& request) {
    std::cout << "[API Server] TTS request received" << std::endl;
    
    if (!scheduler_ || !tts_worker_) {
        return createErrorResponse(APIStatus::SERVICE_UNAVAILABLE, "TTS service not available");
    }
    
    std::string text = "";
    size_t text_pos = request.body.find("\"text\":\"");
    if (text_pos != std::string::npos) {
        size_t start = text_pos + 8;
        size_t end = request.body.find("\"", start);
        if (end != std::string::npos) {
            text = request.body.substr(start, end - start);
        }
    }
    
    if (text.empty()) {
        return createErrorResponse(APIStatus::BAD_REQUEST, "Missing required field: text");
    }
    
    // 提交TTS任务
    auto submission = scheduler_->submitTask(TaskType::TTS_SYNTHESIS, TaskPriority::MEDIUM, 
        [this, text]() -> bool {
            TTSParams params;
            params.text = text;
            std::string output;
            std::vector<uint8_t> data;
            
            if (tts_worker_) {
                return tts_worker_->synthesizeSync(params, output, data);
            }
            return false;
        });
    
    std::string taskId = submission.first;
    
    std::stringstream ss;
    ss << "{" 
       << "\"success\": true," 
       << "\"task_id\": \"" << taskId << "\"," 
       << "\"status\": \"processing\"," 
       << "\"message\": \"TTS request submitted successfully\"" 
       << "}";
    
    return APIResponse(APIStatus::SUCCESS, ss.str());
}

APIResponse APIServer::handleImageRequest(const APIRequest& request) {
    std::cout << "[API Server] Image generation request received" << std::endl;
    
    if (!scheduler_ || !image_worker_) {
        return createErrorResponse(APIStatus::SERVICE_UNAVAILABLE, "Image generation service not available");
    }
    
    std::string prompt = "";
    size_t prompt_pos = request.body.find("\"prompt\":\"");
    if (prompt_pos != std::string::npos) {
        size_t start = prompt_pos + 10;
        size_t end = request.body.find("\"", start);
        if (end != std::string::npos) {
            prompt = request.body.substr(start, end - start);
        }
    }
    
    if (prompt.empty()) {
        return createErrorResponse(APIStatus::BAD_REQUEST, "Missing required field: prompt");
    }
    
    // 提交图像生成任务
    auto submission = scheduler_->submitTask(TaskType::IMAGE_GENERATION, TaskPriority::LOW, 
        [this, prompt]() -> bool {
            ImageGenerationParams params;
            params.prompt = prompt;
            std::string output;
            
            if (image_worker_) {
                return image_worker_->generateSync(params, output);
            }
            return false;
        });
        
    std::string taskId = submission.first;
    
    std::stringstream ss;
    ss << "{" 
       << "\"success\": true," 
       << "\"task_id\": \"" << taskId << "\"," 
       << "\"status\": \"queued\"," 
       << "\"message\": \"Image generation request queued successfully\"" 
       << "}";
    
    return APIResponse(APIStatus::SUCCESS, ss.str());
}

APIResponse APIServer::handleStatusRequest(const APIRequest& request) {
    (void)request;
    LOG_INFO("[API Server] Status request received");
    
    auto stats = scheduler_ ? scheduler_->getSystemStatus() : ResourceIsolationScheduler::SystemStatus{};
    
    std::stringstream ss;
    ss << "{" 
       << "\"version\": \"1.0.0\"," 
       << "\"status\": \"running\"," 
       << "\"uptime\": 0," 
       << "\"resources\": {" 
       << "\"cpu_usage\": " << (tts_worker_ ? tts_worker_->getCPUUtilization() : 0.0f) << "," 
       << "\"gpu_usage\": 0.0," 
       << "\"memory_usage\": 0.0" 
       << "}," 
       << "\"queue_stats\": {" 
       << "\"total_tasks\": " << stats.totalTasks << "," 
       << "\"pending_tasks\": " << stats.pendingTasks << "," 
       << "\"completed_tasks\": " << stats.completedTasks 
       << "}" 
       << "}";
    
    return APIResponse(APIStatus::SUCCESS, ss.str());
}

APIResponse APIServer::handleCancelTask(const APIRequest& request) {
    LOG_INFO("[API Server] Cancel task request received");
    
    if (!scheduler_) {
        return createErrorResponse(APIStatus::SERVICE_UNAVAILABLE, "Scheduler not available");
    }
    
    // 提取任务ID
    std::string task_id_str = "";
    size_t id_pos = request.path.find_last_of('/');
    if (id_pos != std::string::npos) {
        task_id_str = request.path.substr(id_pos + 1);
    }
    
    bool success = scheduler_->cancelTask(task_id_str);
    
    std::stringstream ss;
    ss << "{" 
       << "\"success\": " << (success ? "true" : "false") << "," 
       << "\"message\": \"Task " << (success ? "cancelled successfully" : "not found or already completed") << "\"" 
       << "}";
    
    return APIResponse(APIStatus::SUCCESS, ss.str());
}

APIResponse APIServer::createErrorResponse(APIStatus status, const std::string& message) {
    std::stringstream ss;
    ss << "{" 
       << "\"success\": false," 
       << "\"error\": {" 
       << "\"code\": " << static_cast<int>(status) << "," 
       << "\"message\": \"" << message << "\"" 
       << "}" 
       << "}";
    
    return APIResponse(status, ss.str());
}

std::string APIServer::buildJSONResponse(bool success, const std::string& message, const void* data) {
    (void)data;
    std::stringstream ss;
    ss << "{" 
       << "\"success\": " << (success ? "true" : "false") << "," 
       << "\"message\": \"" << message << "\"" 
       << "}";
    
    return ss.str();
}

std::string toJSON(const APIResponse& response) {
    return response.body;
}

bool parseJSON(const std::string& json, std::unordered_map<std::string, std::string>& result) {
    try {
        std::regex keyValue("\"([^\"]+)\"\\s*:\\s*\"([^\"]+)\"");
        std::smatch match;
        std::string::const_iterator searchStart(json.cbegin());
        
        while (std::regex_search(searchStart, json.cend(), match, keyValue)) {
            result[match[1]] = match[2];
            searchStart = match.suffix().first;
        }
        
        return !result.empty();
    } catch (...) {
        return false;
    }
}

std::shared_ptr<APIServer> createDefaultAPIServer(int port) {
    auto server = std::make_shared<APIServer>(port);
    return server;
}

} // namespace ai_scheduler::api
