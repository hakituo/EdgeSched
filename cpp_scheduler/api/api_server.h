#ifndef API_SERVER_H
#define API_SERVER_H

#include <string>
#include <memory>
#include <functional>
#include <unordered_map>
#include <thread>
#include <atomic>
#include <mutex>
#include <chrono>

namespace ai_scheduler {
class ResourceIsolationScheduler;
class TaskQueue;
class CPUTTSWorker;
class GPULLMWorker;
class GPUImageWorker;
class ITask;
}

namespace ai_scheduler::api {

// API请求和响应结构
enum class APIStatus {
    SUCCESS = 200,
    BAD_REQUEST = 400,
    UNAUTHORIZED = 401,
    NOT_FOUND = 404,
    INTERNAL_ERROR = 500,
    SERVICE_UNAVAILABLE = 503
};

// 资源使用统计结构
struct ResourceStats {
    float cpuUsage;
    float gpuUsage;
    float llmGpuUsage;
    float imgGpuUsage;
    size_t memoryUsage;
    size_t gpuMemoryUsage;
};

// API请求结构
struct APIRequest {
    std::string method;
    std::string path;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    std::unordered_map<std::string, std::string> query_params;
};

// API响应结构
struct APIResponse {
    APIStatus status;
    std::string body;
    std::unordered_map<std::string, std::string> headers;
    
    APIResponse(APIStatus s = APIStatus::SUCCESS, const std::string& b = "")
        : status(s), body(b) {
        headers["Content-Type"] = "application/json";
        headers["Server"] = "AI Scheduler API";
    }
};

// API处理函数类型
typedef std::function<APIResponse(const APIRequest&)> APIHandler;

// API服务器类
class APIServer {
public:
    APIServer(int port = 8080);
    ~APIServer();
    
    // 初始化服务器（包含资源隔离架构）
    bool initialize();
    
    // 设置调度器和worker引用
    void setScheduler(std::shared_ptr<ai_scheduler::ResourceIsolationScheduler> scheduler);
    void setTTSWorker(std::shared_ptr<ai_scheduler::CPUTTSWorker> tts_worker);
    void setLLMWorker(std::shared_ptr<ai_scheduler::GPULLMWorker> llm_worker);
    void setImageWorker(std::shared_ptr<ai_scheduler::GPUImageWorker> image_worker);
    
    // 启动和停止服务器
    bool start();
    void stop();
    
    // 检查服务器状态
    bool isRunning() const;
    int getPort() const;
    
    // API路由注册（内部使用）
    void registerRoutes();
    
    // 任务回调处理
    void onTaskCompleted(std::shared_ptr<ITask> task);
    void onImageProgress(const std::string& taskId, float progress);
    
private:
    // 服务器线程函数
    void serverThread();
    
    // API处理函数
    APIResponse handleHealth(const APIRequest& request);
    APIResponse handleLLMRequest(const APIRequest& request);
    APIResponse handleTTSRequest(const APIRequest& request);
    APIResponse handleImageRequest(const APIRequest& request);
    APIResponse handleStatusRequest(const APIRequest& request);
    APIResponse handleCancelTask(const APIRequest& request);
    APIResponse handleResourceStats(const APIRequest& request);
    APIResponse handleImageProgress(const APIRequest& request);
    
    // 辅助函数
    APIResponse createErrorResponse(APIStatus status, const std::string& message);
    std::string buildJSONResponse(bool success, const std::string& message, const void* data = nullptr);
    
    // 成员变量
    int port_;
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
    std::thread server_thread_;
    
    // 组件引用
    std::shared_ptr<ai_scheduler::ResourceIsolationScheduler> scheduler_;
    std::shared_ptr<ai_scheduler::CPUTTSWorker> tts_worker_;
    std::shared_ptr<ai_scheduler::GPULLMWorker> llm_worker_;
    std::shared_ptr<ai_scheduler::GPUImageWorker> image_worker_;
    
    // 任务进度跟踪
    std::mutex progressMutex_;
    std::unordered_map<std::string, float> imageTaskProgress_;
    std::unordered_map<std::string, std::chrono::system_clock::time_point> taskTimestamps_;
    
    // API路由映射
    std::unordered_map<std::string, APIHandler> routes_;
    
    // 连接处理
    void* server_handle_; // libuv或其他服务器库的句柄
    
    // 认证和安全
    std::string api_key_;
    bool enable_auth_;
};

// JSON辅助函数
std::string toJSON(const APIResponse& response);
bool parseJSON(const std::string& json, std::unordered_map<std::string, std::string>& result);

// 创建默认API服务器实例
std::shared_ptr<APIServer> createDefaultAPIServer(int port = 8080);

} // namespace ai_scheduler::api

#endif // API_SERVER_H
