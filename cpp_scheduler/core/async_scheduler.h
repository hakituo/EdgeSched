#ifndef ASYNC_SCHEDULER_H
#define ASYNC_SCHEDULER_H

#include <uv.h>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <string>
#include <thread>
#include <vector>

namespace ai_scheduler {

// 旧版异步调度器（已弃用，保留兼容）
// 注意：TaskType/TaskPriority 已在 resource_isolation_scheduler.h 中定义
// 此处使用独立命名空间避免冲突
namespace async_legacy {

// 任务类型枚举（旧版）
enum class LegacyTaskType {
    LLM_GPU,
    TTS_CPU,
    IMAGE_GPU_QUEUE
};

// 任务优先级（旧版）
enum class LegacyTaskPriority {
    HIGH,
    MEDIUM,
    LOW
};

} // namespace async_legacy

// 为兼容代码提供别名（指向新版枚举）
using LegacyTaskType = async_legacy::LegacyTaskType;
using LegacyTaskPriority = async_legacy::LegacyTaskPriority;

// 任务基类（旧版）
class Task {
public:
    Task(LegacyTaskType type, LegacyTaskPriority priority = LegacyTaskPriority::MEDIUM);
    virtual ~Task() = default;
    
    // 执行任务的纯虚函数
    virtual void execute() = 0;
    
    // 任务完成回调
    using Callback = std::function<void(bool success, const std::string& result)>;
    void setCallback(Callback callback) { callback_ = callback; }
    void notifyComplete(bool success, const std::string& result);
    
    // 获取任务属性
    LegacyTaskType getType() const { return type_; }
    LegacyTaskPriority getPriority() const { return priority_; }
    uint64_t getId() const { return id_; }
    
private:
    LegacyTaskType type_;
    LegacyTaskPriority priority_;
    uint64_t id_;  // 唯一任务ID
    Callback callback_;
    static std::atomic<uint64_t> next_id_;
};

// 异步调度器类
class AsyncScheduler {
public:
    AsyncScheduler();
    ~AsyncScheduler();
    
    // 初始化调度器
    bool initialize(int num_gpu_workers = 2, int num_cpu_workers = 4);
    
    // 启动事件循环
    void start();
    
    // 停止事件循环
    void stop();

    bool isRunning() const;
    
    // 提交任务
    uint64_t submitTask(std::shared_ptr<Task> task);
    
    // 取消任务
    bool cancelTask(uint64_t task_id);
    
    // 获取libuv循环
    uv_loop_t* getLoop() { return loop_; }
    
private:
    // 内部方法
    void workerThreadFunction(LegacyTaskType worker_type);
    void processTaskQueue();
    
    // libuv相关
    uv_loop_t* loop_;
    uv_async_t* async_handle_;
    
    // 任务队列和管理
    std::mutex task_mutex_;
    std::unordered_map<uint64_t, std::shared_ptr<Task>> tasks_;
    std::vector<std::shared_ptr<Task>> llm_queue_;
    std::vector<std::shared_ptr<Task>> tts_queue_;
    std::vector<std::shared_ptr<Task>> image_queue_;
    
    // 工作线程
    std::vector<std::thread> gpu_workers_;
    std::vector<std::thread> cpu_workers_;
    
    // 状态控制
    std::atomic<bool> running_;
    int num_gpu_workers_;
    int num_cpu_workers_;
};

} // namespace ai_scheduler

#endif // ASYNC_SCHEDULER_H
