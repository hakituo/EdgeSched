#pragma once
#include <memory>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <future>
#include "native_executor.h"
#include "phase_strategy.h"
#include <functional>
#include <unordered_map>
#include <atomic>
#include <string>
#include <chrono>

namespace ai_scheduler {

// 任务类型枚举
enum class TaskType {
    LLM_INFERENCE,    // LLM推理任务（GPU实时）
    TTS_SYNTHESIS,    // TTS语音合成（CPU实时）
    IMAGE_GENERATION, // 图像生成（GPU异步队列）
    NPU_INFERENCE     // NPU推理任务
};

// 任务优先级
enum class TaskPriority {
    HIGH,    // 高优先级（如LLM实时响应）
    MEDIUM,  // 中等优先级
    LOW      // 低优先级
};

// 任务结果状态
enum class TaskStatus {
    PENDING,      // 等待执行
    RUNNING,      // 正在执行
    COMPLETED,    // 执行完成
    FAILED,       // 执行失败
    CANCELLED     // 已取消
};

// 基础任务接口
class ITask {
public:
    ITask(std::string taskId, TaskType type, TaskPriority priority = TaskPriority::MEDIUM)
        : taskId_(std::move(taskId)), type_(type), priority_(priority), status_(TaskStatus::PENDING) {}

    virtual ~ITask() = default;
    virtual void execute() = 0;
    
    virtual TaskType getType() const { return type_; }
    virtual TaskPriority getPriority() const { return priority_; }
    virtual TaskStatus getStatus() const { return status_; }
    virtual void setStatus(TaskStatus status) { status_ = status; }
    virtual std::string getTaskId() const { return taskId_; }
    virtual std::shared_ptr<void> getResult() const = 0;

protected:
    std::string taskId_;
    TaskType type_;
    TaskPriority priority_;
    std::atomic<TaskStatus> status_;
};

// 工作器接口
class IWorker {
public:
    virtual ~IWorker() = default;
    virtual bool initialize() = 0;
    virtual void shutdown() = 0;
    virtual bool canHandle(TaskType type) const = 0;
    virtual void processTask(std::shared_ptr<ITask> task) = 0;
    virtual std::string getWorkerId() const = 0;
    virtual bool isBusy() const = 0;
    virtual bool isReady() const = 0;
};

// 资源隔离调度器
class ResourceIsolationScheduler {
public:
    ResourceIsolationScheduler();
    ~ResourceIsolationScheduler();

    // 初始化调度器
    bool initialize(size_t cpuThreadCount = 4);

    void setNativeExecutorEnabled(bool enabled);
    
    // 关闭调度器
    void shutdown();
    
    // 添加工作器
    bool addWorker(std::shared_ptr<IWorker> worker);
    
    // 提交任务
    void submitTask(std::shared_ptr<ITask> task);

    template<typename F, typename... Args>
    auto submitTask(TaskType type, TaskPriority priority, F&& f, Args&&... args) -> std::pair<std::string, std::future<decltype(f(args...))>>;
    
    // 取消任务
    bool cancelTask(const std::string& taskId);
    
    // 获取任务状态
    TaskStatus getTaskStatus(const std::string& taskId);

    // 获取任务实例
    std::shared_ptr<ITask> getTask(const std::string& taskId);
    
    // Check if scheduler is running
    bool isRunning() const { return running_; }

    // 获取系统状态
    struct SystemStatus {
        size_t totalTasks;
        size_t pendingTasks;
        size_t runningTasks;
        size_t completedTasks;
        size_t failedTasks;
        std::unordered_map<std::string, bool> workerStatus;  // 工作器ID -> 是否忙碌
    };
    
    SystemStatus getSystemStatus();
    
    // 等待所有任务完成
    void waitForAllTasks();
    
    // 资源监控
    struct ResourceUsage {
        float cpuUsage;        // CPU使用率
        float gpuUsage;        // GPU使用率
        size_t memoryUsage;    // 内存使用量（MB）
        size_t gpuMemoryUsage; // GPU内存使用量（MB）
    };
    
    ResourceUsage getResourceUsage();

    // Native Executor Access
    std::shared_ptr<ai_scheduler::NativeExecutor> getNativeExecutor() { return nativeExecutor_; }

    // ============================================================
    // Phase-Aware 调度接口
    // ============================================================

    /// 设置 Phase-Aware 策略
    void setPhaseStrategy(std::shared_ptr<IPhaseAwareStrategy> strategy);

    /// 获取 Phase-Aware 策略
    std::shared_ptr<IPhaseAwareStrategy> getPhaseStrategy() const { return phaseStrategy_; }

    /// 提交带阶段属性的任务
    void submitPhaseTask(std::shared_ptr<ITask> task, const PhaseAttributes& attrs);

    /// 获取 LLM 状态
    LLMState getLLMState() const;

    /// 迁移 LLM 到 CPU
    bool migrateLLMToCPU(bool urgent = false);

    /// 回迁 LLM 到 GPU
    bool restoreLLMToGPU();

    /// 设置迁移回调（对接真实引擎）
    void setMigrationCallbacks(
        std::function<bool(bool urgent)> migrate_fn,
        std::function<bool()> restore_fn);

private:
    // 任务队列管理
    void processTaskQueues();
    
    // 选择合适的工作器
    std::shared_ptr<IWorker> selectWorker(TaskType type);
    
    // 处理异步图像生成任务
    void processImageGenerationQueue();

    
    // 内部数据
    std::vector<std::shared_ptr<IWorker>> workers_;  // 所有工作器
    std::vector<std::shared_ptr<IWorker>> gpuWorkers_;  // GPU工作器
    std::vector<std::shared_ptr<IWorker>> cpuWorkers_;  // CPU工作器
    std::shared_ptr<IWorker> llmWorker_;  // LLM专用工作器
    
    // 任务队列
    std::queue<std::shared_ptr<ITask>> llmTaskQueue_;
    std::queue<std::shared_ptr<ITask>> ttsTaskQueue_;
    std::queue<std::shared_ptr<ITask>> imageTaskQueue_;
    
    // 任务映射
    std::unordered_map<std::string, std::shared_ptr<ITask>> tasks_;
    
    // 线程池和同步原语
    std::vector<std::thread> workerThreads_;
    std::thread imageQueueThread_;
    std::mutex queueMutex_;
    std::mutex imageQueueMutex_;
    std::condition_variable cv_;
    std::condition_variable imageCv_;
    
    // 状态标志
    std::atomic<bool> running_;
    std::atomic<bool> initialized_;

    bool native_executor_enabled_;
    
    // 统计信息
    std::atomic<size_t> totalTasks_;
    std::atomic<size_t> completedTasks_;
    std::atomic<size_t> failedTasks_;

    // Native Executor
    std::shared_ptr<ai_scheduler::NativeExecutor> nativeExecutor_;

    // Phase-Aware 策略
    std::shared_ptr<IPhaseAwareStrategy> phaseStrategy_;

    // Phase-Aware 任务属性映射（task_id -> PhaseAttributes）
    std::unordered_map<std::string, PhaseAttributes> phaseAttributesMap_;
    std::mutex phaseMutex_;

    // 迁移回调
    std::function<bool(bool urgent)> migrateLLMFn_;
    std::function<bool()> restoreLLMFn_;
};

// 具体任务实现
template<typename ResultType>
class Task : public ITask {
public:
    template<typename F, typename... Args>
    Task(TaskType type, TaskPriority priority, F&& f, Args&&... args)
        : ITask("task_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()), type, priority) {
        
        // 创建任务函数
        task_ = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    }
    
    void execute() override {
        try {
            setStatus(TaskStatus::RUNNING);
            result_ = std::make_shared<ResultType>(task_());
            setStatus(TaskStatus::COMPLETED);
        } catch (...) {
            exception_ = std::current_exception();
            setStatus(TaskStatus::FAILED);
        }
    }
    
    std::shared_ptr<void> getResult() const override {
        if (exception_) {
            std::rethrow_exception(exception_);
        }
        return result_;
    }
    
    std::shared_ptr<ResultType> getTypedResult() const {
        if (exception_) {
            std::rethrow_exception(exception_);
        }
        return result_;
    }
    
private:
    std::function<ResultType()> task_;
    std::shared_ptr<ResultType> result_;
    std::exception_ptr exception_;
};

// Void specialization for Task
template<>
class Task<void> : public ITask {
public:
    template<typename F, typename... Args>
    Task(TaskType type, TaskPriority priority, F&& f, Args&&... args)
        : ITask("task_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()), type, priority) {
        task_ = std::bind(std::forward<F>(f), std::forward<Args>(args)...);
    }
    
    void execute() override {
        try {
            setStatus(TaskStatus::RUNNING);
            task_();
            setStatus(TaskStatus::COMPLETED);
        } catch (...) {
            exception_ = std::current_exception();
            setStatus(TaskStatus::FAILED);
        }
    }
    
    std::shared_ptr<void> getResult() const override {
        if (exception_) {
            std::rethrow_exception(exception_);
        }
        return nullptr;
    }

private:
    std::function<void()> task_;
    std::exception_ptr exception_;
};

// 提交任务模板实现
template<typename F, typename... Args>
auto ResourceIsolationScheduler::submitTask(TaskType type, TaskPriority priority, F&& f, Args&&... args) -> std::pair<std::string, std::future<decltype(f(args...))>> {
    using ResultType = decltype(f(args...));
    
    // 创建任务
    auto task = std::make_shared<Task<ResultType>>(type, priority, std::forward<F>(f), std::forward<Args>(args)...);
    std::string taskId = task->getTaskId();

    // 存储任务
    { 
        std::lock_guard<std::mutex> lock(queueMutex_);
        tasks_[taskId] = task;
        totalTasks_++;
        
        // 根据任务类型放入对应的队列
        switch (type) {
            case TaskType::LLM_INFERENCE:
                llmTaskQueue_.push(task);
                break;
            case TaskType::TTS_SYNTHESIS:
                ttsTaskQueue_.push(task);
                break;
            case TaskType::IMAGE_GENERATION:
                {
                    std::lock_guard<std::mutex> imgLock(imageQueueMutex_);
                    imageTaskQueue_.push(task);
                    imageCv_.notify_one();
                }
                break;
        }
    }
    
    // 通知工作线程
    cv_.notify_one();
    
    // 返回future
    return { taskId, std::async(std::launch::deferred, [task]() {
        while (task->getStatus() == TaskStatus::PENDING || task->getStatus() == TaskStatus::RUNNING) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        if constexpr (std::is_void_v<ResultType>) {
            task->getResult();
        } else {
            return *task->getTypedResult();
        }
    }) };
}

} // namespace ai_scheduler
