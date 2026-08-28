#include "resource_isolation_scheduler.h"
#include "../workers/gpu_llm_worker.h"
#include "../monitoring/flight_recorder.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <system_error>
#include <algorithm>
#include <mutex>
#include <condition_variable>

namespace ai_scheduler {

// 实现ResourceIsolationScheduler类
ResourceIsolationScheduler::ResourceIsolationScheduler()
    : running_(false), initialized_(false),
      totalTasks_(0), completedTasks_(0), failedTasks_(0) {
    nativeExecutor_ = std::make_shared<ai_scheduler::NativeExecutor>();
    native_executor_enabled_ = true;
}

ResourceIsolationScheduler::~ResourceIsolationScheduler() {
    shutdown();
}

bool ResourceIsolationScheduler::initialize(size_t cpuThreadCount) {
    if (initialized_) {
        return true;
    }
    
    running_ = true;
    
    // 创建CPU工作线程池
    for (size_t i = 0; i < cpuThreadCount; ++i) {
        workerThreads_.emplace_back([this]() {
            while (running_) {
                processTaskQueues();
            }
        });
    }
    
    // 创建图像生成队列处理线程（单独的异步处理）
    imageQueueThread_ = std::thread([this]() {
        while (running_) {
            processImageGenerationQueue();
        }
    });

    if (native_executor_enabled_ && nativeExecutor_) {
        nativeExecutor_->start();
    }

    initialized_ = true;
    std::cout << "ResourceIsolationScheduler initialized with " 
              << cpuThreadCount << " CPU threads" << std::endl;
    
    return true;
}

void ResourceIsolationScheduler::shutdown() {
    if (!initialized_) {
        return;
    }
    
    running_ = false;
    cv_.notify_all();
    imageCv_.notify_one();

    if (nativeExecutor_) {
        nativeExecutor_->stop();
    }
    
    // 等待所有线程结束
    for (auto& thread : workerThreads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    
    if (imageQueueThread_.joinable()) {
        imageQueueThread_.join();
    }

    
    // 清理工作器
    for (auto& worker : workers_) {
        worker->shutdown();
    }
    
    workers_.clear();
    gpuWorkers_.clear();
    cpuWorkers_.clear();
    llmWorker_ = nullptr;
    
    // 清空任务队列
    { 
        std::lock_guard<std::mutex> lock(queueMutex_);
        while (!llmTaskQueue_.empty()) llmTaskQueue_.pop();
        while (!ttsTaskQueue_.empty()) ttsTaskQueue_.pop();
        tasks_.clear();
    }
    
    { 
        std::lock_guard<std::mutex> lock(imageQueueMutex_);
        while (!imageTaskQueue_.empty()) imageTaskQueue_.pop();
    }

    FlightRecorder::getInstance().log("SHUTDOWN", "Scheduler shutdown completed");
    
    initialized_ = false;
    std::cout << "ResourceIsolationScheduler shutdown completed" << std::endl;
}

void ResourceIsolationScheduler::setNativeExecutorEnabled(bool enabled) {
    if (initialized_) {
        return;
    }
    native_executor_enabled_ = enabled;
}

bool ResourceIsolationScheduler::addWorker(std::shared_ptr<IWorker> worker) {
    if (!worker) {
        return false;
    }
    
    try {
        worker->initialize();
        
        { 
            std::lock_guard<std::mutex> lock(queueMutex_);
            workers_.push_back(worker);
            
            // 根据工作器类型分类
            if (worker->canHandle(TaskType::LLM_INFERENCE) || 
                worker->canHandle(TaskType::IMAGE_GENERATION)) {
                gpuWorkers_.push_back(worker);
                
                // 第一个LLM工作器作为专用LLM处理工作器
                if (worker->canHandle(TaskType::LLM_INFERENCE)) {
                    if (!llmWorker_) {
                        llmWorker_ = worker;
                        std::cout << "LLM dedicated worker set: " << worker->getWorkerId() << std::endl;
                    }
                }
            } else if (worker->canHandle(TaskType::TTS_SYNTHESIS)) {
                cpuWorkers_.push_back(worker);
            }
        }
        
        std::cout << "Worker added: " << worker->getWorkerId() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to add worker: " << e.what() << std::endl;
        return false;
    }
}

void ResourceIsolationScheduler::submitTask(std::shared_ptr<ITask> task) {
    if (!task) return;

    FlightRecorder::getInstance().log("SUBMIT", "Task ID: " + task->getTaskId());

    const auto taskType = task->getType();
    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        tasks_[task->getTaskId()] = task;
        totalTasks_++;

        if (taskType == TaskType::LLM_INFERENCE) {
            llmTaskQueue_.push(task);
        } else if (taskType == TaskType::TTS_SYNTHESIS) {
            ttsTaskQueue_.push(task);
        }
    }

    if (taskType == TaskType::IMAGE_GENERATION) {
        {
            std::lock_guard<std::mutex> imgLock(imageQueueMutex_);
            imageTaskQueue_.push(task);
        }
        imageCv_.notify_one();
        return;
    }

    cv_.notify_one();
}

bool ResourceIsolationScheduler::cancelTask(const std::string& taskId) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    auto it = tasks_.find(taskId);
    if (it != tasks_.end()) {
        auto status = it->second->getStatus();
        if (status == TaskStatus::PENDING) {
            it->second->setStatus(TaskStatus::CANCELLED);
            tasks_.erase(it);
            return true;
        }
        if (status == TaskStatus::RUNNING) {
            it->second->setStatus(TaskStatus::CANCELLED);
            tasks_.erase(it);
            return true;
        }
    }
    return false;
}

TaskStatus ResourceIsolationScheduler::getTaskStatus(const std::string& taskId) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    auto it = tasks_.find(taskId);
    if (it != tasks_.end()) {
        return it->second->getStatus();
    }
    return TaskStatus::CANCELLED; // 任务不存在视为已取消
}

std::shared_ptr<ITask> ResourceIsolationScheduler::getTask(const std::string& taskId) {
    std::lock_guard<std::mutex> lock(queueMutex_);
    auto it = tasks_.find(taskId);
    if (it != tasks_.end()) {
        return it->second;
    }
    return nullptr;
}

ResourceIsolationScheduler::SystemStatus ResourceIsolationScheduler::getSystemStatus() {
    SystemStatus status;
    
    std::lock_guard<std::mutex> lock(queueMutex_);
    status.totalTasks = totalTasks_;
    status.completedTasks = completedTasks_;
    status.failedTasks = failedTasks_;
    
    // 计算等待和运行中的任务数
    status.pendingTasks = 0;
    status.runningTasks = 0;
    
    for (const auto& [id, task] : tasks_) {
        if (task->getStatus() == TaskStatus::PENDING) {
            status.pendingTasks++;
        } else if (task->getStatus() == TaskStatus::RUNNING) {
            status.runningTasks++;
        }
    }
    
    // 获取工作器状态
    for (const auto& worker : workers_) {
        status.workerStatus[worker->getWorkerId()] = worker->isBusy();
    }
    
    return status;
}

void ResourceIsolationScheduler::waitForAllTasks() {
    while (true) {
        bool has_queued = false;
        bool has_pending_or_running = false;
        {
            std::scoped_lock lock(queueMutex_, imageQueueMutex_);
            has_queued = !llmTaskQueue_.empty() || !ttsTaskQueue_.empty() || !imageTaskQueue_.empty();
            for (const auto& kv : tasks_) {
                const auto st = kv.second ? kv.second->getStatus() : TaskStatus::FAILED;
                if (st == TaskStatus::PENDING || st == TaskStatus::RUNNING) {
                    has_pending_or_running = true;
                    break;
                }
            }
        }

        if (!has_queued && !has_pending_or_running) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

ResourceIsolationScheduler::ResourceUsage ResourceIsolationScheduler::getResourceUsage() {
    ResourceUsage usage;
    // 这里可以实现实际的资源监控逻辑
    // 目前返回默认值
    usage.cpuUsage = 0.0f;
    usage.gpuUsage = 0.0f;
    usage.memoryUsage = 0;
    usage.gpuMemoryUsage = 0;
    return usage;
}

void ResourceIsolationScheduler::processTaskQueues() {
    std::shared_ptr<ITask> task = nullptr;
    TaskType taskType = TaskType::TTS_SYNTHESIS; // 默认CPU任务
    
    // 尝试获取任务
    {
        std::unique_lock<std::mutex> lock(queueMutex_);

        cv_.wait(lock, [this]() {
            return !running_ || !llmTaskQueue_.empty() || !ttsTaskQueue_.empty();
        });

        if (!running_) {
            return;
        }

        if (!llmTaskQueue_.empty()) {
            task = llmTaskQueue_.front();
            llmTaskQueue_.pop();
            taskType = TaskType::LLM_INFERENCE;
        } else if (!ttsTaskQueue_.empty()) {
            task = ttsTaskQueue_.front();
            ttsTaskQueue_.pop();
            taskType = TaskType::TTS_SYNTHESIS;
        }

        if (!task) {
            return;
        }

        if (task->getStatus() == TaskStatus::CANCELLED) {
            tasks_.erase(task->getTaskId());
            return;
        }
    }
    
    // 选择合适的工作器处理任务
    std::shared_ptr<IWorker> worker = selectWorker(taskType);
    if (worker) {
        try {
            // 处理任务
            worker->processTask(task);
            
            // 更新统计信息
            if (task->getStatus() == TaskStatus::COMPLETED) {
                completedTasks_++;
            } else if (task->getStatus() == TaskStatus::FAILED) {
                failedTasks_++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing task " << task->getTaskId() << ": " << e.what() << std::endl;
            task->setStatus(TaskStatus::FAILED);
            failedTasks_++;
        }
    } else {
        // 如果没有合适的工作器，将任务放回队列
        std::lock_guard<std::mutex> lock(queueMutex_);
        switch (taskType) {
            case TaskType::LLM_INFERENCE:
                llmTaskQueue_.push(task);
                break;
            case TaskType::TTS_SYNTHESIS:
                ttsTaskQueue_.push(task);
                break;
            default:
                break;
        }
        
        // 短暂休眠避免忙等待
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    
    // 从任务映射中移除完成的任务
    if (task->getStatus() == TaskStatus::CANCELLED) {
        std::lock_guard<std::mutex> lock(queueMutex_);
        tasks_.erase(task->getTaskId());
    }
}

std::shared_ptr<IWorker> ResourceIsolationScheduler::selectWorker(TaskType type) {
    std::lock_guard<std::mutex> lock(queueMutex_);

    // 如果有 phase-aware 策略，使用策略选择 Worker
    if (phaseStrategy_) {
        // 查找任务的阶段属性
        PhaseAttributes default_attrs;
        // 根据 TaskType 推断默认属性
        if (type == TaskType::LLM_INFERENCE) {
            default_attrs = getPhasePreset("llm");
        } else if (type == TaskType::IMAGE_GENERATION) {
            default_attrs = getPhasePreset("image");
        } else if (type == TaskType::TTS_SYNTHESIS) {
            default_attrs = getPhasePreset("tts");
        }

        auto worker = phaseStrategy_->selectWorkerForPhase(
            default_attrs, gpuWorkers_, cpuWorkers_, llmWorker_);
        if (worker) return worker;
        // 策略没找到 Worker，回退到默认逻辑
    }

    // 默认逻辑：LLM 任务使用专用工作器
    if (type == TaskType::LLM_INFERENCE && llmWorker_ && !llmWorker_->isBusy()) {
        return llmWorker_;
    }

    // 根据任务类型选择工作器
    if (type == TaskType::TTS_SYNTHESIS) {
        // 查找空闲的CPU工作器
        for (const auto& worker : cpuWorkers_) {
            if (worker->canHandle(type) && !worker->isBusy()) {
                return worker;
            }
        }
    } else {
        // 查找空闲的GPU工作器（不包括LLM专用工作器）
        for (const auto& worker : gpuWorkers_) {
            if (worker != llmWorker_ && worker->canHandle(type) && !worker->isBusy()) {
                return worker;
            }
        }
    }

    return nullptr; // 没有找到合适的工作器
}

void ResourceIsolationScheduler::processImageGenerationQueue() {
    std::shared_ptr<ITask> task = nullptr;
    
    {
        std::unique_lock<std::mutex> lock(imageQueueMutex_);
        imageCv_.wait(lock, [this]() { return !running_ || !imageTaskQueue_.empty(); });

        if (!running_) {
            return;
        }

        task = imageTaskQueue_.front();
        imageTaskQueue_.pop();
    }

    std::shared_ptr<IWorker> worker = nullptr;
    {
        std::lock_guard<std::mutex> lock(queueMutex_);

        auto it = tasks_.find(task->getTaskId());
        if (it == tasks_.end() || task->getStatus() == TaskStatus::CANCELLED) {
            return;
        }

        for (const auto& w : gpuWorkers_) {
            if (w != llmWorker_ && w->canHandle(TaskType::IMAGE_GENERATION) && !w->isBusy()) {
                worker = w;
                break;
            }
        }
    }

    if (worker) {
        try {
            std::cout << "Processing image generation task on worker: " << worker->getWorkerId() << std::endl;
            worker->processTask(task);

            if (task->getStatus() == TaskStatus::COMPLETED) {
                completedTasks_++;
            } else if (task->getStatus() == TaskStatus::FAILED) {
                failedTasks_++;
            }
        } catch (const std::exception& e) {
            std::cerr << "Error processing image task " << task->getTaskId() << ": " << e.what() << std::endl;
            task->setStatus(TaskStatus::FAILED);
            failedTasks_++;
        }
    } else {
        {
            std::lock_guard<std::mutex> imgLock(imageQueueMutex_);
            imageTaskQueue_.push(task);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    if (task->getStatus() == TaskStatus::CANCELLED) {
        std::lock_guard<std::mutex> lock(queueMutex_);
        tasks_.erase(task->getTaskId());
    }
}

} // namespace ai_scheduler

// ============================================================
// Phase-Aware 调度接口实现
// ============================================================

namespace ai_scheduler {

void ResourceIsolationScheduler::setPhaseStrategy(std::shared_ptr<IPhaseAwareStrategy> strategy) {
    phaseStrategy_ = std::move(strategy);
    std::cout << "Phase-Aware strategy set" << std::endl;
}

void ResourceIsolationScheduler::submitPhaseTask(std::shared_ptr<ITask> task, const PhaseAttributes& attrs) {
    if (!task) return;

    // 保存阶段属性
    {
        std::lock_guard<std::mutex> lock(phaseMutex_);
        phaseAttributesMap_[task->getTaskId()] = attrs;
    }

    // 如果有 phase-aware 策略，根据路由决策分发任务
    if (phaseStrategy_) {
        std::string route = phaseStrategy_->routeTask(
            attrs,
            phaseStrategy_->getLLMState(),
            SchedulerPolicy::V2_PHASE_AWARE  // 默认使用 V2 策略
        );

        if (route == "cpu_backfill") {
            // CPU 任务：直接提交到 TTS 队列（CPU Worker 处理）
            task->setStatus(TaskStatus::PENDING);
            {
                std::lock_guard<std::mutex> lock(queueMutex_);
                tasks_[task->getTaskId()] = task;
                totalTasks_++;
                ttsTaskQueue_.push(task);
            }
            cv_.notify_one();
            return;
        }
        // gpu_burst / resident_gpu / sequential：走标准提交
    }

    // 标准提交
    submitTask(task);
}

LLMState ResourceIsolationScheduler::getLLMState() const {
    if (phaseStrategy_) {
        return phaseStrategy_->getLLMState();
    }
    return LLMState{};
}

bool ResourceIsolationScheduler::migrateLLMToCPU(bool urgent) {
    if (phaseStrategy_) {
        return phaseStrategy_->migrateLLMToCPU(urgent);
    }
    // 直接调用回调
    if (migrateLLMFn_) {
        return migrateLLMFn_(urgent);
    }
    return false;
}

bool ResourceIsolationScheduler::restoreLLMToGPU() {
    if (phaseStrategy_) {
        return phaseStrategy_->restoreLLMToGPU();
    }
    if (restoreLLMFn_) {
        return restoreLLMFn_();
    }
    return false;
}

void ResourceIsolationScheduler::setMigrationCallbacks(
    std::function<bool(bool urgent)> migrate_fn,
    std::function<bool()> restore_fn)
{
    migrateLLMFn_ = std::move(migrate_fn);
    restoreLLMFn_ = std::move(restore_fn);

    // 同时注入到 phase-aware 策略
    if (phaseStrategy_) {
        phaseStrategy_->setMigrationCallbacks(migrateLLMFn_, restoreLLMFn_);
    }
}

} // namespace ai_scheduler
