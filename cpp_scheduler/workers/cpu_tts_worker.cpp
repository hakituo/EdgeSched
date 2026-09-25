#define NOMINMAX
#include "cpu_tts_worker.h"
#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <algorithm>

namespace ai_scheduler {

CPUTTSWorker::CPUTTSWorker(const std::string& workerId, TTSEngineType engineType, int numThreads)
    : workerId_(workerId), engineType_(engineType), numThreads_(numThreads),
      running_(false), initialized_(false), busy_(false),
      activeTasksCount_(0), completedTasksCount_(0),
      totalSynthesisTime_(0), synthesisCount_(0), cpuUtilization_(0.0f) {
}

CPUTTSWorker::~CPUTTSWorker() {
    shutdown();
}

bool CPUTTSWorker::initialize() {
    if (initialized_) return true;
    
    running_ = true;
    initialized_ = true;
    busy_ = false;
    
    // 启动处理线程
    executionThread_ = std::thread([this]() {
        processTaskQueue();
    });
    
    std::cout << "CPU TTS Worker initialized: " << workerId_ << " (Threads: " << numThreads_ << ")" << std::endl;
    return true;
}

void CPUTTSWorker::shutdown() {
    if (!initialized_) return;
    
    running_ = false;
    taskQueueCondition_.notify_all();
    
    if (executionThread_.joinable()) {
        executionThread_.join();
    }
    
    initialized_ = false;
    std::cout << "CPU TTS Worker shutdown: " << workerId_ << std::endl;
}

bool CPUTTSWorker::canHandle(TaskType type) const {
    return type == TaskType::TTS_SYNTHESIS;
}

void CPUTTSWorker::processTask(std::shared_ptr<ITask> task) {
    if (!task) return;
    
    {
        std::lock_guard<std::mutex> lock(taskQueueMutex_);
        taskQueue_.push(task);
    }
    taskQueueCondition_.notify_one();
}

std::string CPUTTSWorker::getWorkerId() const {
    return workerId_;
}

bool CPUTTSWorker::isBusy() const {
    return busy_;
}

bool CPUTTSWorker::isReady() const {
    return initialized_ && running_;
}

void CPUTTSWorker::processTaskQueue() {
    while (running_) {
        std::shared_ptr<ITask> task = nullptr;
        
        {
            std::unique_lock<std::mutex> lock(taskQueueMutex_);
            taskQueueCondition_.wait(lock, [this]() { return !running_ || !taskQueue_.empty(); });
            
            if (!running_) break;
            
            task = taskQueue_.front();
            taskQueue_.pop();
        }
        
        if (task) {
            busy_ = true;
            activeTasksCount_++;
            
            // 执行TTS任务
            auto ttsTask = std::dynamic_pointer_cast<TTSTask>(task);
            if (ttsTask) {
                task->setStatus(TaskStatus::RUNNING);
                
                try {
                    std::string outputPath;
                    std::vector<uint8_t> audioData;
                    if (synthesizeSync(ttsTask->getParams(), outputPath, audioData)) {
                        ttsTask->setAudioOutput(outputPath, audioData);
                        task->setStatus(TaskStatus::COMPLETED);
                        completedTasksCount_++;
                    } else {
                        task->setStatus(TaskStatus::FAILED);
                    }
                } catch (const std::exception& e) {
                    std::cerr << "TTS synthesis failed: " << e.what() << std::endl;
                    task->setStatus(TaskStatus::FAILED);
                }
            }
            
            activeTasksCount_--;
            busy_ = false;
            
            if (taskCompletedCallback_) {
                taskCompletedCallback_(task);
            }
        }
    }
}

bool CPUTTSWorker::synthesizeSync(const TTSParams& params, std::string& outputPath, std::vector<uint8_t>& audioData) {
    auto start = std::chrono::high_resolution_clock::now();
    
    // 模拟合成过程 (根据文本长度)
    size_t durationMs = params.text.length() * 50; // 50ms per char
    if (durationMs < 500) durationMs = 500;
    
    std::this_thread::sleep_for(std::chrono::milliseconds(durationMs));
    
    // 生成模拟音频数据
    audioData.resize(1024, 0); // Mock data
    outputPath = "output_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".wav";
    
    auto end = std::chrono::high_resolution_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    
    totalSynthesisTime_ += elapsed;
    synthesisCount_++;
    
    return true;
}

// Implement other pure virtuals or required methods from header
void CPUTTSWorker::synthesize(const TTSParams& params, TaskPriority priority, std::function<void(std::shared_ptr<TTSTask>)> callback) {
    // Helper to create task and submit
    auto task = std::make_shared<TTSTask>("tts_task_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()), priority, params);
    processTask(task);
    // Callback handling would need more complex logic to map task to callback
}

std::vector<std::string> CPUTTSWorker::getAvailableVoices() const {
    return {"voice_1", "voice_2"};
}

void CPUTTSWorker::setEngineType(TTSEngineType engineType) {
    engineType_ = engineType;
}

void CPUTTSWorker::setNumThreads(int numThreads) {
    numThreads_ = numThreads;
}

float CPUTTSWorker::getAverageSynthesisTime() const {
    if (synthesisCount_ == 0) return 0.0f;
    return (float)totalSynthesisTime_ / synthesisCount_;
}

float CPUTTSWorker::getCPUUtilization() const {
    return cpuUtilization_;
}

size_t CPUTTSWorker::getActiveTasksCount() const {
    return activeTasksCount_;
}

size_t CPUTTSWorker::getCompletedTasksCount() const {
    return completedTasksCount_;
}

std::unique_ptr<ITTSModel> CPUTTSWorker::createModel(TTSEngineType engineType) {
    return nullptr; // Mock
}

bool CPUTTSWorker::synthesizeInternal(const TTSParams& params, std::string& outputPath, std::vector<uint8_t>& audioData) {
    return synthesizeSync(params, outputPath, audioData);
}

std::string CPUTTSWorker::generateOutputFilename(const std::string& prefix) {
    return prefix + ".wav";
}

void CPUTTSWorker::updateCPUUtilization() {
    // Mock
    cpuUtilization_ = 0.1f;
}

} // namespace ai_scheduler
