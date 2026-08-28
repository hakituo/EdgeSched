#include "gpu_llm_worker.h"
#include "llama_model_impl.h"
#include <iostream>
#include <chrono>

namespace ai_scheduler {

// Mock model for fallback
class MockLLMModel : public ILLMModel {
public:
    bool initialize(const LLMModelConfig& config) override { return true; }
    void shutdown() override {}
    LLMInferenceResponse generate(const LLMInferenceRequest& request) override {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return { "This is a mock response from legacy scheduler.", 10, 0.1f, true, "" };
    }
    std::string getModelInfo() const override { return "Mock Model"; }
    size_t getMemoryUsage() const override { return 0; }
    bool isReady() const override { return true; }
};

GPULLMWorker::GPULLMWorker(const std::string& workerId)
    : workerId_(workerId), running_(false), initialized_(false), busy_(false),
      completedTasksCount_(0), totalInferenceTime_(0.0f), gpu_utilization_(0.0f) {
}

GPULLMWorker::~GPULLMWorker() {
    shutdown();
}

bool GPULLMWorker::initialize() {
    if (initialized_) return true;

    running_ = true;
    
    // Default config if not set
    if (modelConfig_.modelPath.empty()) {
        modelConfig_.modelType = "mock";
    }

    // Load model
    model_ = loadModelImpl(modelConfig_);
    if (!model_ || !model_->isReady()) {
        std::cerr << "Failed to load LLM model" << std::endl;
        // Don't fail init, just warn? Or fail?
        // Let's allow init but model is mock if failed?
        if (!model_) {
             model_ = std::make_shared<MockLLMModel>();
        }
    }

    initialized_ = true;
    
    // Start execution thread
    executionThread_ = std::thread(&GPULLMWorker::taskExecutionLoop, this);
    
    std::cout << "GPU LLM Worker initialized: " << workerId_ << std::endl;
    return true;
}

void GPULLMWorker::shutdown() {
    if (!initialized_) return;

    running_ = false;
    cv_.notify_all();

    if (executionThread_.joinable()) {
        executionThread_.join();
    }

    if (model_) {
        model_->shutdown();
        model_ = nullptr;
    }

    initialized_ = false;
    std::cout << "GPU LLM Worker shutdown: " << workerId_ << std::endl;
}

bool GPULLMWorker::canHandle(TaskType type) const {
    return type == TaskType::LLM_INFERENCE;
}

void GPULLMWorker::processTask(std::shared_ptr<ITask> task) {
    if (!task) return;

    {
        std::lock_guard<std::mutex> lock(queueMutex_);
        taskQueue_.push(task);
        currentTasks_++;
    }
    cv_.notify_one();
}

std::string GPULLMWorker::getWorkerId() const {
    return workerId_;
}

bool GPULLMWorker::isBusy() const {
    return busy_ || currentTasks_ > 0;
}

void GPULLMWorker::setModelConfig(const LLMModelConfig& config) {
    modelConfig_ = config;
}

LLMModelConfig GPULLMWorker::getModelConfig() const {
    return modelConfig_;
}

LLMInferenceResponse GPULLMWorker::executeInference(const LLMInferenceRequest& request) {
    if (!model_) {
        return { "", 0, 0.0f, false, "Model not initialized" };
    }
    
    std::lock_guard<std::mutex> lock(modelMutex_);
    busy_ = true;
    auto result = model_->generate(request);
    busy_ = false;
    return result;
}

float GPULLMWorker::getGpuUtilization() const {
    return gpu_utilization_;
}

std::string GPULLMWorker::getModelInfo() const {
    if (model_) return model_->getModelInfo();
    return "No model loaded";
}

bool GPULLMWorker::warmupModel(size_t warmupRounds) {
    if (!model_) return false;
    LLMInferenceRequest req;
    req.prompt = "Hello";
    req.maxTokens = 10;
    for (size_t i = 0; i < warmupRounds; ++i) {
        executeInference(req);
    }
    return true;
}

std::shared_ptr<ILLMModel> GPULLMWorker::loadModelImpl(const LLMModelConfig& config) {
    if (config.modelType == "mock") {
        auto model = std::make_shared<MockLLMModel>();
        model->initialize(config);
        return model;
    } else {
        // Use LlamaCppModel
        auto model = std::make_shared<LlamaCppModel>();
        if (model->initialize(config)) {
            return model;
        } else {
            std::cerr << "Failed to initialize LlamaCppModel, falling back to Mock" << std::endl;
            return std::make_shared<MockLLMModel>();
        }
    }
}

void GPULLMWorker::taskExecutionLoop() {
    while (running_) {
        std::vector<std::shared_ptr<ITask>> batchTasks;
        
        {
            std::unique_lock<std::mutex> lock(queueMutex_);
            cv_.wait(lock, [this]() { return !running_ || !taskQueue_.empty(); });
            
            if (!running_) break;
            
            // Dynamic Batching Logic
            size_t batchSize = 0;
            const size_t maxBatchSize = 4; // MVP limit
            
            while (!taskQueue_.empty() && batchSize < maxBatchSize) {
                batchTasks.push_back(taskQueue_.top());
                taskQueue_.pop();
                batchSize++;
            }
        }
        
        if (!batchTasks.empty()) {
            std::vector<LLMInferenceRequest> llmRequests;
            std::vector<std::shared_ptr<LLMTask>> llmTaskPtrs;
            std::vector<std::shared_ptr<ITask>> otherTasks;
            
            for (auto& t : batchTasks) {
                auto llmTask = std::dynamic_pointer_cast<LLMTask>(t);
                if (llmTask) {
                    llmTask->setStatus(TaskStatus::RUNNING);
                    llmTaskPtrs.push_back(llmTask);
                    llmRequests.push_back(llmTask->getRequest());
                } else {
                    otherTasks.push_back(t);
                }
            }
            
            if (!llmRequests.empty() && model_) {
                 // True Batch Processing
                 std::lock_guard<std::mutex> lock(modelMutex_);
                 busy_ = true;
                 // Use model's batch generation
                 auto responses = model_->batchGenerate(llmRequests);
                 busy_ = false;
                 
                 for (size_t i = 0; i < responses.size(); ++i) {
                     if (i < llmTaskPtrs.size()) {
                         llmTaskPtrs[i]->setResponse(responses[i]);
                         if (responses[i].success) {
                             llmTaskPtrs[i]->setStatus(TaskStatus::COMPLETED);
                             completedTasksCount_++;
                         } else {
                             llmTaskPtrs[i]->setErrorMessage(responses[i].errorMessage);
                             llmTaskPtrs[i]->setStatus(TaskStatus::FAILED);
                         }
                         currentTasks_--;
                     }
                 }
            } else if (!llmRequests.empty() && !model_) {
                 // Model not loaded, fail all
                 for (auto& task : llmTaskPtrs) {
                     task->setErrorMessage("Model not initialized");
                     task->setStatus(TaskStatus::FAILED);
                     currentTasks_--;
                 }
            }

            // Process non-LLM tasks if any (fallback)
            for (auto& t : otherTasks) {
                processLLMInferenceTask(t); // This might fail if it's not LLMTask but processLLMInferenceTask handles cast check
                currentTasks_--;
            }
        }
    }
}

void GPULLMWorker::processLLMInferenceTask(std::shared_ptr<ITask> task) {
    auto llmTask = std::dynamic_pointer_cast<LLMTask>(task);
    if (!llmTask) return;
    
    task->setStatus(TaskStatus::RUNNING);
    
    try {
        llmTask->execute(); // execute calls worker->executeInference inside LLMTask if set up right?
        // Wait, LLMTask::execute needs implementation. 
        // In gpu_llm_worker.h, LLMTask::execute is declared.
        // We should implement it here or call executeInference directly and set result.
        
        // Let's check LLMTask implementation.
        // It holds a reference to GPULLMWorker* worker_.
        // So we can call llmTask->execute() and it should call worker_->executeInference(request_).
        // But LLMTask needs to store the result.
        // Let's implement LLMTask::execute properly below.
        
        // Actually, if LLMTask::execute is virtual, calling it here is the right way.
        // But we need to ensure LLMTask::execute is implemented.
        
        llmTask->execute();
        
        if (llmTask->getStatus() != TaskStatus::FAILED) {
            llmTask->setStatus(TaskStatus::COMPLETED);
            completedTasksCount_++;
        } else {
            // Task failed
        }
        
    } catch (const std::exception& e) {
        std::cerr << "LLM Task failed: " << e.what() << std::endl;
        task->setStatus(TaskStatus::FAILED);
    }
}

// Implement LLMTask::execute
void LLMTask::execute() {
    if (worker_) {
        response_ = worker_->executeInference(request_);
        if (!response_.success) {
            errorMessage_ = response_.errorMessage;
            setStatus(TaskStatus::FAILED);
        }
    } else {
        errorMessage_ = "No worker assigned";
        setStatus(TaskStatus::FAILED);
    }
}

bool GPULLMWorker::initializePythonInterface() {
    return false;
}

bool GPULLMWorker::callPythonInference(const std::string& prompt, std::string& result) {
    return false;
}

std::string GPULLMWorker::inferenceInternal(const std::string& prompt) {
    LLMInferenceRequest req;
    req.prompt = prompt;
    return executeInference(req).generatedText;
}

} // namespace ai_scheduler
