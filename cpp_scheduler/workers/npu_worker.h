#pragma once
#include "../core/resource_isolation_scheduler.h"
#include <memory>
#include <string>
#include <vector>
#include <atomic>
#include <chrono>

namespace ai_scheduler {

// Embedding 任务结构
struct EmbeddingResult {
    std::vector<std::vector<float>> embeddings;
    bool success;
    std::string errorMessage;
};

// Embedding 任务
class EmbeddingTask : public ITask {
public:
    EmbeddingTask(const std::vector<std::string>& texts, const std::string& id = "")
        : ITask(id.empty() ? "emb_task_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) : id, 
               TaskType::NPU_INFERENCE, TaskPriority::HIGH),
          texts_(texts) {}

    void execute() override {
        // NPUWorker will handle the actual execution logic via ORT
    }

    std::shared_ptr<void> getResult() const override {
        return std::make_shared<EmbeddingResult>(result_);
    }

    void setResult(const EmbeddingResult& result) {
        result_ = result;
    }
    
    const std::vector<std::string>& getTexts() const { return texts_; }

private:
    std::vector<std::string> texts_;
    EmbeddingResult result_;
};

// NPU 工作器 - 专门处理 NPU 优化任务 (Embedding/STT)
class NPUWorker : public IWorker {
public:
    NPUWorker(const std::string& workerId = "npu_worker_0");
    ~NPUWorker() override;

    bool initialize() override;
    void shutdown() override;
    
    // NPU Worker 专门处理 NPU_INFERENCE
    bool canHandle(TaskType type) const override;
    
    void processTask(std::shared_ptr<ITask> task) override;
    std::string getWorkerId() const override;
    bool isBusy() const override;
    bool isReady() const override;

private:
    std::string workerId_;
    std::atomic<bool> busy_;
    std::atomic<bool> initialized_;
};

}
