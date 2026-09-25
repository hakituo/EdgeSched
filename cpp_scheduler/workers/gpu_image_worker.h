#pragma once

#include "../core/resource_isolation_scheduler.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <atomic>

namespace ai_scheduler {

class TaskQueue;

enum class ImageModel {
    SD15_TURBO,
    SDXL_TURBO,
    MOBILE_DIFFUSION,
    SVD
};

struct ImageGenerationParams {
    std::string prompt;
    std::string negative_prompt;
    int width;
    int height;
    float guidance_scale;
    int steps;
    int seed;
    ImageModel model;

    ImageGenerationParams()
        : width(512), height(512), guidance_scale(7.5f), steps(20), seed(-1), model(ImageModel::SD15_TURBO) {
    }
};

class GPUImageWorker;

class ImageTask : public ITask {
public:
    ImageTask(const ImageGenerationParams& params, GPUImageWorker* worker = nullptr)
        : ITask("img_task_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()), 
                TaskType::IMAGE_GENERATION, TaskPriority::MEDIUM),
          params_(params), worker_(worker) {
    }
    
    void setWorker(GPUImageWorker* worker) {
        worker_ = worker;
    }

    void execute() override;

    std::shared_ptr<void> getResult() const override {
        if (getStatus() == TaskStatus::FAILED) {
            throw std::runtime_error(errorMessage_);
        }
        return std::make_shared<std::string>(outputPath_);
    }

    const std::string& getOutputPath() const { return outputPath_; }

private:
    ImageGenerationParams params_;
    GPUImageWorker* worker_;
    std::string errorMessage_;
    std::string outputPath_;
};

class GPUImageWorker : public IWorker {
public:
    using Callback = std::function<void(bool success, const std::string& result)>;

    GPUImageWorker(int gpu_id = 1, ImageModel default_model = ImageModel::SD15_TURBO);
    ~GPUImageWorker() override;

    bool initialize() override;
    void shutdown() override;
    bool canHandle(TaskType type) const override {
        return type == TaskType::IMAGE_GENERATION;
    }
    void processTask(std::shared_ptr<ITask> task) override;
    std::string getWorkerId() const override;
    bool isBusy() const override;

    void cleanup();
    void executeTask(const std::string& input, Callback callback);
    bool isReady() const override;

    uint64_t generateImage(const ImageGenerationParams& params, Callback callback);
    
    bool generateSync(const ImageGenerationParams& params, std::string& output_path);

    bool cancelTask(uint64_t task_id);

    std::size_t getQueueSize() const;
    std::size_t getRunningTaskCount() const;
    float getGPUUtilization() const;

    void setDefaultModel(ImageModel model);

private:
    bool generateInternal(const ImageGenerationParams& params, std::string& output_path);
    bool initializePythonInterface();
    bool callPythonGenerate(const ImageGenerationParams& params, std::string& output_path);
    std::string modelToString(ImageModel model) const;

    std::string workerId_;
    std::atomic<bool> initialized_;
    int gpu_id_;
    ImageModel default_model_;
    std::unique_ptr<TaskQueue> task_queue_;

    void* python_module_;
    void* python_generate_func_;

    float gpu_utilization_;
    std::size_t gpu_memory_usage_;
    uint64_t total_generation_time_;
    uint64_t generation_count_;
    std::string output_dir_;
};

} // namespace ai_scheduler
