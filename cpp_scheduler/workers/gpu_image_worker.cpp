#define NOMINMAX
#include "gpu_image_worker.h"
#include "../queue/task_queue.h"
#include <iostream>
#include <thread>
#include <chrono>

namespace ai_scheduler {

GPUImageWorker::GPUImageWorker(int gpu_id, ImageModel default_model)
    : workerId_("gpu_image_worker_" + std::to_string(gpu_id)) {
}

GPUImageWorker::~GPUImageWorker() {
    shutdown();
}

bool GPUImageWorker::initialize() {
    std::cout << "GPU Image Worker initialized: " << workerId_ << std::endl;
    return true;
}

void GPUImageWorker::shutdown() {
    std::cout << "GPU Image Worker shutdown: " << workerId_ << std::endl;
}

void GPUImageWorker::processTask(std::shared_ptr<ITask> task) {
    if (!task) return;
    
    auto imgTask = std::dynamic_pointer_cast<ImageTask>(task);
    if (imgTask) {
        imgTask->setStatus(TaskStatus::RUNNING);
        try {
            imgTask->setWorker(this);
            imgTask->execute();
            imgTask->setStatus(TaskStatus::COMPLETED);
        } catch (...) {
            imgTask->setStatus(TaskStatus::FAILED);
        }
    }
}

std::string GPUImageWorker::getWorkerId() const {
    return workerId_;
}

bool GPUImageWorker::isBusy() const {
    // 简单模拟，实际可能有内部状态
    return false;
}

bool GPUImageWorker::isReady() const {
    return true;
}

void GPUImageWorker::cleanup() {
    // Cleanup resources
}

void GPUImageWorker::executeTask(const std::string& input, Callback callback) {
    // Async execution simulation
    std::thread([this, input, callback]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        callback(true, "generated_image.png");
    }).detach();
}

uint64_t GPUImageWorker::generateImage(const ImageGenerationParams& params, Callback callback) {
    // Sync wrapper or async launch
    std::thread([this, params, callback]() {
        std::string output = "output_" + std::to_string(std::chrono::system_clock::now().time_since_epoch().count()) + ".png";
        std::this_thread::sleep_for(std::chrono::seconds(2)); // Mock generation
        callback(true, output);
    }).detach();
    return 1;
}

bool GPUImageWorker::generateSync(const ImageGenerationParams& params, std::string& output_path) {
    std::this_thread::sleep_for(std::chrono::seconds(2));
    output_path = "output_sync.png";
    return true;
}

bool GPUImageWorker::cancelTask(uint64_t task_id) {
    return true;
}

std::size_t GPUImageWorker::getQueueSize() const {
    return 0;
}

std::size_t GPUImageWorker::getRunningTaskCount() const {
    return 0;
}

float GPUImageWorker::getGPUUtilization() const {
    return 0.5f; // Mock
}

// ImageTask execute implementation
void ImageTask::execute() {
    if (worker_) {
        std::string output;
        if (worker_->generateSync(params_, output)) {
            // Success
             // Store result in outputPath_?
             // ImageTask needs a way to store result. 
             // Accessing private member via friend or public setter?
             // ImageTask has no setter for outputPath_, but getResult returns it.
             // Wait, ITask::getResult is pure virtual. ImageTask implements it.
             // But ImageTask needs to store the result.
             // Looking at ImageTask definition in header:
             // std::string outputPath_;
             // It's private.
             // But ImageTask::execute() is a member function, so it can access outputPath_.
             outputPath_ = output;
        } else {
            errorMessage_ = "Generation failed";
            setStatus(TaskStatus::FAILED);
            throw std::runtime_error("Generation failed");
        }
    } else {
        errorMessage_ = "No worker assigned";
        setStatus(TaskStatus::FAILED);
        throw std::runtime_error("No worker assigned");
    }
}

} // namespace ai_scheduler
