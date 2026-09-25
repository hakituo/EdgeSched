#include "npu_worker.h"
#include <iostream>
#include <thread>

namespace ai_scheduler {

NPUWorker::NPUWorker(const std::string& workerId) 
    : workerId_(workerId), busy_(false), initialized_(false) {}

NPUWorker::~NPUWorker() {
    shutdown();
}

bool NPUWorker::initialize() {
    if (initialized_) return true;

    std::cout << "NPUWorker initializing: " << workerId_ << std::endl;
    
    // 模拟检查 NPU 驱动
    bool npuAvailable = true; 
    if (npuAvailable) {
        std::cout << "NPU device detected (Mock)" << std::endl;
    } else {
        std::cerr << "NPU device not found or driver missing" << std::endl;
        return false;
    }

    initialized_ = true;
    return true;
}

void NPUWorker::shutdown() {
    initialized_ = false;
}

bool NPUWorker::canHandle(TaskType type) const {
    return type == TaskType::NPU_INFERENCE;
}

void NPUWorker::processTask(std::shared_ptr<ITask> task) {
    if (!task) return;

    busy_ = true;
    try {
        task->setStatus(TaskStatus::RUNNING);
        task->execute();
        task->setStatus(TaskStatus::COMPLETED);
    } catch (const std::exception& e) {
        std::cerr << "NPUWorker execution failed: " << e.what() << std::endl;
        task->setStatus(TaskStatus::FAILED);
    } catch (...) {
        std::cerr << "NPUWorker execution failed with unknown error" << std::endl;
        task->setStatus(TaskStatus::FAILED);
    }
    busy_ = false;
}

std::string NPUWorker::getWorkerId() const {
    return workerId_;
}

bool NPUWorker::isBusy() const {
    return busy_;
}

bool NPUWorker::isReady() const {
    return initialized_;
}

}
