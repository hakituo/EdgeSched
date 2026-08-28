#ifndef WORKER_BASE_H
#define WORKER_BASE_H

#include "../core/interfaces.h"
#include <string>
#include <memory>
#include <functional>
#include <atomic>

namespace ai_scheduler {

// 工作器基类
class WorkerBase : public IWorker {
public:
    using Callback = std::function<void(bool success, const std::string& result)>;
    
    WorkerBase(const std::string& name);
    virtual ~WorkerBase() = default;
    
    // 初始化工作器 (from IWorker)
    virtual bool initialize() override = 0;
    
    // 清理资源
    virtual void cleanup() = 0;

    // 关闭工作器 (from IWorker)
    virtual void shutdown() override {
        cleanup();
    }
    
    // 执行任务 (Legacy interface, keep for now)
    virtual void executeTask(const std::string& input, Callback callback) = 0;
    
    // 处理任务 (from IWorker)
    // Default implementation does nothing or throws, subclasses must implement if they support ITask
    virtual void processTask(std::shared_ptr<ITask> task) override {
        // Default implementation: try to execute if it's a closure task
        if (task) {
            task->execute();
        }
    }

    // 检查是否能处理特定类型的任务 (from IWorker)
    virtual bool canHandle(TaskType /*type*/) const override {
        return false; // Default
    }

    // 获取工作器状态
    virtual bool isReady() const = 0;

    // 检查工作器是否忙碌 (from IWorker)
    virtual bool isBusy() const override {
        return !isReady(); // Default implementation, subclasses can override
    }
    
    // 获取工作器名称
    const std::string& getName() const { return name_; }

    // 获取工作器ID (from IWorker)
    virtual std::string getWorkerId() const override {
        return name_;
    }
    
protected:
    std::string name_;
    std::atomic<bool> initialized_;
};

// 工作器工厂类
template <typename T>
class WorkerFactory {
public:
    static std::shared_ptr<T> create() {
        auto worker = std::make_shared<T>();
        if (!worker->initialize()) {
            return nullptr;
        }
        return worker;
    }
};

} // namespace ai_scheduler

#endif // WORKER_BASE_H
