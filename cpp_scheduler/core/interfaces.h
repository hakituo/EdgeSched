#pragma once

/**
 * 统一接口定义
 *
 * 注意：TaskType/TaskPriority/TaskStatus/ITask/IWorker 的主定义
 * 在 resource_isolation_scheduler.h 中。本文件只补充独有的定义。
 */

#include "resource_isolation_scheduler.h"

namespace ai_scheduler {

// 资源消耗定义（本文件独有）
struct ResourceCost {
    size_t cpu_cores = 0;
    size_t ram_mb = 0;
    size_t vram_mb = 0;
    float gpu_compute = 0.0f; // 0.0 - 1.0
};

// 为 ITask 扩展资源消耗预估（非侵入式，通过自由函数）
inline ResourceCost getEstimatedCost(const ITask& /*task*/) {
    return ResourceCost{};
}

// 为 IWorker 扩展资源占用查询（非侵入式，通过自由函数）
inline ResourceCost getResourceUsage(const IWorker& /*worker*/) {
    return ResourceCost{};
}

} // namespace ai_scheduler
