/**
 * Phase-Aware 调度策略实现
 *
 * 核心决策逻辑：
 * 1. CPU-affine 任务（RAG/OCR）：旁路，不触发迁移
 * 2. GPU_PREFERRED 任务（VLM/TTS）：降级 CPU，不迁移 LLM
 * 3. GPU_REQUIRED 任务（Image）：
 *    - LLM 已降级 → 旁路（GPU 已空闲）
 *    - LLM 在推理且快完成 → 等待
 *    - 否则 → 迁移 LLM 释放显存
 */

#include "phase_strategy.h"
#include "resource_isolation_scheduler.h"
#include <algorithm>
#include <iostream>

namespace ai_scheduler {

// ============================================================
// Phase 预设
// ============================================================

PhaseAttributes getPhasePreset(const std::string& phase_name) {
    if (phase_name == "llm") {
        return PhaseAttributes{
            .role = PhaseRole::RESIDENT_MIGRATABLE,
            .device_affinity = DeviceAffinity::GPU_PREFERRED,
            .preemptible = true,
            .queueable = false,
            .migratable = true,
            .vram_usage_mb = 6000,
            .typical_duration_s = 3.0,
            .is_downstream = false,
            .is_upstream = false,
            .can_overlap_with_gpu = false
        };
    } else if (phase_name == "image") {
        return PhaseAttributes{
            .role = PhaseRole::BURST_GPU_PINNED,
            .device_affinity = DeviceAffinity::GPU_REQUIRED,
            .preemptible = false,
            .queueable = true,
            .migratable = false,
            .vram_usage_mb = 4000,
            .typical_duration_s = 7.0,
            .is_downstream = true,
            .is_upstream = false,
            .can_overlap_with_gpu = false
        };
    } else if (phase_name == "vlm") {
        return PhaseAttributes{
            .role = PhaseRole::BURST_GPU_PREFERRED,
            .device_affinity = DeviceAffinity::GPU_PREFERRED,
            .preemptible = true,
            .queueable = true,
            .migratable = false,
            .vram_usage_mb = 2000,
            .typical_duration_s = 2.5,
            .is_downstream = true,
            .is_upstream = false,
            .can_overlap_with_gpu = true
        };
    } else if (phase_name == "tts") {
        return PhaseAttributes{
            .role = PhaseRole::BURST_CPU_LIGHT,
            .device_affinity = DeviceAffinity::CPU_PREFERRED,
            .preemptible = true,
            .queueable = true,
            .migratable = false,
            .vram_usage_mb = 0,
            .typical_duration_s = 2.0,
            .is_downstream = true,
            .is_upstream = false,
            .can_overlap_with_gpu = true
        };
    } else if (phase_name == "rag") {
        return PhaseAttributes{
            .role = PhaseRole::BURST_CPU_LIGHT,
            .device_affinity = DeviceAffinity::CPU_ONLY,
            .preemptible = true,
            .queueable = true,
            .migratable = false,
            .vram_usage_mb = 0,
            .typical_duration_s = 0.8,
            .is_downstream = false,
            .is_upstream = true,
            .can_overlap_with_gpu = true
        };
    } else if (phase_name == "ocr") {
        return PhaseAttributes{
            .role = PhaseRole::BURST_CPU_LIGHT,
            .device_affinity = DeviceAffinity::CPU_ONLY,
            .preemptible = true,
            .queueable = true,
            .migratable = false,
            .vram_usage_mb = 0,
            .typical_duration_s = 1.2,
            .is_downstream = false,
            .is_upstream = true,
            .can_overlap_with_gpu = true
        };
    }

    // 默认：CPU 轻量任务
    return PhaseAttributes{};
}

// ============================================================
// Cost-Aware 决策引擎
// ============================================================

MigrationDecision CostAwareDecisionEngine::decide(
    const PhaseAttributes& burst_attrs,
    const LLMState& llm_state,
    size_t queue_depth,
    double llm_remaining_work_s)
{
    stats_.total_decisions++;
    double pred_duration = burst_attrs.typical_duration_s;

    // 规则 1：CPU-affine 任务旁路，不触发迁移
    if (isCpuAffine(burst_attrs.device_affinity)) {
        stats_.bypass_count++;
        return MigrationDecision{
            .should_migrate = false,
            .reason = "CPU-affine 阶段旁路，不触发迁移",
            .predicted_burst_duration_s = pred_duration,
            .migration_cost = migration_cost_,
            .strategy = "bypass"
        };
    }

    // 规则 2：GPU_PREFERRED 任务降级 CPU，不迁移 LLM
    // 这是 cost-aware 的核心收益：避免迁移 LLM，让小任务走 CPU
    if (burst_attrs.device_affinity == DeviceAffinity::GPU_PREFERRED) {
        stats_.bypass_count++;
        std::string reason = (llm_state.level() == LLMServiceLevel::GPU_NORMAL)
            ? "GPU_PREFERRED 阶段降级 CPU，避免迁移 LLM"
            : "GPU_PREFERRED 阶段走 CPU（LLM 已降级）";
        return MigrationDecision{
            .should_migrate = false,
            .reason = reason,
            .predicted_burst_duration_s = pred_duration,
            .migration_cost = migration_cost_,
            .strategy = "bypass"
        };
    }

    // 规则 3：GPU_REQUIRED 任务必须释放显存
    if (llm_state.level() == LLMServiceLevel::CPU_DEGRADED) {
        // LLM 已降级，GPU 空闲，无需迁移
        stats_.bypass_count++;
        return MigrationDecision{
            .should_migrate = false,
            .reason = "LLM 已在 CPU 降级状态，GPU 空闲",
            .predicted_burst_duration_s = pred_duration,
            .migration_cost = migration_cost_,
            .strategy = "bypass"
        };
    }

    // LLM 在 GPU 上，需要决策
    double migration_cost_s = migration_cost_.totalRoundTripMs() / 1000.0;

    // 规则 3a：LLM 在推理且快完成，短暂等待比迁移划算
    if (llm_remaining_work_s > 0
        && llm_remaining_work_s < short_burst_threshold_s_
        && queue_depth == 0) {
        stats_.wait_count++;
        return MigrationDecision{
            .should_migrate = false,
            .reason = "LLM 即将完成，短暂等待比迁移划算",
            .predicted_burst_duration_s = pred_duration,
            .migration_cost = migration_cost_,
            .wait_cost_s = llm_remaining_work_s,
            .benefit_s = llm_remaining_work_s - migration_cost_s,
            .strategy = "wait"
        };
    }

    // 规则 3b：GPU_REQUIRED 必须迁移（显存被 LLM 占用，burst 无法执行）
    // 这是物理约束：8GB 显存无法同时容纳 LLM + Image
    stats_.migrate_count++;
    return MigrationDecision{
        .should_migrate = true,
        .reason = "GPU_REQUIRED 阶段需要显存，迁移 LLM 释放 GPU",
        .predicted_burst_duration_s = pred_duration,
        .migration_cost = migration_cost_,
        .wait_cost_s = 0.0,
        .benefit_s = -migration_cost_s,
        .strategy = "migrate"
    };
}

// ============================================================
// Always-Migrate 决策引擎
// ============================================================

MigrationDecision AlwaysMigrateDecisionEngine::decide(
    const PhaseAttributes& burst_attrs,
    const LLMState& llm_state,
    size_t queue_depth,
    double llm_remaining_work_s)
{
    stats_.total_decisions++;
    double pred_duration = burst_attrs.typical_duration_s;

    if (isCpuAffine(burst_attrs.device_affinity)) {
        stats_.bypass_count++;
        return MigrationDecision{
            .should_migrate = false,
            .reason = "CPU-affine 阶段旁路",
            .predicted_burst_duration_s = pred_duration,
            .migration_cost = migration_cost_,
            .strategy = "bypass"
        };
    }

    if (llm_state.level() == LLMServiceLevel::CPU_DEGRADED) {
        stats_.bypass_count++;
        return MigrationDecision{
            .should_migrate = false,
            .reason = "LLM 已降级",
            .predicted_burst_duration_s = pred_duration,
            .migration_cost = migration_cost_,
            .strategy = "bypass"
        };
    }

    // 无条件迁移
    stats_.migrate_count++;
    return MigrationDecision{
        .should_migrate = true,
        .reason = "always-migrate 策略：无条件迁移",
        .predicted_burst_duration_s = pred_duration,
        .migration_cost = migration_cost_,
        .strategy = "migrate"
    };
}

// ============================================================
// DefaultPhaseAwareStrategy
// ============================================================

DefaultPhaseAwareStrategy::DefaultPhaseAwareStrategy(const PhaseAwareConfig& config)
    : config_(config)
    , lease_manager_(config.vram_total_mb)
{
    // 根据迁移策略创建决策引擎
    if (config.migration_policy == MigrationPolicy::ALWAYS_MIGRATE) {
        decision_engine_ = std::make_unique<AlwaysMigrateDecisionEngine>(config.migration_cost);
    } else {
        decision_engine_ = std::make_unique<CostAwareDecisionEngine>(
            config.migration_cost,
            config.short_burst_threshold_s,
            config.benefit_threshold_s,
            config.queue_pressure_threshold);
    }
}

std::string DefaultPhaseAwareStrategy::routeTask(
    const PhaseAttributes& attrs,
    const LLMState& llm_state,
    SchedulerPolicy policy)
{
    // SEQUENTIAL 策略：所有任务串行
    if (policy == SchedulerPolicy::SEQUENTIAL) {
        return "sequential";
    }

    // RESIDENT_MIGRATABLE 任务（LLM）：常驻 GPU
    if (attrs.role == PhaseRole::RESIDENT_MIGRATABLE) {
        return "resident_gpu";
    }

    // V1_DUAL_ROLE 策略：CPU 任务也进 GPU 队列
    if (policy == SchedulerPolicy::V1_DUAL_ROLE) {
        return "gpu_burst";
    }

    // V2_PHASE_AWARE 策略
    // CPU-affine 任务：走 backfill
    if (isCpuAffine(attrs.device_affinity)) {
        return "cpu_backfill";
    }

    // GPU_PREFERRED 任务：查询决策是否可降级 CPU
    if (attrs.device_affinity == DeviceAffinity::GPU_PREFERRED) {
        auto decision = decision_engine_->decide(attrs, llm_state, 0, 0.0);
        if (decision.strategy == "bypass") {
            return "cpu_backfill";
        }
    }

    // GPU_REQUIRED 任务：进 GPU burst 队列
    return "gpu_burst";
}

std::shared_ptr<IWorker> DefaultPhaseAwareStrategy::selectWorkerForPhase(
    const PhaseAttributes& attrs,
    const std::vector<std::shared_ptr<IWorker>>& gpu_workers,
    const std::vector<std::shared_ptr<IWorker>>& cpu_workers,
    const std::shared_ptr<IWorker>& llm_worker)
{
    // RESIDENT_MIGRATABLE：使用 LLM 专用 Worker
    if (attrs.role == PhaseRole::RESIDENT_MIGRATABLE) {
        if (llm_worker && !llm_worker->isBusy()) {
            return llm_worker;
        }
        return nullptr;
    }

    // CPU-affine 任务：使用 CPU Worker
    if (isCpuAffine(attrs.device_affinity)) {
        for (const auto& worker : cpu_workers) {
            if (!worker->isBusy()) {
                return worker;
            }
        }
        return nullptr;
    }

    // GPU 任务：使用 GPU Worker（排除 LLM Worker）
    for (const auto& worker : gpu_workers) {
        if (worker != llm_worker && !worker->isBusy()) {
            return worker;
        }
    }

    return nullptr;
}

MigrationDecision DefaultPhaseAwareStrategy::shouldMigrateLLM(
    const PhaseAttributes& burst_attrs,
    const LLMState& llm_state,
    size_t queue_depth)
{
    return decision_engine_->decide(burst_attrs, llm_state, queue_depth, 0.0);
}

bool DefaultPhaseAwareStrategy::migrateLLMToCPU(bool urgent) {
    if (migrate_fn_) {
        bool ok = migrate_fn_(urgent);
        if (ok) {
            llm_state_.enterDegraded();
            // 降级常驻租约
            lease_manager_.downgradeResident("llm_resident");
        }
        return ok;
    }
    // 无回调时，仅更新状态（用于测试）
    llm_state_.enterDegraded();
    lease_manager_.downgradeResident("llm_resident");
    return true;
}

bool DefaultPhaseAwareStrategy::restoreLLMToGPU() {
    llm_state_.enterRestoring();
    if (restore_fn_) {
        bool ok = restore_fn_();
        if (ok) {
            llm_state_.enterNormal();
            // 恢复常驻租约
            lease_manager_.restoreResident("llm_resident", 6000);
        } else {
            // 回迁失败，回退到降级状态
            llm_state_.enterDegraded();
        }
        return ok;
    }
    // 无回调时，仅更新状态（用于测试）
    llm_state_.enterNormal();
    lease_manager_.restoreResident("llm_resident", 6000);
    return true;
}

} // namespace ai_scheduler
