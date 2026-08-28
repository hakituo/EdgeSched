#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "../core/resource_isolation_scheduler.h"
#include "../core/native_executor.h"
#include "../core/phase_strategy.h"
#include "../workers/gpu_llm_worker.h"
// image worker 已移除：C++ 端只做资源调度，图片生成在 Python 侧执行

namespace py = pybind11;
using namespace ai_scheduler;

PYBIND11_MODULE(scheduler_py, m) {
    m.doc() = "AI Scheduler Python Bindings";

    // ============================================================
    // 原有枚举
    // ============================================================

    py::enum_<TaskType>(m, "TaskType")
        .value("LLM_INFERENCE", TaskType::LLM_INFERENCE)
        .value("TTS_SYNTHESIS", TaskType::TTS_SYNTHESIS)
        .value("IMAGE_GENERATION", TaskType::IMAGE_GENERATION)
        .value("NPU_INFERENCE", TaskType::NPU_INFERENCE)
        .export_values();

    py::enum_<TaskStatus>(m, "TaskStatus")
        .value("PENDING", TaskStatus::PENDING)
        .value("RUNNING", TaskStatus::RUNNING)
        .value("COMPLETED", TaskStatus::COMPLETED)
        .value("FAILED", TaskStatus::FAILED)
        .value("CANCELLED", TaskStatus::CANCELLED)
        .export_values();

    py::enum_<TaskPriority>(m, "TaskPriority")
        .value("HIGH", TaskPriority::HIGH)
        .value("MEDIUM", TaskPriority::MEDIUM)
        .value("LOW", TaskPriority::LOW)
        .export_values();

    // ============================================================
    // Phase-Aware 枚举
    // ============================================================

    py::enum_<PhaseRole>(m, "PhaseRole")
        .value("RESIDENT_MIGRATABLE", PhaseRole::RESIDENT_MIGRATABLE)
        .value("BURST_GPU_PINNED", PhaseRole::BURST_GPU_PINNED)
        .value("BURST_GPU_PREFERRED", PhaseRole::BURST_GPU_PREFERRED)
        .value("BURST_CPU_LIGHT", PhaseRole::BURST_CPU_LIGHT)
        .export_values();

    py::enum_<DeviceAffinity>(m, "DeviceAffinity")
        .value("GPU_REQUIRED", DeviceAffinity::GPU_REQUIRED)
        .value("GPU_PREFERRED", DeviceAffinity::GPU_PREFERRED)
        .value("CPU_ONLY", DeviceAffinity::CPU_ONLY)
        .value("CPU_PREFERRED", DeviceAffinity::CPU_PREFERRED)
        .export_values();

    py::enum_<LLMServiceLevel>(m, "LLMServiceLevel")
        .value("GPU_NORMAL", LLMServiceLevel::GPU_NORMAL)
        .value("CPU_DEGRADED", LLMServiceLevel::CPU_DEGRADED)
        .value("RESTORING", LLMServiceLevel::RESTORING)
        .export_values();

    py::enum_<MigrationPolicy>(m, "MigrationPolicy")
        .value("ALWAYS_MIGRATE", MigrationPolicy::ALWAYS_MIGRATE)
        .value("COST_AWARE", MigrationPolicy::COST_AWARE)
        .export_values();

    py::enum_<SchedulerPolicy>(m, "SchedulerPolicy")
        .value("SEQUENTIAL", SchedulerPolicy::SEQUENTIAL)
        .value("V1_DUAL_ROLE", SchedulerPolicy::V1_DUAL_ROLE)
        .value("V2_PHASE_AWARE", SchedulerPolicy::V2_PHASE_AWARE)
        .export_values();

    py::enum_<LeaseType>(m, "LeaseType")
        .value("RESIDENT", LeaseType::RESIDENT)
        .value("BURST", LeaseType::BURST)
        .value("SHARED", LeaseType::SHARED)
        .export_values();

    // ============================================================
    // Phase-Aware 数据结构
    // ============================================================

    py::class_<PhaseAttributes>(m, "PhaseAttributes")
        .def(py::init<>())
        .def_readwrite("role", &PhaseAttributes::role)
        .def_readwrite("device_affinity", &PhaseAttributes::device_affinity)
        .def_readwrite("preemptible", &PhaseAttributes::preemptible)
        .def_readwrite("queueable", &PhaseAttributes::queueable)
        .def_readwrite("migratable", &PhaseAttributes::migratable)
        .def_readwrite("vram_usage_mb", &PhaseAttributes::vram_usage_mb)
        .def_readwrite("typical_duration_s", &PhaseAttributes::typical_duration_s)
        .def_readwrite("is_downstream", &PhaseAttributes::is_downstream)
        .def_readwrite("is_upstream", &PhaseAttributes::is_upstream)
        .def_readwrite("can_overlap_with_gpu", &PhaseAttributes::can_overlap_with_gpu);

    py::class_<MigrationCost>(m, "MigrationCost")
        .def(py::init<>())
        .def_readwrite("kv_cache_save_ms", &MigrationCost::kv_cache_save_ms)
        .def_readwrite("vram_release_ms", &MigrationCost::vram_release_ms)
        .def_readwrite("cpu_load_ms", &MigrationCost::cpu_load_ms)
        .def_readwrite("restore_ms", &MigrationCost::restore_ms)
        .def_readwrite("service_gap_ms", &MigrationCost::service_gap_ms)
        .def("totalDownlinkMs", &MigrationCost::totalDownlinkMs)
        .def("totalRoundTripMs", &MigrationCost::totalRoundTripMs);

    py::class_<MigrationDecision>(m, "MigrationDecision")
        .def(py::init<>())
        .def_readwrite("should_migrate", &MigrationDecision::should_migrate)
        .def_readwrite("reason", &MigrationDecision::reason)
        .def_readwrite("predicted_burst_duration_s", &MigrationDecision::predicted_burst_duration_s)
        .def_readwrite("migration_cost", &MigrationDecision::migration_cost)
        .def_readwrite("wait_cost_s", &MigrationDecision::wait_cost_s)
        .def_readwrite("benefit_s", &MigrationDecision::benefit_s)
        .def_readwrite("strategy", &MigrationDecision::strategy);

    py::class_<GPULease>(m, "GPULease")
        .def(py::init<>())
        .def_readwrite("holder", &GPULease::holder)
        .def_readwrite("holder_role", &GPULease::holder_role)
        .def_readwrite("lease_type", &GPULease::lease_type)
        .def_readwrite("preemptible", &GPULease::preemptible)
        .def_readwrite("duration_pred_s", &GPULease::duration_pred_s)
        .def_readwrite("vram_required_mb", &GPULease::vram_required_mb);

    py::class_<PhaseAwareConfig>(m, "PhaseAwareConfig")
        .def(py::init<>())
        .def_readwrite("policy", &PhaseAwareConfig::policy)
        .def_readwrite("migration_policy", &PhaseAwareConfig::migration_policy)
        .def_readwrite("vram_total_mb", &PhaseAwareConfig::vram_total_mb)
        .def_readwrite("max_concurrent_cpu", &PhaseAwareConfig::max_concurrent_cpu)
        .def_readwrite("migration_cost", &PhaseAwareConfig::migration_cost)
        .def_readwrite("short_burst_threshold_s", &PhaseAwareConfig::short_burst_threshold_s)
        .def_readwrite("benefit_threshold_s", &PhaseAwareConfig::benefit_threshold_s)
        .def_readwrite("queue_pressure_threshold", &PhaseAwareConfig::queue_pressure_threshold);

    // ============================================================
    // LLMState
    // ============================================================

    py::class_<LLMState>(m, "LLMState")
        .def(py::init<>())
        .def("enterDegraded", &LLMState::enterDegraded)
        .def("enterRestoring", &LLMState::enterRestoring)
        .def("enterNormal", &LLMState::enterNormal)
        .def("level", &LLMState::level)
        .def("currentDevice", &LLMState::currentDevice)
        .def("migrationCount", &LLMState::migrationCount)
        .def("totalDegradedTimeS", &LLMState::totalDegradedTimeS);

    // ============================================================
    // GPULeaseManager
    // ============================================================

    py::class_<GPULeaseManager>(m, "GPULeaseManager")
        .def(py::init<size_t>(), py::arg("vram_total_mb") = 8192)
        .def("canAcquire", &GPULeaseManager::canAcquire)
        .def("acquire", &GPULeaseManager::acquire)
        .def("release", &GPULeaseManager::release)
        .def("downgradeResident", &GPULeaseManager::downgradeResident)
        .def("restoreResident", &GPULeaseManager::restoreResident)
        .def("vramTotalMb", &GPULeaseManager::vramTotalMb)
        .def("vramUsedMb", &GPULeaseManager::vramUsedMb)
        .def("vramAvailableMb", &GPULeaseManager::vramAvailableMb);

    // ============================================================
    // Phase-Aware 策略
    // ============================================================

    py::class_<IPhaseAwareStrategy, std::shared_ptr<IPhaseAwareStrategy>>(m, "IPhaseAwareStrategy");

    py::class_<DefaultPhaseAwareStrategy, IPhaseAwareStrategy, std::shared_ptr<DefaultPhaseAwareStrategy>>(m, "DefaultPhaseAwareStrategy")
        .def(py::init<const PhaseAwareConfig&>(), py::arg("config") = PhaseAwareConfig{})
        .def("routeTask", &DefaultPhaseAwareStrategy::routeTask)
        .def("selectWorkerForPhase", &DefaultPhaseAwareStrategy::selectWorkerForPhase)
        .def("shouldMigrateLLM", &DefaultPhaseAwareStrategy::shouldMigrateLLM)
        .def("getLLMState", &DefaultPhaseAwareStrategy::getLLMState, py::return_value_policy::reference_internal)
        .def("getLeaseManager", &DefaultPhaseAwareStrategy::getLeaseManager, py::return_value_policy::reference_internal)
        .def("setMigrationCallbacks", &DefaultPhaseAwareStrategy::setMigrationCallbacks)
        .def("migrateLLMToCPU", &DefaultPhaseAwareStrategy::migrateLLMToCPU, py::arg("urgent") = false)
        .def("restoreLLMToGPU", &DefaultPhaseAwareStrategy::restoreLLMToGPU);

    // Phase 预设函数
    m.def("getPhasePreset", &getPhasePreset);

    // ============================================================
    // 原有结构体
    // ============================================================
    py::class_<LLMInferenceRequest>(m, "LLMInferenceRequest")
        .def(py::init<>())
        .def_readwrite("prompt", &LLMInferenceRequest::prompt)
        .def_readwrite("conversationId", &LLMInferenceRequest::conversationId)
        .def_readwrite("maxTokens", &LLMInferenceRequest::maxTokens)
        .def_readwrite("temperature", &LLMInferenceRequest::temperature)
        .def_readwrite("topK", &LLMInferenceRequest::topK)
        .def_readwrite("topP", &LLMInferenceRequest::topP)
        .def_readwrite("repetitionPenalty", &LLMInferenceRequest::repetitionPenalty)
        .def_readwrite("streamOutput", &LLMInferenceRequest::streamOutput)
        .def_readwrite("onTokenGenerated", &LLMInferenceRequest::onTokenGenerated);

    py::class_<LLMInferenceResponse>(m, "LLMInferenceResponse")
        .def(py::init<>())
        .def_readwrite("generatedText", &LLMInferenceResponse::generatedText)
        .def_readwrite("generatedTokens", &LLMInferenceResponse::generatedTokens)
        .def_readwrite("inferenceTime", &LLMInferenceResponse::inferenceTime)
        .def_readwrite("success", &LLMInferenceResponse::success)
        .def_readwrite("errorMessage", &LLMInferenceResponse::errorMessage);

    // ITask
    py::class_<ITask, std::shared_ptr<ITask>>(m, "ITask")
        .def("getTaskId", &ITask::getTaskId)
        .def("getStatus", &ITask::getStatus);

    // LLMTask
    py::class_<LLMTask, ITask, std::shared_ptr<LLMTask>>(m, "LLMTask")
        .def(py::init<const LLMInferenceRequest&>())
        .def("getResponse", &LLMTask::getResponse);

    // IWorker (needed for addWorker)
    py::class_<IWorker, std::shared_ptr<IWorker>>(m, "IWorker")
        .def("getWorkerId", &IWorker::getWorkerId);

    // GPULLMWorker
    py::class_<GPULLMWorker, IWorker, std::shared_ptr<GPULLMWorker>>(m, "GPULLMWorker")
        .def(py::init<const std::string&>())
        .def("initialize", &GPULLMWorker::initialize)
        .def("shutdown", &GPULLMWorker::shutdown)
        .def("setModelConfig", &GPULLMWorker::setModelConfig);

    // LLMModelConfig
    py::class_<LLMModelConfig>(m, "LLMModelConfig")
        .def(py::init<>())
        .def_readwrite("modelPath", &LLMModelConfig::modelPath)
        .def_readwrite("modelType", &LLMModelConfig::modelType)
        .def_readwrite("quantization", &LLMModelConfig::quantization)
        .def_readwrite("gpuDeviceId", &LLMModelConfig::gpuDeviceId)
        .def_readwrite("maxContextSize", &LLMModelConfig::maxContextSize)
        .def_readwrite("maxBatchSize", &LLMModelConfig::maxBatchSize)
        .def_readwrite("temperature", &LLMModelConfig::temperature)
        .def_readwrite("topK", &LLMModelConfig::topK)
        .def_readwrite("topP", &LLMModelConfig::topP)
        .def_readwrite("repetitionPenalty", &LLMModelConfig::repetitionPenalty)
        .def_readwrite("enableCache", &LLMModelConfig::enableCache)
        .def_readwrite("cacheSize", &LLMModelConfig::cacheSize)
        .def_readwrite("draftModelPath", &LLMModelConfig::draftModelPath)
        .def_readwrite("draftGpuDeviceId", &LLMModelConfig::draftGpuDeviceId)
        .def_readwrite("draftContextSize", &LLMModelConfig::draftContextSize);


    // NativeExecutor
    py::class_<NativeExecutor, std::shared_ptr<NativeExecutor>>(m, "NativeExecutor")
        .def("executeAsync", &NativeExecutor::executeAsync)
        .def("addTimer", &NativeExecutor::addTimer)
        .def("isRunning", &NativeExecutor::isRunning);

    // ResourceIsolationScheduler
    py::class_<ResourceIsolationScheduler, std::shared_ptr<ResourceIsolationScheduler>>(m, "ResourceIsolationScheduler")
        .def(py::init<>())
        .def("initialize", &ResourceIsolationScheduler::initialize, py::arg("cpuThreadCount") = 4)
        .def("setNativeExecutorEnabled", &ResourceIsolationScheduler::setNativeExecutorEnabled)
        .def("shutdown", &ResourceIsolationScheduler::shutdown)
        .def(
            "submitTask",
            [](ResourceIsolationScheduler& scheduler, std::shared_ptr<ITask> task) {
                scheduler.submitTask(std::move(task));
            }
        )
        .def("submitPhaseTask", &ResourceIsolationScheduler::submitPhaseTask)
        .def("cancelTask", &ResourceIsolationScheduler::cancelTask)
        .def("getTaskStatus", &ResourceIsolationScheduler::getTaskStatus)
        .def("getTask", &ResourceIsolationScheduler::getTask)
        .def("addWorker", &ResourceIsolationScheduler::addWorker)
        .def("getNativeExecutor", &ResourceIsolationScheduler::getNativeExecutor)
        .def("setPhaseStrategy", &ResourceIsolationScheduler::setPhaseStrategy)
        .def("getPhaseStrategy", &ResourceIsolationScheduler::getPhaseStrategy)
        .def("getLLMState", &ResourceIsolationScheduler::getLLMState)
        .def("migrateLLMToCPU", &ResourceIsolationScheduler::migrateLLMToCPU, py::arg("urgent") = false)
        .def("restoreLLMToGPU", &ResourceIsolationScheduler::restoreLLMToGPU)
        .def("setMigrationCallbacks", &ResourceIsolationScheduler::setMigrationCallbacks);

}
