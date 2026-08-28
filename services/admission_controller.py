"""
Cost-Aware Admission Controller for EdgeSched (MVP).

Decides whether a burst workload (e.g. image generation, TTS, embedding)
should be admitted immediately (with or without resident-model offload),
queued, rejected, or **routed to CPU** — based on runtime-collected cost
metrics and task affinity profiles.

Key insight: EdgeSched does not route a task to GPU merely because GPU
is available; it routes a task to GPU only when the expected acceleration
outweighs the disruption to resident GPU state.
"""
import copy
import time
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback constants (conservative values from initial experiments)
# ---------------------------------------------------------------------------
FALLBACK_OFFLOAD_S: float = 9.0        # GPU→CPU offload ~8.8 s
FALLBACK_RESTORE_S: float = 11.5       # CPU→GPU restore ~11.2 s
FALLBACK_BURST_S: float = 16.5         # image burst on GPU ~16.2 s
FALLBACK_BURST_VRAM_MB: int = 6700     # SDXL Turbo ~6.46 GB

# Default CPU-to-GPU time ratio for tasks without runtime measurements
_DEFAULT_CPU_MULTIPLIER: float = 3.0


# ---------------------------------------------------------------------------
# Task affinity profiling (static prior + runtime EMA correction)
# ---------------------------------------------------------------------------
@dataclass
class TaskProfile:
    """
    Static resource profile for a known task type.

    Used as the *prior* in hybrid affinity profiling:
        effective_estimate = runtime_EMA   if EMA available
                           = static_prior  otherwise

    Fields:
        task_type:         unique identifier for the task
        estimated_vram_mb: peak GPU VRAM usage (MB) when running on GPU
        device_affinity:   preferred execution target
            - gpu_pinned:        must use GPU (e.g. image generation)
            - gpu_preferred:     benefits from GPU but can run on CPU
            - cpu_preferred:     lightweight, better on CPU
        supports_offload:  whether the task supports offload during execution
        gpu_sensitivity:   0.0-1.0, fraction of performance gained from GPU
                           (1.0 = only GPU gives acceptable latency)
        interactivity:     0.0-1.0, user-visible urgency of the task
        can_run_on_cpu:    whether a CPU execution path exists
        cpu_time_ratio:    expected (cpu_time / gpu_time) ratio
    """
    task_type: str
    estimated_vram_mb: int
    device_affinity: str              # gpu_pinned | gpu_preferred | cpu_preferred
    supports_offload: bool = False
    gpu_sensitivity: float = 1.0
    interactivity: float = 1.0
    can_run_on_cpu: bool = False
    cpu_time_ratio: float = _DEFAULT_CPU_MULTIPLIER


# Static task registry (priors, updated at runtime by EMA)
TASK_REGISTRY: Dict[str, TaskProfile] = {
    "image_gen": TaskProfile(
        task_type="image_gen",
        estimated_vram_mb=6500,
        device_affinity="gpu_pinned",
        supports_offload=True,
        gpu_sensitivity=0.95,
        interactivity=0.8,
        can_run_on_cpu=False,
        cpu_time_ratio=20.0,
    ),
    "tts_f5": TaskProfile(
        task_type="tts_f5",
        estimated_vram_mb=1500,
        device_affinity="gpu_preferred",
        supports_offload=True,
        gpu_sensitivity=0.6,
        interactivity=0.9,
        can_run_on_cpu=True,
        cpu_time_ratio=4.0,
    ),
    "tts_qwen": TaskProfile(
        task_type="tts_qwen",
        estimated_vram_mb=1500,
        device_affinity="gpu_preferred",
        supports_offload=True,
        gpu_sensitivity=0.7,
        interactivity=0.9,
        can_run_on_cpu=True,
        cpu_time_ratio=5.0,
    ),
    "embedding": TaskProfile(
        task_type="embedding",
        estimated_vram_mb=512,
        device_affinity="cpu_preferred",
        supports_offload=False,
        gpu_sensitivity=0.2,
        interactivity=0.3,
        can_run_on_cpu=True,
        cpu_time_ratio=1.5,
    ),
    "ocr": TaskProfile(
        task_type="ocr",
        estimated_vram_mb=256,
        device_affinity="cpu_preferred",
        supports_offload=False,
        gpu_sensitivity=0.15,
        interactivity=0.2,
        can_run_on_cpu=True,
        cpu_time_ratio=1.3,
    ),
    "document": TaskProfile(
        task_type="document",
        estimated_vram_mb=0,
        device_affinity="cpu_preferred",
        supports_offload=False,
        gpu_sensitivity=0.0,
        interactivity=0.5,
        can_run_on_cpu=True,
        cpu_time_ratio=1.0,
    ),
}


@dataclass
class RoutingDecision:
    """Result of the task routing decision."""
    route: str                        # "gpu" | "cpu"
    classification: str               # "gpu_heavy" | "cpu_affine" | "gpu_candidate" | "unknown"
    reason: str = ""
    gpu_benefit_s: float = 0.0        # estimated time saved by using GPU
    disruption_cost_s: float = 0.0    # estimated disruption cost if routed to GPU
    profile: Optional[TaskProfile] = None
    requires_offload: bool = False


# ---------------------------------------------------------------------------
# Enum & result
# ---------------------------------------------------------------------------
class AdmissionDecision(Enum):
    """Admission decision outcomes."""
    ADMIT_NOW = "admit_now"   # Execute immediately (may need offload first)
    QUEUE = "queue"           # Wait in bounded queue
    REJECT = "reject"         # Reject the request


@dataclass
class AdmissionResult:
    """Result of an admission decision."""
    decision: AdmissionDecision
    requires_offload: bool = False
    reason: str = ""
    details: dict = field(default_factory=dict)


class AdmissionRejectedError(RuntimeError):
    """Raised when the admission controller rejects a burst request."""

    def __init__(self, result: AdmissionResult):
        self.result = result
        super().__init__(
            f"Admission rejected: {result.reason} | {result.details}"
        )


# ---------------------------------------------------------------------------
# System state snapshot
# ---------------------------------------------------------------------------
@dataclass
class SystemState:
    """Point-in-time snapshot of system resource state."""
    gpu_free_mb: int = 0
    gpu_total_mb: int = 0
    resident_on_gpu: bool = False
    queue_depth: int = 0
    queue_capacity: int = 9
    burst_running: bool = False


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class AdmissionConfig:
    """Tunable parameters for the admission controller."""
    max_queue_wait_s: float = 120.0       # max acceptable queueing delay
    vram_safety_margin: float = 1.1       # 10 % safety margin on VRAM
    ema_alpha: float = 0.3               # EMA smoothing factor
    degradation_penalty_s: float = 5.0    # UX penalty while LLM is on CPU
    enable_cost_model: bool = True        # False → greedy (always admit)


# ---------------------------------------------------------------------------
# Runtime metrics (EMA tracker)
# ---------------------------------------------------------------------------
class RuntimeMetrics:
    """
    Exponential-moving-average tracker for scheduling-relevant timings.
    All values are measured at runtime; nothing is hard-coded.
    """

    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        self._lock = threading.Lock()

        # EMA values — None until first sample
        self.avg_offload_time_s: Optional[float] = None
        self.avg_restore_time_s: Optional[float] = None
        self.avg_burst_duration_s: Optional[float] = None
        self.avg_burst_vram_mb: Optional[float] = None

        # Static / slowly-changing
        self.resident_vram_mb: int = 0
        self.gpu_total_mb: int = 0

        # Observability
        self.decision_log: List[Dict[str, Any]] = []
        self.sample_count: int = 0

        # Per-task-type EMA tracking (hybrid static-dynamic profiling)
        # key = task_type, value = {"gpu_duration_s": float, "cpu_duration_s": float, "count": int}
        self.task_timings: Dict[str, Dict[str, float]] = {}

    # -- update helpers -------------------------------------------------------

    def _ema(self, current: Optional[float], new_value: float) -> float:
        if current is None:
            return new_value
        return self.alpha * new_value + (1.0 - self.alpha) * current

    def update_offload(self, dt_s: float) -> None:
        with self._lock:
            self.avg_offload_time_s = self._ema(self.avg_offload_time_s, dt_s)
            self.sample_count += 1

    def update_restore(self, dt_s: float) -> None:
        with self._lock:
            self.avg_restore_time_s = self._ema(self.avg_restore_time_s, dt_s)
            self.sample_count += 1

    def update_burst(self, dt_s: float, vram_mb: Optional[int] = None) -> None:
        with self._lock:
            self.avg_burst_duration_s = self._ema(self.avg_burst_duration_s, dt_s)
            if vram_mb is not None:
                self.avg_burst_vram_mb = self._ema(self.avg_burst_vram_mb, float(vram_mb))
            self.sample_count += 1

    def update_task_timing(
        self, task_type: str, duration_s: float, device: str = "gpu"
    ) -> None:
        """Update per-task-type EMA timing for hybrid affinity profiling.

        Args:
            task_type: task identifier (e.g. "image_gen", "tts_f5")
            duration_s: measured execution duration in seconds
            device: "gpu" or "cpu"
        """
        with self._lock:
            if task_type not in self.task_timings:
                self.task_timings[task_type] = {
                    "gpu_duration_s": 0.0,
                    "cpu_duration_s": 0.0,
                    "count": 0,
                }
            entry = self.task_timings[task_type]
            key = f"{device}_duration_s"
            entry[key] = self._ema(entry.get(key) or 0.0, duration_s)
            entry["count"] = entry.get("count", 0) + 1

    def get_effective_profile(self, task_type: str) -> Optional[TaskProfile]:
        """Return a TaskProfile corrected with runtime EMA if available.

        Falls back to static prior from TASK_REGISTRY if no EMA data.
        """
        static = TASK_REGISTRY.get(task_type)
        if not static:
            return None
        with self._lock:
            timings = self.task_timings.get(task_type)
            if not timings or timings.get("count", 0) < 1:
                return static
            # Override with runtime EMA values
            gpu_dur = timings.get("gpu_duration_s")
            cpu_dur = timings.get("cpu_duration_s")
            ratio = _DEFAULT_CPU_MULTIPLIER
            if gpu_dur and gpu_dur > 0.01:
                ratio = (cpu_dur or gpu_dur * _DEFAULT_CPU_MULTIPLIER) / gpu_dur
            prof = copy.copy(static)
            prof.cpu_time_ratio = round(ratio, 2)
            return prof

    def log_decision(self, entry: Dict[str, Any]) -> None:
        with self._lock:
            self.decision_log.append(entry)
            # Keep log bounded
            if len(self.decision_log) > 5000:
                self.decision_log = self.decision_log[-5000:]

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "avg_offload_s": self.avg_offload_time_s,
                "avg_restore_s": self.avg_restore_time_s,
                "avg_burst_s": self.avg_burst_duration_s,
                "avg_burst_vram_mb": self.avg_burst_vram_mb,
                "sample_count": self.sample_count,
            }


# ---------------------------------------------------------------------------
# Admission controller
# ---------------------------------------------------------------------------
class AdmissionController:
    """
    Cost-aware admission controller for burst workloads.

    Replaces unconditional burst-triggered migration with a decision
    that compares estimated completion time under immediate migration
    against expected queueing delay.
    """

    def __init__(self, metrics: RuntimeMetrics, config: Optional[AdmissionConfig] = None):
        self.metrics = metrics
        self.config = config or AdmissionConfig()

    def decide(self, system_state: SystemState) -> AdmissionResult:
        """
        Core decision function.

        Decision order:
        1. GPU idle + sufficient VRAM → ADMIT_NOW (no offload)
        2. Queue full → REJECT
        3. Greedy mode (cost_model off) → ADMIT_NOW (offload if needed)
        4. Cost model: migrate vs queue comparison
        5. Queue wait acceptable → QUEUE
        6. Otherwise → REJECT
        """

        # -- 1. GPU idle + sufficient VRAM → ADMIT_NOW, no offload ----------
        if not system_state.resident_on_gpu and not system_state.burst_running:
            burst_vram = self.metrics.avg_burst_vram_mb or FALLBACK_BURST_VRAM_MB
            required = burst_vram * self.config.vram_safety_margin
            if system_state.gpu_free_mb >= required:
                result = AdmissionResult(
                    decision=AdmissionDecision.ADMIT_NOW,
                    requires_offload=False,
                    reason="gpu_idle_sufficient_vram",
                    details={"gpu_free_mb": system_state.gpu_free_mb,
                             "required_mb": int(required)},
                )
                self._log(system_state, result)
                return result

        # -- 2. Queue full → REJECT -------------------------------------------
        if system_state.queue_depth >= system_state.queue_capacity:
            result = AdmissionResult(
                decision=AdmissionDecision.REJECT,
                reason="queue_full",
                details={"queue_depth": system_state.queue_depth,
                         "queue_capacity": system_state.queue_capacity},
            )
            self._log(system_state, result)
            return result

        # -- 3. Greedy mode ---------------------------------------------------
        if not self.config.enable_cost_model:
            result = AdmissionResult(
                decision=AdmissionDecision.ADMIT_NOW,
                requires_offload=system_state.resident_on_gpu,
                reason="cost_model_disabled_greedy",
            )
            self._log(system_state, result)
            return result

        # -- 4. Compute total completion times --------------------------------
        offload = self.metrics.avg_offload_time_s or FALLBACK_OFFLOAD_S
        restore = self.metrics.avg_restore_time_s or FALLBACK_RESTORE_S
        burst = self.metrics.avg_burst_duration_s or FALLBACK_BURST_S
        penalty = self.config.degradation_penalty_s

        completion_migrate = offload + burst + restore + penalty
        queue_wait = system_state.queue_depth * burst
        completion_queue = queue_wait + burst

        cost_details = {
            "completion_migrate_s": round(completion_migrate, 2),
            "completion_queue_s": round(completion_queue, 2),
            "queue_wait_s": round(queue_wait, 2),
            "offload_s": round(offload, 2),
            "restore_s": round(restore, 2),
            "burst_s": round(burst, 2),
            "penalty_s": round(penalty, 2),
        }

        # -- 5. Migrate vs queue ----------------------------------------------
        if system_state.resident_on_gpu and completion_migrate < completion_queue:
            result = AdmissionResult(
                decision=AdmissionDecision.ADMIT_NOW,
                requires_offload=True,
                reason="migration_cheaper_than_queue",
                details=cost_details,
            )
            self._log(system_state, result)
            return result

        # -- 6. Queue wait acceptable -----------------------------------------
        if queue_wait < self.config.max_queue_wait_s:
            result = AdmissionResult(
                decision=AdmissionDecision.QUEUE,
                reason="queue_cheaper_or_migration_not_worth",
                details=cost_details,
            )
            self._log(system_state, result)
            return result

        # -- 7. Neither migrate nor queue is acceptable -----------------------
        result = AdmissionResult(
            decision=AdmissionDecision.REJECT,
            reason="queue_wait_exceeds_max_and_migration_not_better",
            details={**cost_details,
                     "max_wait_s": self.config.max_queue_wait_s},
        )
        self._log(system_state, result)
        return result

    def route_task(
        self,
        task_type: str,
        system_state: SystemState,
    ) -> RoutingDecision:
        """
        Affinity-aware task routing: GPU vs CPU.

        Core principle: EdgeSched routes a task to GPU only when the
        expected acceleration outweighs the disruption to resident GPU state.

        Routing logic:
        1. cpu_preferred       → always CPU (lightweight auxiliary workloads)
        2. gpu_pinned          → always GPU (must use GPU hardware)
        3. gpu_preferred/any   → cost-benefit analysis:
              gpu_benefit = gpu_duration * gpu_sensitivity
              disruption_cost = offload + restore + degradation (if LLM on GPU)
              if gpu_benefit < disruption_cost and can_run_on_cpu → CPU
              else → GPU (may require offload)

        The effective profile uses runtime EMA when available, falling back
        to the static prior from TASK_REGISTRY (hybrid profiling).
        """
        profile = self.metrics.get_effective_profile(task_type)

        # Unknown task type — default to GPU with offload if needed
        if not profile:
            return RoutingDecision(
                route="gpu",
                classification="unknown",
                reason="unknown_task_type_default_gpu",
                requires_offload=system_state.resident_on_gpu,
            )

        # -- 1. Hard routing rules -----------------------------------------------
        if profile.device_affinity == "cpu_preferred":
            return RoutingDecision(
                route="cpu",
                classification="cpu_affine",
                reason=f"cpu_preferred (vram={profile.estimated_vram_mb}MB, "
                       f"sensitivity={profile.gpu_sensitivity})",
                profile=profile,
            )

        if profile.device_affinity == "gpu_pinned":
            return RoutingDecision(
                route="gpu",
                classification="gpu_heavy",
                reason=f"gpu_pinned (vram={profile.estimated_vram_mb}MB)",
                profile=profile,
                requires_offload=system_state.resident_on_gpu,
            )

        # -- 2. Cost-benefit analysis for gpu_preferred / any -------------------
        # Estimate GPU duration (from EMA or static prior)
        burst_s = self.metrics.avg_burst_duration_s or FALLBACK_BURST_S
        gpu_duration_s = burst_s

        # Estimate CPU duration using the (possibly EMA-corrected) ratio
        cpu_duration_s = gpu_duration_s * profile.cpu_time_ratio

        # GPU benefit = how much faster the task runs on GPU,
        # weighted by how much the task actually benefits from GPU
        gpu_benefit_s = (cpu_duration_s - gpu_duration_s) * profile.gpu_sensitivity

        # Disruption cost = overhead of offloading/restoring LLM
        # Only incurred if LLM is currently on GPU and task needs GPU VRAM
        disruption_cost_s = 0.0
        if system_state.resident_on_gpu:
            offload_s = self.metrics.avg_offload_time_s or FALLBACK_OFFLOAD_S
            restore_s = self.metrics.avg_restore_time_s or FALLBACK_RESTORE_S
            penalty_s = self.config.degradation_penalty_s
            disruption_cost_s = offload_s + restore_s + penalty_s

        # Core decision: is GPU acceleration worth the disruption?
        if profile.can_run_on_cpu and gpu_benefit_s < disruption_cost_s:
            return RoutingDecision(
                route="cpu",
                classification="cpu_affine",
                reason=f"gpu_benefit({gpu_benefit_s:.1f}s) < "
                       f"disruption_cost({disruption_cost_s:.1f}s)",
                gpu_benefit_s=round(gpu_benefit_s, 2),
                disruption_cost_s=round(disruption_cost_s, 2),
                profile=profile,
            )

        # GPU is worth it — but does it need offload?
        needs_offload = system_state.resident_on_gpu
        return RoutingDecision(
            route="gpu",
            classification="gpu_candidate",
            reason=f"gpu_benefit({gpu_benefit_s:.1f}s) >= "
                   f"disruption_cost({disruption_cost_s:.1f}s)",
            gpu_benefit_s=round(gpu_benefit_s, 2),
            disruption_cost_s=round(disruption_cost_s, 2),
            profile=profile,
            requires_offload=needs_offload,
        )

    # -- internal helpers -----------------------------------------------------

    def _log(self, state: SystemState, result: AdmissionResult) -> None:
        entry = {
            "timestamp": time.time(),
            "decision": result.decision.value,
            "requires_offload": result.requires_offload,
            "reason": result.reason,
            "gpu_free_mb": state.gpu_free_mb,
            "queue_depth": state.queue_depth,
            "resident_on_gpu": state.resident_on_gpu,
            "burst_running": state.burst_running,
            "details": result.details,
            "metrics_snapshot": self.metrics.snapshot(),
        }
        self.metrics.log_decision(entry)
        logger.debug(
            "[Admission] decision={} offload={} reason={} | {}",
            result.decision.value,
            result.requires_offload,
            result.reason,
            result.details,
        )

    def get_decision_stats(self) -> Dict[str, Any]:
        """Return summary statistics over the decision log."""
        log = self.metrics.decision_log
        if not log:
            return {"total": 0}

        total = len(log)
        counts: Dict[str, int] = {}
        for entry in log:
            d = entry.get("decision", "unknown")
            counts[d] = counts.get(d, 0) + 1

        reject_count = counts.get("reject", 0)
        return {
            "total": total,
            "counts": counts,
            "ratios": {k: round(v / total, 3) for k, v in counts.items()},
            "reject_rate": round(reject_count / total, 3),
        }


# ---------------------------------------------------------------------------
# Module-level singletons
# ---------------------------------------------------------------------------
_runtime_metrics: Optional[RuntimeMetrics] = None
_admission_controller: Optional[AdmissionController] = None
# 控制器初始化会在锁内获取运行指标单例，因此必须允许同线程重入。
_singleton_lock = threading.RLock()


def get_runtime_metrics() -> RuntimeMetrics:
    """Return the process-wide RuntimeMetrics singleton."""
    global _runtime_metrics
    if _runtime_metrics is not None:
        return _runtime_metrics
    with _singleton_lock:
        if _runtime_metrics is None:
            _runtime_metrics = RuntimeMetrics()
        return _runtime_metrics


def get_admission_controller() -> AdmissionController:
    """Return the process-wide AdmissionController singleton."""
    global _admission_controller
    if _admission_controller is not None:
        return _admission_controller
    with _singleton_lock:
        if _admission_controller is None:
            metrics = get_runtime_metrics()
            config = AdmissionConfig()
            _admission_controller = AdmissionController(metrics, config)
        return _admission_controller
