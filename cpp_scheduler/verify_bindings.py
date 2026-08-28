#!/usr/bin/env python3
"""验证 Phase-Aware C++ 绑定"""
import sys
sys.path.insert(0, '/home/leslie/mvp_core/cpp_scheduler/build')

import scheduler_py as sp

# 验证枚举
print("=== Phase-Aware 枚举 ===")
print(f"PhaseRole.RESIDENT_MIGRATABLE: {sp.PhaseRole.RESIDENT_MIGRATABLE}")
print(f"PhaseRole.BURST_GPU_PINNED: {sp.PhaseRole.BURST_GPU_PINNED}")
print(f"DeviceAffinity.GPU_REQUIRED: {sp.DeviceAffinity.GPU_REQUIRED}")
print(f"DeviceAffinity.CPU_ONLY: {sp.DeviceAffinity.CPU_ONLY}")
print(f"LLMServiceLevel.GPU_NORMAL: {sp.LLMServiceLevel.GPU_NORMAL}")
print(f"MigrationPolicy.COST_AWARE: {sp.MigrationPolicy.COST_AWARE}")
print(f"SchedulerPolicy.V2_PHASE_AWARE: {sp.SchedulerPolicy.V2_PHASE_AWARE}")
print(f"LeaseType.RESIDENT: {sp.LeaseType.RESIDENT}")

# 验证数据结构
print("\n=== Phase-Aware 数据结构 ===")
attrs = sp.PhaseAttributes()
attrs.role = sp.PhaseRole.BURST_GPU_PINNED
attrs.device_affinity = sp.DeviceAffinity.GPU_REQUIRED
attrs.vram_usage_mb = 6700
print(f"PhaseAttributes role: {attrs.role}, affinity: {attrs.device_affinity}, vram: {attrs.vram_usage_mb}")

cost = sp.MigrationCost()
cost.kv_cache_save_ms = 150
cost.restore_ms = 200
print(f"MigrationCost totalRoundTrip: {cost.totalRoundTripMs()}")

# 验证策略
print("\n=== Phase-Aware 策略 ===")
strategy = sp.DefaultPhaseAwareStrategy()
print("DefaultPhaseAwareStrategy created OK")

# 验证预设
print("\n=== Phase 预设 ===")
llm_preset = sp.getPhasePreset("llm")
print(f"LLM preset role: {llm_preset.role}, affinity: {llm_preset.device_affinity}")
image_preset = sp.getPhasePreset("image")
print(f"Image preset role: {image_preset.role}, affinity: {image_preset.device_affinity}")
tts_preset = sp.getPhasePreset("tts")
print(f"TTS preset role: {tts_preset.role}, affinity: {tts_preset.device_affinity}")
rag_preset = sp.getPhasePreset("rag")
print(f"RAG preset role: {rag_preset.role}, affinity: {rag_preset.device_affinity}")

# 验证 LLMState
print("\n=== LLMState ===")
llm_state = sp.LLMState()
print(f"Initial level: {int(llm_state.level())}")
llm_state.enterDegraded()
print(f"After enterDegraded: {int(llm_state.level())}")
llm_state.enterNormal()
print(f"After enterNormal: {int(llm_state.level())}")

# 验证 GPULeaseManager
print("\n=== GPULeaseManager ===")
lease_mgr = sp.GPULeaseManager(8192)
print(f"Available VRAM: {lease_mgr.vramAvailableMb()}")

# 验证 Cost-Aware 决策
print("\n=== Cost-Aware 决策引擎 ===")
config = sp.PhaseAwareConfig()
config.policy = sp.SchedulerPolicy.V2_PHASE_AWARE
strategy2 = sp.DefaultPhaseAwareStrategy(config)
burst_attrs = sp.getPhasePreset("image")
decision = strategy2.shouldMigrateLLM(burst_attrs, llm_state, 0)
print(f"Should migrate for image burst: {decision.should_migrate}, reason: {decision.reason}")

print("\nALL TESTS PASSED!")
