#include "biological_system.h"
#include "nvram_binding.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <ctime>

namespace ai_scheduler {

BiologicalSystem::BiologicalSystem() 
    : energy_(1.0f), sleep_debt_(0.0f), phase_(CircadianPhase::ACTIVE) {
}

BiologicalSystem::~BiologicalSystem() {
}

void BiologicalSystem::initialize() {
    lastUpdate_ = std::chrono::system_clock::now();
    neurotransmitters_.dopamine = config_.baseline_dopamine;
    neurotransmitters_.serotonin = config_.baseline_serotonin;
    neurotransmitters_.norepinephrine = config_.baseline_norepinephrine;
    neurotransmitters_.oxytocin = config_.baseline_oxytocin;
    neurotransmitters_.cortisol = config_.baseline_cortisol;
    energy_ = 1.0f;
    sleep_debt_ = 0.0f;

    // Initialize NVRAM binding
    // Assuming default path or passed via config. For MVP, we use default relative path logic or fixed.
    // In production, this should come from config.
    // Since we are inside legacy/mvp_core/cpp_scheduler/core, we need to point to legacy/mvp_core/data/nvram.dat
    // But working directory might be different. 
    // Best effort: Try absolute path if known, or relative.
    // We'll let the singleton manage it, assume caller initializes it or we try default.
    NVRAMContext::getInstance().initialize("data/nvram.dat"); // Try relative first
    
    // Check if we should restore from NVRAM?
    // Design doc says: "Restart should restore state".
    BioState state = NVRAMContext::getInstance().getBioState();
    if (state.timestamp > 0) {
        // Valid state found
        neurotransmitters_.dopamine = state.dopamine;
        energy_ = state.energy;
        // Mood is derived, so we don't set it directly back to components unless we reverse map.
        // For MVP, we trust the in-memory reset + NVRAM sync forward.
        // Or if we want strict restore:
        // dopamine is direct. energy is direct.
        // mood is (serotonin + dopamine)/2. So serotonin = 2*mood - dopamine.
        neurotransmitters_.serotonin = 2.0 * state.mood - state.dopamine;
        
        std::cout << "BiologicalSystem: Restored from NVRAM. Dopamine=" << neurotransmitters_.dopamine << std::endl;
    }
}

void BiologicalSystem::update(float deltaTimeSeconds) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    decayNeurotransmitters(deltaTimeSeconds);
    updateCircadianRhythm();
    
    // Natural energy decay if awake
    if (phase_ != CircadianPhase::SLEEP) {
        energy_ -= config_.energy_awake_decay * deltaTimeSeconds;
        sleep_debt_ += config_.sleep_debt_awake_gain * deltaTimeSeconds;
    } else {
        energy_ += config_.energy_sleep_recover * deltaTimeSeconds;
        sleep_debt_ -= config_.sleep_debt_sleep_recover * deltaTimeSeconds;
    }
    
    energy_ = std::clamp(energy_, 0.0f, 1.0f);
    sleep_debt_ = std::clamp(sleep_debt_, 0.0f, 1.0f);

    // Sync to NVRAM
    // Mood = (Serotonin + Dopamine) / 2
    double mood = (neurotransmitters_.serotonin + neurotransmitters_.dopamine) / 2.0;
    NVRAMContext::getInstance().updateBioState(
        neurotransmitters_.dopamine,
        energy_,
        mood
    );
}

void BiologicalSystem::decayNeurotransmitters(float deltaTime) {
    auto decay_to = [this, deltaTime](float& val, float baseline) {
        float diff = baseline - val;
        val += diff * config_.decay_rate * deltaTime;
        val = std::clamp(val, 0.0f, 1.0f);
    };

    decay_to(neurotransmitters_.dopamine, config_.baseline_dopamine);
    decay_to(neurotransmitters_.serotonin, config_.baseline_serotonin);
    decay_to(neurotransmitters_.norepinephrine, config_.baseline_norepinephrine);
    decay_to(neurotransmitters_.oxytocin, config_.baseline_oxytocin);
    decay_to(neurotransmitters_.cortisol, config_.baseline_cortisol);
}

void BiologicalSystem::updateCircadianRhythm() {
    // Simple simulation based on system time
    auto now = std::chrono::system_clock::now();
    time_t t = std::chrono::system_clock::to_time_t(now);
    struct tm* tm = std::localtime(&t);
    
    int hour = tm->tm_hour;
    
    if (hour >= 7 && hour < 9) phase_ = CircadianPhase::WAKE;
    else if (hour >= 9 && hour < 21) phase_ = CircadianPhase::ACTIVE;
    else if (hour >= 21 && hour < 23) phase_ = CircadianPhase::TIRED;
    else phase_ = CircadianPhase::SLEEP;
}

Neurotransmitter BiologicalSystem::getNeurotransmitters() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return neurotransmitters_;
}

void BiologicalSystem::adjustNeurotransmitter(const std::string& name, float delta) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (name == "dopamine") neurotransmitters_.dopamine += delta;
    else if (name == "serotonin") neurotransmitters_.serotonin += delta;
    else if (name == "norepinephrine") neurotransmitters_.norepinephrine += delta;
    else if (name == "oxytocin") neurotransmitters_.oxytocin += delta;
    else if (name == "cortisol") neurotransmitters_.cortisol += delta;
    
    // Clamp
    auto clamp = [](float& val) { val = std::clamp(val, 0.0f, 1.0f); };
    clamp(neurotransmitters_.dopamine);
    clamp(neurotransmitters_.serotonin);
    clamp(neurotransmitters_.norepinephrine);
    clamp(neurotransmitters_.oxytocin);
    clamp(neurotransmitters_.cortisol);
}

float BiologicalSystem::getEnergy() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return energy_;
}

void BiologicalSystem::consumeEnergy(float amount) {
    std::lock_guard<std::mutex> lock(mutex_);
    energy_ -= amount;
    energy_ = std::clamp(energy_, 0.0f, 1.0f);
}

void BiologicalSystem::recoverEnergy(float amount) {
    std::lock_guard<std::mutex> lock(mutex_);
    energy_ += amount;
    energy_ = std::clamp(energy_, 0.0f, 1.0f);
}

float BiologicalSystem::getSleepDebt() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sleep_debt_;
}

CircadianPhase BiologicalSystem::getCircadianPhase() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return phase_;
}

float BiologicalSystem::calculateCognitiveDelay(float complexity) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    float baseDelay = config_.cognitive_base_delay;
    
    // Energy impact: Low energy -> higher delay
    float energyFactor = 1.0f + (1.0f - energy_) * config_.cognitive_energy_scale;
    
    // Neurotransmitter impact
    // High dopamine -> faster (lower delay)
    // Low serotonin -> slower (hesitation)
    float dopamineFactor = 1.0f - (neurotransmitters_.dopamine - config_.baseline_dopamine) * config_.cognitive_dopamine_scale;
    float serotoninFactor = 1.0f - (neurotransmitters_.serotonin - config_.baseline_serotonin) * config_.cognitive_serotonin_scale;
    float cortisolFactor = 1.0f + (neurotransmitters_.cortisol - config_.baseline_cortisol) * config_.cognitive_cortisol_scale;
    float sleepDebtFactor = 1.0f + sleep_debt_ * config_.cognitive_sleep_debt_scale;
    
    // Complexity scaling
    float complexityDelay = complexity * config_.cognitive_complexity_scale;
    
    float totalDelay = (baseDelay + complexityDelay) * energyFactor * dopamineFactor * serotoninFactor * cortisolFactor * sleepDebtFactor;
    
    return std::max(0.1f, totalDelay);
}

BiologicalConfig BiologicalSystem::getConfig() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return config_;
}

void BiologicalSystem::setConfig(const BiologicalConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
}

} // namespace ai_scheduler
