#pragma once
#include "gpu_llm_worker.h"
#include <string>
#include <vector>
#include <memory>
#include <unordered_map>
#include <deque>
#include <cstdint>

// Forward declarations for llama.cpp types
struct llama_model;
struct llama_context;
struct llama_token_data;
struct llama_sampler;
struct llama_batch;

namespace ai_scheduler {

class LlamaCppModel : public ILLMModel {
public:
    LlamaCppModel();
    ~LlamaCppModel() override;

    bool initialize(const LLMModelConfig& config) override;
    void shutdown() override;
    LLMInferenceResponse generate(const LLMInferenceRequest& request) override;
    std::string getModelInfo() const override;
    size_t getMemoryUsage() const override;
    bool isReady() const override;

    /// KV Cache Offload：将 KV Cache 序列化到磁盘，释放 GPU 显存
    /// 用于 Phase-Aware 迁移：LLM GPU→CPU 时保存 KV Cache
    bool offloadKVCache(const std::string& path) override;

    /// KV Cache Restore：从磁盘恢复 KV Cache 到 GPU
    /// 用于 Phase-Aware 回迁：LLM CPU→GPU 时恢复 KV Cache
    bool restoreKVCache(const std::string& path) override;

private:
    // Internal helper methods
    std::vector<int> tokenize(const std::string& text, bool add_bos);
    std::string detokenize(const std::vector<int>& tokens);

    int32_t getSeqId(const std::string& conversationId);
    void touchSeq(int32_t seqId);
    void evictIfNeeded();
    
    // Llama.cpp context
    llama_model* model_ = nullptr;
    llama_context* ctx_ = nullptr;
    
    // Draft model for speculative decoding
    llama_model* draft_model_ = nullptr;
    llama_context* draft_ctx_ = nullptr;

    // llama_sampler* sampler_ = nullptr; // Removed for compatibility with older llama.cpp
    
    // Configuration
    LLMModelConfig config_;
    bool ready_ = false;

    size_t max_sessions_ = 0;
    std::unordered_map<std::string, int32_t> conv_to_seq_;
    std::unordered_map<int32_t, std::string> seq_to_conv_;
    std::unordered_map<int32_t, std::vector<int>> seq_tokens_;
    std::deque<int32_t> lru_;
    
    // Buffers
    std::vector<int> prompt_tokens_;
    std::vector<int> generated_tokens_;
};

} // namespace ai_scheduler
