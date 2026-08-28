#include "llama_model_impl.h"
#include <iostream>
#include <algorithm>
#include <thread>
#include <cstring>
#include <cmath>
#include <random>
#include <vector>
#include "llama.h"

namespace ai_scheduler {

static inline bool _utf8_is_cont(unsigned char c) {
    return (c & 0xC0) == 0x80;
}

static void _utf8_consume(std::string& pending, std::string& out, bool flush_incomplete) {
    size_t i = 0;
    while (i < pending.size()) {
        unsigned char c = (unsigned char)pending[i];
        size_t len = 0;

        if ((c & 0x80) == 0) {
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            len = 4;
        } else {
            out.append("\xEF\xBF\xBD", 3);
            i += 1;
            continue;
        }

        if (i + len > pending.size()) {
            if (flush_incomplete) {
                out.append("\xEF\xBF\xBD", 3);
                i = pending.size();
            }
            break;
        }

        bool ok = true;
        for (size_t j = 1; j < len; ++j) {
            if (!_utf8_is_cont((unsigned char)pending[i + j])) {
                ok = false;
                break;
            }
        }

        if (!ok) {
            out.append("\xEF\xBF\xBD", 3);
            i += 1;
            continue;
        }

        out.append(pending, i, len);
        i += len;
    }

    if (i > 0) {
        pending.erase(0, i);
    }
}

static void llama_batch_clear(struct llama_batch & batch) {
    batch.n_tokens = 0;
}

static void llama_batch_add(struct llama_batch & batch, llama_token id, llama_pos pos, const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits ? 1 : 0;
    batch.n_tokens++;
}

static llama_token sample_from_logits(
    const float* logits,
    int n_vocab,
    int top_k,
    float top_p,
    float temperature
) {
    if (!logits || n_vocab <= 0) {
        return 0;
    }

    if (temperature <= 0.0f) {
        int best = 0;
        float best_logit = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > best_logit) {
                best_logit = logits[i];
                best = i;
            }
        }
        return (llama_token)best;
    }

    int k = top_k > 0 ? top_k : 40;
    if (k > n_vocab) {
        k = n_vocab;
    }
    float p = top_p > 0.0f ? top_p : 1.0f;
    if (p > 1.0f) {
        p = 1.0f;
    }

    std::vector<int> idx(n_vocab);
    for (int i = 0; i < n_vocab; i++) {
        idx[i] = i;
    }

    std::nth_element(
        idx.begin(),
        idx.begin() + k,
        idx.end(),
        [logits](int a, int b) { return logits[a] > logits[b]; }
    );
    idx.resize(k);
    std::sort(
        idx.begin(),
        idx.end(),
        [logits](int a, int b) { return logits[a] > logits[b]; }
    );

    float max_logit = logits[idx[0]] / temperature;
    std::vector<float> probs;
    probs.reserve(idx.size());
    float sum = 0.0f;
    for (int id : idx) {
        float v = logits[id] / temperature;
        float ev = std::exp(v - max_logit);
        probs.push_back(ev);
        sum += ev;
    }
    if (sum <= 0.0f) {
        return (llama_token)idx[0];
    }
    for (float& v : probs) {
        v /= sum;
    }

    if (p < 1.0f) {
        float cumulative = 0.0f;
        size_t keep = 0;
        for (size_t i = 0; i < probs.size(); i++) {
            cumulative += probs[i];
            keep++;
            if (cumulative >= p) {
                break;
            }
        }
        if (keep < 1) {
            keep = 1;
        }
        idx.resize(keep);
        probs.resize(keep);
        float renorm = 0.0f;
        for (float v : probs) {
            renorm += v;
        }
        if (renorm > 0.0f) {
            for (float& v : probs) {
                v /= renorm;
            }
        }
    }

    static thread_local std::mt19937 rng{ std::random_device{}() };
    std::discrete_distribution<int> dist(probs.begin(), probs.end());
    int pick = dist(rng);
    if (pick < 0 || pick >= (int)idx.size()) {
        pick = 0;
    }
    return (llama_token)idx[pick];
}

static bool g_llama_backend_initialized = false;

LlamaCppModel::LlamaCppModel() {
    if (!g_llama_backend_initialized) {
        llama_backend_init();
        g_llama_backend_initialized = true;
    }
}

LlamaCppModel::~LlamaCppModel() {
    shutdown();
}

bool LlamaCppModel::initialize(const LLMModelConfig& config) {
    if (ready_) return true;

    config_ = config;
    std::cout << "Loading llama model from: " << config_.modelPath << std::endl;

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = config_.gpuDeviceId >= 0 ? 999 : 0;
    model_params.main_gpu = config_.gpuDeviceId >= 0 ? config_.gpuDeviceId : 0;

    model_ = llama_model_load_from_file(config_.modelPath.c_str(), model_params);
    if (!model_) {
        std::cerr << "Failed to load llama model: " << config_.modelPath << std::endl;
        return false;
    }

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = config_.maxContextSize > 0 ? (uint32_t)config_.maxContextSize : 4096;
            ctx_params.n_batch = config_.maxBatchSize > 0 ? (uint32_t)config_.maxBatchSize : 512;

            max_sessions_ = 0;
            conv_to_seq_.clear();
            seq_to_conv_.clear();
            seq_tokens_.clear();
            lru_.clear();
            if (config_.enableCache && config_.cacheSize > 0) {
                max_sessions_ = config_.cacheSize;
            }

            uint32_t n_seq_max = 1;
            if (max_sessions_ > 0) {
                n_seq_max = (uint32_t)(max_sessions_ + 1);
                if (n_seq_max < 2) n_seq_max = 2;
                if (n_seq_max > 128) n_seq_max = 128;
            }
            ctx_params.n_seq_max = n_seq_max;

    ctx_ = llama_init_from_model(model_, ctx_params);
    if (!ctx_) {
        std::cerr << "Failed to create llama context" << std::endl;
        llama_model_free(model_);
        model_ = nullptr;
        return false;
    }

    if (!config_.draftModelPath.empty()) {
        std::cout << "Loading draft model from: " << config_.draftModelPath << std::endl;
        llama_model_params draft_params = llama_model_default_params();
        draft_params.n_gpu_layers = config_.draftGpuDeviceId >= 0 ? 999 : 0;
        draft_params.main_gpu = config_.draftGpuDeviceId >= 0 ? config_.draftGpuDeviceId : 0;

        draft_model_ = llama_model_load_from_file(config_.draftModelPath.c_str(), draft_params);
        if (draft_model_) {
            llama_context_params draft_ctx_params = llama_context_default_params();
            draft_ctx_params.n_ctx = config_.draftContextSize > 0 ? (uint32_t)config_.draftContextSize : 512;
            draft_ctx_params.n_batch = config_.maxBatchSize > 0 ? (uint32_t)config_.maxBatchSize : 512;
            draft_ctx_params.n_seq_max = 1;

            draft_ctx_ = llama_init_from_model(draft_model_, draft_ctx_params);
            if (!draft_ctx_) {
                std::cerr << "Failed to create draft context" << std::endl;
                llama_model_free(draft_model_);
                draft_model_ = nullptr;
            } else {
                std::cout << "Draft model loaded successfully." << std::endl;
            }
        } else {
            std::cerr << "Failed to load draft model: " << config_.draftModelPath << std::endl;
        }
    }

    ready_ = true;
    std::cout << "Llama model loaded successfully." << std::endl;
    return true;
}

void LlamaCppModel::shutdown() {
    if (ctx_) {
        llama_free(ctx_);
        ctx_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }

    if (draft_ctx_) {
        llama_free(draft_ctx_);
        draft_ctx_ = nullptr;
    }
    if (draft_model_) {
        llama_model_free(draft_model_);
        draft_model_ = nullptr;
    }

    ready_ = false;

    max_sessions_ = 0;
    conv_to_seq_.clear();
    seq_to_conv_.clear();
    seq_tokens_.clear();
    lru_.clear();
}

std::vector<int> LlamaCppModel::tokenize(const std::string& text, bool add_bos) {
    const llama_vocab* vocab = llama_model_get_vocab(model_);
    int n_tokens = (int)text.length() + 2;
    std::vector<llama_token> tokens(n_tokens);

    int32_t n = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(), tokens.data(), (int32_t)tokens.size(), add_bos, false);

    if (n < 0) {
        tokens.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int32_t)text.length(), tokens.data(), (int32_t)tokens.size(), add_bos, false);
    }

    std::vector<int> result;
    if (n > 0) {
        result.reserve(n);
        for (int i = 0; i < n; i++) {
            result.push_back(tokens[i]);
        }
    }
    return result;
}

std::string LlamaCppModel::detokenize(const std::vector<int>& tokens) {
    if (tokens.empty()) return "";

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    std::string result;
    for (int token : tokens) {
        char buf[256];
        int n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        if (n < 0) {
            n = llama_token_to_piece(vocab, token, buf, sizeof(buf), 0, true);
        }
        if (n > 0) {
            result.append(buf, n);
        }
    }
    return result;
}

int32_t LlamaCppModel::getSeqId(const std::string& conversationId) {
    auto it = conv_to_seq_.find(conversationId);
    if (it != conv_to_seq_.end()) {
        return it->second;
    }

    if (max_sessions_ > 0) {
        while (conv_to_seq_.size() >= max_sessions_ && !lru_.empty()) {
            int32_t evict_id = lru_.front();
            lru_.pop_front();

            auto conv_it = seq_to_conv_.find(evict_id);
            if (conv_it != seq_to_conv_.end()) {
                conv_to_seq_.erase(conv_it->second);
                seq_to_conv_.erase(conv_it);
            }
            seq_tokens_.erase(evict_id);

            if (ctx_) {
                llama_memory_seq_rm(llama_get_memory(ctx_), (llama_seq_id)evict_id, 0, -1);
            }
        }
    }

    int32_t max_seq = (int32_t)llama_n_seq_max(ctx_);
    if (max_seq <= 0) max_seq = 1;

    int32_t seq = -1;
    for (int32_t i = 0; i < max_seq; i++) {
        if (seq_to_conv_.find(i) == seq_to_conv_.end()) {
            seq = i;
            break;
        }
    }

    if (seq == -1) {
        seq = 0;
    }

    conv_to_seq_[conversationId] = seq;
    seq_to_conv_[seq] = conversationId;
    lru_.push_back(seq);
    return seq;
}

void LlamaCppModel::touchSeq(int32_t seqId) {
    for (auto it = lru_.begin(); it != lru_.end();) {
        if (*it == seqId) {
            it = lru_.erase(it);
            continue;
        }
        ++it;
    }
    lru_.push_back(seqId);
}

void LlamaCppModel::evictIfNeeded() {
    if (max_sessions_ == 0) {
        return;
    }

    while (conv_to_seq_.size() > max_sessions_ && !lru_.empty()) {
        int32_t evict_id = lru_.front();
        lru_.pop_front();

        auto conv_it = seq_to_conv_.find(evict_id);
        if (conv_it != seq_to_conv_.end()) {
            conv_to_seq_.erase(conv_it->second);
            seq_to_conv_.erase(conv_it);
        }
        seq_tokens_.erase(evict_id);

        if (ctx_) {
            llama_memory_seq_rm(llama_get_memory(ctx_), (llama_seq_id)evict_id, 0, -1);
        }
    }
}

LLMInferenceResponse LlamaCppModel::generate(const LLMInferenceRequest& request) {
    if (!ready_) {
        return { "", 0, 0.0f, false, "Model not ready" };
    }

    if (request.shouldStop && request.shouldStop()) {
        return { "", 0, 0.0f, false, "Cancelled" };
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    const bool use_cache =
        max_sessions_ > 0 && config_.enableCache && !request.conversationId.empty();

    llama_seq_id seq_id = 0;
    std::vector<int>* cached_tokens = nullptr;
    if (use_cache) {
        int32_t sid = getSeqId(request.conversationId);
        seq_id = (llama_seq_id)sid;
        touchSeq(sid);
        cached_tokens = &seq_tokens_[sid];
    } else {
        seq_id = 0;
        llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, 0, -1);
    }

    prompt_tokens_ = tokenize(request.prompt, true);

    int n_ctx = llama_n_ctx(ctx_);
    if ((int)prompt_tokens_.size() > n_ctx - 4) {
        return { "", 0, 0.0f, false, "Prompt too long for context" };
    }

    const size_t max_tokens = request.maxTokens > 0 ? request.maxTokens : 512;
    const float temperature = request.temperature > 0.0f ? request.temperature : config_.temperature;
    const int top_k = request.topK > 0 ? request.topK : config_.topK;
    const float top_p = request.topP > 0.0f ? request.topP : config_.topP;

    int32_t n_seq_max = (int32_t)llama_n_seq_max(ctx_);
    if (n_seq_max <= 0) {
        n_seq_max = 1;
    }

    const int batch_size = (int)(config_.maxBatchSize > 0 ? config_.maxBatchSize : 512);
    llama_batch batch = llama_batch_init(batch_size, 0, 1);
    llama_batch draft_batch = llama_batch_init(batch_size, 0, 1);

    bool use_speculative = (draft_ctx_ != nullptr) && (temperature < 0.3f);
    const int n_draft = 5;

    int n_past = 0;
    if (use_cache && cached_tokens) {
        size_t common = 0;
        const size_t max_common = (std::min)(cached_tokens->size(), prompt_tokens_.size());
        while (common < max_common && (*cached_tokens)[common] == prompt_tokens_[common]) {
            common += 1;
        }
        if (common < cached_tokens->size()) {
            llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, (llama_pos)common, -1);
            cached_tokens->resize(common);
        }
        n_past = (int)common;

        if (use_speculative) {
            llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, 0, -1);
        }
    } else {
        if (use_speculative) {
            llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, 0, -1);
        }
    }

    for (size_t i = (size_t)n_past; i < prompt_tokens_.size();) {
        llama_batch_clear(batch);
        const size_t remaining = prompt_tokens_.size() - i;
        const size_t chunk = remaining > (size_t)batch_size ? (size_t)batch_size : remaining;
        for (size_t j = 0; j < chunk; j++) {
            const size_t pos = i + j;
            llama_batch_add(batch, (llama_token)prompt_tokens_[pos], (llama_pos)pos, { seq_id }, false);
        }
        if (i + chunk == prompt_tokens_.size() && batch.n_tokens > 0) {
            batch.logits[batch.n_tokens - 1] = true;
        }

        if (llama_decode(ctx_, batch) != 0) {
            llama_batch_free(batch);
            llama_batch_free(draft_batch);
            return { "", 0, 0.0f, false, "Failed to decode prompt" };
        }

        if (use_speculative) {
            llama_batch_clear(draft_batch);
            for (size_t j = 0; j < chunk; j++) {
                const size_t pos = i + j;
                llama_batch_add(draft_batch, (llama_token)prompt_tokens_[pos], (llama_pos)pos, { 0 }, j == chunk - 1);
            }
            if (llama_decode(draft_ctx_, draft_batch) != 0) {
                use_speculative = false;
            }
        }

        i += chunk;
    }

    int n_cur = (int)prompt_tokens_.size();
    if (use_cache && cached_tokens) {
        cached_tokens->assign(prompt_tokens_.begin(), prompt_tokens_.end());
    }

    int n_decode = 0;
    std::string generated_text;
    std::string pending_stream_buffer;
    std::vector<int> output_tokens;

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const llama_vocab* draft_vocab = use_speculative ? llama_model_get_vocab(draft_model_) : nullptr;

    const size_t max_tokens_by_ctx =
        n_ctx > n_cur + 1 ? (size_t)(n_ctx - n_cur - 1) : (size_t)0;
    const size_t max_tokens_effective = (std::min)(max_tokens, max_tokens_by_ctx);

    llama_sampler * smpl = nullptr;
    if (temperature <= 0.0f) {
        smpl = llama_sampler_init_greedy();
    } else {
        auto sparams = llama_sampler_chain_default_params();
        smpl = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(top_k));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p > 0.0f ? top_p : 1.0f, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
    }

    while ((size_t)n_decode < max_tokens_effective) {
        if (request.shouldStop && request.shouldStop()) {
            break;
        }

        if (use_speculative) {
            std::vector<llama_token> drafts;
            drafts.reserve(n_draft);

            for (int i = 0; i < n_draft; ++i) {
                auto* logits = llama_get_logits(draft_ctx_);
                int n_vocab_draft = llama_vocab_n_tokens(draft_vocab);
                llama_token draft_token = sample_from_logits(logits, n_vocab_draft, top_k, top_p, temperature);

                int n_vocab_target = llama_vocab_n_tokens(vocab);
                if (draft_token >= n_vocab_target) {
                    use_speculative = false;
                    break;
                }

                drafts.push_back(draft_token);

                llama_batch_clear(draft_batch);
                llama_batch_add(draft_batch, draft_token, n_cur + i, { 0 }, true);
                if (llama_decode(draft_ctx_, draft_batch) != 0) {
                    use_speculative = false;
                    break;
                }
            }

            if (!use_speculative) continue;

            llama_batch_clear(batch);
            auto* logits = llama_get_logits(ctx_);
            int n_vocab_target = llama_vocab_n_tokens(vocab);

            llama_token target_token = sample_from_logits(logits, n_vocab_target, top_k, top_p, temperature);

            std::vector<llama_token> accepted;
            bool mismatched = false;

            if (target_token == drafts[0]) {
                accepted.push_back(target_token);

                llama_batch_clear(batch);
                for (size_t i = 0; i < drafts.size(); ++i) {
                    llama_batch_add(batch, drafts[i], n_cur + i, { seq_id }, true);
                }

                if (llama_decode(ctx_, batch) != 0) {
                    break;
                }

                for (size_t i = 0; i < drafts.size(); ++i) {
                    float* ith_logits = llama_get_logits_ith(ctx_, (int32_t)i);
                    llama_token next_pred = sample_from_logits(ith_logits, n_vocab_target, top_k, top_p, temperature);

                    if (i < drafts.size() - 1) {
                        if (next_pred == drafts[i+1]) {
                            accepted.push_back(next_pred);
                        } else {
                            accepted.push_back(next_pred);
                            mismatched = true;

                            llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, n_cur + i + 1, -1);
                            llama_memory_seq_rm(llama_get_memory(ctx_), seq_id, n_cur + i + 1, -1);
                            break;
                        }
                    } else {
                        accepted.push_back(next_pred);
                    }
                }

            } else {
                accepted.push_back(target_token);
                mismatched = true;
                llama_memory_seq_rm(llama_get_memory(draft_ctx_), 0, n_cur, -1);
            }

            for (llama_token t : accepted) {
                output_tokens.push_back(t);
                char buf[256];
                int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
                if (n > 0) {
                    pending_stream_buffer.append(buf, n);
                    std::string emitted;
                    _utf8_consume(pending_stream_buffer, emitted, false);
                    if (!emitted.empty()) {
                        generated_text.append(emitted);
                        if (request.streamOutput && request.onTokenGenerated) {
                            request.onTokenGenerated(emitted);
                        }
                    }
                }

                if (llama_vocab_is_eog(vocab, t) || t == llama_vocab_eos(vocab)) {
                    n_decode = (int)max_tokens_effective;
                    break;
                }
            }

            n_cur += (int)accepted.size();
            n_decode += (int)accepted.size();

            llama_token last_accepted = accepted.back();
            llama_batch_clear(batch);
            llama_batch_add(batch, last_accepted, n_cur - 1, { seq_id }, true);
            if (llama_decode(ctx_, batch) != 0) {
                break;
            }

            llama_batch_clear(draft_batch);
            llama_batch_add(draft_batch, last_accepted, n_cur - 1, { 0 }, true);
            llama_decode(draft_ctx_, draft_batch);
            (void)mismatched;

        } else {
            llama_token new_token_id = smpl ? llama_sampler_sample(smpl, ctx_, -1) : llama_vocab_eos(vocab);

            if (llama_vocab_is_eog(vocab, new_token_id) || new_token_id == llama_vocab_eos(vocab)) {
                break;
            }

            output_tokens.push_back(new_token_id);

            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n > 0) {
                pending_stream_buffer.append(buf, n);
                std::string emitted;
                _utf8_consume(pending_stream_buffer, emitted, false);
                if (!emitted.empty()) {
                    generated_text.append(emitted);
                    if (request.streamOutput && request.onTokenGenerated) {
                        request.onTokenGenerated(emitted);
                    }
                }
            }

            llama_batch_clear(batch);
            llama_batch_add(batch, new_token_id, n_cur, { seq_id }, true);

            n_cur++;
            n_decode++;

            if (llama_decode(ctx_, batch) != 0) {
                break;
            }
        }
    }

    if (smpl) {
        llama_sampler_free(smpl);
        smpl = nullptr;
    }

    llama_batch_free(batch);
    llama_batch_free(draft_batch);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float> duration = end_time - start_time;

    if (use_cache && cached_tokens) {
        if (!output_tokens.empty()) {
            cached_tokens->insert(cached_tokens->end(), output_tokens.begin(), output_tokens.end());
        }
        touchSeq((int32_t)seq_id);
        evictIfNeeded();
    }

    if (request.shouldStop && request.shouldStop()) {
        return { generated_text, (size_t)n_decode, duration.count(), false, "Cancelled" };
    }

    if (!pending_stream_buffer.empty()) {
        std::string tail;
        _utf8_consume(pending_stream_buffer, tail, true);
        if (!tail.empty()) {
            generated_text.append(tail);
        }
    }

    return { generated_text, (size_t)n_decode, duration.count(), true, "" };
}

std::string LlamaCppModel::getModelInfo() const {
    char buf[128];
    llama_model_desc(model_, buf, sizeof(buf));
    return std::string("LlamaCppModel: ") + buf;
}

size_t LlamaCppModel::getMemoryUsage() const {
    return llama_model_size(model_);
}

bool LlamaCppModel::isReady() const {
    return ready_;
}

bool LlamaCppModel::offloadKVCache(const std::string& path) {
    if (!ctx_) {
        std::cerr << "offloadKVCache: 无有效 llama_context" << std::endl;
        return false;
    }

    // 使用 llama.cpp 的 state save API
    // llama_state_get_size / llama_state_get_data / llama_state_set_data
    // 注意：这些 API 在不同 llama.cpp 版本间变化较大
    // TODO: 在 WSL2 编译环境中根据实际 llama.cpp 版本实现
    std::cout << "offloadKVCache: 保存 KV Cache 到 " << path << std::endl;

    // 获取 state 大小
    const size_t state_size = llama_state_get_size(ctx_);
    if (state_size == 0) {
        std::cerr << "offloadKVCache: state 大小为 0" << std::endl;
        return false;
    }

    // 分配缓冲区并保存 state
    std::vector<uint8_t> state_buf(state_size);
    const size_t saved = llama_state_get_data(ctx_, state_buf.data(), state_size);
    if (saved != state_size) {
        std::cerr << "offloadKVCache: 保存数据不完整 (" << saved << "/" << state_size << ")" << std::endl;
        return false;
    }

    // 写入文件
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        std::cerr << "offloadKVCache: 无法打开文件 " << path << std::endl;
        return false;
    }
    size_t written = fwrite(state_buf.data(), 1, state_size, f);
    fclose(f);

    if (written != state_size) {
        std::cerr << "offloadKVCache: 写入不完整 (" << written << "/" << state_size << ")" << std::endl;
        return false;
    }

    std::cout << "offloadKVCache: 成功保存 " << state_size << " 字节" << std::endl;
    return true;
}

bool LlamaCppModel::restoreKVCache(const std::string& path) {
    if (!ctx_) {
        std::cerr << "restoreKVCache: 无有效 llama_context" << std::endl;
        return false;
    }

    std::cout << "restoreKVCache: 从 " << path << " 恢复 KV Cache" << std::endl;

    // 读取文件
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        std::cerr << "restoreKVCache: 无法打开文件 " << path << std::endl;
        return false;
    }

    fseek(f, 0, SEEK_END);
    const size_t file_size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (file_size == 0) {
        fclose(f);
        std::cerr << "restoreKVCache: 文件为空" << std::endl;
        return false;
    }

    std::vector<uint8_t> state_buf(file_size);
    size_t read_bytes = fread(state_buf.data(), 1, file_size, f);
    fclose(f);

    if (read_bytes != file_size) {
        std::cerr << "restoreKVCache: 读取不完整" << std::endl;
        return false;
    }

    // 恢复 state
    const size_t restored = llama_state_set_data(ctx_, state_buf.data(), file_size);
    if (restored != file_size) {
        std::cerr << "restoreKVCache: 恢复数据不完整 (" << restored << "/" << file_size << ")" << std::endl;
        return false;
    }

    std::cout << "restoreKVCache: 成功恢复 " << file_size << " 字节" << std::endl;
    return true;
}

} // namespace ai_scheduler

