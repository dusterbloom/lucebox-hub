#include "lsa_runtime.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_set>
#include <utility>

namespace dflash::common {
namespace {

void sort_unique(std::vector<int> & values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

std::vector<int> difference(const std::vector<int> & lhs,
                            const std::vector<int> & rhs) {
    std::vector<int> out;
    std::set_difference(lhs.begin(), lhs.end(), rhs.begin(), rhs.end(),
                        std::back_inserter(out));
    return out;
}

}  // namespace

LsaRuntime::LsaRuntime(LsaConfig config, LsaRetriever & retriever)
    : config_(config), retriever_(retriever) {}

bool LsaRuntime::add_chunk(LsaChunk chunk, std::string & error) {
    error.clear();
    if (chunk.id < 0) {
        error = "chunk id must be non-negative";
        return false;
    }
    if (chunk.token_begin < 0 || chunk.token_end <= chunk.token_begin) {
        error = "chunk token range is invalid";
        return false;
    }
    if ((int)chunk.index_key.size() != retriever_.key_size()) {
        error = "index key size does not match retriever";
        return false;
    }
    if (!std::all_of(chunk.index_key.begin(), chunk.index_key.end(),
                     [](float value) { return std::isfinite(value); })) {
        error = "index key contains a non-finite value";
        return false;
    }
    const auto duplicate = std::find_if(
        chunks_.begin(), chunks_.end(),
        [&](const LsaChunk & existing) { return existing.id == chunk.id; });
    if (duplicate != chunks_.end()) {
        error = "chunk id already exists";
        return false;
    }
    if (!chunks_.empty() && chunk.token_begin < chunks_.back().token_end) {
        error = "chunks must be added in non-overlapping token order";
        return false;
    }

    chunks_.push_back(std::move(chunk));
    ++catalog_version_;
    return true;
}

void LsaRuntime::clear() {
    chunks_.clear();
    hot_chunks_.clear();
    stats_ = {};
    ++catalog_version_;
    ++hot_version_;
    next_trigger_token_ = 0;
}

bool LsaRuntime::plan(int committed_tokens, const std::vector<float> & hidden,
                      LsaPlan & out, std::string & error) {
    out = {};
    error.clear();

    if (config_.retrieval_interval <= 0) {
        error = "retrieval interval must be positive";
        return false;
    }
    if (config_.top_k < 0 || config_.attention_sink_chunks < 0 ||
        config_.recent_chunks < 0) {
        error = "LSA chunk counts must be non-negative";
        return false;
    }
    if (committed_tokens < 0) {
        error = "committed token count must be non-negative";
        return false;
    }
    if (committed_tokens < next_trigger_token_) {
        out.keep = hot_chunks_;
        return true;
    }
    if ((int)hidden.size() != retriever_.hidden_size()) {
        error = "hidden size does not match retriever";
        return false;
    }

    out.triggered = true;
    out.catalog_version = catalog_version_;
    out.hot_version = hot_version_;
    out.committed_tokens = committed_tokens;

    std::vector<float> scores;
    if (!retriever_.score(hidden, chunks_, scores, error)) {
        if (error.empty()) error = "retriever scoring failed";
        return false;
    }
    if (scores.size() != chunks_.size()) {
        error = "retriever score count does not match chunk count";
        return false;
    }
    if (!std::all_of(scores.begin(), scores.end(),
                     [](float value) { return std::isfinite(value); })) {
        error = "retriever returned a non-finite score";
        return false;
    }

    if (config_.top_k > 0) {
        std::vector<size_t> order(chunks_.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
            [&](size_t lhs, size_t rhs) {
                if (scores[lhs] != scores[rhs]) return scores[lhs] > scores[rhs];
                return chunks_[lhs].id < chunks_[rhs].id;
            });
        const size_t count =
            std::min(order.size(), static_cast<size_t>(config_.top_k));
        for (size_t i = 0; i < count; ++i) {
            out.keep.push_back(chunks_[order[i]].id);
        }
    } else {
        if (!std::isfinite(config_.threshold)) {
            error = "retrieval threshold must be finite";
            return false;
        }
        for (size_t i = 0; i < chunks_.size(); ++i) {
            if (scores[i] >= config_.threshold) {
                out.keep.push_back(chunks_[i].id);
            }
        }
    }

    const size_t sink_count = std::min(
        chunks_.size(), static_cast<size_t>(config_.attention_sink_chunks));
    for (size_t i = 0; i < sink_count; ++i) {
        out.keep.push_back(chunks_[i].id);
    }

    const size_t recent_count = std::min(
        chunks_.size(), static_cast<size_t>(config_.recent_chunks));
    for (size_t i = chunks_.size() - recent_count; i < chunks_.size(); ++i) {
        out.keep.push_back(chunks_[i].id);
    }

    sort_unique(out.keep);
    out.load = difference(out.keep, hot_chunks_);
    out.evict = difference(hot_chunks_, out.keep);
    return true;
}

bool LsaRuntime::commit(const LsaPlan & plan, LsaResidency & residency,
                        std::string & error) {
    error.clear();
    if (!plan.triggered) {
        error = "cannot commit an untriggered LSA plan";
        return false;
    }
    if (plan.catalog_version != catalog_version_ ||
        plan.hot_version != hot_version_) {
        error = "LSA plan is stale";
        return false;
    }
    if (!residency.transition(plan.load, plan.evict, error)) {
        if (error.empty()) error = "LSA residency transition failed";
        return false;
    }

    hot_chunks_ = plan.keep;
    ++hot_version_;
    next_trigger_token_ =
        (plan.committed_tokens / config_.retrieval_interval + 1) *
        config_.retrieval_interval;
    ++stats_.retrieval_cycles;
    stats_.loaded_chunks += plan.load.size();
    stats_.evicted_chunks += plan.evict.size();
    stats_.max_hot_chunks =
        std::max(stats_.max_hot_chunks,
                 static_cast<uint64_t>(hot_chunks_.size()));
    return true;
}

}  // namespace dflash::common
