// Logit-equivalence gate hooks (C1.3-05). Both are env-driven and inert unless
// the env var names a file:
//   LUCE_LOGITS_DUMP=<file>   append the last-position logits of every step
//   LUCE_TEACHER_FORCE=<file> int32 tokens that replace the sampled token, so
//                             two runs follow one token path
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace luce::common {

class LogitsDump {
public:
    static LogitsDump from_env(const char * v) {
        LogitsDump d;
        if (v && *v) d.path_ = v;
        return d;
    }
    bool enabled() const { return !path_.empty(); }
    bool append(const float * logits, size_t n) const {
        if (!enabled()) return false;
        FILE * f = std::fopen(path_.c_str(), "ab");
        if (!f) return false;
        const bool ok = std::fwrite(logits, sizeof(float), n, f) == n;
        std::fclose(f);
        return ok;
    }
private:
    std::string path_;
};

class TeacherForce {
public:
    static TeacherForce from_env(const char * v) {
        TeacherForce t;
        if (!v || !*v) return t;
        FILE * f = std::fopen(v, "rb");
        if (!f) return t;
        int32_t x;
        while (std::fread(&x, sizeof x, 1, f) == 1) t.toks_.push_back(x);
        std::fclose(f);
        return t;
    }
    bool enabled() const { return !toks_.empty(); }
    // Forced token for generation step i (0 = the first token), -1 when none.
    int32_t at(size_t i) const { return i < toks_.size() ? toks_[i] : -1; }
private:
    std::vector<int32_t> toks_;
};

}  // namespace luce::common
