// KVFlash C1.4 item 3: one slot's parked session on disk. A bundle is the
// session's history, its recurrent slab at the history end, its prompt-end
// checkpoint and the pager's serialize() blob, behind an identity hash of the
// server that wrote it (model file, KV types, geometry). Pure: pack/unpack are
// byte-exact and validate every length before they copy; the file helpers are
// a tmp-then-rename write and a whole-file read.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

namespace luce::common {

struct KvFlashSlotBundle {
    uint64_t identity = 0;
    uint32_t ckpt_pos = 0;                                  // 0: no prompt-end checkpoint
    std::vector<int32_t> history;                           // the tokens the rows and the slab hold
    std::vector<std::vector<uint8_t>> ssm, conv;            // the slab after history
    std::vector<std::vector<uint8_t>> ckpt_ssm, ckpt_conv;  // the slab after history[0:ckpt_pos)
    std::vector<uint8_t> pager;                             // KvFlashPager::serialize()
};

inline uint64_t kvflash_fnv1a(std::string_view s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ULL; }
    return h;
}

inline std::string kvflash_bundle_path(const std::string & dir, int slot) {
    return dir + "/slot" + std::to_string(slot) + ".kvb";
}

namespace detail {
constexpr uint64_t kBundleMagic = 0x31544f4c5346564bULL;   // "KVFSLOT1" little-endian
template <class T> void put(std::vector<uint8_t> & out, const T & v) {
    const uint8_t * p = reinterpret_cast<const uint8_t *>(&v);
    out.insert(out.end(), p, p + sizeof(T));
}
inline void put_bytes(std::vector<uint8_t> & out, const std::vector<uint8_t> & v) {
    put(out, (uint64_t)v.size());
    out.insert(out.end(), v.begin(), v.end());
}
inline void put_list(std::vector<uint8_t> & out, const std::vector<std::vector<uint8_t>> & l) {
    put(out, (uint32_t)l.size());
    for (const auto & v : l) put_bytes(out, v);
}
struct Cursor {
    const uint8_t * p; size_t n, at = 0;
    template <class T> bool get(T & v) {
        if (n - at < sizeof(T)) return false;
        std::memcpy(&v, p + at, sizeof(T)); at += sizeof(T);
        return true;
    }
    bool get_bytes(std::vector<uint8_t> & v) {
        uint64_t len = 0;
        if (!get(len) || len > n - at) return false;
        v.assign(p + at, p + at + len); at += (size_t)len;
        return true;
    }
    bool get_list(std::vector<std::vector<uint8_t>> & l) {
        uint32_t cnt = 0;
        if (!get(cnt) || cnt > 4096) return false;
        l.resize(cnt);
        for (auto & v : l) if (!get_bytes(v)) return false;
        return true;
    }
};
}  // namespace detail

inline std::vector<uint8_t> kvflash_bundle_pack(const KvFlashSlotBundle & b) {
    std::vector<uint8_t> out;
    detail::put(out, detail::kBundleMagic);
    detail::put(out, b.identity);
    detail::put(out, b.ckpt_pos);
    detail::put(out, (uint32_t)b.history.size());
    const uint8_t * h = reinterpret_cast<const uint8_t *>(b.history.data());
    out.insert(out.end(), h, h + b.history.size() * sizeof(int32_t));
    for (const auto * l : {&b.ssm, &b.conv, &b.ckpt_ssm, &b.ckpt_conv}) detail::put_list(out, *l);
    detail::put_bytes(out, b.pager);
    return out;
}

// False (out untouched) on a bad magic, another identity, a truncated or
// oversized section, an empty history or a checkpoint past the history.
inline bool kvflash_bundle_unpack(const uint8_t * data, size_t n, uint64_t identity, KvFlashSlotBundle & out) {
    detail::Cursor c{data, n};
    uint64_t magic = 0;
    KvFlashSlotBundle b;
    uint32_t nh = 0;
    if (!data || !c.get(magic) || magic != detail::kBundleMagic || !c.get(b.identity) || b.identity != identity ||
        !c.get(b.ckpt_pos) || !c.get(nh) || nh == 0 || nh > (n - c.at) / sizeof(int32_t) || b.ckpt_pos > nh) return false;
    b.history.resize(nh);
    std::memcpy(b.history.data(), data + c.at, nh * sizeof(int32_t)); c.at += nh * sizeof(int32_t);
    for (auto * l : {&b.ssm, &b.conv, &b.ckpt_ssm, &b.ckpt_conv}) if (!c.get_list(*l)) return false;
    if (!c.get_bytes(b.pager) || c.at != n) return false;
    out = std::move(b);
    return true;
}

// Whole file to `path` through path.tmp + rename: a reader sees the old bundle or the new one, never a torn one.
inline bool kvflash_bundle_write(const std::string & path, const std::vector<uint8_t> & bytes) {
    const std::string tmp = path + ".tmp";
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size() && std::fflush(f) == 0 && ::fsync(fileno(f)) == 0;
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed || std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
    return true;
}

inline bool kvflash_bundle_read(const std::string & path, std::vector<uint8_t> & out) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = std::fseek(f, 0, SEEK_END) == 0;
    const long size = ok ? std::ftell(f) : -1;
    ok = ok && size >= 0 && std::fseek(f, 0, SEEK_SET) == 0;
    if (ok) { out.resize((size_t)size); ok = std::fread(out.data(), 1, out.size(), f) == out.size(); }
    std::fclose(f);
    return ok;
}

}  // namespace luce::common
