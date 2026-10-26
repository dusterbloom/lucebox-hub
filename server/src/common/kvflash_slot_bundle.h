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
#include <new>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/file.h>
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

// Everything before the pager bytes, with the pager length: the pack is head + pager, so a
// writer can stream the pager blob after the head instead of holding a second copy of it.
inline std::vector<uint8_t> kvflash_bundle_head(const KvFlashSlotBundle & b, size_t pager_bytes) {
    std::vector<uint8_t> out;
    detail::put(out, detail::kBundleMagic);
    detail::put(out, b.identity);
    detail::put(out, b.ckpt_pos);
    detail::put(out, (uint32_t)b.history.size());
    const uint8_t * h = reinterpret_cast<const uint8_t *>(b.history.data());
    out.insert(out.end(), h, h + b.history.size() * sizeof(int32_t));
    for (const auto * l : {&b.ssm, &b.conv, &b.ckpt_ssm, &b.ckpt_conv}) detail::put_list(out, *l);
    detail::put(out, (uint64_t)pager_bytes);
    return out;
}

inline std::vector<uint8_t> kvflash_bundle_pack(const KvFlashSlotBundle & b) {
    std::vector<uint8_t> out = kvflash_bundle_head(b, b.pager.size());
    out.insert(out.end(), b.pager.begin(), b.pager.end());
    return out;
}

// The largest bundle a server can write: a history of history_max tokens, the two slabs of
// slab_bytes with their list overhead, a pager blob of pager_max bytes.
inline size_t kvflash_bundle_max_bytes(size_t history_max, size_t slab_bytes, size_t pager_max) {
    return 24 + history_max * 4 + 2 * (slab_bytes + 2 * (4 + 4096 * 8)) + 8 + pager_max;
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

// `head` then whatever `more(put)` writes, to `path` through path.tmp.<pid> + rename: a reader
// sees the old bundle or the new one, never a torn one, and two servers never share a tmp inode.
template <class More>
inline bool kvflash_bundle_write_stream(const std::string & path, const std::vector<uint8_t> & head, More && more) {
    const std::string tmp = path + ".tmp." + std::to_string((long)::getpid());
    FILE * f = std::fopen(tmp.c_str(), "wb");
    if (!f) return false;
    auto put = [f](const uint8_t * d, size_t n) { return std::fwrite(d, 1, n, f) == n; };
    const bool ok = put(head.data(), head.size()) && more(put) && std::fflush(f) == 0 && ::fsync(fileno(f)) == 0;
    const bool closed = std::fclose(f) == 0;
    if (!ok || !closed || std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); return false; }
    return true;
}

inline bool kvflash_bundle_write(const std::string & path, const std::vector<uint8_t> & bytes) {
    return kvflash_bundle_write_stream(path, bytes, [](auto &) { return true; });
}

// False (out untouched) when the file is missing, unreadable, over max_bytes or too big to hold.
inline bool kvflash_bundle_read(const std::string & path, std::vector<uint8_t> & out, size_t max_bytes = SIZE_MAX) {
    FILE * f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    bool ok = std::fseek(f, 0, SEEK_END) == 0;
    const long size = ok ? std::ftell(f) : -1;
    ok = ok && size >= 0 && (size_t)size <= max_bytes && std::fseek(f, 0, SEEK_SET) == 0;
    std::vector<uint8_t> buf;
    if (ok) {
        try { buf.resize((size_t)size); } catch (const std::bad_alloc &) { ok = false; }
        ok = ok && std::fread(buf.data(), 1, buf.size(), f) == buf.size();
    }
    std::fclose(f);
    if (ok) out = std::move(buf);
    return ok;
}

// One server per bundle directory: an exclusive flock on dir/.lock held while the returned fd is
// open; -1 when another server holds it (or the directory cannot be opened).
inline int kvflash_bundle_dir_lock(const std::string & dir) {
    const int fd = ::open((dir + "/.lock").c_str(), O_RDWR | O_CREAT | O_CLOEXEC, 0644);
    if (fd < 0) return -1;
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) { ::close(fd); return -1; }
    return fd;
}

}  // namespace luce::common
