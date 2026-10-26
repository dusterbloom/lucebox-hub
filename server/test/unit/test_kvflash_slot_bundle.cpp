// Pure unit tests for the C1.4 per-slot disk bundle (kvflash_slot_bundle.h):
// byte-exact pack/unpack, every refusal, the atomic file write. No GPU.
#include "CppUnitTestFramework.hpp"
#include "../../src/common/kvflash_slot_bundle.h"

#include <cstdio>

using namespace luce::common;

namespace {
struct KvflashSlotBundleFixture {};

KvFlashSlotBundle sample() {
    KvFlashSlotBundle b;
    b.identity = kvflash_fnv1a("model|size|mtime|q8_0|q8_0|rot0|64|32|8192");
    b.ckpt_pos = 3;
    b.history = {11, 22, 33, 44, 55};
    b.ssm = {{1, 2, 3}, {}, {4}};
    b.conv = {{5, 6}};
    b.ckpt_ssm = {{7, 7, 7}, {}, {8}};
    b.ckpt_conv = {{9, 9}};
    b.pager = {0xde, 0xad, 0xbe, 0xef, 0x00, 0x01};
    return b;
}

bool same(const KvFlashSlotBundle & a, const KvFlashSlotBundle & b) {
    return a.identity == b.identity && a.ckpt_pos == b.ckpt_pos && a.history == b.history && a.ssm == b.ssm &&
           a.conv == b.conv && a.ckpt_ssm == b.ckpt_ssm && a.ckpt_conv == b.ckpt_conv && a.pager == b.pager;
}
}  // namespace

TEST_CASE(KvflashSlotBundleFixture, pack_unpack_is_byte_exact_and_checks_the_identity) {
    const KvFlashSlotBundle b = sample();
    const std::vector<uint8_t> bytes = kvflash_bundle_pack(b);
    CHECK(std::memcmp(bytes.data(), "KVFSLOT1", 8) == 0);
    KvFlashSlotBundle out;
    REQUIRE(kvflash_bundle_unpack(bytes.data(), bytes.size(), b.identity, out));
    CHECK(same(out, b));
    CHECK(kvflash_bundle_pack(out) == bytes);
    out = KvFlashSlotBundle{};
    CHECK(!kvflash_bundle_unpack(bytes.data(), bytes.size(), b.identity + 1, out));   // another server's state
    CHECK(out.history.empty());                                                       // untouched on refusal
    CHECK(kvflash_fnv1a("a") != kvflash_fnv1a("b"));
}

TEST_CASE(KvflashSlotBundleFixture, truncated_oversized_empty_or_inconsistent_bundles_are_refused) {
    const KvFlashSlotBundle b = sample();
    const std::vector<uint8_t> bytes = kvflash_bundle_pack(b);
    KvFlashSlotBundle out;
    for (size_t n = 0; n < bytes.size(); n += 7) CHECK(!kvflash_bundle_unpack(bytes.data(), n, b.identity, out));
    std::vector<uint8_t> longer = bytes;
    longer.push_back(0);
    CHECK(!kvflash_bundle_unpack(longer.data(), longer.size(), b.identity, out));     // trailing bytes
    std::vector<uint8_t> bad = bytes;
    bad[0] ^= 1;
    CHECK(!kvflash_bundle_unpack(bad.data(), bad.size(), b.identity, out));           // magic
    bad = bytes;
    bad[8 + 8 + 4] = 0;                                                               // history count 0
    CHECK(!kvflash_bundle_unpack(bad.data(), bad.size(), b.identity, out));
    bad = bytes;
    bad[8 + 8 + 4] = 0xff; bad[8 + 8 + 5] = 0xff;                                     // history count past the blob
    CHECK(!kvflash_bundle_unpack(bad.data(), bad.size(), b.identity, out));
    KvFlashSlotBundle far = b;
    far.ckpt_pos = 6;                                                                 // past the history
    const std::vector<uint8_t> fb = kvflash_bundle_pack(far);
    CHECK(!kvflash_bundle_unpack(fb.data(), fb.size(), b.identity, out));
    CHECK(!kvflash_bundle_unpack(nullptr, 0, b.identity, out));
    KvFlashSlotBundle lean = b;                                                       // no checkpoint, empty lists
    lean.ckpt_pos = 0; lean.ckpt_ssm.clear(); lean.ckpt_conv.clear();
    const std::vector<uint8_t> lb = kvflash_bundle_pack(lean);
    REQUIRE(kvflash_bundle_unpack(lb.data(), lb.size(), b.identity, out));
    CHECK(same(out, lean));
}

TEST_CASE(KvflashSlotBundleFixture, write_is_tmp_then_rename_and_read_gives_the_bytes_back) {
    const std::string path = "kvflash_slot_bundle_unit_test.kvb";
    const std::vector<uint8_t> bytes = kvflash_bundle_pack(sample());
    REQUIRE(kvflash_bundle_write(path, bytes));
    std::vector<uint8_t> back;
    REQUIRE(kvflash_bundle_read(path, back));
    CHECK(back == bytes);
    CHECK(std::fopen((path + ".tmp").c_str(), "rb") == nullptr);                      // no leftover tmp
    const std::vector<uint8_t> other = {1, 2, 3};
    REQUIRE(kvflash_bundle_write(path, other));                                       // overwrite in place
    REQUIRE(kvflash_bundle_read(path, back));
    CHECK(back == other);
    std::remove(path.c_str());
    CHECK(!kvflash_bundle_read(path, back));
    CHECK(!kvflash_bundle_write("/nonexistent-dir-kvflash/x.kvb", bytes));
    CHECK(kvflash_bundle_path("/d", 3) == "/d/slot3.kvb");
}

TEST_CASE(KvflashSlotBundleFixture, the_head_plus_the_pager_blob_is_the_pack_and_a_streamed_write_reads_back) {
    const KvFlashSlotBundle b = sample();
    const std::vector<uint8_t> bytes = kvflash_bundle_pack(b);
    std::vector<uint8_t> head = kvflash_bundle_head(b, b.pager.size());
    CHECK(head.size() == bytes.size() - b.pager.size());
    CHECK(head.size() == kvflash_bundle_head_bytes(b));
    CHECK(head.capacity() == head.size());                                                      // one allocation, no growth
    head.insert(head.end(), b.pager.begin(), b.pager.end());
    CHECK(head == bytes);
    const std::string path = "kvflash_slot_bundle_unit_test_stream.kvb";
    REQUIRE(kvflash_bundle_write_stream(path, kvflash_bundle_head(b, b.pager.size()), [&](auto & put) {
        return put(b.pager.data(), 3) && put(b.pager.data() + 3, b.pager.size() - 3);
    }));
    std::vector<uint8_t> back;
    REQUIRE(kvflash_bundle_read(path, back));
    CHECK(back == bytes);
    CHECK(std::fopen((path + ".tmp." + std::to_string(::getpid())).c_str(), "rb") == nullptr);   // no leftover tmp
    CHECK(!kvflash_bundle_write_stream(path, head, [](auto &) { return false; }));                // a failed part: no file change
    REQUIRE(kvflash_bundle_read(path, back));
    CHECK(back == bytes);
    std::remove(path.c_str());
}

TEST_CASE(KvflashSlotBundleFixture, a_file_over_the_cap_is_refused_and_a_bundle_dir_has_one_owner) {
    const std::vector<uint8_t> bytes = kvflash_bundle_pack(sample());
    const std::string path = "kvflash_slot_bundle_unit_test_cap.kvb";
    REQUIRE(kvflash_bundle_write(path, bytes));
    std::vector<uint8_t> back = {7};
    CHECK(!kvflash_bundle_read(path, back, bytes.size() - 1));
    CHECK(back == std::vector<uint8_t>{7});                                                     // untouched on refusal
    CHECK(kvflash_bundle_read(path, back, bytes.size()));
    CHECK(back == bytes);
    std::remove(path.c_str());
    CHECK(kvflash_bundle_max_bytes(5, 3 * 3 + 2 * 2, 6) >= bytes.size());                        // history 5, the four slabs, pager 6
    const int fd = kvflash_bundle_dir_lock(".");
    REQUIRE(fd >= 0);
    CHECK(kvflash_bundle_dir_lock(".") < 0);                                                    // a second owner is refused
    ::close(fd);
    const int fd2 = kvflash_bundle_dir_lock(".");
    CHECK(fd2 >= 0);
    ::close(fd2);
    std::remove("./.lock");
    CHECK(kvflash_bundle_dir_lock("/nonexistent-dir-kvflash") < 0);
}

TEST_CASE(KvflashSlotBundleFixture, the_writer_thread_writes_jobs_in_order_and_drain_waits_for_them) {
    const KvFlashSlotBundle b = sample();
    const std::vector<uint8_t> all = kvflash_bundle_pack(b);
    const std::vector<uint8_t> head = kvflash_bundle_head(b, b.pager.size());
    const std::string path = "kvflash_slot_bundle_unit_test_writer.kvb";
    std::vector<std::pair<int, bool>> done;
    {
        KvFlashBundleWriter w([&](const KvFlashBundleWriter::Job & j, bool ok, double) { done.push_back({j.slot, ok}); });
        w.submit({path, head, {0xff}, 1});                                                      // an older save of the slot
        w.submit({path, head, b.pager, 2});                                                     // the newer one lands last
        w.submit({"/nonexistent-dir-kvflash/x.kvb", head, b.pager, 3});                         // reported, does not stop the writer
        w.submit({path + ".2", head, b.pager, 4});
        w.wait(2);                                                                              // jobs 1 and 2 are on disk (FIFO)
        std::vector<uint8_t> first;
        REQUIRE(kvflash_bundle_read(path, first));
        CHECK(first == all);
        w.wait(42);                                                                             // no job of that slot: at once
        w.drain();
        CHECK(w.pending() == 0);
        REQUIRE(done.size() == 4);
        CHECK(done[0] == std::make_pair(1, true));
        CHECK(done[1] == std::make_pair(2, true));
        CHECK(done[2] == std::make_pair(3, false));
        CHECK(done[3] == std::make_pair(4, true));
        w.submit({path + ".3", head, b.pager, 5});                                              // the destructor drains
    }
    REQUIRE(done.size() == 5);
    std::vector<uint8_t> back;
    REQUIRE(kvflash_bundle_read(path, back));
    CHECK(back == all);
    REQUIRE(kvflash_bundle_read(path + ".3", back));
    CHECK(back == all);
    std::remove(path.c_str());
    std::remove((path + ".2").c_str());
    std::remove((path + ".3").c_str());
}
