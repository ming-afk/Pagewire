// control_diskann.cpp -- Q-learning static-workload rewire on a disk-resident
// DiskANN index (train_batch / do_rewire / B_buf LRU / method registry / sigma
// query stream), ported from cpp_port/control_vamana.cpp's in-RAM Vamana CSR
// onto PQFlashIndex's disk-aware regime (PQ codes + on-disk sector records,
// cached_beam_search, O_DIRECT read-modify-write edits).
//
// Requires an index built by apps/build_disk_index (float data, L2 metric):
//   <prefix>_disk.index, <prefix>_pq_pivots.bin, <prefix>_pq_compressed.bin
// Reuses np_random.hpp / npio.hpp from the hnswlib control.cpp port for a
// byte-identical synthetic query stream.
//
// Linux only: LinuxAlignedFileReader (O_DIRECT + libaio), as PQFlashIndex uses.
#include <cstdio>
#include <cstdint>
#include <cassert>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <limits>
#include <random>
#include <memory>
#include <iterator>
#include <functional>
#include <fstream>
#include <filesystem>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <list>
#include <exception>

#include <fcntl.h>
#include <unistd.h>

#include "aligned_file_reader.h"
#include "linux_aligned_file_reader.h"

#ifdef DISKANN_URING
#include <liburing.h>
class UringAlignedFileReader : public AlignedFileReader
{
  public:
    static constexpr unsigned kQueueDepth = 1024; // == libaio's MAX_EVENTS

    UringAlignedFileReader() = default;
    ~UringAlignedFileReader() override
    {
        if (ring_ready) { io_uring_queue_exit(&ring); ring_ready = false; }
    }

    IOContext &get_ctx() override { return unused_ctx; }
    void register_thread() override {}
    void deregister_thread() override {}
    void deregister_all_threads() override {}

    void open(const std::string &fname) override
    {
        file_desc = ::open(fname.c_str(), O_DIRECT | O_RDONLY | O_LARGEFILE);
        if (file_desc == -1) throw std::runtime_error("io_uring reader: cannot open " + fname);
        const int rc = io_uring_queue_init(kQueueDepth, &ring, 0);
        if (rc < 0) throw std::runtime_error(std::string("io_uring_queue_init failed: ") + ::strerror(-rc));
        ring_ready = true;
        registered = (io_uring_register_files(&ring, &file_desc, 1) == 0);
        fprintf(stderr, "Opened file : %s  [READER=io_uring depth=%u registered_fd=%s]\n",
                fname.c_str(), kQueueDepth, registered ? "yes" : "no");
    }

    void close() override
    {
        if (file_desc != -1) { ::close(file_desc); file_desc = -1; }
    }

    void read(std::vector<AlignedRead> &read_reqs, IOContext &, bool async = false) override
    {
        if (async) throw std::runtime_error("io_uring reader: async mode not supported");
        std::lock_guard<std::mutex> lk(ring_mu);
        const size_t total = read_reqs.size();
        for (size_t base = 0; base < total; base += kQueueDepth)
        {
            const size_t n = std::min((size_t)kQueueDepth, total - base);
            for (size_t i = 0; i < n; i++)
            {
                io_uring_sqe *sqe = io_uring_get_sqe(&ring);
                if (sqe == nullptr) throw std::runtime_error("io_uring: submission queue unexpectedly full");
                AlignedRead &r = read_reqs[base + i];
                io_uring_prep_read(sqe, registered ? 0 : file_desc, r.buf, (unsigned)r.len, r.offset);
                if (registered) sqe->flags |= IOSQE_FIXED_FILE;
                // carry the expected length so a short read cannot pass silently
                io_uring_sqe_set_data(sqe, (void *)(uintptr_t)r.len);
            }
            const int submitted = io_uring_submit_and_wait(&ring, (unsigned)n);
            if (submitted < 0)
                throw std::runtime_error(std::string("io_uring_submit_and_wait failed: ") + ::strerror(-submitted));
            if ((size_t)submitted != n)
                throw std::runtime_error("io_uring: submitted " + std::to_string(submitted) + " of " +
                                         std::to_string(n) + " requests");
            for (size_t harvested = 0; harvested < n; harvested++)
            {
                io_uring_cqe *cqe = nullptr;
                const int rc = io_uring_wait_cqe(&ring, &cqe);
                if (rc < 0) throw std::runtime_error(std::string("io_uring_wait_cqe failed: ") + ::strerror(-rc));
                const int res = cqe->res;
                const uint64_t want = (uint64_t)(uintptr_t)io_uring_cqe_get_data(cqe);
                io_uring_cqe_seen(&ring, cqe);
                if (res < 0) throw std::runtime_error(std::string("io_uring read failed: ") + ::strerror(-res));
                if ((uint64_t)res != want)
                    throw std::runtime_error("io_uring short read: got " + std::to_string(res) + " of " +
                                             std::to_string(want) + " bytes");
            }
        }
    }

  private:
    int file_desc = -1;
    io_uring ring{};
    bool ring_ready = false;
    bool registered = false;
    std::mutex ring_mu;
    IOContext unused_ctx = (IOContext)-1; // the interface hands one back; uring has no use for it
};
#endif // DISKANN_URING

// --io-uring. Only reachable in a build that defines DISKANN_URING and links
// liburing; every other target rejects the flag rather than silently ignoring it
// and reporting libaio numbers under a uring label.
static bool g_use_io_uring = false;
#include "defaults.h"
#include "neighbor.h"
#include "pq.h"
#include "utils.h"

#include "qlrewire_support/npio.hpp"
#include "qlrewire_support/np_random.hpp"

using Clock = std::chrono::steady_clock;
static double secs_since(Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

static double g_write_delay_us = 0.0;    // charge per write sector, microseconds
static double g_write_delay_prob = 1.0;  // probability a given sector is charged
static std::mt19937_64 g_write_delay_rng(0x5eed5eed);
static long g_write_delay_charged = 0;   // sectors actually charged
static long g_write_delay_offered = 0;   // sectors the charge was offered
static double g_write_delay_s = 0.0;     // wall time spent inside the charge

static void charge_write_delay(size_t bytes) {
    if (g_write_delay_us <= 0.0) return;
    const long sectors = (long)(bytes / diskann::defaults::SECTOR_LEN);
    const Clock::time_point t0 = Clock::now();
    for (long i = 0; i < sectors; i++) {
        g_write_delay_offered++;
        if (g_write_delay_prob < 1.0 &&
            std::generate_canonical<double, 53>(g_write_delay_rng) >= g_write_delay_prob)
            continue;
        g_write_delay_charged++;
        const Clock::time_point deadline =
            Clock::now() + std::chrono::nanoseconds((long long)(g_write_delay_us * 1000.0));
        const auto margin = std::chrono::microseconds(120);
        if (deadline - Clock::now() > margin)
            std::this_thread::sleep_for((deadline - Clock::now()) - margin);
        while (Clock::now() < deadline) { /* spin the tail for accuracy */ }
    }
    g_write_delay_s += secs_since(t0);
}

static double g_read_delay_us = 0.0;     // charge per device read request, microseconds
static double g_read_delay_prob = 1.0;   // probability a given read request is charged
static std::mt19937_64 g_read_delay_rng(0x6eed6eed);
static long g_read_delay_offered = 0;
static long g_read_delay_charged = 0;
static double g_read_delay_s = 0.0;
static long g_read_delay_start_query = 0;
static long g_read_delay_query = 0;

static void charge_read_delay(long requests) {
    if (g_read_delay_us <= 0.0 || g_read_delay_query < g_read_delay_start_query) return;
    const Clock::time_point t0 = Clock::now();
    for (long i = 0; i < requests; i++) {
        g_read_delay_offered++;
        if (g_read_delay_prob < 1.0 &&
            std::generate_canonical<double, 53>(g_read_delay_rng) >= g_read_delay_prob)
            continue;
        g_read_delay_charged++;
        const Clock::time_point deadline =
            Clock::now() + std::chrono::nanoseconds((long long)(g_read_delay_us * 1000.0));
        const auto margin = std::chrono::microseconds(120);
        if (deadline - Clock::now() > margin)
            std::this_thread::sleep_for((deadline - Clock::now()) - margin);
        while (Clock::now() < deadline) { /* spin the tail for accuracy */ }
    }
    g_read_delay_s += secs_since(t0);
}

static bool g_in_memory = false;
static std::vector<char> g_ram_image;
static inline char *ram_at(uint64_t offset) { return g_ram_image.data() + offset; }

static ssize_t pread_with_delay(int fd, void *buf, size_t bytes, off_t offset) {
    // No device, so no delay to charge: the write/read delay rig prices a real
    // sector, and pricing a memcpy would invent an I/O cost the run never paid.
    if (g_in_memory) { std::memcpy(buf, ram_at((uint64_t)offset), bytes); return (ssize_t)bytes; }
    const ssize_t n = ::pread(fd, buf, bytes, offset);
    if (n == (ssize_t)bytes) charge_read_delay(1);
    return n;
}

// Write seam, mirroring pread_with_delay. Every ::pwrite in this file goes
// through here so the in-memory mode cannot miss a commit path.
static ssize_t pwrite_backend(int fd, const void *buf, size_t bytes, off_t offset) {
    if (g_in_memory) { std::memcpy(ram_at((uint64_t)offset), buf, bytes); return (ssize_t)bytes; }
    return ::pwrite(fd, buf, bytes, offset);
}

enum RSelPart { RS_SETUP, RS_FETCH, RS_ROW, RS_POOL, RS_GC, RS_COST, RS_SCORE, RS_SORT,
                RS_PROTECT, RS_SELECT, RS_DIFF, RS_GAIN_GATE, RS_EDIT, RS_LAMBDA,
                RS_REPORT, RS_MISC, RS_N };
static const char *RS_NAMES[RS_N] = { "setup", "fetch", "row", "pool", "gc", "cost", "score", "sort",
                                      "protect", "select", "diff", "gain_gate", "edit", "lambda",
                                      "report", "misc" };

struct TimeProfile {
    double search = 0.0;         // g.search(): the serving beam search
    double search_io = 0.0;      // device I/O inside the serving beam search window
    double nav = 0.0;            // dead: the Q-ranked training walk it timed is gone.
                                 // Kept at 0 so the TIMEPROF line keeps its `nav=`
                                 // key -- plot_time_decomposition.py requires the
                                 // group, and every stored log carries it.
    double update = 0.0;         // Q / c / B_buf updates
    double rewire_select = 0.0;  // do_rewire minus flush_edits
    double rewire_commit = 0.0;  // flush_edits: the read-modify-write commits
    double rsel[RS_N] = {};      // do_rewire sub-buckets; misc reconciles to rewire_select
    double rsel_io[RS_N] = {};   // wall time inside device I/O for the same sub-buckets
    double rewire_commit_io = 0.0;
    // The served loop's own bookkeeping, previously invisible inside `other`.
    // Reported on their own line so the TIMEPROF line stays byte-compatible with
    // every stored log and with plot_time_decomposition.py.
    double serve_sets = 0.0;     // V_t / V_t_set / distinct-sector / ground-truth sets
    double serve_dump = 0.0;     // --dump-query-results row formatting
    double layout = 0.0;         // layout_step (Algorithm 4)
    double audit = 0.0;          // audit_step (Algorithm 5)
    double governor = 0.0;       // governor_step (Algorithm 6)
    double qlearn_total() const { return nav + update + rewire_select + rewire_commit; }
    double pagewire_total() const { return layout + audit + governor; }
    double rsel_sum(bool include_misc) const {
        double s = 0.0;
        for (int i = 0; i < (include_misc ? RS_N : RS_MISC); i++) s += rsel[i];
        return s;
    }
    double rsel_io_sum(bool include_misc) const {
        double s = 0.0;
        for (int i = 0; i < (include_misc ? RS_N : RS_MISC); i++) s += rsel_io[i];
        return s;
    }
};
static TimeProfile g_tp;

// Accumulates its scope's wall time into one bucket.
struct TimeSpan {
    double &bucket;
    Clock::time_point t0;
    explicit TimeSpan(double &b) : bucket(b), t0(Clock::now()) {}
    ~TimeSpan() { bucket += secs_since(t0); }
    TimeSpan(const TimeSpan &) = delete;
    TimeSpan &operator=(const TimeSpan &) = delete;
};

// --search-only: run the serving beam search and stop. No Q-ranked walk, no
// score update, no rewire. This is the arm that measures DiskANN with nothing
// attached, which the existing --no-rewire does NOT give (it skips do_rewire but
// still runs the walk and the update).
static bool g_search_only = false;
// --skip-rewire-commit: training walk, Q updates, audits and direct repairs all
// run, but do_rewire (the rewire commit) is skipped, isolating audit repair.
static bool g_skip_rewire_commit = false;
// Frozen-mode unseen-edge estimator cadence. Learning still observes every query;
// after ADAPT_STOP only each Nth query materializes adjacency rows and updates the
// drift signature estimator. 1 preserves the original exact-window behavior.
static int g_detector_sample_every = 1;

static bool g_q_update_traversed_only = false;

static bool g_label_from_gt = false;

// --quick-gt: use calculate_quick_gt (a live L=1000 search against the index
// being trained, see its definition) as the source of ground_truth_ids
// instead of the precomputed exact-GT file (batch_g). Off by default, so
// recall_val and the --gt-label-rate label still mean agreement with the
// true nearest neighbors unless this is explicitly requested.
static bool g_quick_gt = false;

static double g_gt_label_rate = -1.0;
static unsigned g_gt_label_seed = 777777;
static std::mt19937_64 g_gt_label_rng(777777);
static std::uniform_real_distribution<double> g_gt_label_unif01(0.0, 1.0);

static bool g_covisit_scope_expanded = false;
// Pairs log_covisitation enumerated since the last LayoutStep, reported on the
// layout line: the two scopes differ by the square of the miss rate, so the
// volume is the first thing to check when comparing arms.
static long g_covisit_pairs = 0;

static int g_dim = 0;             // index data dimension (set at load)
// External queries live in the dataset's native dimension. For a MIPS index that
// is one less than g_dim: build_disk_index reduces MIPS to L2 by appending a
// coordinate sqrt(max_norm^2 - ||x||^2) to every base vector, so the on-disk
// vectors and PQ pivots are (native+1)-dimensional while the .npz caches are not.
static int g_query_dim = 0;
static bool g_mips = false;       // index carries _max_base_norm.bin (built --dist_fn mips)
static int g_k = 100;             // ground-truth width used by the training stream
static const int GT_STRIDE = 100; // proxy_gt / eval_gt row width in the cache npz

static std::string g_data_type = "float";
static size_t data_type_elem_size(const std::string &dt) {
    if (dt == "float") return sizeof(float);
    if (dt == "int8" || dt == "uint8") return sizeof(uint8_t);
    throw std::runtime_error("unknown --data_type '" + dt + "' (expected float, int8, or uint8)");
}
// Decode `n` on-disk elements of type g_data_type at `raw` into `out` as float,
// widening int8/uint8 the same way PQ table construction already does elsewhere
// in this file. Used at every full-precision coordinate read so the rest of the
// pipeline (which does float re-ranking math) is unchanged regardless of dtype.
static void decode_coords_to_float(const char *raw, size_t n, float *out) {
    if (g_data_type == "float") {
        std::memcpy(out, raw, n * sizeof(float));
    } else if (g_data_type == "int8") {
        const int8_t *p = (const int8_t *)raw;
        for (size_t i = 0; i < n; i++) out[i] = (float)p[i];
    } else { // uint8
        const uint8_t *p = (const uint8_t *)raw;
        for (size_t i = 0; i < n; i++) out[i] = (float)p[i];
    }
}

struct IOCounters {
    long reads = 0;       // node fetches issued
    long read_sectors = 0;
    long writes = 0;      // sectors written back
    double io_s = 0.0;    // wall time inside reader->read / pwrite
    // Pagewire eq. (1) accounting. `reads`/`read_sectors` keep their old meaning
    // (one per id requested) so every number in the existing logs stays
    // comparable; `sector_reqs` is what the device actually saw, i.e. the sum
    // over queries of |pi(V(q))|.
    long sector_reqs = 0;    // AlignedReads actually submitted
    long qbuf_hits = 0;      // ids served from a sector an EARLIER round fetched
    long call_coalesced = 0; // ids sharing a sector with another id in the same batch
    // Served from C_blk, the CROSS-query resident set (HotNodeCache). Kept apart
    // from qbuf_hits so eq. (1) stays additively decomposable: every requested id
    // lands in exactly one of {cblk_hits, qbuf_hits, call_coalesced, sector_reqs},
    // because a sector resolved from C_blk is never queued nor qbuf-claimed.
    long cblk_hits = 0;
    long cblk_admits = 0;    // misses read straight into a C_blk slot
    long cblk_frozen = 0;
    long rmw_preads = 0;
    long rmw_preads_resident = 0;
    long device_reads = 0;   // sectors the device was asked to read
    long device_writes = 0;  // sectors the device was asked to write
    // Split of the same fetches by whether the requested id's row has ever been
    // rewired (g_row_rewired). A device read is attributed to the id that
    // triggered it, not to the other rows sharing its sector.
    long ids_on_rewired = 0;          // requested ids whose row has been rewired
    long sector_reqs_on_rewired = 0;  // device reads whose requesting id is a rewired row
    // --uncacheable-node-fraction / --uncacheable-node-list: requested ids that
    // were forced to the device because their node is uncacheable. Each is also
    // counted in sector_reqs.
    long uncacheable_forced_reads = 0;
    void clear() {
        ids_on_rewired = sector_reqs_on_rewired = 0;
        uncacheable_forced_reads = 0;
        reads = read_sectors = writes = 0;
        sector_reqs = qbuf_hits = call_coalesced = cblk_hits = cblk_admits = 0;
        cblk_frozen = 0;
        rmw_preads = rmw_preads_resident = 0;
        device_reads = device_writes = 0;
        io_s = 0.0;
    }
};
static IOCounters g_io_traversal, g_io_rewire, g_io_reset,
                  g_io_layout, g_io_audit, g_io_floor;
// calculate_quick_gt's extra search charges here, kept off the traversal/
// rewire/layout/audit counters so it does not perturb any existing I/O report.
static IOCounters g_io_quickgt;
static IOCounters *g_io = &g_io_traversal; // phase currently being charged
// Rows whose adjacency any commit has changed, cumulative over the run. Read by
// read_nodes to split each phase's fetches into on-rewired-row vs. not.
static std::vector<uint8_t> g_row_rewired;
static long g_rewired_row_count = 0;
static double g_uncacheable_node_fraction = 0.0;
static uint64_t g_uncacheable_node_seed = 0x5eed5eedULL;   // --uncacheable-node-seed
// --uncacheable-node-list FILE (text, one node id per line). An explicit set,
// e.g. the workload's hottest nodes ranked by a --dump-node-fetch-counts
// pre-pass. When non-empty it replaces the hash rule above.
static std::unordered_set<uint32_t> g_uncacheable_node_set;
// --dump-node-fetch-counts FILE: per-node count of in-query fetch requests
// (the c(v) census population), written as "id count" lines at FINAL, before
// the held-out post-test, so it covers training traversal only.
static std::string g_node_fetch_count_path;
static std::unordered_map<uint32_t, uint64_t> g_node_fetch_count;
static inline bool is_uncacheable_node(uint32_t node_id) {
    if (!g_uncacheable_node_set.empty()) return g_uncacheable_node_set.count(node_id) != 0;
    if (g_uncacheable_node_fraction <= 0.0) return false;
    uint64_t mixed = (uint64_t)node_id ^ g_uncacheable_node_seed;   // splitmix64
    mixed += 0x9e3779b97f4a7c15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
    mixed ^= mixed >> 31;
    return (double)(mixed >> 11) * 0x1.0p-53 < g_uncacheable_node_fraction;
}
static void mark_row_rewired(uint32_t row_id, uint64_t num_points) {
    if (g_row_rewired.empty()) g_row_rewired.assign(num_points, 0);
    if (!g_row_rewired[row_id]) { g_row_rewired[row_id] = 1; g_rewired_row_count++; }
}
// Requests read_nodes' sector dedup removed, summed over the run: the direct
// measure of how much co-location the read path actually managed to collect.
static long g_coalesced_reads = 0;
// eq. (1) for the *serving* pass alone. sector_reqs sums over both passes
// train_batch runs (the eval-facing search and the Q-ranked walk), so it cannot
// be compared against g_distinct_sectors, which I_PI derives from the serving
// search's expanded set only. This isolates the comparable half.
static long g_search_sector_reqs = 0;
// eq. (1) for the eval_recall passes. They issue the same g.search as the
// serving pass but run outside train_batch, and they charge to traversal like
// everything else -- so without this the "walk" residual silently absorbs them.


// Every sector the device was asked to move, summed over all five phases. This
// is the single I/O axis: it does not care what caused the transfer, only that
// it happened. Printed cumulatively at every eval checkpoint so a run's whole
// trajectory can be read off one log without reconstructing conventions.
static long device_io_total() {
    long total = 0;
    for (const IOCounters *c : {&g_io_traversal, &g_io_rewire, &g_io_reset,
                                &g_io_layout, &g_io_audit})
        total += c->device_reads + c->device_writes;
    return total;
}
static void report_device_io(const char *tag, long gqc) {
    fprintf(stderr, "  IODEV: %s q=%ld traversal[r=%ld w=%ld] rewire[r=%ld w=%ld] "
                    "layout[r=%ld w=%ld] audit[r=%ld w=%ld] reset[r=%ld w=%ld] total=%ld\n",
            tag, gqc,
            g_io_traversal.device_reads, g_io_traversal.device_writes,
            g_io_rewire.device_reads, g_io_rewire.device_writes,
            g_io_layout.device_reads, g_io_layout.device_writes,
            g_io_audit.device_reads, g_io_audit.device_writes,
            g_io_reset.device_reads, g_io_reset.device_writes,
            device_io_total());
    fprintf(stderr, "  REWIREDREADS: %s q=%ld rewired_rows=%ld trav_ids=%ld trav_ids_on_rewired=%ld "
                    "trav_sector_reqs=%ld trav_sector_reqs_on_rewired=%ld\n",
            tag, gqc, g_rewired_row_count, g_io_traversal.reads, g_io_traversal.ids_on_rewired,
            g_io_traversal.sector_reqs, g_io_traversal.sector_reqs_on_rewired);
    if (g_uncacheable_node_fraction > 0.0 || !g_uncacheable_node_set.empty())
        fprintf(stderr, "  UNCACHEABLE: %s q=%ld fraction=%.3f list_size=%zu trav_forced_reads=%ld "
                        "rewire_forced_reads=%ld audit_forced_reads=%ld\n",
                tag, gqc, g_uncacheable_node_fraction, g_uncacheable_node_set.size(),
                g_io_traversal.uncacheable_forced_reads,
                g_io_rewire.uncacheable_forced_reads, g_io_audit.uncacheable_forced_reads);
    // The realised charge, not the requested one. Asking for it every time the
    // device line is printed is the point: a nominal 200 us that lands at 250 us
    // (sleep overshoot) or at 100 us (a write path that never reached a pwrite)
    // would otherwise be invisible, and the whole arm would be mislabelled.
    if (g_write_delay_us > 0.0)
        fprintf(stderr, "  WRITEDELAY: %s q=%ld offered=%ld charged=%ld frac=%.4f "
                        "requested_us=%.1f realised_us=%.1f total_s=%.3f\n",
                tag, gqc, g_write_delay_offered, g_write_delay_charged,
                g_write_delay_offered ? (double)g_write_delay_charged / (double)g_write_delay_offered : 0.0,
                g_write_delay_us,
                g_write_delay_charged ? g_write_delay_s * 1e6 / (double)g_write_delay_charged : 0.0,
                g_write_delay_s);
    if (g_read_delay_us > 0.0)
        fprintf(stderr, "  READDELAY: %s q=%ld offered=%ld charged=%ld frac=%.4f "
                        "requested_us=%.1f realised_us=%.1f total_s=%.3f\n",
                tag, gqc, g_read_delay_offered, g_read_delay_charged,
                g_read_delay_offered ? (double)g_read_delay_charged / (double)g_read_delay_offered : 0.0,
                g_read_delay_us,
                g_read_delay_charged ? g_read_delay_s * 1e6 / (double)g_read_delay_charged : 0.0,
                g_read_delay_s);
    // The same five phases in SECONDS rather than sectors: io_s is accumulated
    // around reader->read(reqs) and the write calls, so this is device time and
    // nothing else. Sectors alone cannot answer whether a phase is I/O bound --
    // the per-read cost is not a constant, it falls with the depth of the libaio
    // batch, so a dense-miss phase and a sparse-miss phase moving the same number
    // of sectors do not cost the same. Until now this was reachable only through
    // --per-query-csv, which requires --b-rewire 1 and therefore could not be
    // read on any arm whose batch size is part of the arm.
    fprintf(stderr, "  IOTIME: %s q=%ld traversal=%.3f rewire=%.3f layout=%.3f "
                    "audit=%.3f reset=%.3f total=%.3f\n",
            tag, gqc, g_io_traversal.io_s, g_io_rewire.io_s, g_io_layout.io_s,
            g_io_audit.io_s, g_io_reset.io_s,
            g_io_traversal.io_s + g_io_rewire.io_s + g_io_layout.io_s +
                g_io_audit.io_s + g_io_reset.io_s);
}

static void report_time_profile(const char *tag, long gqc, double train_s) {
    const double q = g_tp.qlearn_total(), p = g_tp.pagewire_total();
    const double bucketed_s = g_tp.search + q + p;
    const double search_cpu_s = g_tp.search - g_tp.search_io;
    fprintf(stderr,
            "  TIMEPROF: %s q=%ld train_s=%.3f search=%.3f qlearn=%.3f"
            "[nav=%.3f update=%.3f rsel=%.3f rcommit=%.3f]"
            " pagewire=%.3f[layout=%.3f audit=%.3f gov=%.3f]"
            " other=%.3f bucketed_s=%.3f\n",
            tag, gqc, train_s, g_tp.search, q,
            g_tp.nav, g_tp.update, g_tp.rewire_select, g_tp.rewire_commit,
            p, g_tp.layout, g_tp.audit, g_tp.governor,
            train_s - bucketed_s, bucketed_s);
    fprintf(stderr, "  TIMEPROF_SEARCH: %s q=%ld total=%.3f io=%.3f cpu=%.3f\n",
            tag, gqc, g_tp.search, g_tp.search_io, search_cpu_s);
    fprintf(stderr, "  TIMEPROF_SERVE: %s q=%ld sets=%.3f dump=%.3f\n",
            tag, gqc, g_tp.serve_sets, g_tp.serve_dump);
    fprintf(stderr, "  TIMEPROF_RSEL: %s q=%ld total=%.3f detail=%.3f[",
            tag, gqc, g_tp.rewire_select, g_tp.rsel_sum(true));
    for (int i = 0; i < RS_N; i++)
        fprintf(stderr, "%s=%.3f%s", RS_NAMES[i], g_tp.rsel[i], i + 1 == RS_N ? "" : " ");
    fprintf(stderr, "]\n");
    fprintf(stderr, "  TIMEPROF_RSEL_IO: %s q=%ld total=%.3f detail=%.3f[",
            tag, gqc, g_tp.rsel_io_sum(true), g_tp.rsel_io_sum(true));
    for (int i = 0; i < RS_N; i++)
        fprintf(stderr, "%s=%.3f%s", RS_NAMES[i], g_tp.rsel_io[i], i + 1 == RS_N ? "" : " ");
    fprintf(stderr, "]\n");
    fprintf(stderr, "  TIMEPROF_RSEL_CPU: %s q=%ld total=%.3f detail=%.3f[",
            tag, gqc, g_tp.rewire_select - g_tp.rsel_io_sum(true),
            g_tp.rsel_sum(true) - g_tp.rsel_io_sum(true));
    for (int i = 0; i < RS_N; i++)
        fprintf(stderr, "%s=%.3f%s", RS_NAMES[i], g_tp.rsel[i] - g_tp.rsel_io[i], i + 1 == RS_N ? "" : " ");
    fprintf(stderr, "]\n");
    fprintf(stderr, "  TIMEPROF_RCOMMIT_IO: %s q=%ld total=%.3f io=%.3f cpu=%.3f\n",
            tag, gqc, g_tp.rewire_commit, g_tp.rewire_commit_io,
            g_tp.rewire_commit - g_tp.rewire_commit_io);
}

static void record_rewire_select(double rsel_inc, double detail_before,
                                 double rsel_io_inc, double io_detail_before) {
    const double measured_detail_inc = g_tp.rsel_sum(false) - detail_before;
    const double measured_io_inc = g_tp.rsel_io_sum(false) - io_detail_before;
    g_tp.rewire_select += rsel_inc;
    g_tp.rsel[RS_MISC] += rsel_inc - measured_detail_inc;
    g_tp.rsel_io[RS_MISC] += rsel_io_inc - measured_io_inc;
}

// Charges every I/O issued in its scope to one phase counter (traversal /
// rewire / reset), mirroring the majflt brackets of the Vamana harness.
struct IOPhase {
    IOCounters *prev;
    explicit IOPhase(IOCounters &c) : prev(g_io) { g_io = &c; }
    ~IOPhase() { g_io = prev; }
};

struct QueryBlockBuffer {
    bool enabled = false;      // --query-block-buffer
    size_t cap_blocks = 512;   // 2 MB at 4KB blocks; ~4x nu_bar (~132) on T2I-10M

    // One slab for the whole run, not per query: opening a scope costs a map
    // clear, not an allocation. Misses are read straight into it (see claim),
    // so admission never copies.
    char *slab = nullptr;
    uint64_t bytes_per_node = 0;
    size_t used = 0;                               // slab slots taken this query
    bool active = false;
    // Inside a QueryScope, regardless of whether the buffer itself is enabled.
    // `active` folds in `enabled`, so it cannot be the c(v) census gate: with the
    // buffer off there is no cross-round dedup and c(v) collapses to ~1 -- which
    // is the control arm the measurement needs, not a reason to stop measuring.
    bool in_query = false;
    std::unordered_map<uint64_t, char *> resident; // sector no -> its bytes in slab

    void reserve(uint64_t bpn) {
        if (!enabled || slab != nullptr) return;
        bytes_per_node = bpn;
        diskann::alloc_aligned((void **)&slab, cap_blocks * bpn, diskann::defaults::SECTOR_LEN);
    }
    void begin() { in_query = true; active = (enabled && slab != nullptr); used = 0; resident.clear(); }
    void end() { in_query = false; active = false; used = 0; resident.clear(); }

    char *find(uint64_t sector) const {
        if (!active) return nullptr;
        auto it = resident.find(sector);
        return it == resident.end() ? nullptr : it->second;
    }
    // Hands out the next free slab slot for `sector` and publishes it in
    // `resident` immediately; the caller points its AlignedRead at the slot, so
    // the sector is admitted by the read itself. Returns nullptr when the buffer
    // is off or full, in which case the caller falls back to per-call scratch and
    // the query degrades to today's behaviour rather than growing without bound.
    char *claim(uint64_t sector) {
        if (!active || used >= cap_blocks) return nullptr;
        char *slot = slab + used * bytes_per_node;
        resident.emplace(sector, slot);
        used++;
        return slot;
    }
    void release() {
        if (slab) { diskann::aligned_free(slab); slab = nullptr; }
    }
};

struct HotNodeCache {
    bool enabled = false;          // --hot-node-cache-unbounded
    bool writethrough = false;
    bool async_writeback = false;  // --async-rewire-commit: C_blk is the dirty overlay
    uint64_t bytes_per_node = 0;
    size_t ceiling_sectors = 0;    // --hot-cache-ceiling; hitting it is fatal
    size_t cap_sectors = 0;
    long frozen_misses = 0;        // misses claim() declined because the cap was hit

    bool lru = false;              // --hot-cache-lru
    long evictions = 0;
    // Largest dirty set seen while claim() was at the cap, i.e. the high-water
    // mark of the part of the cache eviction was not allowed to touch. Reported
    // against cap_sectors: if this approaches the cap the arm reverted to the
    // freeze behaviour for a while and its hit rate is not the policy's.
    long max_dirty_at_claim = 0;
    mutable uint64_t call_epoch = 0;
    using RecencyEntry = std::pair<uint64_t, uint64_t>;   // (sector, epoch)
    mutable std::list<RecencyEntry> recency;              // MRU at front
    mutable std::unordered_map<uint64_t, std::list<RecencyEntry>::iterator> recency_pos;

    static constexpr size_t kSlotsPerChunk = 8192;   // 32 MB at 4KB sectors
    std::vector<char *> chunks;
    size_t used = 0;                               // bump pointer, in slots
    std::unordered_map<uint64_t, char *> resident; // sector no -> its bytes
    mutable std::mutex mu;
    std::condition_variable cv, idle_cv;
    std::deque<uint64_t> writeq;
    std::unordered_map<uint64_t, uint64_t> version;
    std::unordered_set<uint64_t> dirty;
    std::thread writer;
    std::exception_ptr writer_error;
    bool stop_writer = false;
    long inflight = 0, async_enqueued = 0, async_writes = 0, async_rewrites = 0;
    double async_io_s = 0.0, async_wait_s = 0.0;

    // Per-batch reporting, read out before drop() and reset by it. The IOCounters
    // are cumulative over a method, so the per-batch line is a delta against the
    // snapshot the previous drop took.
    long batches = 0;
    size_t peak_sectors = 0;
    long snap_trav_hit = 0, snap_trav_admit = 0;
    long snap_rew_hit = 0, snap_rew_admit = 0;
    long snap_qbuf_hit = 0, snap_rmw = 0, snap_rmw_res = 0;
    long snap_trav_frozen = 0, snap_rew_frozen = 0;
    // A method boundary clears the IOCounters, so the snapshots must go with them
    // or the first batch of the next method reports a negative delta.
    void reset_snapshots() {
        snap_trav_hit = snap_trav_admit = snap_rew_hit = snap_rew_admit = 0;
        snap_qbuf_hit = snap_rmw = snap_rmw_res = 0;
        snap_trav_frozen = snap_rew_frozen = 0;
    }

    void reserve(uint64_t bpn) { if (enabled) bytes_per_node = bpn; }

    // Move a resident sector to the MRU end and stamp it with the call in
    // flight. Called from every hit and every admission, so the list order is
    // exactly last-use order and the stamp pins the current call's working set.
    void touch_locked(uint64_t sector) const {
        if (!lru) return;
        auto it = recency_pos.find(sector);
        if (it == recency_pos.end()) {
            recency.emplace_front(sector, call_epoch);
            recency_pos.emplace(sector, recency.begin());
        } else {
            it->second->second = call_epoch;
            recency.splice(recency.begin(), recency, it->second);
        }
    }
    // One read_nodes call is one epoch. Nothing this call has touched can be
    // evicted while it is still holding pointers into those slots.
    void begin_call() {
        if (!lru) return;
        std::lock_guard<std::mutex> lk(mu);
        call_epoch++;
    }
    // Least-recently-used first, skipping the two ineligible classes above, and
    // returning the victim's slot for the caller to reuse. Scanning from the
    // back, epochs are non-decreasing towards the front, so the first pinned
    // entry means everything ahead of it is pinned too and the scan can stop.
    char *evict_one_locked() {
        for (auto it = recency.rbegin(); it != recency.rend(); ++it) {
            if (it->second == call_epoch) break;       // pinned by the call in flight
            const uint64_t victim = it->first;
            if (dirty.count(victim)) continue;         // write-back still owes this one
            auto res = resident.find(victim);
            if (res == resident.end()) continue;       // index disagreement; skip, do not trust
            char *slot = res->second;
            resident.erase(res);
            recency_pos.erase(victim);
            recency.erase(std::next(it).base());
            evictions++;
            return slot;
        }
        return nullptr;
    }
    char *find_locked(uint64_t sector) const {
        if (!enabled) return nullptr;
        auto it = resident.find(sector);
        if (it == resident.end()) return nullptr;
        touch_locked(sector);
        return it->second;
    }
    char *find(uint64_t sector) const {
        std::lock_guard<std::mutex> lk(mu);
        return find_locked(sector);
    }
    // Never fails while under the ceiling: growing is the policy. Exceeding the
    // ceiling aborts rather than degrading to a miss -- a silent fallback would
    // make the arm mean "unbounded until it wasn't", and the hit rate would no
    // longer be the ceiling it is reported as.
    char *claim(uint64_t sector) {
        std::lock_guard<std::mutex> lk(mu);
        if (!enabled) return nullptr;
        auto it = resident.find(sector);
        if (it != resident.end()) return it->second;
        // The cap is checked before the ceiling: a capped run is meant to reach its
        // limit and keep going, so it must never reach the tripwire the uncapped
        // run aborts on.
        if (cap_sectors != 0 && used >= cap_sectors) {
            // At the cap: with LRU on, make room by evicting; without it, freeze
            // exactly as before. `used` stays at the cap either way, so the
            // arena, the ceiling check and peak_sectors all keep their meaning.
            if (lru && (long)dirty.size() > max_dirty_at_claim)
                max_dirty_at_claim = (long)dirty.size();
            char *reused = lru ? evict_one_locked() : nullptr;
            if (reused == nullptr) { frozen_misses++; return nullptr; }
            resident.emplace(sector, reused);
            touch_locked(sector);
            return reused;
        }
        if (used >= ceiling_sectors)
            throw std::runtime_error("hot-node cache exceeded --hot-cache-ceiling (" +
                                     std::to_string(ceiling_sectors) + " sectors); raise it or "
                                     "shorten the batch rather than letting the arm degrade");
        const size_t chunk = used / kSlotsPerChunk, slot_in_chunk = used % kSlotsPerChunk;
        if (chunk == chunks.size()) {
            char *mem = nullptr;
            diskann::alloc_aligned((void **)&mem, kSlotsPerChunk * bytes_per_node,
                                   diskann::defaults::SECTOR_LEN);
            chunks.push_back(mem);
        }
        char *slot = chunks[chunk] + slot_in_chunk * bytes_per_node;
        resident.emplace(sector, slot);
        touch_locked(sector);
        used++;
        return slot;
    }
    void mark_dirty_locked(uint64_t sector) {
        dirty.insert(sector);
        version[sector]++;
        writeq.push_back(sector);
        async_enqueued++;
        cv.notify_one();
    }
    void check_writer_error() {
        if (writer_error) std::rethrow_exception(writer_error);
    }
    void wait_idle() {
        if (!async_writeback || !writer.joinable()) return;
        auto t0 = Clock::now();
        std::unique_lock<std::mutex> lk(mu);
        idle_cv.wait(lk, [&] { return writeq.empty() && inflight == 0; });
        async_wait_s += secs_since(t0);
        check_writer_error();
    }
    void start_async_writer(int fd, uint64_t bytes) {
        if (!async_writeback) return;
        writer = std::thread([this, fd, bytes] {
            char *buf = nullptr;
            diskann::alloc_aligned((void **)&buf, bytes, diskann::defaults::SECTOR_LEN);
            try {
                for (;;) {
                    uint64_t sector = 0, v = 0;
                    {
                        std::unique_lock<std::mutex> lk(mu);
                        cv.wait(lk, [&] { return stop_writer || !writeq.empty(); });
                        if (stop_writer && writeq.empty()) break;
                        sector = writeq.front();
                        writeq.pop_front();
                        auto it = resident.find(sector);
                        if (it == resident.end()) { idle_cv.notify_all(); continue; }
                        v = version[sector];
                        inflight++;
                        std::memcpy(buf, it->second, bytes);
                    }
                    auto t0 = Clock::now();
                    if (pwrite_backend(fd, buf, bytes, sector * diskann::defaults::SECTOR_LEN) != (ssize_t)bytes)
                        throw std::runtime_error("async rewire write-back: pwrite failed");
                    charge_write_delay(bytes);
                    const double io_s = secs_since(t0);
                    {
                        std::lock_guard<std::mutex> lk(mu);
                        async_io_s += io_s;
                        async_writes++;
                        if (version[sector] == v) dirty.erase(sector);
                        else { writeq.push_back(sector); async_rewrites++; cv.notify_one(); }
                        inflight--;
                        idle_cv.notify_all();
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> lk(mu);
                writer_error = std::current_exception();
                writeq.clear();
                inflight = 0;
                idle_cv.notify_all();
            }
            diskann::aligned_free(buf);
        });
    }
    void stop_async_writer_now() {
        if (!writer.joinable()) return;
        try { wait_idle(); } catch (...) {}
        {
            std::lock_guard<std::mutex> lk(mu);
            stop_writer = true;
            cv.notify_all();
        }
        writer.join();
        check_writer_error();
    }
    // End of batch. The map is emptied and the bump pointer rewound; the chunks
    // themselves stay and are overwritten by the next batch. Nothing can read the
    // old bytes because nothing can find them.
    void drop() {
        wait_idle();
        std::lock_guard<std::mutex> lk(mu);
        peak_sectors = std::max(peak_sectors, used);
        resident.clear();
        recency.clear();
        recency_pos.clear();
        version.clear();
        dirty.clear();
        used = 0;
        batches++;
    }
    // Method boundary: same eviction, but it is not a batch and must not be
    // counted as one, and the per-method totals start from zero.
    void reset_for_method() {
        wait_idle();
        std::lock_guard<std::mutex> lk(mu);
        resident.clear();
        recency.clear();
        recency_pos.clear();
        version.clear();
        dirty.clear();
        writeq.clear();
        used = 0;
        batches = 0;
        peak_sectors = 0;
        reset_snapshots();
    }
    double resident_mb() const { return (double)(used * bytes_per_node) / 1048576.0; }
    double peak_mb() const { return (double)(peak_sectors * bytes_per_node) / 1048576.0; }
    void release() {
        try { stop_async_writer_now(); }
        catch (const std::exception &e) { fprintf(stderr, "async writer shutdown failed: %s\n", e.what()); }
        std::lock_guard<std::mutex> lk(mu);
        for (char *c : chunks) diskann::aligned_free(c);
        chunks.clear();
        resident.clear();
        version.clear();
        dirty.clear();
        used = 0;
    }
};

// Brackets one query. Everything read inside is charged to a single trajectory.
// do_rewire is deliberately left outside any scope: a rewire batch is not a
// query, and sharing the buffer with it would flatter its I/O.
struct QueryScope {
    QueryBlockBuffer &b;
    explicit QueryScope(QueryBlockBuffer &qb) : b(qb) { b.begin(); }
    ~QueryScope() { b.end(); }
    QueryScope(const QueryScope &) = delete;
    QueryScope &operator=(const QueryScope &) = delete;
};

struct DiskCostStat {
    double cost_ewma = 0.0;     // meaningless while observation_count == 0
    long observation_count = 0; // 0 == never expanded, therefore never measured
};
// One hash finaliser for both flat tables below. Multiply-xor-fold, so the low
// bits a power-of-two mask keeps are a function of every input bit -- node ids
// are dense and sequential, and taking their low bits raw would cluster.
static inline uint32_t flat_hash_of_node(uint32_t k) {
    uint64_t z = (uint64_t)k * 0x9E3779B97F4A7C15ull;
    return (uint32_t)(z >> 32) ^ (uint32_t)z;
}

struct FlatNodeSet {
    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;   // not a valid node id
    std::vector<uint32_t> slots;
    uint32_t slot_mask = 0;
    size_t entry_count = 0;
    void init(size_t expected_entries) {
        size_t capacity = 16;
        while (capacity < expected_entries * 4) capacity <<= 1;
        slots.assign(capacity, EMPTY_KEY);
        slot_mask = (uint32_t)(capacity - 1);
        entry_count = 0;
    }
    void clear() {
        std::fill(slots.begin(), slots.end(), EMPTY_KEY);
        entry_count = 0;
    }
    size_t size() const { return entry_count; }
    size_t count(int key) const {
        uint32_t k = (uint32_t)key, i = flat_hash_of_node(k) & slot_mask;
        while (slots[i] != EMPTY_KEY) {
            if (slots[i] == k) return 1;
            i = (i + 1) & slot_mask;
        }
        return 0;
    }
    void insert(int key) {
        uint32_t k = (uint32_t)key, i = flat_hash_of_node(k) & slot_mask;
        while (slots[i] != EMPTY_KEY) {
            if (slots[i] == k) return;
            i = (i + 1) & slot_mask;
        }
        slots[i] = k;
        entry_count++;
        if (entry_count * 2 > slots.size()) grow();
    }
    // unordered_set::insert(k).second, which two of the train_batch call sites
    // consume: true when the key was NOT already present.
    bool insert_new(int key) {
        uint32_t k = (uint32_t)key, i = flat_hash_of_node(k) & slot_mask;
        while (slots[i] != EMPTY_KEY) {
            if (slots[i] == k) return false;
            i = (i + 1) & slot_mask;
        }
        slots[i] = k;
        entry_count++;
        if (entry_count * 2 > slots.size()) grow();
        return true;
    }
    template <class Iter> void insert(Iter first, Iter last) {
        for (; first != last; ++first) insert((int)*first);
    }
    void grow() {
        std::vector<uint32_t> old_slots;
        old_slots.swap(slots);
        slots.assign(old_slots.size() * 2, EMPTY_KEY);
        slot_mask = (uint32_t)(slots.size() - 1);
        entry_count = 0;
        for (uint32_t k : old_slots) if (k != EMPTY_KEY) insert((int)k);
    }
};


struct DiskCostTable {
    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;   // not a valid node id
    std::vector<uint32_t> keys;
    std::vector<DiskCostStat> vals;
    uint32_t slot_mask = 0;
    size_t entry_count = 0;
    DiskCostTable() { allocate(1024); }
    void allocate(size_t capacity) {
        keys.assign(capacity, EMPTY_KEY);
        vals.assign(capacity, DiskCostStat());
        slot_mask = (uint32_t)(capacity - 1);
        entry_count = 0;
    }
    void clear() { allocate(1024); }
    size_t size() const { return entry_count; }
    DiskCostStat *find(uint32_t key) {
        uint32_t i = flat_hash_of_node(key) & slot_mask;
        while (keys[i] != EMPTY_KEY) {
            if (keys[i] == key) return &vals[i];
            i = (i + 1) & slot_mask;
        }
        return nullptr;
    }
    DiskCostStat &operator[](uint32_t key) {
        uint32_t i = flat_hash_of_node(key) & slot_mask;
        while (keys[i] != EMPTY_KEY) {
            if (keys[i] == key) return vals[i];
            i = (i + 1) & slot_mask;
        }
        keys[i] = key;
        entry_count++;
        // Kept at or below half load: a grow re-finds the key afterwards, so the
        // reference handed back is always into the live table.
        if (entry_count * 2 > keys.size()) { grow(); return *find(key); }
        return vals[i];
    }
    void grow() {
        std::vector<uint32_t> old_keys;
        std::vector<DiskCostStat> old_vals;
        old_keys.swap(keys);
        old_vals.swap(vals);
        allocate(old_keys.size() * 2);
        for (size_t i = 0; i < old_keys.size(); i++)
            if (old_keys[i] != EMPTY_KEY) (*this)[old_keys[i]] = old_vals[i];
    }
};

// Keyed on TARGET node; lives across queries, cleared per method. This cost
// census is intentionally separate from §6's optional source-node learner cache.
static DiskCostTable g_expansion_cost_by_node;
static double g_expansion_cost_ewma_rate = 0.1; // --expansion-cost-beta
static bool g_track_expansion_cost = false;     // --track-expansion-cost

static bool g_harvest_cosector = false;   // --harvest-cosector

static long g_search_ep_override = -1;
static bool g_audit_ep_last_result = false;
static long g_last_serving_ep = -1;
static double g_harvest_ratio = 1.0;      // --harvest-ratio: fraction of harvested
                                           // co-occupants (closest-first) whose
                                           // neighbours also get PQ-scored and
                                           // pushed into the frontier

// One observation of "expanding v cost a fetch", folded into v's marginal.
static inline void observe_expansion_cost(uint32_t v, int paid_a_fetch) {
    DiskCostStat &s = g_expansion_cost_by_node[v];
    // The first observation seeds the EWMA at itself rather than at a prior, so
    // observation_count == 0 keeps meaning "no measurement" and never "a
    // measurement of 1.0". A never-expanded node has an empty statistic, not an
    // uncertain one, and eq. (6) has to make that choice explicitly rather than
    // inherit a prior invented here.
    s.cost_ewma = (s.observation_count == 0)
                      ? (double)paid_a_fetch
                      : (1.0 - g_expansion_cost_ewma_rate) * s.cost_ewma +
                            g_expansion_cost_ewma_rate * (double)paid_a_fetch;
    s.observation_count++;
}

// One definition of "measured", shared by the coverage report and by pricing.
// Returns -1 for a node the census has never observed. If the two ever disagreed
// on what counts as measured, the report would be describing a different
// population than eq. (6) actually acts on.
static inline double lookup_expansion_cost(uint32_t v) {
    DiskCostStat *s = g_expansion_cost_by_node.find(v);
    if (s == nullptr || s->observation_count == 0) return -1.0;
    return s->cost_ewma;
}

static double g_expansion_cost_theta = 0.0; // --expansion-cost-theta

struct IoPriceController {
    bool enabled = false;                    // --io-target-per-query
    double io_target_per_query = 0.0;        // I*, device sector reads per query
    double dual_ascent_step_gamma = 0.0;
    double gamma_override = 0.0;             // --dual-ascent-gamma
    double gamma_floor = 0.0;                // anneal stops here so drift stays trackable
    double deadband_sectors_per_query = 0.0; // --io-target-deadband
    bool one_sided = false;                  // --io-target-one-sided
    int previous_target_gap_sign = 0;
    long steps_taken = 0;
    double last_nu_bar = 0.0;                // for Thm 5.2's lambda*nu_bar deficit bound

    // Deferred to the first boundary: nu_bar is not knowable before a batch runs.
    void warm_start_step_size(double mean_expansions_per_query) {
        if (gamma_override > 0.0) dual_ascent_step_gamma = gamma_override;
        else if (mean_expansions_per_query > 0.0)
            dual_ascent_step_gamma = 0.1 / mean_expansions_per_query;
        else dual_ascent_step_gamma = 0.1;
        gamma_floor = dual_ascent_step_gamma / 64.0;
        fprintf(stderr, "  IOPRICE: warm start gamma=%.6g (%s), floor=%.6g, "
                        "nu_bar=%.2f expansions/query, target=%.2f sectors/query\n",
                dual_ascent_step_gamma,
                gamma_override > 0.0 ? "--dual-ascent-gamma" : "0.1/nu_bar",
                gamma_floor, mean_expansions_per_query, io_target_per_query);
    }

    // Alg. 3 line 1, before selection, so this batch commits at the new price.
    void refresh_price(double measured_io_per_query, double mean_expansions_per_query,
                       long gqc) {
        if (!enabled) return;
        if (steps_taken == 0) warm_start_step_size(mean_expansions_per_query);
        last_nu_bar = mean_expansions_per_query;
        const double effective_target = one_sided
            ? std::min(measured_io_per_query, io_target_per_query)
            : io_target_per_query;
        const double target_gap = measured_io_per_query - effective_target;
        const bool inside_deadband = deadband_sectors_per_query > 0.0 &&
                                     std::fabs(target_gap) < deadband_sectors_per_query;
        const double theta_before = g_expansion_cost_theta;
        if (!inside_deadband) {
            // A sign flip means the step overshot; halve it.
            const int gap_sign = (target_gap > 0.0) - (target_gap < 0.0);
            if (previous_target_gap_sign != 0 && gap_sign != 0 &&
                gap_sign != previous_target_gap_sign)
                dual_ascent_step_gamma = std::max(dual_ascent_step_gamma / 2.0, gamma_floor);
            if (gap_sign != 0) previous_target_gap_sign = gap_sign;
            // Projection matters: theta < 0 would pay for expensive edges.
            g_expansion_cost_theta = std::max(0.0, g_expansion_cost_theta +
                                                   dual_ascent_step_gamma * target_gap);
        }
        steps_taken++;
        fprintf(stderr, "  IOPRICE: q=%ld io_per_query=%.2f target=%.2f gap=%+.2f "
                        "gamma=%.6g theta=%.6g->%.6g nu_bar=%.2f%s\n",
                gqc, measured_io_per_query, io_target_per_query, target_gap,
                dual_ascent_step_gamma, theta_before, g_expansion_cost_theta,
                mean_expansions_per_query,
                inside_deadband ? " [deadband]"
                                : (one_sided && target_gap == 0.0 ? " [one-sided hold]" : ""));
    }
};
static IoPriceController g_io_price_controller;

enum class UnmeasuredCostPolicy {
    PoolMean, // neutral: the mean of this pool's measured candidates. Invents nothing.
    Free,     // optimistic: unmeasured looks costless
    Full      // pessimistic: unmeasured looks like a guaranteed fetch
};
static double g_rewire_margin_lambda = 0.0; // --rewire-margin-lambda

static bool g_early_rewire_commit = false;
// --rewire-commit-direct. The edit overlay still exists for the duration of one
// concrete decision, but it is flushed immediately after each directed removal
// or bidirectional addition. That isolates the value of batching adj_edit across
// a whole do_rewire pass without changing the bytes written for a decision.
static bool g_rewire_commit_direct = false;
static long g_commit_flush_calls = 0;
static long g_commit_flush_rows = 0;

static double g_lambda_auto_quantile = 0.0; // --lambda-auto[-quantile], 0 = off
static double g_lambda_auto_scale = 0.5;    // --lambda-auto-scale
static double g_lambda_auto_ewma = 0.0;     // --lambda-auto-ewma, smooths WHICHEVER auto
                                            // rule is armed; 0 = track the quantile

static double g_lambda_noise_floor_quantile = 0.0; // --lambda-noise-floor[-quantile], 0 = off

static bool g_lambda_calibrate_first_batch = false;
static bool g_lambda_calibrate_every_batch = false;

static bool g_rewire_unidirectional = false;
// Rows admitted by the gate as the PRIMARY endpoint u in the current batch.
// ADJEDIT uses it to split dirtied rows into u-rows and reciprocal v-rows, which
// is the measurement that decides whether the reciprocal insert is what stops
// the write cut tracking q.
static std::unordered_set<uint32_t> g_admitted_u_rows;

static double place_strictly_above(const std::vector<double> &sorted_samples, size_t k) {
    const double value = sorted_samples[k];
    size_t next = k + 1;
    while (next < sorted_samples.size() && sorted_samples[next] == value) next++;
    if (next == sorted_samples.size())
        return std::nextafter(value, std::numeric_limits<double>::infinity());
    const double midpoint = 0.5 * (value + sorted_samples[next]);
    if (midpoint > value && midpoint < sorted_samples[next]) return midpoint;
    const double stepped = std::nextafter(value, sorted_samples[next]);
    // The two samples are adjacent doubles, so nothing lies strictly between them.
    // Fall back to the quantile value itself: a strict g > lambda still vetoes
    // exactly the mass at or below it, whereas stepping onto the next sample would
    // veto one atom too many. The margin is what is lost, not the veto fraction.
    return stepped < sorted_samples[next] ? stepped : value;
}
// --dump-gain-terms <csv>: Algorithm 3 line 5's two operands, per changed row.
// The GAIN census only ever reported their difference, so the shape of
// sum_{N_out\S} Qhat_theta -- the term that decides whether lambda lands on an
// atom -- was not observable from any existing log.
static FILE *g_gain_terms_file = nullptr;

static bool g_recall_prior = false; // --recall-prior
static double mean_or(const std::vector<double> &v, double fallback) {
    if (v.empty()) return fallback;
    double s = 0.0;
    for (double x : v) s += x;
    return s / (double)v.size();
}
static double recall_prior_for_batch(const std::vector<double> &batch_recalls, double Q_0,
                                     const std::string &reward) {
    if (!g_recall_prior || reward != "recall" || batch_recalls.empty()) return Q_0;
    return mean_or(batch_recalls, Q_0);
}

static UnmeasuredCostPolicy g_unmeasured_cost_policy = UnmeasuredCostPolicy::PoolMean;
static const char *unmeasured_cost_policy_name() {
    switch (g_unmeasured_cost_policy) {
        case UnmeasuredCostPolicy::Free: return "free";
        case UnmeasuredCostPolicy::Full: return "full";
        default: return "pool_mean";
    }
}

struct ExpansionCostCoverage {
    long unmeasured_incumbent = 0, unmeasured_challenger = 0;
    long measured_incumbent = 0, measured_challenger = 0;
    long measured_below_one = 0;
    double measured_cost_sum = 0.0;
    long histogram[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}; // c over [0,1], bin width 0.1
    long pools_total = 0, pools_with_spread = 0;
    double pool_spread_sum = 0.0;

    // One candidate of one pool, with its cost already resolved by
    // lookup_expansion_cost (negative == never measured).
    void observe_candidate(bool is_incumbent, double cost) {
        if (cost < 0.0) {
            (is_incumbent ? unmeasured_incumbent : unmeasured_challenger)++;
            return;
        }
        (is_incumbent ? measured_incumbent : measured_challenger)++;
        measured_cost_sum += cost;
        if (cost < 1.0) measured_below_one++;
        histogram[std::min(9, std::max(0, (int)(cost * 10.0)))]++; // c == 1.0 -> last bin
    }
    // A pool is reorderable only if at least two of its candidates were measured
    // AND they disagree; one measured candidate among unmeasured ones is not a
    // spread, since eq. (6) has nothing to compare it against.
    void observe_pool(long measured_in_pool, double min_cost, double max_cost) {
        pools_total++;
        if (measured_in_pool >= 2 && max_cost > min_cost) {
            pools_with_spread++;
            pool_spread_sum += max_cost - min_cost;
        }
    }
    void print() const {
        const long measured = measured_incumbent + measured_challenger;
        const long unmeasured = unmeasured_incumbent + unmeasured_challenger;
        fprintf(stderr,
                "  CVCOV: nodes_measured=%zu candidates=%ld measured=%ld(inc %ld/chal %ld)"
                " unmeasured=%ld(inc %ld/chal %ld)\n",
                g_expansion_cost_by_node.size(), measured + unmeasured, measured,
                measured_incumbent, measured_challenger, unmeasured, unmeasured_incumbent,
                unmeasured_challenger);
        fprintf(stderr,
                "  CVCOV: mean_c=%.4f frac_reorderable=%.4f pools=%ld spread_pools=%ld(%.4f)"
                " mean_spread=%.4f\n",
                measured ? measured_cost_sum / measured : 0.0,
                measured ? (double)measured_below_one / measured : 0.0, pools_total,
                pools_with_spread, pools_total ? (double)pools_with_spread / pools_total : 0.0,
                pools_with_spread ? pool_spread_sum / pools_with_spread : 0.0);
        fprintf(stderr, "  CVCOV: hist=");
        for (int i = 0; i < 10; i++) fprintf(stderr, "%s%ld", i ? "," : "[", histogram[i]);
        fprintf(stderr, "]\n");
    }
};

// UCB/LCB-gated eviction: rank incumbents by their upper confidence bound and
// challengers by their lower bound, so a single-query unconverged challenger
// cannot displace an established incumbent on noise alone. (Same flag and
// semantics as control_vamana.cpp.)
static bool g_ucb_gate_rewire = false;

static bool g_no_cost_coverage = false;

// ── disk-resident DiskANN index: PQ codes in RAM, graph + full vectors on SSD ──
struct DiskANNIndex {
    // layout metadata, read from the _disk.index header exactly as
    // PQFlashIndex::load_from_separate_paths does
    uint64_t num_points = 0, data_dim = 0, max_node_len = 0, nnodes_per_sector = 0;
    uint64_t max_degree = 0;          // R, the degree cap the on-disk record can hold
    uint64_t disk_bytes_per_point = 0;
    uint64_t num_sectors_per_node = 1;
    uint32_t medoid = 0;              // search entry point (== VamanaGraph::start)
    std::vector<uint32_t> medoid_candidates;

    // PQ side: codes stay resident in RAM for the whole run, which is the entire
    // point of the DiskANN regime -- navigation never touches the disk.
    uint64_t n_chunks = 0;
    std::vector<uint8_t> pq_codes;    // num_points * n_chunks
    diskann::FixedChunkPQTable pq_table;
    std::vector<float> pq_codeword_pair_dists; // [chunk][source code][target code]

    std::shared_ptr<AlignedFileReader> reader;
    QueryBlockBuffer qbuf;            // per-query block dedup; off unless --query-block-buffer
    HotNodeCache hotcache;            // C_blk, cross-query; off unless --hot-node-cache
    std::string disk_index_file;
    bool mips = false;                // MIPS->L2 reduction in effect (see g_mips)
    bool track_orig = false;          // capture pristine rows so restore_dirty() can undo
    int write_fd = -1;                // O_DIRECT|O_RDWR|O_LARGEFILE handle used to commit rewires
    bool writeback = true;            // false => edits stay in the RAM overlay only
    bool owns_work_copy = false;      // disk_index_file is a scratch copy we may delete
    bool keep_work_copy = false;      // leave the (rewired) copy behind for inspection

    // Adjacency edited since the last flush. Reads consult it first, so an
    // in-flight edit is visible before it is committed; on flush the rows are
    // written back to their sectors and dropped, which keeps later reads of
    // those nodes charged to disk like every other node.
    std::unordered_map<uint32_t, std::vector<uint32_t>> adj_edit;
    // Pristine (pre-edit) adjacency of every node ever edited, captured on first
    // touch -- the disk analogue of VamanaGraph::dirty + restore_dirty(), which
    // lets --isolated reset the graph between queries without re-reading it all.
    std::unordered_map<uint32_t, std::vector<uint32_t>> adj_orig;

    long edges_added = 0, edges_deleted = 0;

    void load(const std::string &index_prefix, bool writeback_, const std::string &work_copy,
              const std::string &reuse_work_copy) {
        writeback = writeback_;
        const std::string source_disk_index_file = index_prefix + "_disk.index";
        disk_index_file = reuse_work_copy.empty() ? source_disk_index_file : reuse_work_copy;
        std::string pivots_file = index_prefix + "_pq_pivots.bin";
        std::string codes_file = index_prefix + "_pq_compressed.bin";
        if (file_exists(source_disk_index_file + "_pq_pivots.bin"))
            throw std::runtime_error("disk-PQ indexes are unsupported: this harness needs "
                                     "full-precision vectors on disk for exact re-ranking");

        // PQ pivots define the true data dimension; codes give num_points/n_chunks.
        size_t pq_ncentroids, pq_dim;
        diskann::get_bin_metadata(pivots_file, pq_ncentroids, pq_dim, METADATA_SIZE);
        if (pq_ncentroids != 256) throw std::runtime_error("expected 256 PQ centroids");
        data_dim = pq_dim;
        g_dim = (int)data_dim;
        // MIPS indexes are self-identifying: build_disk_index writes the rescaling
        // factor next to the graph. Detect rather than flag, so a mismatched
        // --dist_fn cannot silently produce L2 rankings on a MIPS graph.
        g_mips = file_exists(source_disk_index_file + "_max_base_norm.bin");
        mips = g_mips;
        g_query_dim = g_mips ? (int)data_dim - 1 : (int)data_dim;
        disk_bytes_per_point = data_dim * data_type_elem_size(g_data_type);

        // Read the PQ-codes bin directly into pq_codes instead of going through
        // diskann::load_bin (new[] into a raw T*) + assign()-then-delete[]: that
        // path keeps two full-size copies alive at once, doubling peak RAM for
        // no reason at PQ-codes sizes that matter (tens of GB at billion-point
        // scale). Format matches load_bin_impl: int32 npts, int32 dim, then
        // npts*dim raw bytes, no other header.
        {
            std::ifstream codes_reader(codes_file, std::ios::binary);
            if (!codes_reader.is_open()) throw std::runtime_error("cannot open " + codes_file);
            int32_t npts_i32 = 0, nchunks_i32 = 0;
            codes_reader.read((char *)&npts_i32, sizeof(int32_t));
            codes_reader.read((char *)&nchunks_i32, sizeof(int32_t));
            num_points = (size_t)npts_i32;
            n_chunks = (size_t)nchunks_i32;
            pq_codes.resize(num_points * n_chunks);
            codes_reader.read((char *)pq_codes.data(), (std::streamsize)pq_codes.size());
            if (!codes_reader) throw std::runtime_error("short read on " + codes_file);
        }
        pq_table.load_pq_centroid_bin(pivots_file.c_str(), n_chunks);
        if (pq_table.supports_codeword_pair_distances()) {
            pq_codeword_pair_dists.resize((size_t)n_chunks * 256 * 256);
            pq_table.populate_codeword_pair_distances(pq_codeword_pair_dists.data());
        }

        // _disk.index header: [nr nc][npts ndims][medoid][max_node_len][nnodes_per_sector]...
        std::ifstream meta(disk_index_file, std::ios::binary);
        if (!meta.is_open()) throw std::runtime_error("cannot open " + disk_index_file);
        auto rd32 = [&meta] { uint32_t v; meta.read((char *)&v, sizeof(v)); return v; };
        auto rd64 = [&meta] { uint64_t v; meta.read((char *)&v, sizeof(v)); return v; };
        rd32(); rd32();                          // nr, nc of the bin-format header
        uint64_t disk_nnodes = rd64();
        rd64();                                  // disk ndims (== data_dim without disk-PQ)
        uint64_t medoid_id = rd64();
        max_node_len = rd64();
        nnodes_per_sector = rd64();
        meta.close();
        if (disk_nnodes != num_points)
            throw std::runtime_error("point-count mismatch between PQ codes and disk index");
        medoid = (uint32_t)medoid_id;
        medoid_candidates = {medoid};
        std::string medoids_file = disk_index_file + "_medoids.bin";
        if (file_exists(medoids_file)) {
            uint32_t *medoids = nullptr;
            size_t nmed, mdim;
            diskann::load_bin<uint32_t>(medoids_file, medoids, nmed, mdim);
            if (nmed != 1)
                std::cerr << "info: index has " << nmed
                          << " medoids; search() enters from whichever is PQ-nearest "
                             "to each query\n";
            medoid = medoids[0];
            medoid_candidates.assign(medoids, medoids + nmed);
            delete[] medoids;
        }
        max_degree = ((max_node_len - disk_bytes_per_point) / sizeof(uint32_t)) - 1;
        num_sectors_per_node =
            nnodes_per_sector > 0 ? 1 : DIV_ROUND_UP(max_node_len, diskann::defaults::SECTOR_LEN);

        // In-memory: pull the whole index into RAM once, here, and never touch
        // the device again. This is the single permitted disk access of an
        // in-memory session -- the equivalent of the Vamana harness loading its
        // CSR at start -- and it happens before any query is served, so it is
        // outside every measured phase.
        if (g_in_memory) {
            auto t0 = Clock::now();
            const uint64_t fsz = (uint64_t)std::filesystem::file_size(disk_index_file);
            g_ram_image.resize(fsz);
            std::ifstream img(disk_index_file, std::ios::binary);
            if (!img.is_open()) throw std::runtime_error("cannot open " + disk_index_file);
            img.read(g_ram_image.data(), (std::streamsize)fsz);
            if ((uint64_t)img.gcount() != fsz)
                throw std::runtime_error("short read loading index image into RAM");
            img.close();
            fprintf(stderr, "IN-MEMORY: loaded %s into RAM (%.2f GB, %.1fs); "
                            "C_blk forced off, no device I/O after this point\n",
                    disk_index_file.c_str(), (double)fsz / 1e9, secs_since(t0));
            return;
        }

        if (!work_copy.empty()) {
            auto t0 = Clock::now();
            std::filesystem::copy_file(disk_index_file, work_copy,
                                       std::filesystem::copy_options::overwrite_existing);
            double copy_s = secs_since(t0);
            auto t1 = Clock::now();
            for (const std::string &f : {disk_index_file, work_copy}) {
                int fd = ::open(f.c_str(), O_RDONLY | O_LARGEFILE);
                if (fd < 0) continue;
                if (f == work_copy) ::fsync(fd);
                ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
                ::close(fd);
            }
            ::sync();
            double sync_s = secs_since(t1);
            fprintf(stderr, "work copy: %s -> %s (%.1f GB, copy %.1fs + sync/evict %.1fs)\n",
                    disk_index_file.c_str(), work_copy.c_str(),
                    (double)std::filesystem::file_size(work_copy) / 1e9, copy_s, sync_s);
            disk_index_file = work_copy;
            owns_work_copy = true;
        }
        // The backend is the whole point of --io-uring, so it is chosen here and
        // nowhere else; both paths then open the same file with the same flags.
        if (g_use_io_uring) {
#ifdef DISKANN_URING
            reader.reset(new UringAlignedFileReader());
#else
            throw std::runtime_error("--io-uring requires a build with DISKANN_URING and liburing "
                                     "(target control_diskann_uring); this binary has libaio only");
#endif
        } else {
            reader.reset(new LinuxAlignedFileReader());
        }
        reader->open(disk_index_file);
        reader->register_thread();
        // One greppable tag on both paths, so a log can never be attributed to the
        // wrong backend after the fact.
        fprintf(stderr, "READER: %s\n", g_use_io_uring ? "io_uring" : "libaio");
        if (writeback) {
            // Same flags as LinuxAlignedFileReader (O_DIRECT | O_LARGEFILE), except
            // O_RDWR in place of O_RDONLY: this handle exists to commit rewires, and
            // its read-modify-write reads must bypass the page cache exactly as the
            // reader's do, or the RMW half would be served from cache and undercount.
            write_fd = ::open(disk_index_file.c_str(), O_RDWR | O_DIRECT | O_LARGEFILE);
            if (write_fd < 0) throw std::runtime_error("cannot open index for rewire write-back");
        }
    }

    ~DiskANNIndex() {
        if (swap_buf_a) diskann::aligned_free(swap_buf_a);
        if (swap_buf_b) diskann::aligned_free(swap_buf_b);
        delete[] swap_rec;
        qbuf.release();
        hotcache.release();
        if (write_fd >= 0) ::close(write_fd);
        if (reader) reader->close();   // null under --in-memory
        if (owns_work_copy && !keep_work_copy) {
            std::error_code ec;
            std::filesystem::remove(disk_index_file, ec);
        }
    }

    std::unordered_map<uint32_t, uint32_t> reloc;      // id -> slot
    std::unordered_map<uint32_t, uint32_t> slot_owner; // slot -> id
    long layout_swaps = 0;

    std::vector<uint32_t> epoch_missed;
    bool collect_epoch_misses = false;  // set from --layout-adapt

    inline uint32_t slot_of(uint32_t node_id) const {
        auto it = reloc.find(node_id);
        return it == reloc.end() ? node_id : it->second;
    }
    inline uint32_t owner_of_slot(uint32_t slot) const {
        auto it = slot_owner.find(slot);
        return it == slot_owner.end() ? slot : it->second;
    }

    // The disk file contains the rewritten records. This sidecar is the other
    // half of a relocated layout: its non-identity logical-id -> physical-slot
    // permutation. slot_owner is reconstructed because it is the inverse.
    void save_layout_map(const std::string &path) const {
        FILE *f = fopen(path.c_str(), "wb");
        if (!f) throw std::runtime_error("cannot open layout map for write: " + path);
        const uint64_t magic = 0x5041474557495245ULL; // "PAGEWIRE"
        const uint64_t version = 1;
        const uint64_t n = num_points;
        std::vector<std::pair<uint32_t, uint32_t>> rows(reloc.begin(), reloc.end());
        std::sort(rows.begin(), rows.end());
        const uint64_t count = rows.size();
        if (fwrite(&magic, sizeof(magic), 1, f) != 1 ||
            fwrite(&version, sizeof(version), 1, f) != 1 ||
            fwrite(&n, sizeof(n), 1, f) != 1 ||
            fwrite(&count, sizeof(count), 1, f) != 1 ||
            (!rows.empty() && fwrite(rows.data(), sizeof(rows[0]), rows.size(), f) != rows.size())) {
            fclose(f);
            throw std::runtime_error("short write saving layout map: " + path);
        }
        fclose(f);
        fprintf(stderr, "LAYOUTMAP: saved %llu moved nodes -> %s\n",
                (unsigned long long)count, path.c_str());
    }

    void load_layout_map(const std::string &path) {
        FILE *f = fopen(path.c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open layout map for read: " + path);
        uint64_t magic, version, n, count;
        if (fread(&magic, sizeof(magic), 1, f) != 1 ||
            fread(&version, sizeof(version), 1, f) != 1 ||
            fread(&n, sizeof(n), 1, f) != 1 ||
            fread(&count, sizeof(count), 1, f) != 1 ||
            magic != 0x5041474557495245ULL || version != 1 || n != num_points) {
            fclose(f);
            throw std::runtime_error("invalid layout map: " + path);
        }
        reloc.clear();
        slot_owner.clear();
        for (uint64_t i = 0; i < count; i++) {
            std::pair<uint32_t, uint32_t> row;
            if (fread(&row, sizeof(row), 1, f) != 1 || row.first >= num_points ||
                row.second >= num_points || row.first == row.second ||
                !reloc.emplace(row).second || !slot_owner.emplace(row.second, row.first).second) {
                fclose(f);
                throw std::runtime_error("corrupt layout map: " + path);
            }
        }
        fclose(f);
        fprintf(stderr, "LAYOUTMAP: loaded %llu moved nodes <- %s\n",
                (unsigned long long)count, path.c_str());
    }

    // The kappa nodes currently sharing node_id's sector, in slot order. This is
    // pi(u) in Algorithm 4: a set of *ids*, resolved through the relocation map,
    // since which node occupies a slot changes as swaps land. Callers must have
    // established nnodes_per_sector >= 2 (layout_step does).
    void block_of(uint32_t node_id, std::vector<uint32_t> &occupants) const {
        occupants.clear();
        const uint32_t kappa = (uint32_t)nnodes_per_sector;
        const uint32_t base = (slot_of(node_id) / kappa) * kappa;
        for (uint32_t j = 0; j < kappa && base + j < (uint32_t)num_points; j++)
            occupants.push_back(owner_of_slot(base + j));
    }

    // sector holding node_id, and the node's offset within the fetched buffer
    // (multi-node sectors pack nnodes_per_sector records; wide nodes get their
    // own run of sectors starting at offset 0) -- same arithmetic as PQFlashIndex,
    // applied to the node's current slot rather than its id. Sector indices and
    // byte offsets stay 64-bit (a 13.7 GB file overruns 2^32); the slot does not
    // need to, since it shares the id's range.
    inline uint64_t node_sector(uint64_t node_id) const {
        const uint64_t s = slot_of((uint32_t)node_id);
        return 1 + (nnodes_per_sector > 0 ? s / nnodes_per_sector
                                          : s * num_sectors_per_node);
    }
    inline char *node_offset(char *sector_buf, uint64_t node_id) const {
        return sector_buf + (nnodes_per_sector == 0
                                 ? 0
                                 : ((uint64_t)slot_of((uint32_t)node_id) % nnodes_per_sector) *
                                       max_node_len);
    }

    void swap_slots(uint32_t v, uint32_t w) {
        if (v == w) return;
        if (!writeback)
            throw std::runtime_error("layout adaptation needs --writeback: relocation is a "
                                     "physical sector move, not a RAM overlay edit");
        // No per-swap wait_idle() here any more: the caller (layout_step) drains
        // once before the pass starts, and every commit below either goes through
        // the SAME resident buffer + dirty bit the async writer already resolves
        // through `resident`/`version` (self-correcting on a re-dirty, see
        // start_async_writer), or is a synchronous pwrite on this thread. Neither
        // needs a drain between swaps.
        const uint64_t bytes = num_sectors_per_node * diskann::defaults::SECTOR_LEN;
        const uint64_t sv = node_sector(v), sw = node_sector(w);
        const bool two = (sw != sv);
        ensure_swap_buffers(bytes);

        // Both halves go through C_blk whenever it holds them. Reading around the
        // cache would leave the resident copy on pre-swap bytes, and under
        // write-through/write-back that copy is later flushed over the new layout,
        // so relocation has to patch in place rather than beside it. Coherence then
        // stops depending on dropping the cache after LayoutStep, and a resident
        // half costs no pread -- the same saving flush_edits already takes. Locked:
        // the async writer thread also touches hotcache.resident/dirty.
        char *cached_v = nullptr, *cached_w = nullptr;
        char *bv = nullptr, *bw = nullptr;
        if (layout_commit_batched) {
            // Staged: the image may already carry earlier swaps from this pass,
            // and its read (if any) was charged on first touch.
            bv = layout_stage_image(sv);
            bw = two ? layout_stage_image(sw) : bv;
        } else {
            {
                std::lock_guard<std::mutex> lk(hotcache.mu);
                cached_v = hotcache.find_locked(sv);
                cached_w = two ? hotcache.find_locked(sw) : cached_v;
            }
            bv = cached_v ? cached_v : swap_buf_a;
            bw = two ? (cached_w ? cached_w : swap_buf_b) : bv;

            auto t0 = Clock::now();
            if (cached_v == nullptr) {
                if (pread_with_delay(write_fd, bv, bytes, sv * diskann::defaults::SECTOR_LEN) !=
                    (ssize_t)bytes)
                    throw std::runtime_error("layout swap: pread failed");
                g_io->device_reads += (long)num_sectors_per_node;
            }
            if (two && cached_w == nullptr) {
                if (pread_with_delay(write_fd, bw, bytes, sw * diskann::defaults::SECTOR_LEN) !=
                    (ssize_t)bytes)
                    throw std::runtime_error("layout swap: pread failed");
                g_io->device_reads += (long)num_sectors_per_node;
            }
            // the read half's synchronous cost, charged either way
            g_io->io_s += secs_since(t0);
        }

        // The modify half: node_offset resolves the *old* slots, which is what we
        // want -- the map is not republished until both writes are away or queued.
        char *pv = node_offset(bv, v);
        char *pw = node_offset(bw, w);
        std::memcpy(swap_rec, pv, max_node_len);
        std::memcpy(pv, pw, max_node_len);
        std::memcpy(pw, swap_rec, max_node_len);

        auto commit_half = [&](char *buf, uint64_t sector, bool resident) {
            if (hotcache.async_writeback && resident) {
                std::lock_guard<std::mutex> lk(hotcache.mu);
                hotcache.mark_dirty_locked(sector); // buf IS the resident copy, already patched
                return;
            }
            auto tw0 = Clock::now();
            if (pwrite_backend(write_fd, buf, bytes, sector * diskann::defaults::SECTOR_LEN) != (ssize_t)bytes)
                throw std::runtime_error("layout swap: pwrite failed");
            charge_write_delay(bytes);
            g_io->io_s += secs_since(tw0);
        };
        if (!layout_commit_batched) {
            commit_half(bv, sv, cached_v != nullptr);
            g_io->device_writes += (long)num_sectors_per_node;
            if (two) {
                commit_half(bw, sw, cached_w != nullptr);
                g_io->device_writes += (long)num_sectors_per_node;
            }
        }
        // Same convention as flush_edits: `writes` counts sectors written --
        // queued counts as written, since the async writer WILL flush it, just off
        // the critical path -- and `read_sectors` absorbs the RMW read half;
        // `reads` counts only fetches issued through the reader.
        if (!layout_commit_batched) {
            const long halves = two ? 2 : 1;
            const long resident_halves = (cached_v ? 1 : 0) + ((two && cached_w) ? 1 : 0);
            g_io->writes += (long)num_sectors_per_node * halves;
            g_io->read_sectors += (long)num_sectors_per_node * (halves - resident_halves);
            g_io->rmw_preads += halves;
            g_io->rmw_preads_resident += resident_halves;
        }

        // Publish. Both old slots are captured first, because the first assign
        // mutates `reloc` and the second still needs v's pre-swap slot. Entries
        // that fall back to the identity are dropped, which is what keeps the
        // map proportional to the relocated set rather than to N.
        const uint32_t old_v = slot_of(v), old_w = slot_of(w);
        auto assign = [&](uint32_t id, uint32_t slot) {
            if (slot == id) { reloc.erase(id); slot_owner.erase(slot); }
            else { reloc[id] = slot; slot_owner[slot] = id; }
        };
        assign(v, old_w);
        assign(w, old_v);
        layout_swaps++;
#ifndef NDEBUG
        // reloc and slot_owner must stay exact inverses.
        assert(owner_of_slot(slot_of(v)) == v && owner_of_slot(slot_of(w)) == w);
#endif
    }

    struct LayoutStageEntry { char *buf; bool owned; bool resident; };
    std::unordered_map<uint64_t, LayoutStageEntry> layout_stage;
    bool layout_commit_batched = false;

    char *layout_stage_image(uint64_t sector) {
        auto it = layout_stage.find(sector);
        if (it != layout_stage.end()) return it->second.buf;
        const uint64_t bytes = num_sectors_per_node * diskann::defaults::SECTOR_LEN;
        char *resident = nullptr;
        {
            std::lock_guard<std::mutex> lk(hotcache.mu);
            resident = hotcache.find_locked(sector);
        }
        if (resident != nullptr) {
            layout_stage.emplace(sector, LayoutStageEntry{resident, false, true});
            g_io->rmw_preads_resident++;
            return resident;
        }
        char *buf = nullptr;
        diskann::alloc_aligned((void **)&buf, bytes, diskann::defaults::SECTOR_LEN);
        auto t0 = Clock::now();
        if (pread_with_delay(write_fd, buf, bytes, sector * diskann::defaults::SECTOR_LEN) !=
            (ssize_t)bytes)
            throw std::runtime_error("layout swap (batched): pread failed");
        g_io->io_s += secs_since(t0);
        g_io->device_reads += (long)num_sectors_per_node;
        g_io->read_sectors += (long)num_sectors_per_node;
        g_io->rmw_preads++;
        layout_stage.emplace(sector, LayoutStageEntry{buf, true, false});
        return buf;
    }

    size_t layout_stage_size() const { return layout_stage.size(); }

    // Called once at the end of each LayoutStep pass. Writes are charged per
    // DISTINCT sector here, which is the whole point of the mode.
    void flush_layout_stage() {
        if (layout_stage.empty()) return;
        const uint64_t bytes = num_sectors_per_node * diskann::defaults::SECTOR_LEN;
        for (auto &kv : layout_stage) {
            LayoutStageEntry &e = kv.second;
            if (e.resident && hotcache.async_writeback) {
                std::lock_guard<std::mutex> lk(hotcache.mu);
                hotcache.mark_dirty_locked(kv.first); // already patched in place
            } else {
                auto t0 = Clock::now();
                if (pwrite_backend(write_fd, e.buf, bytes, kv.first * diskann::defaults::SECTOR_LEN) !=
                    (ssize_t)bytes)
                    throw std::runtime_error("layout swap (batched): pwrite failed");
                charge_write_delay(bytes);
                g_io->io_s += secs_since(t0);
            }
            g_io->device_writes += (long)num_sectors_per_node;
            g_io->writes += (long)num_sectors_per_node;
            if (e.owned) diskann::aligned_free(e.buf);
        }
        layout_stage.clear();
    }

  private:
    // Scratch for swap_slots, allocated once: two sector buffers and one record
    // buffer. Reused rather than allocated per swap -- LayoutStep calls this in a
    // loop, and aligned_alloc per swap would dominate the memcpy it serves.
    char *swap_buf_a = nullptr, *swap_buf_b = nullptr, *swap_rec = nullptr;
    void ensure_swap_buffers(uint64_t bytes) {
        if (swap_buf_a) return;
        diskann::alloc_aligned((void **)&swap_buf_a, bytes, diskann::defaults::SECTOR_LEN);
        diskann::alloc_aligned((void **)&swap_buf_b, bytes, diskann::defaults::SECTOR_LEN);
        swap_rec = new char[max_node_len];
    }

  public:

    using HarvestCB = std::function<void(uint32_t, const float *, const uint32_t *, uint32_t)>;

    void read_nodes(const std::vector<uint32_t> &ids, std::vector<std::vector<float>> *coords,
                    std::vector<std::vector<uint32_t>> *nbrs,
                    const HarvestCB *harvest_cb = nullptr) {
        if (ids.empty()) return;
        const uint64_t bytes_per_node = num_sectors_per_node * diskann::defaults::SECTOR_LEN;
        qbuf.reserve(bytes_per_node);
        hotcache.reserve(bytes_per_node);
        hotcache.begin_call();   // pins this call's working set against eviction

        char *buf = nullptr;
        diskann::alloc_aligned((void **)&buf, ids.size() * bytes_per_node, diskann::defaults::SECTOR_LEN);

        std::vector<char *> block(ids.size(), nullptr);
        std::unordered_map<uint64_t, char *> queued;
        std::vector<AlignedRead> reqs;
        reqs.reserve(ids.size());
        // One representative requested id per entry in `reqs`, same order, so the
        // post-read harvest step can recover which sector (via block_of) each
        // freshly-issued read actually fetched.
        std::vector<uint32_t> fresh_rep_id;
        if (harvest_cb) fresh_rep_id.reserve(ids.size());
        long hits = 0, coalesced = 0, cblk = 0, admits = 0, frozen = 0;
        size_t scratch_slot = 0;
        // c(v) census (eq. (5)): one observation per ID, not per sector, since the
        // unit is an expansion of v. `in_query` rather than `active` so the census
        // still runs with the block buffer off -- that arm has no cross-round dedup
        // and should measure c ~ 1, which is a result and not a reason to stop
        // measuring. It excludes do_rewire's batch fetches, which are not a query
        // trajectory; the trajectories that are one -- the serving search and
        // audit_step's, each opening its own QueryScope -- are included.
        const bool census = g_track_expansion_cost && qbuf.in_query;
        long ids_on_rewired = 0, sector_reqs_on_rewired = 0, uncacheable_forced = 0;
        for (size_t i = 0; i < ids.size(); i++) {
            const uint64_t sec = node_sector(ids[i]);
            const bool on_rewired = !g_row_rewired.empty() && g_row_rewired[ids[i]];
            if (on_rewired) ids_on_rewired++;
            if (census && !g_node_fetch_count_path.empty()) g_node_fetch_count[ids[i]]++;
            // --uncacheable-node-fraction / --uncacheable-node-list: bypass every
            // cache level and never publish the sector into C_blk, `queued` or
            // qbuf. The read lands in per-call scratch, which is sized for
            // ids.size() slots, so it cannot overflow even if every id is
            // uncacheable. Coherence holds because this read goes to the device,
            // which write-through keeps current.
            if (is_uncacheable_node(ids[i])) {
                char *slot = buf + scratch_slot++ * bytes_per_node;
                block[i] = slot;
                uncacheable_forced++;
                if (census) observe_expansion_cost(ids[i], 1);
                if (collect_epoch_misses) epoch_missed.push_back((uint32_t)ids[i]);
                reqs.emplace_back(sec * diskann::defaults::SECTOR_LEN, bytes_per_node, slot);
                if (on_rewired) sector_reqs_on_rewired++;
                if (harvest_cb) fresh_rep_id.push_back(ids[i]);
                continue;
            }
            // C_blk first: it is the outermost level, and resolving here keeps the
            // four outcomes disjoint. NOT gated on `in_query`: the cache's scope is
            // the whole batch, so do_rewire's fetch_neighbors and its reciprocal-
            // endpoint reads both hit what the batch's queries already paid for.
            // That is half of what the arm measures -- the touched-set re-reads are
            // exactly the traffic a cross-query cache should remove.
            if (char *hot = hotcache.find(sec)) {
                block[i] = hot; cblk++;
                if (census) observe_expansion_cost(ids[i], 0); // resident in C_blk
                continue;
            }
            auto qit = queued.find(sec);
            if (qit != queued.end()) {
                block[i] = qit->second; coalesced++;
                if (census) observe_expansion_cost(ids[i], 0); // same batch, already claimed
                continue;
            }
            if (char *carried = qbuf.find(sec)) {
                block[i] = carried; hits++;
                if (census) observe_expansion_cost(ids[i], 0); // an earlier round of this query
                continue;
            }
            // Read straight into a C_blk slot when the cache is on, so admission is
            // the read itself and costs nothing extra; otherwise into the query
            // buffer, and failing that per-call scratch. With the cache on the
            // qbuf layer is subsumed -- anything it held, C_blk holds too, and
            // C_blk is consulted first -- so qbuf_hits must read 0. That identity
            // is asserted in the end-of-run QBUF line.
            char *slot = nullptr;
            // claim() returns nullptr only when a cap froze the cache; an admission
            // that did not happen must not be counted as one, or the frozen tail of
            // the run would report a collapsing hit rate against admissions it never
            // made. Uncapped, claim() never returns nullptr and this is the old line.
            if (hotcache.enabled) { slot = hotcache.claim(sec); if (slot != nullptr) admits++; else frozen++; }
            if (slot == nullptr) slot = qbuf.claim(sec);
            if (slot == nullptr) slot = buf + scratch_slot++ * bytes_per_node;
            queued.emplace(sec, slot);
            block[i] = slot;
            if (census) observe_expansion_cost(ids[i], 1);     // paid a fetch
            if (collect_epoch_misses) epoch_missed.push_back((uint32_t)ids[i]);
            reqs.emplace_back(sec * diskann::defaults::SECTOR_LEN, bytes_per_node, slot);
            if (on_rewired) sector_reqs_on_rewired++;
            if (harvest_cb) fresh_rep_id.push_back(ids[i]);
        }

        if (!reqs.empty()) {
            auto t0 = Clock::now();
            if (g_in_memory) {
                for (const AlignedRead &r : reqs) std::memcpy(r.buf, ram_at(r.offset), r.len);
            } else {
            reader->read(reqs, reader->get_ctx());
            charge_read_delay((long)reqs.size());
            }
            g_io->io_s += secs_since(t0);
            g_io->device_reads += (long)reqs.size() * (long)num_sectors_per_node;
        }
        // MERGE NOTE. This branch's pre-merge read_nodes charged `reads` the
        // deduped slot count; cpp_main charges it the ids requested. We keep
        // cpp_main's meaning, so across the two lines of work:
        //   `reads`       == exp 2's "traversal reads" (layout-invariant, ids asked for)
        //   `sector_reqs` == exp 3's "traversal reads" (what the device saw)
        // Read old logs against the matching column, not against `reads` alone.
        g_io->reads += (long)ids.size();
        g_io->read_sectors += (long)(ids.size() * num_sectors_per_node);
        g_io->sector_reqs += (long)reqs.size();
        g_io->qbuf_hits += hits;
        g_io->call_coalesced += coalesced;
        g_io->cblk_hits += cblk;
        g_io->cblk_admits += admits;
        g_io->cblk_frozen += frozen;
        g_io->ids_on_rewired += ids_on_rewired;
        g_io->sector_reqs_on_rewired += sector_reqs_on_rewired;
        g_io->uncacheable_forced_reads += uncacheable_forced;
        // g_coalesced_reads kept in step with its pre-merge definition (same-call
        // dedup only), so the existing I_PI report line stays meaningful.
        g_coalesced_reads += coalesced;

        if (coords) coords->resize(ids.size());
        if (nbrs) nbrs->resize(ids.size());
        for (size_t i = 0; i < ids.size(); i++) {
            char *node_buf = node_offset(block[i], ids[i]);
            if (coords) {
                (*coords)[i].resize(data_dim);
                decode_coords_to_float(node_buf, data_dim, (*coords)[i].data());
            }
            if (nbrs) {
                // An edited-but-not-yet-committed row wins over what is on disk.
                auto it = adj_edit.find(ids[i]);
                if (it != adj_edit.end()) { (*nbrs)[i] = it->second; continue; }
                const uint32_t *nh = (const uint32_t *)(node_buf + disk_bytes_per_point);
                (*nbrs)[i].assign(nh + 1, nh + 1 + *nh);
            }
        }

        // Harvest: every sector in `reqs` was paid for by one requested id
        // (`fresh_rep_id[r]`) but physically holds nnodes_per_sector records.
        // Score the OTHER occupants against the query for free, mirroring
        // Starling page_search.cpp:277-306's "compute remaining nodes in the
        // pages that are fetched" step. Skip an occupant that is itself one of
        // this call's requested ids -- it is decoded by the ordinary loop above
        // and must not be handed to the callback twice.
        if (harvest_cb && nnodes_per_sector >= 2) {
            std::vector<uint32_t> occupants;
            // Reused across occupants: harvest_cb is called synchronously and
            // finishes with the buffer before the next iteration overwrites it.
            std::vector<float> decoded_buf(data_dim);
            for (size_t r = 0; r < reqs.size(); r++) {
                char *sector_buf = (char *)reqs[r].buf;
                block_of(fresh_rep_id[r], occupants);
                for (uint32_t occ : occupants) {
                    if (std::find(ids.begin(), ids.end(), occ) != ids.end()) continue;
                    char *node_buf = node_offset(sector_buf, occ);
                    decode_coords_to_float(node_buf, data_dim, decoded_buf.data());
                    const float *fp = decoded_buf.data();
                    auto it = adj_edit.find(occ);
                    if (it != adj_edit.end()) {
                        (*harvest_cb)(occ, fp, it->second.data(), (uint32_t)it->second.size());
                    } else {
                        const uint32_t *nh = (const uint32_t *)(node_buf + disk_bytes_per_point);
                        (*harvest_cb)(occ, fp, nh + 1, *nh);
                    }
                }
            }
        }
        diskann::aligned_free(buf);
    }

    std::unordered_map<uint32_t, std::vector<uint32_t>> fetch_neighbors(const std::vector<int> &ids) {
        std::unordered_map<uint32_t, std::vector<uint32_t>> out;
        std::vector<uint32_t> need;
        for (int id : ids) {
            auto it = adj_edit.find((uint32_t)id);
            if (it != adj_edit.end()) out[(uint32_t)id] = it->second;
            else if (!out.count((uint32_t)id)) need.push_back((uint32_t)id);
        }
        std::sort(need.begin(), need.end());
        need.erase(std::unique(need.begin(), need.end()), need.end());
        // Chunked so a single call never exceeds the reader's I/O depth.
        const size_t kMaxBatch = diskann::defaults::MAX_N_SECTOR_READS;
        for (size_t off = 0; off < need.size(); off += kMaxBatch) {
            std::vector<uint32_t> chunk(need.begin() + off,
                                        need.begin() + std::min(need.size(), off + kMaxBatch));
            std::vector<std::vector<uint32_t>> nb;
            read_nodes(chunk, nullptr, &nb);
            for (size_t i = 0; i < chunk.size(); i++) out[chunk[i]] = std::move(nb[i]);
        }
        return out;
    }

    std::vector<uint32_t> get_neighbors(uint32_t u) {
        auto m = fetch_neighbors({(int)u});
        return m[u];
    }

    std::vector<uint32_t> &edit_row(uint32_t u, const std::vector<uint32_t> &cur) {
        auto it = adj_edit.find(u);
        if (it != adj_edit.end()) return it->second;
        if (track_orig) adj_orig.emplace(u, cur); // first touch: remember the pristine row
        return adj_edit.emplace(u, cur).first->second;
    }

    void add_directed(uint32_t u, uint32_t v, const std::vector<uint32_t> &u_nbrs);
    void add_edge_bidirectional(uint32_t u, uint32_t v, const std::vector<uint32_t> &u_nbrs,
                                const std::vector<uint32_t> &v_nbrs) {
        add_directed(u, v, u_nbrs);
        add_directed(v, u, v_nbrs);
    }
    void remove_edge(uint32_t u, uint32_t v, const std::vector<uint32_t> &u_nbrs) {
        std::vector<uint32_t> &row = edit_row(u, u_nbrs);
        auto it = std::find(row.begin(), row.end(), v);
        if (it == row.end()) return;
        row.erase(it);
    }

    // Commit the overlay to disk, grouped by sector so co-resident nodes cost one
    // read-modify-write between them (a multi-node sector must be re-read before
    // patching, exactly as a real updater would have to).
    void flush_edits() {
        if (!writeback || adj_edit.empty()) return;
        TimeSpan commit_ts(g_tp.rewire_commit);
        const double commit_io_before = g_io->io_s;
        std::unordered_map<uint64_t, std::vector<uint32_t>> by_sector;
        for (auto &kv : adj_edit) by_sector[node_sector(kv.first)].push_back(kv.first);
        g_commit_flush_calls++;
        g_commit_flush_rows += (long)adj_edit.size();
        const uint64_t bytes = num_sectors_per_node * diskann::defaults::SECTOR_LEN;
        // How big the staged overlay actually gets. adj_edit holds one entry per
        // DISTINCT node edited since the last flush -- the node's own turn in the
        // loop plus every time it was some other node's reciprocal -- so its size
        // is what by_sector coalesces down to, and the gap between the two is the
        // whole reason the batched commit is cheap. row_bytes is the payload only
        // (degree * 4), not the map node or vector object overhead.
        if (!g_rewire_commit_direct) {
            size_t edit_bytes = 0;
            for (auto &kv : adj_edit) edit_bytes += kv.second.size() * sizeof(uint32_t);
            size_t u_rows = 0;
            for (auto &kv : adj_edit) if (g_admitted_u_rows.count(kv.first)) u_rows++;
            fprintf(stderr, "  ADJEDIT_SPLIT: dirtied=%zu u_rows=%zu v_rows=%zu admitted_u=%zu\n",
                    adj_edit.size(), u_rows, adj_edit.size() - u_rows, g_admitted_u_rows.size());
            fprintf(stderr, "  ADJEDIT: nodes=%zu sectors=%zu nodes_per_sector=%.3f "
                            "row_bytes=%zu mean_deg=%.2f\n",
                    adj_edit.size(), by_sector.size(),
                    by_sector.empty() ? 0.0 : (double)adj_edit.size() / by_sector.size(),
                    edit_bytes,
                    adj_edit.empty() ? 0.0 : (double)edit_bytes / (4.0 * adj_edit.size()));
        }
        // The one place the staged rows are written into a sector image. Factored
        // out so the overlay commit and the read-modify-write commit cannot drift
        // apart; the bytes it writes are identical in both.
        auto patch_sector = [&](char *sector_buf, const std::vector<uint32_t> &nodes) {
            for (uint32_t u : nodes) {
                char *node_buf = node_offset(sector_buf, u);
                uint32_t *nh = (uint32_t *)(node_buf + disk_bytes_per_point);
                const std::vector<uint32_t> &row = adj_edit[u];
                *nh = (uint32_t)row.size();
                std::memcpy(nh + 1, row.data(), row.size() * sizeof(uint32_t));
            }
        };
        // One read-modify-write commit of a single sector, on the calling thread.
        // This is what the no-cache arm pays for every dirty sector, and what a
        // capped run pays for the dirty sectors the freeze kept out of C_blk.
        auto commit_sector_rmw = [&](char *scratch, uint64_t sec, const std::vector<uint32_t> &nodes) {
            const uint64_t off = sec * diskann::defaults::SECTOR_LEN;
            auto t0 = Clock::now();
            g_io->rmw_preads++;                // what the device actually saw
            g_io->device_reads += (long)num_sectors_per_node;
            if (pread_with_delay(write_fd, scratch, bytes, off) != (ssize_t)bytes)
                throw std::runtime_error("rewire write-back: pread failed");
            patch_sector(scratch, nodes);
            if (pwrite_backend(write_fd, scratch, bytes, off) != (ssize_t)bytes)
                throw std::runtime_error("rewire write-back: pwrite failed");
            charge_write_delay(bytes);
            g_io->io_s += secs_since(t0);
            g_io->device_writes += (long)num_sectors_per_node;
            g_io->writes += (long)num_sectors_per_node;
            g_io->read_sectors += (long)num_sectors_per_node; // the RMW read half
        };
        if (hotcache.async_writeback) {
            // Uncapped, every dirty sector is resident by construction: a row can
            // only be edited if do_rewire read it, and that read was admitted. A cap
            // breaks exactly that -- past the freeze the read is not admitted -- so
            // those sectors take the ordinary RMW commit instead of the overlay.
            // sync_buf is allocated on first need, so an uncapped run never has one.
            char *sync_buf = nullptr;
            for (auto &kv : by_sector) {
                const uint64_t sec = kv.first;
                bool served_by_overlay = false;
                {
                    std::lock_guard<std::mutex> lk(hotcache.mu);
                    char *sector_buf = hotcache.find_locked(sec);
                    if (sector_buf != nullptr) {
                        patch_sector(sector_buf, kv.second);
                        hotcache.mark_dirty_locked(sec);
                        served_by_overlay = true;
                    }
                }
                if (served_by_overlay) {
                    g_io->rmw_preads_resident++;
                    g_io->device_writes += (long)num_sectors_per_node;
                    g_io->writes += (long)num_sectors_per_node;
                    continue;
                }
                if (hotcache.cap_sectors == 0)
                    throw std::runtime_error("async rewire commit needs dirty sectors resident in C_blk");
                if (sync_buf == nullptr)
                    diskann::alloc_aligned((void **)&sync_buf, bytes, diskann::defaults::SECTOR_LEN);
                commit_sector_rmw(sync_buf, sec, kv.second);
            }
            if (sync_buf != nullptr) diskann::aligned_free(sync_buf);
            adj_edit.clear();
            g_tp.rewire_commit_io += g_io->io_s - commit_io_before;
            return;
        }
        char *buf = nullptr;
        diskann::alloc_aligned((void **)&buf, bytes, diskann::defaults::SECTOR_LEN);
        for (auto &kv : by_sector) {
            uint64_t off = kv.first * diskann::defaults::SECTOR_LEN;
            // Write-through: when C_blk is holding this sector, its bytes are the
            // current on-disk bytes (flush_edits is the only writer and visits each
            // sector once), so they can be patched and written straight back and
            // the pread is pure waste. Patching also leaves the resident copy
            // correct, which is the entire reason the drop can be relaxed.
            char *sector_buf = nullptr;
            char *resident = hotcache.find(kv.first);
            if (resident != nullptr) g_io->rmw_preads_resident++;
            auto t0 = Clock::now();
            if (resident != nullptr && hotcache.writethrough) {
                sector_buf = resident;             // no pread: already have it
            } else {
                sector_buf = buf;
                g_io->rmw_preads++;                // what the device actually saw
                g_io->device_reads += (long)num_sectors_per_node;
                if (pread_with_delay(write_fd, sector_buf, bytes, off) != (ssize_t)bytes)
                    throw std::runtime_error("rewire write-back: pread failed");
            }
            patch_sector(sector_buf, kv.second);
            if (pwrite_backend(write_fd, sector_buf, bytes, off) != (ssize_t)bytes)
                throw std::runtime_error("rewire write-back: pwrite failed");
            charge_write_delay(bytes);
            g_io->device_writes += (long)num_sectors_per_node;
            // Without write-through the resident copy now holds PRE-write bytes and
            // is stale until the drop. That is why lifetime 0 is refused unless
            // write-through is on.
            (void)resident;
            g_io->io_s += secs_since(t0);
            g_io->writes += (long)num_sectors_per_node;
            g_io->read_sectors += (long)num_sectors_per_node; // the RMW read half
        }
        diskann::aligned_free(buf);
        adj_edit.clear(); // committed: subsequent reads of these nodes go to disk again
        g_tp.rewire_commit_io += g_io->io_s - commit_io_before;
    }

    // Undo every edit since the last restore, from the pristine rows captured on
    // first touch -- O(touched) like VamanaGraph::restore_dirty, not O(N).
    void restore_dirty() {
        if (adj_orig.empty()) return;
        hotcache.wait_idle();
        adj_edit = adj_orig;      // stage the originals ...
        flush_edits();            // ... and commit them back (or keep them in RAM)
        if (!writeback) adj_edit.clear();
        adj_orig.clear();
    }

    struct PQQuery {
        const DiskANNIndex *idx = nullptr;
        std::vector<float> chunk_dists; // 256 * n_chunks
        float dist(uint32_t id) const {
            const uint8_t *code = idx->pq_codes.data() + (size_t)id * idx->n_chunks;
            float acc = 0.0f;
            for (uint64_t ch = 0; ch < idx->n_chunks; ch++) acc += chunk_dists[256 * ch + code[ch]];
            return acc;
        }
    };
    // MIPS->L2 query transform, byte-identical to PQFlashIndex::cached_beam_search:
    // L2-normalise the native dims, then append a 0 as the extra coordinate. The
    // resulting augmented query ranks by L2 exactly as the original ranks by inner
    // product. For an L2 index this is a plain copy.
    std::vector<float> augment_query(const float *q) const {
        std::vector<float> aug((size_t)data_dim, 0.0f);
        if (!mips) { std::copy(q, q + data_dim, aug.begin()); return aug; }
        const uint64_t native = data_dim - 1;
        double norm_sq = 0.0;
        for (uint64_t i = 0; i < native; i++) norm_sq += (double)q[i] * (double)q[i];
        const double qnorm = std::sqrt(norm_sq);
        const double inv = (qnorm > 0.0) ? 1.0 / qnorm : 0.0;
        for (uint64_t i = 0; i < native; i++) aug[i] = (float)(q[i] * inv);
        aug[native] = 0.0f; // the coordinate that absorbs the base-norm slack
        return aug;
    }
    // prepare_query takes an already-augmented data_dim vector; node_query feeds it
    // a reconstructed base vector, which is already in the augmented space. The
    // native-space entry point prepare_external_query went with the Q-ranked walk
    // that was its only caller -- search() augments inline (see augment_query above).
    PQQuery prepare_query(const float *q) const {
        PQQuery pq;
        pq.idx = this;
        std::vector<float> rotated(q, q + data_dim);
        const_cast<diskann::FixedChunkPQTable &>(pq_table).preprocess_query(rotated.data());
        pq.chunk_dists.resize(256 * n_chunks);
        const_cast<diskann::FixedChunkPQTable &>(pq_table)
            .populate_chunk_distances(rotated.data(), pq.chunk_dists.data());
        return pq;
    }
    // Node-as-query: reconstruct u from its PQ code, then treat it like a query.
    // Replaces VamanaGraph::dist_uv (exact L2 between two base vectors); reading
    // u's true vector would cost a disk fetch per ranked node, and its neighbours'
    // true vectors one fetch each on top of that.
    PQQuery node_query(uint32_t u) const {
        std::vector<float> recon(data_dim);
        const_cast<diskann::FixedChunkPQTable &>(pq_table)
            .inflate_vector(const_cast<uint8_t *>(pq_codes.data() + (size_t)u * n_chunks), recon.data());
        return prepare_query(recon.data());
    }

    float codeword_pair_distance(uint32_t u, uint32_t v) const {
        const uint8_t *u_code = pq_codes.data() + (size_t)u * n_chunks;
        const uint8_t *v_code = pq_codes.data() + (size_t)v * n_chunks;
        float acc = 0.0f;
        for (uint64_t chunk = 0; chunk < n_chunks; chunk++)
            acc += pq_codeword_pair_dists[((size_t)chunk * 256 + u_code[chunk]) * 256 + v_code[chunk]];
        return acc;
    }

    static float exact_l2(const float *a, const float *b, uint64_t dim) {
        float acc = 0.0f;
        for (uint64_t i = 0; i < dim; i++) { float d = a[i] - b[i]; acc += d * d; }
        return acc;
    }

    void search(const float *query, uint64_t K, uint64_t ef, uint64_t beam_width,
                std::vector<int> &result_ids, std::vector<int> &visited_order,
                std::vector<int> *expanded_order = nullptr,
                std::vector<std::vector<uint32_t>> *expanded_nbrs = nullptr) {
        // Everything this search reads is one trajectory for eq. (1) purposes.
        QueryScope qs(qbuf);
        // One augmentation per search: reused by both the PQ table and the exact
        // re-rank, which must compare against the same augmented vector the on-disk
        // full-precision coordinates live in.
        std::vector<float> aug = augment_query(query);
        const float *query_aug = aug.data();
        PQQuery pq = prepare_query(query_aug);
        diskann::NeighborPriorityQueue retset((size_t)ef);
        std::vector<diskann::Neighbor> full_retset;
        std::unordered_set<uint32_t> visited;
        visited_order.clear();
        if (expanded_nbrs) expanded_nbrs->clear();

        uint32_t default_entry_point = medoid;
        if (medoid_candidates.size() > 1) {
            float best_dist = pq.dist(medoid_candidates[0]);
            default_entry_point = medoid_candidates[0];
            for (size_t mi = 1; mi < medoid_candidates.size(); mi++) {
                const float d = pq.dist(medoid_candidates[mi]);
                if (d < best_dist) { best_dist = d; default_entry_point = medoid_candidates[mi]; }
            }
        }
        const uint32_t entry_point =
            (g_search_ep_override >= 0 && (uint64_t)g_search_ep_override < num_points)
                ? (uint32_t)g_search_ep_override
                : default_entry_point;
        retset.insert(diskann::Neighbor(entry_point, pq.dist(entry_point)));
        visited.insert(entry_point);
        visited_order.push_back((int)entry_point);

        std::vector<uint32_t> frontier;
        std::vector<std::vector<float>> fcoords;
        std::vector<std::vector<uint32_t>> fnbrs;

        struct HarvestCand { uint32_t id; float dist; std::vector<uint32_t> nbrs; };
        std::vector<HarvestCand> harvested;
        HarvestCB harvest_cb = [&](uint32_t id, const float *coords, const uint32_t *nb, uint32_t nn) {
            if (!visited.insert(id).second) return; // already seen this query
            visited_order.push_back((int)id);
            float d = exact_l2(query_aug, coords, data_dim);
            full_retset.emplace_back(id, d);
            harvested.push_back({id, d, std::vector<uint32_t>(nb, nb + nn)});
        };

        while (retset.has_unexpanded_node()) {
            frontier.clear();
            while (retset.has_unexpanded_node() && frontier.size() < beam_width)
                frontier.push_back(retset.closest_unexpanded().id);
            if (expanded_order)
                for (uint32_t id : frontier) expanded_order->push_back((int)id);
            harvested.clear();
            read_nodes(frontier, &fcoords, &fnbrs, g_harvest_cosector ? &harvest_cb : nullptr);
            // Appended after the read, in lockstep with expanded_order above, so
            // the two stay index-aligned across beam rounds. Copied rather than
            // moved: fnbrs is reused by the next round's read_nodes.
            if (expanded_nbrs)
                for (size_t i = 0; i < frontier.size(); i++) expanded_nbrs->push_back(fnbrs[i]);
            for (size_t i = 0; i < frontier.size(); i++) {
                full_retset.emplace_back(frontier[i], exact_l2(query_aug, fcoords[i].data(), data_dim));
                for (uint32_t v : fnbrs[i]) {
                    if (!visited.insert(v).second) continue;
                    visited_order.push_back((int)v);
                    retset.insert(diskann::Neighbor(v, pq.dist(v)));
                }
            }
            if (!harvested.empty()) {
                std::sort(harvested.begin(), harvested.end(),
                          [](const HarvestCand &a, const HarvestCand &b) { return a.dist < b.dist; });
                size_t take = (size_t)std::llround(g_harvest_ratio * (double)harvested.size());
                take = std::min(take, harvested.size());
                for (size_t hi = 0; hi < take; hi++) {
                    for (uint32_t v : harvested[hi].nbrs) {
                        if (!visited.insert(v).second) continue;
                        visited_order.push_back((int)v);
                        retset.insert(diskann::Neighbor(v, pq.dist(v)));
                    }
                }
            }
        }
        std::sort(full_retset.begin(), full_retset.end());
        result_ids.clear();
        for (size_t i = 0; i < full_retset.size() && i < K; i++)
            result_ids.push_back((int)full_retset[i].id);
    }
};

void DiskANNIndex::add_directed(uint32_t u, uint32_t v, const std::vector<uint32_t> &u_nbrs) {
    if (u == v) return;
    std::vector<uint32_t> &row = edit_row(u, u_nbrs);
    if (std::find(row.begin(), row.end(), v) != row.end()) return; // already an edge
    if (row.size() < max_degree) { row.push_back(v); return; }
    // At the cap: re-rank all d+1 candidates by (PQ) distance to u and keep the
    // closest R -- v itself may be the one dropped.
    PQQuery uq = node_query(u);
    std::vector<std::pair<float, uint32_t>> cands;
    cands.reserve(row.size() + 1);
    for (uint32_t w : row) cands.emplace_back(uq.dist(w), w);
    cands.emplace_back(uq.dist(v), v);
    std::sort(cands.begin(), cands.end());
    cands.resize(max_degree);
    row.clear();
    for (auto &pr : cands) row.push_back(pr.second);
}

// Length-tiered edge protection + reachability audits + governor (exp_tr.tex
// section 4.4 + section 9, level-only mechanism: dynamic live-longest-edge
// ranking, not the paper's fixed-at-build Definition 4.3; see VIOLATIONS.md).
static bool g_edge_protection = false;  // --edge-protection: turn the whole mechanism on (no paper symbol)
static bool g_dump_protect_mask = false;
static std::unordered_map<int, std::vector<uint32_t>> g_prev_mask;   // node -> last mask
static std::unordered_map<int, std::unordered_set<uint32_t>> g_orig_row; // node -> row at first rewire
static long g_pm_rewires = 0, g_pm_pinned = 0, g_pm_bite = 0, g_pm_fresh = 0;
static double g_pm_churn_sum = 0.0, g_pm_len_ratio_sum = 0.0;
static long g_pm_churn_n = 0, g_pm_len_n = 0;

static int g_k_p_init = 4;              // --protect-k-p-init: k_p^(0), the cold-start level
static bool g_protect_schedule = false;
static long g_protect_schedule_every_queries = 200;
static long g_protect_schedule_every_batches = 0;
static int g_protect_schedule_step = 2;
static int g_protect_schedule_min = 0;
// Counts governor_step() calls when --protect-schedule-every-batches > 0, so
// the ladder tracks do_rewire cycles directly instead of a query count that
// drifts if the last batch is short.
static long g_protect_schedule_batches_done = 0;
// --protect-fix-k-p: freeze k_p at --protect-k-p-init for the whole run,
// overriding any adaptive/schedule flag also passed. Window and repair-volume
// bookkeeping still reset per batch so the miss-rate logging stays meaningful.
// Isolates audit-triggered restoration from the level schedule.
static bool g_fix_k_p = false;
// --protect-audit-passive: audits still fire and still classify
// attributable/unattributable -- so the governor's signal is untouched -- but
// the attributable branch's B_buf touch is suppressed, so no repair ever
// reaches the graph. The other half of the same ablation.
static bool g_audit_passive = false;
static int g_K_max = 0;                 // --protect-k-p-max: K_max, record depth; 0 => R/2 from max_degree at load
static double g_p_a = 0.01;             // --protect-audit-rate: p_a, audits per served query
static double g_epsilon_r = 0.01;       // --protect-slo: eps_r, the target attributable-miss rate
static double g_varsigma = 0.003;       // --protect-slack: varsigma, the margin a demotion must clear eps_r by
static long g_T_dwell = 2000;           // --protect-dwell: T_dwell, queries required between level changes
static double g_alpha = 1.2;            // --protect-alpha: alpha, the audit's progress test (Alg. 5 line 6).
                                        // The skeleton arm's RobustPrune takes the same paper symbol through
                                        // its own knob, g_skeleton_alpha.
static double g_m_pq = 0.0;             // --protect-pq-margin: m_pq in Algorithm 5's reserve progress test
static long g_chi_rep = 1;              // --protect-chi-rep: chi_rep, repair volume per window that force-promotes
static bool g_promote_on_lcb = true;
static bool g_gov_window_accumulate = false;
static bool g_gov_symmetric = false;
static double g_gov_mr_beta = 0.34;    // --protect-mr-beta: EWMA rate for the per-batch miss rate
static bool g_gov_observe = false;
static bool g_gov_anneal = false;
static double g_gov_anneal_tau = 0.5;  // --protect-anneal-tau
static int g_gov_anneal_step = 2;      // --protect-anneal-step
static int g_gov_anneal_min = 0;       // --protect-anneal-min: hard floor the anneal never goes below
static long g_gov_anneal_linear_n = 0;
static double g_gov_anneal_deleted_frac = 0.0;
static long g_protect_warmup_freeze = 0;
static bool g_protect_auto_slo = false;
static double g_slo_margin = 0.02;
static long g_slo_warmup_n = 0;
static bool g_slo_frozen = false;          // eps_r fixed, warm-up over
static long g_slo_warm_attrib = 0;         // warm-up accumulator: attributable trials
static long g_slo_warm_total = 0;          // warm-up accumulator: all trials
// --protect-auto-slo / --protect-slo-warmup reachability check: a governor whose
// eps_r sits above every UCB the window ever produces prints exactly the same
// hold/demote lines as one with nothing to act on. Track the largest UCB seen
// and say so once, after enough batches that the absence is real.
static double g_slo_max_ucb_seen = 0.0;
static long g_slo_batches_seen = 0;
static bool g_slo_unreachable_warned = false;
// --protect-demote-n: the window size demotion requires. Was hard-coded to 30.
// Note this is NOT the binding constraint -- with a clean window the UCB test
// needs n >= ln(1/delta)/(eps_r - varsigma) = 428 regardless of this gate.
static long g_demote_n = 30;
static double g_delta = 0.05;           // --protect-delta: delta, the confidence level the miss-rate bound holds at
static int g_audit_ef = 0;              // --protect-audit-ef: 0 => inherit serving ef
static unsigned long long g_floor_seed = 0xB0B0ull;

static double g_delta_red = 0.10;
static double g_delta_sw = 0.05;
static int g_candlog_ef = 0;
// Maximum number of qualifying retained-log candidates recruited for one
// unattributable node-target diagnostic. One preserves the paper-style default.
static int g_log_rescue_max = 1;
// --protect-log-rescue-all-path: repair every qualifying node on y's search
// path (up to g_log_rescue_max edges each) instead of only the first one.
static bool g_log_rescue_all_path = false;
// --protect-pin-repeat: Algorithm 5 lines 9-10's "pin on repeat" -- how many
// times the SAME (u, candidate) pair must be recruited from the log before it
// is pinned into S(u) unconditionally (Definition 4.1's S = P_kp(u) union pin).
static int g_pin_repeat = 2;

static std::vector<uint32_t> live_rank_of(DiskANNIndex &g, uint32_t u, const std::vector<uint32_t> &live_neighbors,
                                          const DiskANNIndex::PQQuery *uq_in = nullptr) {
    // `uq_in`: a PQQuery for this same u that the caller already built. node_query
    // is a pure function of pq_codes and pq_table, neither of which a rewire pass
    // mutates, so a caller-supplied one is the same object this would construct.
    DiskANNIndex::PQQuery uq_local;
    if (!uq_in) { uq_local = g.node_query(u); uq_in = &uq_local; }
    // Distances are resolved once each instead of once per comparison (2*d*log d
    // calls for d = |row|, every one of them a random gather into the 0.32 GB PQ
    // code array). The comparator sees the identical sequence of outcomes, so the
    // permutation std::sort produces is unchanged.
    std::vector<std::pair<float, uint32_t>> keyed;
    keyed.reserve(live_neighbors.size());
    for (uint32_t v : live_neighbors) keyed.emplace_back(uq_in->dist(v), v);
    std::sort(keyed.begin(), keyed.end(),
              [](const std::pair<float, uint32_t> &a, const std::pair<float, uint32_t> &b) {
                  return a.first > b.first;
              });
    std::vector<uint32_t> rank;
    rank.reserve(keyed.size());
    for (auto &kv : keyed) rank.push_back(kv.second);
    return rank;
}

struct AuditWindow {
    long attributable = 0, unattributable = 0, pass = 0;
    long red_redundant = 0, red_total = 0, sw_sole = 0, sw_total = 0;
    long total() const { return attributable + unattributable + pass; }
    double miss_rate() const { long n = total(); return n == 0 ? 0.0 : (double)attributable / n; }
    double half_width() const {
        long n = total();
        if (n == 0) return 1.0;                                  // nothing observed at all
        if (attributable == 0) return std::log(1.0 / g_delta) / (double)n;
        if (n < 30) return 1.0;                                  // Wald is not usable this small
        double p = miss_rate();
        return 1.96 * std::sqrt(p * (1.0 - p) / (double)n);
    }
    double ucb() const { return miss_rate() + half_width(); }
    // The paper's promote-side bound: the miss rate must be confidently ABOVE
    // eps_r, not merely consistent with it. A window with no observed misses has
    // miss_rate 0, so lcb is 0 and cannot promote at any n -- which is the whole
    // point, since ln(1/delta)/n keeps the UCB above a 0.01 SLO until n >= 300.
    double lcb() const { return std::max(0.0, miss_rate() - half_width()); }
    double redundancy() const { return red_total == 0 ? 0.0 : (double)red_redundant / (double)red_total; }
    double sole_witness_rate() const { return sw_total == 0 ? 1.0 : (double)sw_sole / (double)sw_total; }
    void clear() {
        attributable = unattributable = pass = 0;
        red_redundant = red_total = sw_sole = sw_total = 0;
    }
};

struct Governor {
    int k_p = 0;                  // the level, Algorithm 6's governed quantity
    int K_max = 0;                // record depth, the ceiling a promotion may reach
    int k_ps = 0;                 // k_ps, the last-safe level a promotion may never fall below
    long dwell = 0;               // queries served since the last level change
    long promotions = 0, demotions = 0;
    AuditWindow window;           // Algorithm 6's window statistics
    double mr_ewma = -1.0;        // --protect-symmetric: EWMA of per-batch attributable-miss rate (-1 = unseen)
    long audits_total = 0;        // --protect-symmetric: cumulative audit count, gates the first move
    int anneal_floor = 0;         // --protect-anneal: sticky floor, raised whenever demonstrated harm forces a step up
};
static Governor g_governor;
// Repairs since the last governor_step: an attributable miss recruits its
// rescuing edge into B_buf immediately (targeted repair, independent of
// whether the level moves), and a nonzero count here also force-promotes --
// the fast half of the paper's asymmetric raise-fast/lower-slow design.
static long g_repair_volume = 0;
// Set at the end of do_rewire, read by governor_step's --protect-anneal branch:
// how many of this batch's hot rows the learner actually reordered. The ratio
// rows_changed/hot_nodes is the online settling signal of Theorem 5.3 (writes
// decay under a stationary workload).
static long g_last_rows_changed = 0;
static long g_last_hot_nodes = 0;
static long g_last_deleted = 0;
static long g_first_batch_deleted = 0;    // batch-1 value, the denominator

static std::string g_audit_dump_csv;
static FILE *g_audit_dump_f = nullptr;
static long g_audit_dump_gqc = 0;   // ordinal audit-trial counter, this run only
static long g_audit_pass = 0;       // search reached y
static long g_audit_degenerate = 0; // nothing expanded at all
static long g_audit_nobuf = 0;      // search failed, u not in B_buf
static long g_audit_norescue = 0;   // B_buf[u] exists, no candidate clears alpha
static long g_audit_attrib = 0;     // attributable: a rescuer cleared alpha

// Comparison arms: longest/shortest-fraction protection + navigability
// skeleton (PageWire Algorithm 2), ported from
// cpp_port/control_vamana_pctprotect.cpp. Fixed rules, no governor/audit loop;
// no reciprocal-side-eviction guard (same protection-blind overflow prune as
// --edge-protection, to isolate the selection rule across arms).
static bool g_pct_protect = false;
static bool g_pct_protect_longest = true; // true: protect longest; false: protect shortest
static double g_pct_protect_frac = 0.30;

static bool g_skeleton_protect = false;
static double g_skeleton_alpha = 1.2;
static int g_skeleton_rp = -1;              // <=0 -> R/2 (paper's typical split)
static int g_skeleton_profile_queries = -1; // <=0 -> reuse n_queries
static IOCounters g_io_skeleton;            // one-time extraction cost, kept out of traversal/rewire

static std::unordered_map<uint32_t, std::unordered_set<uint32_t>> g_skel_members;

static std::vector<uint32_t> robust_prune_pq(DiskANNIndex &g, uint32_t u,
                                             const std::unordered_set<uint32_t> &C, double alpha, int stop_after) {
    struct Cand { uint32_t id; float d2; bool alive; };
    std::vector<Cand> pool;
    pool.reserve(C.size());
    DiskANNIndex::PQQuery uq = g.node_query(u);
    for (uint32_t id : C) if (id != u) pool.push_back({id, uq.dist(id), true});
    double alpha2 = alpha * alpha;
    std::vector<uint32_t> result;
    int remaining = (int)pool.size();
    while (remaining > 0) {
        if (stop_after > 0 && (int)result.size() > stop_after) break;
        int best = -1; float bestd2 = 0;
        for (int i = 0; i < (int)pool.size(); i++) {
            if (!pool[i].alive) continue;
            if (best < 0 || pool[i].d2 < bestd2) { best = i; bestd2 = pool[i].d2; }
        }
        pool[best].alive = false; remaining--;
        result.push_back(pool[best].id);
        DiskANNIndex::PQQuery bq = g.node_query(pool[best].id);
        for (auto &cnd : pool) {
            if (!cnd.alive) continue;
            float d2_pstar_c = bq.dist(cnd.id);
            if (alpha2 * d2_pstar_c <= cnd.d2) { cnd.alive = false; remaining--; }
        }
    }
    return result;
}

static void extract_skeleton(DiskANNIndex &g, const float *qv, int n_profile, int K, int ef,
                             uint64_t disk_beam_width) {
    IOPhase ph(g_io_skeleton);
    int R = (int)g.max_degree;
    int Rp = g_skeleton_rp > 0 ? g_skeleton_rp : std::max(1, R / 2);
    const size_t CAP_C = 50;
    std::unordered_map<uint32_t, std::unordered_set<uint32_t>> covisit;
    std::vector<int> result_ids, visited;
    for (int qi = 0; qi < n_profile; qi++) {
        g.search(qv + (size_t)qi * g_query_dim, (uint64_t)K, (uint64_t)ef, disk_beam_width, result_ids, visited);
        std::vector<int> V_t;
        { std::unordered_set<int> seen; for (int x : visited) if (seen.insert(x).second) V_t.push_back(x); }
        for (int u : V_t) {
            auto &s = covisit[(uint32_t)u];
            if (s.size() >= CAP_C) continue;
            for (int w : V_t) {
                if (w == u) continue;
                if (s.size() >= CAP_C) break;
                s.insert((uint32_t)w);
            }
        }
    }
    fprintf(stderr, "  [skeleton] profiled %d queries: %zu distinct nodes visited (reads=%ld io_s=%.1f)\n",
            n_profile, covisit.size(), g_io_skeleton.reads, g_io_skeleton.io_s);

    long n_full_protect = 0, n_partial = 0;
    long total_c = 0, total_winners = 0, total_frozen = 0;
    long processed = 0;
    auto t_prune0 = Clock::now();
    for (auto &kv : covisit) {
        uint32_t u = kv.first;
        auto &C = kv.second;
        if (C.empty()) continue;
        std::vector<uint32_t> row = g.get_neighbors(u);
        for (uint32_t v : row) C.insert(v);
        total_c += (long)C.size();
        std::vector<uint32_t> winners = robust_prune_pq(g, u, C, g_skeleton_alpha, Rp);
        total_winners += (long)winners.size();
        processed++;
        if (processed % 20000 == 0)
            fprintf(stderr, "  [skeleton] robust_prune: %ld/%zu nodes (%.1fs elapsed)\n",
                    processed, covisit.size(), secs_since(t_prune0));
        std::unordered_set<uint32_t> &frozen = g_skel_members[u];
        if ((int)winners.size() > Rp) {
            for (uint32_t v : row) frozen.insert(v); // fallback: freeze the whole pristine row
            n_full_protect++;
            total_frozen += (long)row.size();
        } else {
            std::unordered_set<uint32_t> winset(winners.begin(), winners.end());
            long nf = 0;
            for (uint32_t v : row) if (winset.count(v)) { frozen.insert(v); nf++; }
            n_partial++;
            total_frozen += nf;
        }
    }
    fprintf(stderr, "  [skeleton] nodes: %zu covisited, %ld full-protect-fallback (winners>Rp=%d), "
                    "%ld partial; mean|C|=%.1f mean winners=%.1f mean frozen-edges=%.1f "
                    "(prune %.1fs, extraction reads=%ld io_s=%.1f)\n",
            covisit.size(), n_full_protect, Rp, n_partial,
            covisit.empty() ? 0.0 : (double)total_c / covisit.size(),
            covisit.empty() ? 0.0 : (double)total_winners / covisit.size(),
            covisit.empty() ? 0.0 : (double)total_frozen / covisit.size(),
            secs_since(t_prune0), g_io_skeleton.reads, g_io_skeleton.io_s);
}

// ── Q-learning state (identical to control_vamana.cpp) ──
// Q[u][v]: learned value and visit count of taking edge u->v while chasing a
// query. The count is used both as the UCB denominator and as the
// incremental-mean step size 1/(1+c) in the Q-update.
struct EdgeStats { double q; long c; };

struct DetectorCalibration {
    enum Event { NONE, ADAPT_STOP, DRIFT_START };
    FILE *file = nullptr;
    int window_q = 100;
    long freeze_q = -1;
    bool control = false;
    bool adapting = true;
    double stop_threshold = 0.21;
    int stop_windows = 3;
    long min_adapt_q = 500;
    double drift_threshold = 0.70;
    long epoch_start_q = 0;
    int stable_windows = 0;
    Event boundary_event = NONE;
    double q_delta_sum = 0.0;
    double q_innovation_sum = 0.0;
    long q_updates = 0;
    long buffer_novel = 0;
    long novelty_sample_queries = 0;
    std::unordered_set<uint64_t> signature;
    std::unordered_set<uint64_t> window_edges;

    void reset() {
        q_delta_sum = q_innovation_sum = 0.0;
        q_updates = buffer_novel = novelty_sample_queries = 0;
        signature.clear();
        window_edges.clear();
        adapting = true;
        epoch_start_q = 0;
        stable_windows = 0;
        boundary_event = NONE;
    }

    bool learning() const { return !control || adapting; }

    void q_update(double old_q, double new_q, double reward) {
        if (!file) return;
        q_delta_sum += std::abs(new_q - old_q);
        q_innovation_sum += std::abs(reward - old_q);
        q_updates++;
    }

    void buffer_edge(long q, int u, int v) {
        if (!file) return;
        const uint64_t edge = (uint64_t)(uint32_t)u << 32 | (uint32_t)v;
        if (!window_edges.insert(edge).second) return;
        if ((control && adapting) || (!control && q <= freeze_q)) {
            if (signature.insert(edge).second) buffer_novel++;
        } else if (!signature.count(edge)) {
            buffer_novel++;
        }
    }

    Event finish_query(long q) {
        if (!file || q % window_q != 0) return NONE;
        const double delta = q_updates ? q_delta_sum / q_updates : 0.0;
        const double innovation = q_updates ? q_innovation_sum / q_updates : 0.0;
        const double novelty = window_edges.empty()
            ? 0.0 : (double)buffer_novel / (double)window_edges.size();
        Event event = NONE;
        if (control && adapting) {
            stable_windows = q - epoch_start_q >= min_adapt_q && delta < stop_threshold
                                 ? stable_windows + 1 : 0;
            if (stable_windows >= stop_windows) {
                adapting = false;
                event = ADAPT_STOP;
            }
        } else if (control && novelty > drift_threshold) {
            adapting = true;
            epoch_start_q = q;
            stable_windows = 0;
            event = DRIFT_START;
        }
        const char *mode = control ? (adapting ? "adapting" : "frozen")
                                   : (q <= freeze_q ? "building" : "frozen");
        const char *event_name = event == ADAPT_STOP ? "ADAPT_STOP"
                                : event == DRIFT_START ? "DRIFT_START" : "";
        fprintf(file, "%ld,%s,%s,%d,%ld,%.12g,%.12g,%zu,%ld,%.12g,%zu,%ld\n",
                q, mode, event_name, stable_windows, q_updates, delta, innovation,
                window_edges.size(), buffer_novel, novelty, signature.size(), novelty_sample_queries);
        fflush(file);
        fprintf(stderr,
                "  DETECTORCAL: q=%ld mode=%s event=%s stable=%d q_updates=%ld q_delta=%.9g "
                "q_innovation=%.9g buffer_unique=%zu buffer_novel=%ld "
                "buffer_novel_frac=%.6f signature=%zu novelty_samples=%ld\n",
                q, mode, event_name, stable_windows, q_updates, delta, innovation,
                window_edges.size(), buffer_novel, novelty,
                signature.size(), novelty_sample_queries);
        if (event == ADAPT_STOP)
            fprintf(stderr, "  DETECTOR_EVENT,event=ADAPT_STOP,q=%ld,q_delta=%.9g,threshold=%.9g,windows=%d\n",
                    q, delta, stop_threshold, stop_windows);
        if (event == DRIFT_START)
            fprintf(stderr, "  DETECTOR_EVENT,event=DRIFT_START,q=%ld,unseen_edge_fraction=%.9g,threshold=%.9g\n",
                    q, novelty, drift_threshold);
        q_delta_sum = q_innovation_sum = 0.0;
        q_updates = buffer_novel = novelty_sample_queries = 0;
        window_edges.clear();
        if (event == DRIFT_START) signature.clear();
        boundary_event = event;
        return event;
    }

    Event take_event() {
        const Event event = boundary_event;
        boundary_event = NONE;
        return event;
    }
};
static DetectorCalibration g_detector_calibration;

struct QRow {
    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;   // not a valid node id
    std::vector<uint32_t> keys;
    std::vector<EdgeStats> vals;
    uint32_t slot_mask = 0;
    size_t entry_count = 0;
    void allocate(size_t capacity) {
        keys.assign(capacity, EMPTY_KEY);
        vals.assign(capacity, EdgeStats{0.0, 0L});
        slot_mask = (uint32_t)(capacity - 1);
        entry_count = 0;
    }
    size_t size() const { return entry_count; }
    EdgeStats *find(int key) {
        if (keys.empty()) return nullptr;
        uint32_t k = (uint32_t)key, i = flat_hash_of_node(k) & slot_mask;
        while (keys[i] != EMPTY_KEY) {
            if (keys[i] == k) return &vals[i];
            i = (i + 1) & slot_mask;
        }
        return nullptr;
    }
    // unordered_map::emplace(key, seed).first->second: the existing value if the
    // key is present, otherwise the seed, stored.
    EdgeStats &emplace_or_get(int key, const EdgeStats &seed) {
        if (keys.empty()) allocate(16);
        uint32_t k = (uint32_t)key, i = flat_hash_of_node(k) & slot_mask;
        while (keys[i] != EMPTY_KEY) {
            if (keys[i] == k) return vals[i];
            i = (i + 1) & slot_mask;
        }
        keys[i] = k;
        vals[i] = seed;
        entry_count++;
        // Kept at or below half load: a grow re-finds the key, so the reference
        // handed back is always into the live table.
        if (entry_count * 2 > keys.size()) { grow(); return *find(key); }
        return vals[i];
    }
    void grow() { rebuild(keys.size() * 2, nullptr); }
    // Drop every entry whose key is not in `keep`, in one sweep.
    template <class Set> void retain_in(const Set &keep) {
        if (entry_count == 0) return;
        rebuild(keys.size(), &keep);
    }
  private:
    template <class Set> void rebuild(size_t capacity, const Set *keep) {
        std::vector<uint32_t> old_keys;
        std::vector<EdgeStats> old_vals;
        old_keys.swap(keys);
        old_vals.swap(vals);
        allocate(capacity);
        for (size_t i = 0; i < old_keys.size(); i++) {
            if (old_keys[i] == EMPTY_KEY) continue;
            if (keep && !keep->count((int)old_keys[i])) continue;
            emplace_or_get((int)old_keys[i], old_vals[i]);
        }
    }
    void rebuild(size_t capacity, std::nullptr_t) {
        rebuild<std::unordered_set<int>>(capacity, nullptr);
    }
};
// Qmap used to be the standalone map type (std::unordered_map<int, QRow>);
// QRow now lives nested in HotLearnerCache::Entry instead, so there is no
// map-of-QRow type to alias any more -- only the per-node QRow value survives.

static std::vector<uint8_t> g_row_dirty;
static inline void mark_row_dirty(int u) {
    if (u < 0) return;
    if ((size_t)u >= g_row_dirty.size()) g_row_dirty.resize((size_t)u + 1024, 0);
    g_row_dirty[(size_t)u] = 1;
}
static long g_dirty_batch_index = 0;

static long g_nav_damage_at = -1;           // --nav-damage-at: cumulative query (gqc) to fire at, one-shot
static double g_nav_damage_frac = 0.0;      // --nav-damage-frac: fraction of num_points to sample
static unsigned g_nav_damage_seed = 12345;  // --nav-damage-seed
static bool g_nav_damage_strip_all = false; // --nav-damage-strip-all: remove EVERY evictable edge, not just the longest
static int g_nav_damage_longest = 0;
static bool g_nav_damage_done = false;
static std::string g_nav_damage_dump_csv;   // --nav-damage-dump-csv: one row per removed (u,v) edge
static FILE *g_nav_damage_dump_f = nullptr;
// --protect-repair-known-damage-only is an oracle diagnostic for injected
// damage experiments: accept a repair only when it reinstates an edge this run
// actually deleted. It measures the ceiling of target selection separately
// from detection and must not be used as a deployable policy.
static bool g_protect_repair_known_damage_only = false;
static std::unordered_map<uint32_t, std::unordered_set<uint32_t>> g_nav_damage_removed;
// --nav-damage-target-ids <path>: same longest-N removal as --nav-damage-longest,
// but scoped to a caller-supplied node-ID list (one id per line) instead of the
// whole index -- e.g. the locality a specific query set touches.
static std::vector<int> g_nav_damage_target_ids;
static int g_nav_damage_target_n = 12; // --nav-damage-target-n: N for the target-id list
// --nav-damage-target-frac F: with --nav-damage-target-ids, remove the longest
// round(F * degree) out-edges of each target instead of a fixed N.
static double g_nav_damage_target_frac = 0.0;
// --audit-target-ids <path>: forces audit_step's target y onto this list
// (round-robin by trial count) instead of drawing uniformly over the whole
// graph -- lets damage placed at a small locality actually get exercised by
// audit trials instead of relying on a ~(|list|/num_points) natural hit rate.
static std::vector<uint32_t> g_audit_target_ids;
static uint64_t g_audit_target_ids_trial_count = 0;
static int g_nav_damage_prescan_ef = 0;
static bool g_nav_damage_prescan_use_side_eval_a = false;
static bool g_nav_damage_prescan_force_audit_target = false;
// --nav-damage-prescan-target-cap N: retain only the N most frequently expanded
// prescan nodes. This makes the damage/audit budget concentrate on the portion
// of the scored workload it can actually revisit; 0 retains the old full union.
static int g_nav_damage_prescan_target_cap = 0;

struct NavDamageStats {
    long nodes_sampled = 0;
    long nodes_damaged = 0;
    long nodes_skipped_fully_protected = 0; // deg <= k_p_now: nothing evictable
    long edges_removed = 0;
};

static void remember_restoration_record(DiskANNIndex &g, uint32_t u, const std::vector<uint32_t> &row);

static bool g_nav_damage_in_edges = false;
static NavDamageStats inject_in_edge_damage(DiskANNIndex &g, const std::vector<int> &target_ids) {
    NavDamageStats st;
    std::unordered_set<uint32_t> target_set(target_ids.begin(), target_ids.end());
    std::unordered_map<uint32_t, std::vector<uint32_t>> in_neighbours_of_target;
    const int64_t N = (int64_t)g.num_points;
    const size_t kChunk = 65536;
    for (int64_t chunk_begin = 0; chunk_begin < N; chunk_begin += (int64_t)kChunk) {
        const int64_t chunk_end = std::min<int64_t>(N, chunk_begin + (int64_t)kChunk);
        std::vector<int> chunk_ids;
        chunk_ids.reserve((size_t)(chunk_end - chunk_begin));
        for (int64_t u = chunk_begin; u < chunk_end; u++) chunk_ids.push_back((int)u);
        auto rows = g.fetch_neighbors(chunk_ids);
        for (auto &[u, row] : rows)
            for (uint32_t v : row)
                if (target_set.count(v) && v != u) in_neighbours_of_target[v].push_back(u);
    }
    const double frac = g_nav_damage_target_frac > 0.0 ? g_nav_damage_target_frac : 1.0;
    for (int yid : target_ids) {
        const uint32_t y = (uint32_t)yid;
        st.nodes_sampled++;
        auto it = in_neighbours_of_target.find(y);
        if (it == in_neighbours_of_target.end()) continue;
        std::vector<uint32_t> &in_neighbours = it->second;
        DiskANNIndex::PQQuery yq = g.node_query(y);
        std::sort(in_neighbours.begin(), in_neighbours.end(),
                  [&](uint32_t a, uint32_t b) { return yq.dist(a) > yq.dist(b); }); // longest first
        const size_t n_remove = std::min(in_neighbours.size(),
                                         (size_t)std::lround(frac * (double)in_neighbours.size()));
        for (size_t i = 0; i < n_remove; i++) {
            const uint32_t u = in_neighbours[i];
            std::vector<uint32_t> row = g.get_neighbors(u);
            remember_restoration_record(g, u, row);
            g.remove_edge(u, y, row);
            g_nav_damage_removed[u].insert(y);
            mark_row_dirty((int)u);
            st.edges_removed++;
            if (g_nav_damage_dump_f)
                fprintf(g_nav_damage_dump_f, "%lld,%lld,%.6f\n", (long long)u, (long long)y,
                        (double)yq.dist(u));
        }
        fprintf(stderr, "  [nav-damage-in] y=%u in_degree=%zu removed=%zu\n", y, in_neighbours.size(), n_remove);
        st.nodes_damaged++;
    }
    g.flush_edits();
    if (g_nav_damage_dump_f) fflush(g_nav_damage_dump_f);
    return st;
}

static NavDamageStats inject_navigability_damage(DiskANNIndex &g, double frac, unsigned seed,
                                                 bool strip_all, int n_longest = 0,
                                                 const std::vector<int> *target_ids = nullptr) {
    NavDamageStats st;
    if (g_nav_damage_in_edges && target_ids && !target_ids->empty())
        return inject_in_edge_damage(g, *target_ids);
    if (frac <= 0.0 && n_longest <= 0 && (!target_ids || target_ids->empty())) return st;
    const int64_t N = (int64_t)g.num_points;
    int64_t n_target = std::max<int64_t>(1, std::lround(frac * (double)N));
    n_target = std::min(n_target, N);
    std::vector<int> ids;
    if (target_ids && !target_ids->empty()) {
        // Caller-supplied locality: no sampling, no whole-index scan.
        ids = *target_ids;
    } else if (n_longest > 0) {
        // Every node: no sampling, so no RNG draw and no seed dependence.
        ids.reserve((size_t)N);
        for (int64_t u = 0; u < N; u++) ids.push_back((int)u);
    } else {
        std::mt19937_64 rng(seed);
        std::uniform_int_distribution<int64_t> pick(0, N - 1);
        std::unordered_set<int64_t> sampled;
        sampled.reserve((size_t)n_target * 2);
        while ((int64_t)sampled.size() < n_target) sampled.insert(pick(rng));
        ids.reserve(sampled.size());
        for (int64_t u : sampled) ids.push_back((int)u);
        std::sort(ids.begin(), ids.end());
    }
    // One batched fetch rather than one single-node read per target: do_rewire
    // already knows its whole touched set up front and reads it this way. The
    // whole-index pass is chunked because holding 10M adjacency rows at once
    // would cost several GB of map overhead on top of the rows themselves.
    const size_t kChunk = 65536;
    const int k_p_now = g_edge_protection ? g_governor.k_p : 0;
    std::unordered_map<uint32_t, std::vector<uint32_t>> rows;
    for (size_t chunk_begin = 0; chunk_begin < ids.size(); chunk_begin += kChunk) {
    const size_t chunk_end = std::min(ids.size(), chunk_begin + kChunk);
    std::vector<int> chunk_ids(ids.begin() + chunk_begin, ids.begin() + chunk_end);
    rows = g.fetch_neighbors(chunk_ids);
    for (int uid : chunk_ids) {
        st.nodes_sampled++;
        const uint32_t u = (uint32_t)uid;
        auto rit = rows.find(u);
        if (rit == rows.end()) continue;
        const std::vector<uint32_t> &row = rit->second;
        const int dgu = (int)row.size();
        if (n_longest <= 0 && dgu <= k_p_now) { st.nodes_skipped_fully_protected++; continue; }
        // Damage precedes learning. Capture the pristine record here so a fully
        // stripped node still has its deleted shortcuts available to audit.
        remember_restoration_record(g, u, row);
        // Longest-first, the same convention and the same ranking function the
        // protection prefix uses: ranks [0, k_p_now) are protected, ranks
        // [k_p_now, dgu) are what the learner is free to evict.
        // One PQQuery per damaged node, shared by the ranking and the dump, so
        // the distance written to the CSV is the same number the rank was made
        // from rather than a second, independently recomputed one.
        DiskANNIndex::PQQuery uq = g.node_query(u);
        const std::vector<uint32_t> rank = live_rank_of(g, u, row, &uq);
        // n_longest: ranks [0, n_longest) -- the longest edges, i.e. exactly the
        // ones the protection prefix would pin. Otherwise the learner-evictable
        // suffix [k_p_now, ...) as before.
        const int rk_begin = n_longest > 0 ? 0 : k_p_now;
        const int n_remove = (target_ids && !target_ids->empty() && g_nav_damage_target_frac > 0.0)
                                 ? (int)std::lround(g_nav_damage_target_frac * (double)dgu)
                                 : n_longest;
        const int rk_end = n_longest > 0 ? std::min(dgu, n_remove)
                                         : (strip_all ? dgu : std::min(dgu, k_p_now + 1));
        for (int rk = rk_begin; rk < rk_end; rk++) {
            const uint32_t v = rank[rk];
            g.remove_edge(u, v, row);
            g_nav_damage_removed[u].insert(v);
            st.edges_removed++;
            if (g_nav_damage_dump_f)
                fprintf(g_nav_damage_dump_f, "%lld,%lld,%.6f\n", (long long)u, (long long)v,
                        (double)uq.dist(v));
        }
        mark_row_dirty((int)u);
        st.nodes_damaged++;
    }
    }
    if (g_nav_damage_dump_f) fflush(g_nav_damage_dump_f);
    return st;
}

static bool g_rewire_dirty_rows = false;

struct PQDistanceCache {
    static constexpr uint32_t EMPTY_KEY = 0xFFFFFFFFu;
    std::vector<uint32_t> keys;
    std::vector<float> vals;
    uint32_t slot_mask = 0;
    size_t entry_count = 0;

    float *find(int key) {
        if (keys.empty()) return nullptr;
        uint32_t k = (uint32_t)key, i = flat_hash_of_node(k) & slot_mask;
        while (keys[i] != EMPTY_KEY) {
            if (keys[i] == k) return &vals[i];
            i = (i + 1) & slot_mask;
        }
        return nullptr;
    }
    void insert(int key, float value) {
        if (keys.empty()) allocate(16);
        uint32_t k = (uint32_t)key, i = flat_hash_of_node(k) & slot_mask;
        while (keys[i] != EMPTY_KEY) {
            if (keys[i] == k) { vals[i] = value; return; }
            i = (i + 1) & slot_mask;
        }
        keys[i] = k;
        vals[i] = value;
        entry_count++;
        if (entry_count * 2 > keys.size()) grow();
    }
    size_t size() const { return entry_count; }
    void clear() {
        std::fill(keys.begin(), keys.end(), EMPTY_KEY);
        entry_count = 0;
    }
  private:
    void allocate(size_t capacity) {
        keys.assign(capacity, EMPTY_KEY);
        vals.resize(capacity);
        slot_mask = (uint32_t)(capacity - 1);
        entry_count = 0;
    }
    void grow() {
        std::vector<uint32_t> old_keys;
        std::vector<float> old_vals;
        old_keys.swap(keys);
        old_vals.swap(vals);
        allocate(old_keys.size() * 2);
        for (size_t i = 0; i < old_keys.size(); i++)
            if (old_keys[i] != EMPTY_KEY) insert((int)old_keys[i], old_vals[i]);
    }
};

// Per-node fixed-capacity LRU of ground-truth ids "wanted" while a query's path
// passed through this node; at rewire time this is the candidate pool beyond the
// node's current adjacency.
struct LRU {
    std::vector<int> order; // front = oldest
    PQDistanceCache pq_dist;
    // Returns true when the SET of members changed. do_rewire folds B_buf[u] into
    // C_u as a set, so refreshing the recency of a member it already holds is
    // invisible to any ranking; only an insert (and the eviction an insert can
    // trigger) changes what the row is scored against.
    bool touch(int w, int cap) {
        auto it = std::find(order.begin(), order.end(), w);
        const bool was_present = (it != order.end());
        if (was_present) order.erase(it); // present already: refresh recency
        order.push_back(w);
        if ((int)order.size() > cap) order.erase(order.begin());
        return !was_present;
    }
    template <class Set> bool touch_all(const Set &ws, int cap) {
        if (ws.empty()) return false;
        size_t keep = 0, members_present = 0;
        for (size_t i = 0; i < order.size(); i++) {
            if (ws.count(order[i])) { members_present++; continue; }
            order[keep++] = order[i];
        }
        order.resize(keep);
        for (int w : ws) order.push_back(w);
        if ((int)order.size() > cap)
            order.erase(order.begin(), order.begin() + ((int)order.size() - cap));
        return members_present < ws.size();
    }
};
// Bmap likewise retired (std::unordered_map<int, LRU>) -- LRU now lives
// nested in HotLearnerCache::Entry, same reasoning as Qmap above.

struct HotLearnerCache {
    size_t cap = 0;
    uint64_t clock = 0;
    struct Entry { uint32_t hits = 0; uint64_t last = 0; QRow q; LRU buf; };
    struct Key {
        uint32_t hits;
        uint64_t last;
        int node;
        bool operator>(const Key &o) const {
            if (hits != o.hits) return hits > o.hits;
            return last > o.last;
        }
    };
    std::unordered_map<int, Entry> tab;
    std::vector<Key> heap;

    bool enabled() const { return cap > 0; }
    bool contains(int u) const { return !enabled() || tab.count(u); }
    void reserve() { if (enabled()) { tab.reserve(cap); heap.reserve(2 * cap); } }

    // Replaces the old Qmap::operator[] / Bmap::operator[] call sites: same
    // default-insert-on-miss semantics, now against the one shared map.
    QRow &qrow(int u) { return tab[u].q; }
    LRU &lru(int u) { return tab[u].buf; }
    Entry *find(int u) {
        auto it = tab.find(u);
        return it == tab.end() ? nullptr : &it->second;
    }

    void push_key(int u, const Entry &e) {
        heap.push_back({e.hits, e.last, u});
        std::push_heap(heap.begin(), heap.end(), std::greater<Key>());
    }

    std::vector<int> touch(const std::vector<int> &nodes) {
        if (!enabled()) return nodes;
        for (int u : nodes) {
            Entry &e = tab[u];
            if (e.hits != std::numeric_limits<uint32_t>::max()) e.hits++;
            e.last = ++clock;
            push_key(u, e);
        }
        // §6 H.Touch(V_t): bound learner state only; eviction drops the
        // whole collocated row (Q and B_buf together, one erase), never the
        // graph row.  Lazy LFU heap avoids rescanning H on every eviction.
        while (tab.size() > cap) {
            Key k;
            for (;;) {
                std::pop_heap(heap.begin(), heap.end(), std::greater<Key>());
                k = heap.back();
                heap.pop_back();
                auto it = tab.find(k.node);
                if (it != tab.end() && it->second.hits == k.hits && it->second.last == k.last) break;
            }
            tab.erase(k.node);
        }
        if (heap.size() > 4 * tab.size() + 1024) {
            heap.clear();
            heap.reserve(2 * cap);
            for (auto &kv : tab) heap.push_back({kv.second.hits, kv.second.last, kv.first});
            std::make_heap(heap.begin(), heap.end(), std::greater<Key>());
        }
        std::vector<int> hot;
        hot.reserve(nodes.size());
        for (int u : nodes) if (tab.count(u)) hot.push_back(u);
        return hot;
    }

    std::vector<int> keys() const {
        std::vector<int> out;
        out.reserve(tab.size());
        for (auto &kv : tab) out.push_back(kv.first);
        return out;
    }
};

// Lazy in-memory stand-in for the paper's immutable restoration record: the
// K_max longest build-row edges, by PQ length from u. The on-disk harness did
// not add Pagewire's sector-slack record, so we capture rows before this run
// first edits them and use that frozen rank order for Algorithm 5.
static std::unordered_map<uint32_t, std::vector<uint32_t>> g_restoration_record;
// Use on-demand ef=50 searches for repair proposals instead of retaining any
// historical per-node restoration state.
static bool g_protect_search_repair = false;
static long g_audit_search_proposals = 0;

static void remember_restoration_record(DiskANNIndex &g, uint32_t u, const std::vector<uint32_t> &row) {
    if (!g_edge_protection || g_protect_search_repair || g_governor.K_max <= 0 || g_restoration_record.count(u)) return;
    std::vector<uint32_t> rec = row;
    DiskANNIndex::PQQuery uq = g.node_query(u);
    std::sort(rec.begin(), rec.end(), [&](uint32_t a, uint32_t b) {
        float da = uq.dist(a), db = uq.dist(b);
        if (da != db) return da > db;
        return a < b;
    });
    if ((int)rec.size() > g_governor.K_max) rec.resize((size_t)g_governor.K_max);
    g_restoration_record.emplace(u, std::move(rec));
}

static std::unordered_map<uint32_t, std::vector<uint32_t>> g_candidate_log;
// Repeat counter per (u, recruited candidate) and the resulting permanent pins
// -- Algorithm 5's "pin on repeat". Consulted by do_rewire so a pinned edge
// survives regardless of k_p or score (Definition 4.1's S(u) = P_kp(u) u pin).
static std::unordered_map<uint32_t, std::unordered_map<uint32_t, int>> g_rescue_repeats;
static std::unordered_map<uint32_t, std::unordered_set<uint32_t>> g_pinned_edges;
// Nodes whose retained prefix fails to cover the approximate log even at
// K_max -- Definition 4.1's "flagged": full out-list stays protected, learned
// rewiring disabled at that node (enforced in do_rewire).
static std::unordered_set<uint32_t> g_flagged_nodes;
static long g_flagged_count = 0;
static long g_audit_log_rescue = 0;   // Algorithm 5 lines 9-10: log rescue fired

static bool g_protect_direct_repair = false;
static bool g_protect_direct_repair_force = false;
// Also add each directly-committed repair to the pinned set, so the next
// do_rewire cannot select it away (Definition 4.1's S(u) = P_kp(u) u pin).
static bool g_protect_direct_repair_pin = false;
// --protect-direct-repair-to-y: every forced audit repair (u, v) also queues
// (u, y), an edge straight from the repaired node to the audit target y.
static bool g_protect_direct_repair_to_y = false;
static int64_t g_audit_current_y = -1; // y of the audit_step in progress
static bool g_protect_direct_repair_retest = false;
static std::string g_audit_repair_dump_csv;
static FILE *g_audit_repair_dump_f = nullptr;
static std::string g_audit_retest_dump_csv;
static FILE *g_audit_retest_dump_f = nullptr;
static float g_protect_repair_min_improvement = 0.0f;
static std::vector<std::pair<uint32_t, uint32_t>> g_direct_repair_queue;
static void queue_audit_direct_repair(uint32_t u, uint32_t v) {
    g_direct_repair_queue.emplace_back(u, v);
    if (g_protect_direct_repair_to_y && g_audit_current_y >= 0 && u != (uint32_t)g_audit_current_y &&
        v != (uint32_t)g_audit_current_y)
        g_direct_repair_queue.emplace_back(u, (uint32_t)g_audit_current_y);
}
static std::vector<std::pair<long, uint32_t>> g_direct_repair_retest_targets;
static bool g_protect_repair_y_log_fill = false;
static std::unordered_set<uint32_t> g_audit_failed_targets;
static long g_y_log_fill_calls = 0, g_y_log_fill_zero_budget = 0, g_y_log_fill_added = 0, g_y_log_fill_evicted = 0;
static long g_direct_repair_applied = 0;   // edge present in u's row after the edit
static long g_direct_repair_existing = 0;  // (u,v) was already an edge
static long g_direct_repair_capped = 0;    // add_directed's cap re-rank dropped v
static long g_repair_margin_rejected = 0;
static long g_repair_margin_accepted = 0;
static double g_repair_margin_accepted_sum = 0.0;
static double g_repair_margin_rejected_sum = 0.0;
static long g_direct_repair_retest_reached = 0;
// --protect-repair-path-to-y: on a failed audit, queue (u, y) for EVERY node u
// the audit search expanded, with no candidate filter.
static bool g_protect_repair_path_to_y = false;
// --audit-retest-queries <bin>: row i is the held-out query whose top-1 GT is
// --audit-target-ids[i]. Retest searches it from the same entry point too.
static std::vector<std::vector<float>> g_audit_retest_queries;
// Per-batch retest outcomes (the last batch's numbers, not cumulative).
static long g_retest_n = 0, g_retest_y_top1 = 0, g_retest_testq_n = 0, g_retest_testq_top1 = 0;

static void record_direct_repair(uint32_t y, uint32_t u, uint32_t v, float improvement) {
    if (g_audit_repair_dump_f)
        fprintf(g_audit_repair_dump_f, "%ld,%u,%u,%u,%.9g\n", g_audit_dump_gqc, y, u, v, improvement);
    if (g_protect_direct_repair_retest) g_direct_repair_retest_targets.emplace_back(g_audit_dump_gqc, y);
}

static bool repair_is_pending(uint32_t u, uint32_t v) {
    return std::find(g_direct_repair_queue.begin(), g_direct_repair_queue.end(),
                     std::make_pair(u, v)) != g_direct_repair_queue.end();
}

static bool repair_is_admissible(uint32_t u, uint32_t v) {
    return !g_protect_repair_known_damage_only ||
           (g_nav_damage_removed.count(u) && g_nav_damage_removed[u].count(v));
}

static bool repair_has_effect(float dist_u_y, float dist_v_y) {
    const float improvement = dist_u_y - dist_v_y;
    if (improvement >= g_protect_repair_min_improvement) {
        g_repair_margin_accepted++;
        g_repair_margin_accepted_sum += improvement;
        return true;
    }
    g_repair_margin_rejected++;
    g_repair_margin_rejected_sum += improvement;
    return false;
}

static void certify_node(DiskANNIndex &g, uint32_t u, const std::vector<uint32_t> &log) {
    auto rit = g_restoration_record.find(u);
    if (rit == g_restoration_record.end() || rit->second.empty() || log.empty()) return;
    const std::vector<uint32_t> &record = rit->second;
    DiskANNIndex::PQQuery uq = g.node_query(u);
    const float r_Kmax = uq.dist(record.back());
    for (uint32_t y : log) {
        if (y == u) continue;
        const float dy = uq.dist(y);
        if (dy <= r_Kmax) continue; // inside the landing radius: not a coverage requirement
        bool covered = false;
        for (uint32_t v : record) {
            if (uq.dist(v) <= dy / (float)g_alpha) { covered = true; break; }
        }
        if (!covered) {
            if (g_flagged_nodes.insert(u).second) g_flagged_count++;
            return;
        }
    }
}

static std::vector<uint32_t> compute_candidate_log(DiskANNIndex &g, uint32_t u, uint64_t disk_beam_width) {
    std::vector<std::vector<float>> ucoord;
    g.read_nodes({u}, &ucoord, nullptr);
    std::vector<int> result_ids, visited, expanded;
    const uint64_t ef = g_candlog_ef > 0 ? (uint64_t)g_candlog_ef
                                         : (uint64_t)std::max(64, 8 * std::max(1, g_governor.K_max));
    // Self-lookup idiom already used by audit_step (y's own vector as the
    // "query") and measure_build_floor: feed the on-disk coordinates straight
    // into search(), which re-augments them itself.
    g.search(ucoord[0].data(), /*K=*/1, ef, disk_beam_width, result_ids, visited, &expanded);
    DiskANNIndex::PQQuery uq = g.node_query(u);
    std::vector<std::pair<float, uint32_t>> by_dist;
    std::unordered_set<uint32_t> seen{u};
    for (int id : expanded) {
        uint32_t v = (uint32_t)id;
        if (seen.insert(v).second) by_dist.emplace_back(uq.dist(v), v);
    }
    std::sort(by_dist.begin(), by_dist.end());
    std::vector<uint32_t> log;
    log.reserve(by_dist.size());
    for (auto &kv : by_dist) log.push_back(kv.second);
    return log;
}

static const std::vector<uint32_t> &get_candidate_log(DiskANNIndex &g, uint32_t u, uint64_t disk_beam_width) {
    auto it = g_candidate_log.find(u);
    if (it != g_candidate_log.end()) return it->second;
    std::vector<uint32_t> log = compute_candidate_log(g, u, disk_beam_width);
    certify_node(g, u, log);
    auto ins = g_candidate_log.emplace(u, std::move(log));
    return ins.first->second;
}

static bool g_y_log_predamage = false;
static std::vector<std::vector<uint32_t>> g_y_log_overlap_gt;

static void build_predamage_y_logs(DiskANNIndex &g, uint64_t disk_beam_width,
                                   std::unordered_map<uint32_t, std::vector<uint32_t>> &predamage_logs) {
    if (!g_y_log_predamage && g_y_log_overlap_gt.empty()) return;
    for (uint32_t y : g_audit_target_ids)
        if (!predamage_logs.count(y)) predamage_logs.emplace(y, compute_candidate_log(g, y, disk_beam_width));
}

static void use_predamage_y_logs(DiskANNIndex &g, uint64_t disk_beam_width,
                                 const std::unordered_map<uint32_t, std::vector<uint32_t>> &predamage_logs) {
    if (!g_y_log_overlap_gt.empty()) {
        double pre_hits = 0, post_hits = 0, pre_size = 0, post_size = 0;
        size_t n = 0;
        for (size_t i = 0; i < g_audit_target_ids.size() && i < g_y_log_overlap_gt.size(); i++) {
            const uint32_t y = g_audit_target_ids[i];
            std::unordered_set<uint32_t> gt(g_y_log_overlap_gt[i].begin(), g_y_log_overlap_gt[i].end());
            const std::vector<uint32_t> &pre = predamage_logs.at(y);
            const std::vector<uint32_t> post = compute_candidate_log(g, y, disk_beam_width);
            long pre_in = 0, post_in = 0;
            for (uint32_t v : pre) pre_in += gt.count(v);
            for (uint32_t v : post) post_in += gt.count(v);
            fprintf(stderr, "  [y-log-overlap] i=%zu y=%u gt_k=%zu predamage=%ld/%zu postdamage=%ld/%zu\n",
                    i, y, gt.size(), pre_in, pre.size(), post_in, post.size());
            pre_hits += pre_in; post_hits += post_in; pre_size += pre.size(); post_size += post.size(); n++;
        }
        if (n)
            fprintf(stderr, "  [y-log-overlap] mean over %zu targets: predamage gt_hits=%.2f log_size=%.1f "
                            "postdamage gt_hits=%.2f log_size=%.1f\n",
                    n, pre_hits / n, pre_size / n, post_hits / n, post_size / n);
    }
    if (g_y_log_predamage)
        for (const auto &kv : predamage_logs) {
            certify_node(g, kv.first, kv.second);
            g_candidate_log[kv.first] = kv.second;
        }
}

using CoRow = std::unordered_map<int, long>;
using CoMap = std::unordered_map<int, CoRow>;

// Split in two so a caller with many lookups against the same node holds the row
// once: layout_step asks ~45 times per pair over ~7 distinct outer keys.
static const CoRow *co_row(const CoMap &co, int a) {
    auto it = co.find(a);
    return it == co.end() ? nullptr : &it->second;
}

static int g_layout_baff_cap = 0;

static long co_at(const CoRow *row, int b) {
    if (row == nullptr) return 0;
    auto jt = row->find(b);
    return jt == row->end() ? 0 : jt->second;
}

// Data_logging (every query): one increment per unordered pair of the trajectory.
// Quadratic in |V(q)|, which the paper avoids by capping enumeration at the
// sketch capacity; with |V(q)| ~ ef this costs a few 10^4 hash bumps per query
// and is not measurable against the disk reads it accompanies.
static long log_covisitation(CoMap &co, const std::vector<int> &V_q) {
    // i + 1 < size, not i < size: the last element's body is empty, and touching
    // co[] for it would insert a row the pre-hoist loop never created.
    for (size_t i = 0; i + 1 < V_q.size(); i++) {
        CoRow &row_i = co[V_q[i]];
        for (size_t j = i + 1; j < V_q.size(); j++) {
            row_i[V_q[j]] += 1;
            co[V_q[j]][V_q[i]] += 1;
        }
    }
    const long n = (long)V_q.size();
    return n * (n - 1) / 2;
}

// I(pi) instrumentation, accumulated over the training stream: distinct sectors
// touched per query summed, against expanded nodes summed. Their ratio is the
// per-query block-touch rate LayoutStep is trying to push down; it starts at 1.0
// (every expansion its own sector) and falls toward 1/kappa as packing improves.
static long g_distinct_sectors = 0, g_expanded_nodes = 0;

// --dump-query-results: q, then the search's K result ids, then the query's K
// ground-truth ids. Recall is scored from this file offline (train_recall.py),
// so no measurement pass ever runs a query against the index during training.
static FILE *g_query_results_file = nullptr;
// How many N_out(u) candidates the line-11 update credited vs skipped as
// untraversed. Both count post-dedup pool members; B_buf-only entries are in
// neither, so credited+skipped is the size of the N_out half of P(u).
static long g_qupdate_nout_credited = 0, g_qupdate_nout_skipped = 0;
// B_buf(u) credits, which only --q-update-full-pool issues. Reported separately
// because B_buf is capped at 3R against N_out's R, so it, not the V_t filter, is
// the larger part of the pre-line-11 update volume.
static long g_qupdate_bbuf_credited = 0;
static bool g_q_update_full_pool = false;

// --buffer-cap-mult: multiplier on max_degree (R) for the B_buf slack cap,
// i.e. buffer_cap = mult * R. DEFAULT 3 (the value that was previously
// hardcoded). Exposed to test whether B_buf's candidate-pool size is a
// binding constraint on the learner's recall gain.
static double g_buffer_cap_mult = 3.0;

// eps-greedy nav: one explore coin per source node per step.
static std::mt19937_64 g_nav_rng(42);
static std::uniform_real_distribution<double> g_unif01(0.0, 1.0);

struct MethodCfg {
    std::string label;
    std::string reward;   // "recall": shared per-query recall credited to every candidate in P(u);
                          // "binary": 1/0 per edge on whether its target is in the query's label T_t;
                          // "one":    constant 1 credited to every candidate in P(u)
};

static double edge_reward(const std::string &reward, double recall_val,
                          const std::unordered_set<int> &label_ids, int v) {
    if (reward == "recall") return recall_val;
    if (reward == "one") return 1.0;
    return label_ids.count(v) ? 1.0 : 0.0; // "binary"
}

// calculate_quick_gt: an alternative source of "ground truth" for train_batch,
// used only when --quick-gt is set (default: off, see g_quick_gt). Not the
// precomputed exact GT file -- a second, deeper search against the live index
// (L=1000, same k and beam width as the serving search), on the premise that
// a wide-enough beam is a usable stand-in for the oracle at a fraction of its
// cost. When enabled it feeds BOTH recall_val (so "recall" means agreement
// with this search, not with the true nearest neighbors) and, when a
// --gt-label-rate draw fires, T_t. Charged to its own IOCounters
// (g_io_quickgt) so this extra search never perturbs the traversal/rewire/
// layout/audit I/O already reported elsewhere. Returns the ranked ids (not a
// set) so callers can still dump them in GT column order.
static constexpr uint64_t kQuickGtEf = 1000;
static std::vector<int> calculate_quick_gt(DiskANNIndex &g, const float *q_vec, int k,
                                            uint64_t disk_beam_width) {
    IOPhase quickgt_phase(g_io_quickgt);
    std::vector<int> quick_gt_ids, quick_gt_visited;
    g.search(q_vec, (uint64_t)k, kQuickGtEf, disk_beam_width, quick_gt_ids, quick_gt_visited);
    return quick_gt_ids;
}

// train_batch: port of control_vamana.cpp train_batch. The Q-learning logic is
// unchanged; what changes is where distances and neighbours come from -- PQ
// lookups in RAM for distances, one batched sector read per beam step for
// adjacency.
// batch_g (the precomputed exact-GT file) is the default source of
// ground_truth_ids; --quick-gt switches to calculate_quick_gt instead (see
// g_quick_gt's comment), for both recall_val and T_t.
static long train_batch(DiskANNIndex &g, const float *batch_v, const int64_t *batch_g, int nbatch,
                        HotLearnerCache &learner_hot, double Q_0, const std::string &reward, long gqc,
                        int K, int ef, int buffer_cap, uint64_t disk_beam_width,
                        std::vector<double> *out_recalls = nullptr, CoMap *co_out = nullptr) {
    FlatNodeSet pool_seen; // hoisted: dedups N_out(u) against B_buf[u]
    // Sized to what it actually holds -- the credited members of one node's pool,
    // at most N_out (R) plus, under --q-update-full-pool, B_buf (3R) -- because
    // clear() costs capacity, not count, and this one is cleared once per node of
    // V_t n H. It grows itself if a pool ever exceeds this.
    pool_seen.init(64);
    // Hoisted for the same reason: these were rebuilt from scratch for every
    // query, one allocation set per query over a 5000-query run. All three are
    // membership- or size-only (insert / count / size), never iterated, so
    // reusing a cleared container cannot leak a different order into anything.
    std::vector<int> V_t;
    std::vector<size_t> V_t_row;
    FlatNodeSet V_t_set;
    V_t_set.init((size_t)ef); // the expanded set, cleared once per query
    std::unordered_set<uint64_t> distinct_secs;
    std::unordered_set<int> ground_truth_ids;  // sourced from batch_g, or from calculate_quick_gt if --quick-gt
    std::vector<int> quick_gt_ids;             // same query's GT, in ranked order (for the dump file)
    std::string dump_row;
    for (int qi = 0; qi < nbatch; qi++) {
        const bool detector_learning = g_detector_calibration.learning();
        const float *q_vec = batch_v + (size_t)qi * g_query_dim;
        const int64_t *gt_row = batch_g + (size_t)qi * g_k; // only read when !g_quick_gt
        gqc += 1;
        const bool detector_sample = detector_learning || g_detector_sample_every == 1 ||
                                     gqc % g_detector_sample_every == 0;

        // The single trajectory §6 and §7 both read off: the stock DiskANN beam
        // search. Expanding a node already decodes its adjacency row, so
        // expanded_nbrs carries N_out(u) for every u the search touched, free.
        std::vector<int> result_ids, visited, expanded;
        std::vector<std::vector<uint32_t>> expanded_nbrs;
        const long sector_reqs_before_search = g_io->sector_reqs;
        const double io_s_before_search = g_io->io_s;
        const size_t miss_mark = g.epoch_missed.size();
        // Rows are requested only when the learner will consume them: under
        // --search-only nothing below the early-out reads expanded_nbrs, and
        // copying them would charge the DiskANN-alone reference arm for work it
        // does not do. That arm must stay byte-comparable to a pre-§6 binary.
        { TimeSpan ts(g_tp.search);
          g.search(q_vec, (uint64_t)K, (uint64_t)ef, disk_beam_width, result_ids, visited,
                   &expanded, (!g_search_only && detector_sample) ? &expanded_nbrs : nullptr); }
        // full_retset is sorted by EXACT distance over the expanded (and
        // harvested) nodes, so result_ids[0] is the closest node this query
        // expanded -- the audit entry point under --protect-audit-ep-last-result.
        if (g_audit_ep_last_result && !result_ids.empty()) g_last_serving_ep = result_ids[0];
        g_search_sector_reqs += g_io->sector_reqs - sector_reqs_before_search;
        g_tp.search_io += g_io->io_s - io_s_before_search;
        // V_t is the search's *expanded* set, not the broader discovered set: only
        // an expanded node costs a sector fetch, and only an expanded node has its
        // adjacency row in hand. This is the same set §7 defines co() over, so §6
        // and §7 now share one definition of V(q) instead of carrying two.
        // V_t_row keeps each survivor's position, so N_out(V_t[i]) is
        // expanded_nbrs[V_t_row[i]] -- no id->row map needed.
        // V_t_set is the same set as V_t; kept for the O(1) membership test the
        // traversed-only line-11 pool needs (N_out(u) n V_t).
        V_t.clear(); V_t_row.clear(); V_t_set.clear();
        { TimeSpan sets_ts(g_tp.serve_sets);
        for (size_t i = 0; i < expanded.size(); i++)
            if (V_t_set.insert_new(expanded[i])) {
                V_t.push_back(expanded[i]); V_t_row.push_back(i);
            } }
        // §6 H.Touch(V_t): admits and renews, and evicts the collocated Q/B_buf
        // row of whatever falls out of H. Learner state only -- an evicted
        // node keeps its graph row. Called for that effect; V_t n H is read
        // back per node below via contains(), so the returned subset is not
        // needed. learner_hot is always a live object now (Q/B_buf storage
        // lives inside it), so the old null check is gone -- only enabled()
        // (cap > 0) still gates whether H actually bounds anything.
        if (detector_learning && learner_hot.enabled())
            learner_hot.touch(V_t);
        // Which trajectory the affinity comes from, per --covisit-scope.
        //
        // `miss` (the default, and what every arm before this flag ran): the
        // sectors this query actually had to fetch. Once C_blk holds a sector a
        // second touch is served whether or not the two nodes are co-located, so
        // an expansion that hit the cache is not co-location LayoutStep can
        // convert into a saved read.
        //
        // `expanded`: the whole expanded set, which is what section 7 defines
        // co() over. The argument above is about one query in isolation; a pair
        // co-visited while resident is still evidence about the pair, and it is
        // exactly the evidence a big cache deletes before the layout pass sees
        // it. Whether that evidence is worth acting on is the measurement.
        if (detector_learning && co_out)
            g_covisit_pairs += g_covisit_scope_expanded
                                   ? log_covisitation(*co_out, V_t)
                                   : [&] {
                                         std::vector<int> M_t;
                                         M_t.reserve(g.epoch_missed.size() - miss_mark);
                                         for (size_t mi = miss_mark; mi < g.epoch_missed.size(); mi++)
                                             M_t.push_back((int)g.epoch_missed[mi]);
                                         return log_covisitation(*co_out, M_t);
                                     }();
        // I(pi) as Lemma F.1 models it: one I/O per distinct block touched, which
        // is what block packing can reduce. `reads` still cannot show a layout
        // effect -- it counts ids requested, which is layout-invariant -- but
        // since the query block buffer landed, `sector_reqs` can: with the buffer
        // on and uncapped, g_search_sector_reqs must equal g_distinct_sectors.
        // The two are computed by independent routes (one from the request stream,
        // one from the expanded set), so a mismatch is a real defect in one of them.
        { TimeSpan sets_ts(g_tp.serve_sets);
          distinct_secs.clear();
          for (int x : V_t) distinct_secs.insert(g.node_sector((uint32_t)x));
          g_distinct_sectors += (long)distinct_secs.size();
          g_expanded_nodes += (long)V_t.size();
          // ground_truth_ids defaults to the precomputed exact-GT file
          // (batch_g/gt_row); --quick-gt switches to calculate_quick_gt's
          // live search instead (see g_quick_gt's comment). quick_gt_ids is
          // populated either way so the dump file below has a single path.
          if (g_quick_gt) {
              quick_gt_ids = calculate_quick_gt(g, q_vec, g_k, disk_beam_width);
          } else {
              quick_gt_ids.assign(gt_row, gt_row + g_k);
          }
          ground_truth_ids.clear();
          ground_truth_ids.insert(quick_gt_ids.begin(), quick_gt_ids.end()); }
        if (g_query_results_file) {
            // One fwrite per query instead of 2 + K + g_k fprintf calls (202 here).
            // The bytes are identical: same format specifiers, same order.
            TimeSpan dump_ts(g_tp.serve_dump);
            char num[32];
            dump_row.clear();
            snprintf(num, sizeof(num), "%ld", gqc);      // gqc is already this query, 1-based
            dump_row += num;
            for (int j = 0; j < K; j++) {
                snprintf(num, sizeof(num), ",%d", j < (int)result_ids.size() ? result_ids[j] : -1);
                dump_row += num;
            }
            // GT columns hold quick_gt_ids: the file GT by default, or
            // calculate_quick_gt's ranked ids under --quick-gt; -1 pads any
            // row where the quick search (if used) returned fewer than g_k ids.
            for (int j = 0; j < g_k; j++) {
                snprintf(num, sizeof(num), ",%lld",
                         (long long)(j < (int)quick_gt_ids.size() ? quick_gt_ids[j] : -1));
                dump_row += num;
            }
            dump_row += '\n';
            fwrite(dump_row.data(), 1, dump_row.size(), g_query_results_file);
        }
        int inter = 0; for (int r : result_ids) if (ground_truth_ids.count(r)) inter++;
        double recall_val = (double)inter / K;
        if (out_recalls) out_recalls->push_back(recall_val);
        // --search-only: everything below this line is the learner. Leaving here
        // gives the DiskANN-alone reference the other arms are measured against.
        if (g_search_only) continue;
        // The prior every edge FIRST SEEN during this query is seeded at. Held
        // fixed for the whole query, so an edge's seed records the difficulty of
        // the query that discovered it rather than one degenerate constant.
        const double edge_prior = (g_recall_prior && reward == "recall") ? recall_val : Q_0;
        // Algorithm 1 lines 7-8, default branch (no probe): the label T_t is
        // the free self-label, the top-k of the re-ranked examined set the
        // search already returned. --gt-label-rate p ADDS ground_truth_ids
        // (the file GT by default, or calculate_quick_gt's output under
        // --quick-gt) on top of the self-label (never replaces it) on a
        // per-query Bernoulli(p) draw: p=0 is self-label only; p>0, each
        // query independently unions in that GT on top of the self-label
        // with probability p. p<0 (unset) falls back to the g_label_from_gt
        // switch: --label-gt (self-label UNION gt-label every query) or
        // --self-label (self-label only).
        const double gt_label_rate = (g_gt_label_rate >= 0.0) ? g_gt_label_rate : (g_label_from_gt ? 1.0 : 0.0);
        const bool use_gt_label = (gt_label_rate >= 1.0) ? true
                                 : (gt_label_rate <= 0.0) ? false
                                 : (g_gt_label_unif01(g_gt_label_rng) < gt_label_rate);
        std::unordered_set<int> T_t(result_ids.begin(), result_ids.end());
        // ground_truth_ids was already computed above (for recall_val too,
        // from batch_g by default or from calculate_quick_gt under
        // --quick-gt), so the draw just reuses it -- no second GT lookup
        // either way, and --gt-label-rate/--label-gt work the same under
        // both sources.
        if (use_gt_label) T_t.insert(ground_truth_ids.begin(), ground_truth_ids.end());

        // While adaptation is frozen, the foreground still forms the candidate
        // allocations needed by the drift detector, but it must not mutate Q,
        // B_buf, H, co-visitation state, or the graph. A drift event ends the
        // current serving batch so the next outer-loop batch restarts learning.
        if (!detector_learning) {
            if (detector_sample) {
                TimeSpan detector_ts(g_tp.update);
                g_detector_calibration.novelty_sample_queries++;
                for (int u : V_t) {
                    if (!learner_hot.contains(u)) continue;
                    for (int w : T_t) g_detector_calibration.buffer_edge(gqc, u, w);
                }
            }
            if (g_detector_calibration.finish_query(gqc) != DetectorCalibration::NONE)
                break;
            continue;
        }

        // Algorithm 1 lines 11-13, covisited-only variant: each u in V_t n H
        // credits only (u,v) pairs with v in N_out(u) n V_t -- v must be an
        // actual out-edge AND itself expanded this query. B_buf[u] is no
        // longer scored here at all unless --q-update-full-pool is set, which
        // adds it back as the second half of the union. Either way do_rewire
        // still ranks the full pool N_out(u) u B_buf[u] as C_u, so in the
        // default path a recruited B_buf candidate can be selected for commit
        // while carrying only its seed prior Q_0, never an update from this
        // loop. The reward channel is unchanged; what changes is which edges
        // receive it.
        TimeSpan update_ts(g_tp.update);
        auto update_edge = [&](QRow &q_row, int v) {
            EdgeStats &st = q_row.emplace_or_get(v, EdgeStats{edge_prior, 0L});
            const double r = edge_reward(reward, recall_val, T_t, v);
            const double old_q = st.q;
            st.c += 1;
            st.q += (1.0 / (1.0 + st.c)) * (r - st.q);
            g_detector_calibration.q_update(old_q, st.q, r);
        };
        for (size_t i = 0; i < V_t.size(); i++) {
            const int u = V_t[i];
            // V_t n H. contains() is true for every node when H is disabled, so
            // --learner-hot-cap 0 keeps learning everywhere, as before §6.
            if (!learner_hot.contains(u)) continue;
            // qrow(u): one tab[u] lookup, default-inserting the collocated
            // {hits,last,q,buf} row on first touch (same as the old Qmap[u]).
            QRow &q_row = learner_hot.qrow(u);
            bool learner_state_changed = false;
            pool_seen.clear();
            for (uint32_t v : expanded_nbrs[V_t_row[i]]) {
                // The V_t filter is UNCONDITIONAL: an out-neighbour qualifies
                // only if the search actually expanded it this query. Line 11's
                // pool is
                //     P(u) = (N_out(u) n V_t) u B_buf(u)
                // and --q-update-full-pool adds the B_buf half below; it does
                // not relax this filter. A candidate in both halves is one
                // candidate and is credited once, which is what pool_seen is
                // for -- note pool_seen is populated only by CREDITED members,
                // so an untraversed out-neighbour that also sits in B_buf[u] is
                // still credited, through the B_buf branch. That is the union
                // above: B_buf(u) is outside the intersection by construction.
                if (!V_t_set.count((int)v)) {
                    g_qupdate_nout_skipped += 1;
                    continue;
                }
                if (pool_seen.insert_new((int)v)) {
                    g_qupdate_nout_credited += 1;
                    update_edge(q_row, (int)v);
                    learner_state_changed = true;   // Q[u][v] moved
                }
            }
            // The other half of the pool, and the larger half: B_buf[u] is
            // capped at 3R = 192 against N_out's R = 64, so this loop is where
            // most of the old code's exploration lived. Skipped entirely in the
            // default (covisited-only) path. Not V_t-filtered by design -- a
            // recruited candidate has no out-edge to be traversed through yet,
            // so filtering it on V_t would credit nothing and the flag would be
            // a no-op.
            if (g_q_update_full_pool) {
                // u's row already exists (qrow(u) above created it), so this
                // find() always hits -- kept as a find rather than tab[u] only
                // because an empty buf.order (the common case before line
                // 14-15 below populates it) must iterate zero times, not
                // default-construct a second time.
                HotLearnerCache::Entry *e = learner_hot.find(u);
                if (e)
                    for (int w : e->buf.order)
                        if (pool_seen.insert_new(w)) {
                            g_qupdate_bbuf_credited += 1;
                            update_edge(q_row, w);
                            learner_state_changed = true;   // Q[u][w] moved
                        }
            }
            // Lines 14-15, unchanged: B_buf[u] is not scored above, but it is
            // still populated here for do_rewire's ranking and future queries.
            LRU &buf = learner_hot.lru(u);
            // C_u membership moved. touch_all is touch() for every w in T_t, in
            // T_t's own iteration order, folded into a single pass over `order`.
            for (int w : T_t) g_detector_calibration.buffer_edge(gqc, u, w);
            if (buf.touch_all(T_t, buffer_cap)) learner_state_changed = true;
            if (learner_state_changed) mark_row_dirty(u);
        }
        if (g_detector_calibration.finish_query(gqc) != DetectorCalibration::NONE)
            break;
    }
    return gqc;
}

struct CommitGainCensus {
    static const int LADDER = 8;
    // Brackets the per-edge price gap (theta * spread, ~1e-3 at theta=0.001)
    // and the measured quality gap (~0.26), an order of magnitude per rung.
    double lambda_rung[LADDER] = {0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0};
    long rows_suppressed[LADDER] = {0, 0, 0, 0, 0, 0, 0, 0};
    long changed_rows = 0;
    long nonpositive_rows = 0;
    long exact_zero_rows = 0;
    double g_sum = 0.0, g_min = 0.0, g_max = 0.0;
    std::vector<double> samples;

    void observe(double commit_gain) {
        if (changed_rows == 0) { g_min = g_max = commit_gain; }
        else { g_min = std::min(g_min, commit_gain); g_max = std::max(g_max, commit_gain); }
        changed_rows++;
        g_sum += commit_gain;
        if (commit_gain <= 0.0) nonpositive_rows++;
        if (commit_gain == 0.0) exact_zero_rows++;
        // "would be suppressed" is the negation of the commit test g > lambda.
        for (int i = 0; i < LADDER; i++) if (!(commit_gain > lambda_rung[i])) rows_suppressed[i]++;
        samples.push_back(commit_gain);
    }
    double quantile(double q) {
        if (samples.empty()) return 0.0;
        size_t k = (size_t)(q * (double)(samples.size() - 1));
        return samples[k]; // caller sorts first
    }
    void report() {
        if (changed_rows == 0) { fprintf(stderr, "  GAIN: changed_rows=0\n"); return; }
        std::sort(samples.begin(), samples.end());
        fprintf(stderr,
                "  GAIN: changed_rows=%ld mean=%.6g min=%.6g p10=%.6g p50=%.6g p90=%.6g max=%.6g"
                " nonpositive=%ld(%.4f) exactzero=%ld(%.4f)\n",
                changed_rows, g_sum / (double)changed_rows, g_min, quantile(0.10),
                quantile(0.50), quantile(0.90), g_max, nonpositive_rows,
                (double)nonpositive_rows / (double)changed_rows, exact_zero_rows,
                (double)exact_zero_rows / (double)changed_rows);
        fprintf(stderr, "  GAINLADDER:");
        for (int i = 0; i < LADDER; i++)
            fprintf(stderr, " %g:%ld(%.4f)", lambda_rung[i], rows_suppressed[i],
                    (double)rows_suppressed[i] / (double)changed_rows);
        fprintf(stderr, "\n");
    }
};

static void do_rewire(DiskANNIndex &g, HotLearnerCache &learner_hot, double Q_0,
                      double rewire_edge_prior, long gqc) {
    auto rsel_t0 = Clock::now();
    double rsel_io0 = g_io->io_s;
    auto lap = [&](int p) {
        const double io_now = g_io->io_s;
        g_tp.rsel[p] += secs_since(rsel_t0);
        g_tp.rsel_io[p] += io_now - rsel_io0;
        rsel_t0 = Clock::now();
        rsel_io0 = io_now;
    };
    std::vector<int> node_vec = learner_hot.keys();
    {
        long dirty_in_node_vec = 0;
        for (int u : node_vec)
            if (u >= 0 && (size_t)u < g_row_dirty.size() && g_row_dirty[(size_t)u]) dirty_in_node_vec++;
        fprintf(stderr, "  DIRTY: batch=%ld node_vec=%zu dirty=%ld frac=%.4f\n",
                ++g_dirty_batch_index, node_vec.size(), dirty_in_node_vec,
                node_vec.empty() ? 0.0 : (double)dirty_in_node_vec / (double)node_vec.size());
        if (g_rewire_dirty_rows) {
            // Order-preserving compaction: node_vec is sorted and uniqued above and
            // stays that way, so the rows that survive are visited in exactly the
            // order the full pass would have visited them in.
            size_t keep = 0;
            for (size_t i = 0; i < node_vec.size(); i++) {
                const int u = node_vec[i];
                if (u >= 0 && (size_t)u < g_row_dirty.size() && g_row_dirty[(size_t)u])
                    node_vec[keep++] = u;
            }
            node_vec.resize(keep);
        }
        // Cleared here, not at the end: the edits this pass makes must stay marked
        // for the NEXT pass to see.
        std::fill(g_row_dirty.begin(), g_row_dirty.end(), 0);
    }
    lap(RS_SETUP);
    long added = 0, deleted = 0;
    const int R = (int)g.max_degree;
    ExpansionCostCoverage cost_coverage;
    const bool report_cost_coverage = g_track_expansion_cost && !g_no_cost_coverage;
    // Rides the same flag: every arm that prices also wants to know what lambda
    // would have cost it, and g is meaningful at theta == 0 too (there it is the
    // pure Q difference, so a nonpositive g is UCB churn with no pricing excuse).
    CommitGainCensus gain_census;
    const bool report_commit_gain = g_track_expansion_cost;
    const bool gate_by_margin = (g_rewire_margin_lambda > 0.0);
    const bool lambda_auto = (g_lambda_auto_quantile > 0.0);
    const bool lambda_noise_floor = (g_lambda_noise_floor_quantile > 0.0);
    std::vector<double> boundary_gaps; // Delta_flat, one per saturated pool
    std::vector<double> absolute_gains; // |g|, one per changed row (section 10 step 4)
    long rows_suppressed_by_gate = 0;
    long pq_dist_cache_hits = 0, pq_dist_cache_misses = 0;
    const bool price_by_cost = (g_expansion_cost_theta != 0.0);
    long rows_changed = 0;
    struct Ranked { int v; double score; double dist; double priced_utility; bool from_fallback; };
    // One batched read for all touched nodes' current rows, instead of a fetch
    // per node -- the touched set is known up front, so there is no reason to
    // serialize its I/O.
    std::unordered_map<uint32_t, std::vector<uint32_t>> live;
    live = g.fetch_neighbors(node_vec);
    lap(RS_FETCH);
    auto flush_direct_decision = [&] {
        if (!g_rewire_commit_direct || g.adj_edit.empty()) return;
        for (auto &kv : g.adj_edit) {
            auto lit = live.find(kv.first);
            if (lit != live.end()) lit->second = kv.second;
        }
        g.flush_edits();
        rsel_t0 = Clock::now();
        rsel_io0 = g_io->io_s;
    };

    const size_t scratch_cap = (size_t)(R + 3 * R);
    std::vector<uint32_t> N_out;
    FlatNodeSet N_out_set;
    std::unordered_set<int> C_u;
    std::vector<int> candidates;
    std::vector<double> candidate_cost;
    std::vector<Ranked> ranked;
    FlatNodeSet N_new;
    std::vector<int> to_del, to_add;
    // --lambda-calibrate-first-batch: batch-1 decisions withheld for the second pass.
    struct PendingDecision { int u; std::vector<int> to_add, to_del; double gain; };
    std::vector<PendingDecision> pending_decisions;
    g_admitted_u_rows.clear();
    const bool calibrate_first_batch =
        lambda_noise_floor &&
        (g_lambda_calibrate_every_batch ||
         (g_lambda_calibrate_first_batch && g_rewire_margin_lambda == 0.0));
    FlatNodeSet add_set, del_set;
    N_out.reserve((size_t)R);
    N_out_set.init((size_t)R * 2);
    C_u.reserve(scratch_cap);
    candidates.reserve(scratch_cap);
    candidate_cost.reserve(scratch_cap);
    ranked.reserve(scratch_cap);
    N_new.init((size_t)R * 2);
    to_del.reserve((size_t)R);
    to_add.reserve((size_t)R);
    add_set.init((size_t)R);
    del_set.init((size_t)R);

    for (int u : node_vec) {
        N_out.clear();
        N_out_set.clear();
        C_u.clear();
        candidates.clear();
        candidate_cost.clear();
        ranked.clear();
        N_new.clear();
        to_del.clear();
        to_add.clear();
        add_set.clear();
        del_set.clear();

        // `live` is only a read cache: if u already picked up a reciprocal edge
        // earlier in this same pass, that pending row is the current one.
        auto eit = g.adj_edit.find((uint32_t)u);
        N_out = (eit != g.adj_edit.end()) ? eit->second : live[(uint32_t)u];
        remember_restoration_record(g, (uint32_t)u, N_out);
        // Definition 4.1: a node the certification check (certify_node, run
        // lazily off the approximate candidate log) could never cover even at
        // K_max is flagged -- its full out-list stays protected and learned
        // rewiring is disabled there, so this row is left untouched this batch.
        if (g_edge_protection && g_flagged_nodes.count((uint32_t)u)) continue;
        N_out_set.insert(N_out.begin(), N_out.end());
        lap(RS_ROW);
        // Candidate pool: what u links to now, plus its B_buf (targets it wished
        // it could reach directly). Rewiring never invents a candidate outside
        // what training observed.
        //
        // `entry` is the ONE lookup for u's collocated Q/B_buf row -- it used
        // to be two independent find(u) calls (bit on B_buf, qit on Q) on two
        // separate maps that, per the HotLearnerCache comment above, are
        // never out of sync for a node reached through the normal training
        // path. This is the specific hash-lookup reduction the collocation
        // was done for: do_rewire's row loop is where Q-learn's wall time
        // concentrates (see timeprofile-rewire-select-dominates), and it ran
        // once per row before this change.
        C_u.insert(N_out.begin(), N_out.end());
        HotLearnerCache::Entry *entry = learner_hot.find(u);
        if (entry) for (int w : entry->buf.order) C_u.insert(w);
        // Bound stale distances from candidates that churned out of this row.
        // Clearing loses only cache warmth; PQ codes and every decision remain
        // unchanged. Stable rows retain all entries across rewire batches.
        if (entry && entry->buf.pq_dist.size() > 2 * scratch_cap)
            entry->buf.pq_dist.clear();
        lap(RS_POOL);
        // GC Q entries for edges that fell out of C_u so the maps stay bounded.
        if (entry) entry->q.retain_in(C_u);
        lap(RS_GC);

        // Pass 1: resolve every candidate's cost once. This cannot be folded into
        // the ranking pass below, because the PoolMean policy needs the pool's mean
        // before any single candidate's price can be formed.
        double pool_min_cost = 0.0, pool_max_cost = 0.0, pool_measured_cost_sum = 0.0;
        long pool_measured = 0;
        candidates.assign(C_u.begin(), C_u.end());
        candidate_cost.assign(candidates.size(), -1.0);
        // Nothing downstream reads a cost when theta == 0 and the census is off:
        // the unmeasured-cost policy then maps every entry to the same substitute
        // price, which theta multiplies to zero. Leaving candidate_cost at -1.0 is
        // exactly what Pass 2 already handles for a never-measured candidate.
        const bool cost_is_consumed = price_by_cost || report_cost_coverage;
        for (size_t i = 0; cost_is_consumed && i < candidates.size(); i++) {
            const double cost = lookup_expansion_cost((uint32_t)candidates[i]);
            candidate_cost[i] = cost;
            if (cost < 0.0) continue;
            if (pool_measured == 0) { pool_min_cost = pool_max_cost = cost; }
            else {
                pool_min_cost = std::min(pool_min_cost, cost);
                pool_max_cost = std::max(pool_max_cost, cost);
            }
            pool_measured_cost_sum += cost;
            pool_measured++;
        }
        // With nothing measured the mean is unused in practice: every candidate then
        // takes the same substitute price, so the theta term cancels inside the pool.
        const double pool_mean_cost = pool_measured ? pool_measured_cost_sum / pool_measured : 0.0;
        if (report_cost_coverage) cost_coverage.observe_pool(pool_measured, pool_min_cost, pool_max_cost);
        lap(RS_COST);

        // Pass 2: score and rank.
        std::unique_ptr<DiskANNIndex::PQQuery> uq;
        auto ensure_uq = [&]() -> DiskANNIndex::PQQuery & {
            if (!uq) uq = std::make_unique<DiskANNIndex::PQQuery>(g.node_query((uint32_t)u));
            return *uq;
        };
        // uq.dist(v) gathers v's PQ code from a 0.32 GB array at a pseudo-random
        // offset, so it misses to DRAM once per candidate while the chunk-distance
        // table it indexes into stays in L1. The candidate ids are all known up
        // front, so the misses can be issued ahead of the loop that needs them.
        // A prefetch is a hint with no architectural effect -- it cannot change a
        // distance, a score, or a selection.
        const uint8_t *pq_base = g.pq_codes.data();
        const uint64_t pq_stride = g.n_chunks;
        const size_t PQ_PREFETCH_DIST = 8;
        for (size_t ci = 0; ci < candidates.size(); ci++) {
            const int v = candidates[ci];
            float pq_dist;
            float *cached_dist = entry ? entry->buf.pq_dist.find(v) : nullptr;
            if (cached_dist) {
                pq_dist = *cached_dist;
                pq_dist_cache_hits++;
            } else {
                if (ci + PQ_PREFETCH_DIST < candidates.size())
                    __builtin_prefetch(pq_base + (size_t)candidates[ci + PQ_PREFETCH_DIST] * pq_stride, 0, 1);
                pq_dist = g.pq_codeword_pair_dists.empty()
                              ? ensure_uq().dist((uint32_t)v)
                              : g.codeword_pair_distance((uint32_t)u, (uint32_t)v);
                if (entry) entry->buf.pq_dist.insert(v, pq_dist);
                pq_dist_cache_misses++;
            }
            double q = rewire_edge_prior;
            bool q_from_fallback = true;
            long visit_count_of_candidate = 0;
            if (entry) {
                const EdgeStats *st = entry->q.find(v);
                if (st) {
                    q = st->q;
                    visit_count_of_candidate = st->c;
                    q_from_fallback = false;
                }
            }
            // Incumbency is asked for twice (UCB gate, then the CVCOV census) and
            // N_out_set is an unordered_set probe each time. Resolved once, and
            // only when one of the two consumers is actually on -- at
            // theta > 0 with the census off this probe disappears entirely.
            const bool candidate_is_incumbent =
                (g_ucb_gate_rewire || report_cost_coverage) ? (N_out_set.count(v) != 0)
                                                            : false;
            double score = q;
            if (g_ucb_gate_rewire) {
                double bonus = std::sqrt(2.0 * std::log((double)std::max(gqc, 2L)) /
                                         (double)std::max(visit_count_of_candidate, 1L));
                score = candidate_is_incumbent ? (q + bonus)    // incumbent: rank by UCB
                                               : (q - bonus);   // challenger: rank by LCB
            }
            if (report_cost_coverage)
                cost_coverage.observe_candidate(candidate_is_incumbent, candidate_cost[ci]);
            double priced_utility;
            if (price_by_cost) {
                double resolved_cost = candidate_cost[ci];
                if (resolved_cost < 0.0) {
                    switch (g_unmeasured_cost_policy) {
                        case UnmeasuredCostPolicy::Free: resolved_cost = 0.0; break;
                        case UnmeasuredCostPolicy::Full: resolved_cost = 1.0; break;
                        default: resolved_cost = pool_mean_cost; break;
                    }
                }
                // eq. (6). Left strictly untouched at theta == 0 so the control arm is
                // the pre-pricing code path, not a re-derivation of it.
                score -= g_expansion_cost_theta * resolved_cost;
                // Algorithm 3 line 5's operand: priced utility WITHOUT the UCB bonus.
                priced_utility = q - g_expansion_cost_theta * resolved_cost;
            } else {
                priced_utility = q;
            }
            ranked.push_back({v, score, (double)pq_dist, priced_utility, q_from_fallback});
        }
        lap(RS_SCORE);
        // Primary key is the (possibly confidence-adjusted) score; ties broken by
        // geometric closeness.
        std::sort(ranked.begin(), ranked.end(), [](const Ranked &a, const Ranked &b) {
            if (a.score != b.score) return a.score > b.score;
            return a.dist < b.dist;
        });
        // Delta_flat at u. Unsaturated pools are skipped: an empty slot has no
        // boundary, and filling it is free (implementation note (iii)).
        if (lambda_auto && (int)ranked.size() > R)
            boundary_gaps.push_back(ranked[R - 1].priced_utility - ranked[R].priced_utility);
        lap(RS_SORT);
        // Live-referenced length-tiered protection: rank u's out-edges as they
        // stand entering this rewire (N_out, before any of this pass's own
        // edits) longest-first, and pin the top k_p unconditionally, ahead of
        // anything admitted by score. Recomputed fresh every rewire, so a long
        // edge the learner added last round is just as eligible for
        // protection as one from the original build -- see the section header
        // for why that distinction is the whole point of this version.
        if (g_edge_protection) {
            DiskANNIndex::PQQuery &uq_ref = ensure_uq();
            std::vector<uint32_t> lrank = live_rank_of(g, (uint32_t)u, N_out, &uq_ref);
            std::vector<uint32_t> mask;
            for (int i = 0; i < g_governor.k_p && i < (int)lrank.size() && (int)N_new.size() < R; i++) {
                N_new.insert((int)lrank[i]);
                mask.push_back(lrank[i]);
            }
            // Definition 4.1: S(u) = P_kp(u) union pin. Edges Algorithm 5's log
            // rescue has recruited --protect-pin-repeat times are retained
            // unconditionally, independent of k_p and of this rewire's scoring.
            auto pin_it = g_pinned_edges.find((uint32_t)u);
            if (pin_it != g_pinned_edges.end()) {
                for (uint32_t v : pin_it->second) {
                    if ((int)N_new.size() >= R) break;
                    if (!N_new.count((int)v)) {
                        N_new.insert((int)v);
                        mask.push_back(v);
                    }
                }
            }
            if (g_dump_protect_mask && !mask.empty()) {
                // Score-only top-R: what selection would have kept with no protection.
                std::unordered_set<int> score_top;
                for (int i = 0; i < (int)ranked.size() && (int)score_top.size() < R; i++)
                    score_top.insert(ranked[i].v);
                auto oit = g_orig_row.find(u);
                if (oit == g_orig_row.end())
                    oit = g_orig_row.emplace(u, std::unordered_set<uint32_t>(N_out.begin(), N_out.end())).first;
                const DiskANNIndex::PQQuery &uq_len = uq_ref; // same u, same pure function
                double pinned_len = 0.0, row_len = 0.0;
                for (uint32_t v : mask) {
                    pinned_len += uq_len.dist(v);
                    if (!score_top.count((int)v)) g_pm_bite += 1;
                    if (!oit->second.count(v)) g_pm_fresh += 1;
                }
                for (uint32_t v : N_out) row_len += uq_len.dist(v);
                if (!N_out.empty() && row_len > 0.0) {
                    g_pm_len_ratio_sum += (pinned_len / (double)mask.size()) / (row_len / (double)N_out.size());
                    g_pm_len_n += 1;
                }
                auto pit = g_prev_mask.find(u);
                if (pit != g_prev_mask.end()) {
                    std::unordered_set<uint32_t> prev(pit->second.begin(), pit->second.end());
                    int inter = 0;
                    for (uint32_t v : mask) if (prev.count(v)) inter++;
                    const size_t uni = prev.size() + mask.size() - (size_t)inter;
                    if (uni) { g_pm_churn_sum += (double)inter / (double)uni; g_pm_churn_n += 1; }
                }
                g_prev_mask[u] = mask;
                g_pm_rewires += 1;
                g_pm_pinned += (long)mask.size();
            }
        } else if (g_pct_protect) {
            // Comparison arm: pin a fixed g_pct_protect_frac of u's CURRENT live
            // out-edges (longest or shortest, live_rank_of sorts longest-first)
            // unconditionally, ahead of anything admitted by Q-rank below.
            DiskANNIndex::PQQuery &uq_ref = ensure_uq();
            std::vector<uint32_t> lrank = live_rank_of(g, (uint32_t)u, N_out, &uq_ref);
            int d = (int)lrank.size();
            int n_protect = (int)std::ceil(g_pct_protect_frac * (double)d);
            if (n_protect > d) n_protect = d;
            if (g_pct_protect_longest) {
                for (int i = 0; i < n_protect && (int)N_new.size() < R; i++) N_new.insert((int)lrank[i]);
            } else {
                for (int i = d - n_protect; i < d && (int)N_new.size() < R; i++) N_new.insert((int)lrank[i]);
            }
        } else if (g_skeleton_protect) {
            // Comparison arm: pin whichever of u's skeleton edges (extract_skeleton,
            // run once before training) are still present in its current row. An
            // already-evicted skeleton edge is not restored -- see the section header.
            auto sit = g_skel_members.find((uint32_t)u);
            if (sit != g_skel_members.end())
                for (uint32_t v : sit->second)
                    if (N_out_set.count((int)v) && (int)N_new.size() < R) N_new.insert((int)v);
        }
        lap(RS_PROTECT);
        for (int i = 0; i < (int)ranked.size() && (int)N_new.size() < R; i++) N_new.insert(ranked[i].v);
        lap(RS_SELECT);
        // Diff old vs new into concrete add/remove ops.
        for (uint32_t v : N_out) if (!N_new.count((int)v)) to_del.push_back((int)v);
        for (auto &r : ranked) if (N_new.count(r.v) && !N_out_set.count(r.v)) to_add.push_back(r.v);
        lap(RS_DIFF);
        const bool selection_changed = (!to_del.empty() || !to_add.empty());
        // Algorithm 3 line 5. Needed whenever the census reports it OR the gate acts on it.
        double commit_gain = 0.0;
        double sum_new = 0.0, sum_old = 0.0;   // the two operands, kept apart
        // How much of each operand carried no Q entry and was priced at the
        // fallback prior. A row where both sides are entirely fallback-priced
        // cancels to g == 0 whatever the prior's VALUE is, so this is what
        // separates residual inert mass from mass the new prior could still move.
        int n_add_fallback = 0, n_del_fallback = 0;
        bool gated_out = false;
        if (selection_changed &&
            (report_commit_gain || gate_by_margin || lambda_noise_floor || g_gain_terms_file)) {
            add_set.insert(to_add.begin(), to_add.end());
            del_set.insert(to_del.begin(), to_del.end());
            for (const Ranked &r : ranked) {
                if (add_set.count(r.v)) { sum_new += r.priced_utility; n_add_fallback += r.from_fallback; }
                else if (del_set.count(r.v)) { sum_old += r.priced_utility; n_del_fallback += r.from_fallback; }
            }
            commit_gain = sum_new - sum_old;
        }
        // Observed BEFORE the gate on purpose: the census must keep describing the
        // full distribution of proposed changes, so the ladder stays comparable
        // across arms with different lambda.
        if (report_commit_gain && selection_changed) gain_census.observe(commit_gain);
        // Same reason, and the reason this is the pre-gate site: the noise floor
        // must describe every proposed row, or it would be a quantile of the rows
        // the previous lambda already let through.
        if (lambda_noise_floor && selection_changed) absolute_gains.push_back(std::fabs(commit_gain));
        // Written BEFORE the gate, like the census, so the dump describes the full
        // distribution of proposed changes rather than the survivors.
        if (g_gain_terms_file && selection_changed)
            fprintf(g_gain_terms_file, "%ld,%d,%d,%d,%zu,%zu,%.17g,%.17g,%.17g,%d,%d\n",
                    gqc, u, (int)N_out.size(), R, to_add.size(), to_del.size(),
                    sum_new, sum_old, commit_gain, n_add_fallback, n_del_fallback);
        // --lambda-calibrate-first-batch: withhold, so pass 2 can gate on a
        // lambda derived from this batch's own |g| distribution.
        if (calibrate_first_batch && selection_changed) {
            pending_decisions.push_back({u, to_add, to_del, commit_gain});
            gated_out = true;
        }
        // Algorithm 3 line 6.  Skipped while calibrating: pass 2 is the gate, and
        // letting the standing (stale) lambda also filter pass 1 would apply two
        // different thresholds to the same batch.
        if (!calibrate_first_batch && selection_changed && gate_by_margin &&
            !(commit_gain > g_rewire_margin_lambda)) {
            rows_suppressed_by_gate++;
            gated_out = true;
        }
        // --protect-warmup-freeze: suppress every commit until query N. All the
        // selection work above still runs (so estimates and the noise-floor
        // sample are unaffected), only the disk edit is withheld.
        if (g_protect_warmup_freeze > 0 && gqc <= g_protect_warmup_freeze) gated_out = true;
        lap(RS_GAIN_GATE);
        if (gated_out) continue; // no edits enqueued for u -- its row is never rewritten
        if (selection_changed) { rows_changed++; mark_row_rewired((uint32_t)u, g.num_points); }
        g_admitted_u_rows.insert((uint32_t)u);
        if (g_rewire_commit_direct) {
            std::vector<uint32_t> &u_live = live[(uint32_t)u];
            for (int v : to_del) {
                g.remove_edge((uint32_t)u, (uint32_t)v, u_live);
                deleted++;
                flush_direct_decision();
            }
            for (int v : to_add) {
                if (g_rewire_unidirectional) {
                    // No reciprocal row to read, cap, or write.
                    g.add_directed((uint32_t)u, (uint32_t)v, u_live);
                } else {
                    // The reciprocal endpoint's row has to be read before it can be capped.
                    std::vector<uint32_t> v_nbrs = g.get_neighbors((uint32_t)v);
                    remember_restoration_record(g, (uint32_t)v, v_nbrs);
                    g.add_edge_bidirectional((uint32_t)u, (uint32_t)v, u_live, v_nbrs);
                }
                added++;
                flush_direct_decision();
            }
        } else {
            for (int v : to_del) { g.remove_edge((uint32_t)u, (uint32_t)v, N_out); deleted++; mark_row_dirty(u); mark_row_dirty(v); }
            for (int v : to_add) {
                if (g_rewire_unidirectional) {
                    g.add_directed((uint32_t)u, (uint32_t)v, N_out);
                } else {
                    // The reciprocal endpoint's row has to be read before it can be capped.
                    std::vector<uint32_t> v_nbrs = g.get_neighbors((uint32_t)v);
                    remember_restoration_record(g, (uint32_t)v, v_nbrs);
                    g.add_edge_bidirectional((uint32_t)u, (uint32_t)v, N_out, v_nbrs);
                }
                added++;
                // The reciprocal endpoint's row is mutated too, so it is dirty
                // even though nothing wrote its Q or B_buf.
                mark_row_dirty(u); mark_row_dirty(v);
            }
        }
        lap(RS_EDIT);
        // --early-rewire-commit: commit this node's edits now, while the sectors
        // do_rewire read for it are still resident, rather than accumulating the
        // whole pass. flush_edits then finds them in C_blk and takes the overlay
        // path, so no RMW pread is issued.
        //
        // `live` must be refreshed BEFORE the flush. The top of this loop treats
        // adj_edit as authoritative for a node that already picked up a
        // reciprocal edge earlier in this pass, falling back to `live` only when
        // adj_edit has nothing -- and flush_edits clears adj_edit. Without this,
        // a node edited as some earlier node's reciprocal would be re-read from a
        // stale `live` row and that edge would be silently dropped. Only keys
        // already in `live` are refreshed: `live` is read only for u in node_vec,
        // and fetch_neighbors gave it every one of those.
        if (g_early_rewire_commit && !g.adj_edit.empty()) {
            for (auto &kv : g.adj_edit) {
                auto lit = live.find(kv.first);
                if (lit != live.end()) lit->second = kv.second;
            }
            g.flush_edits();
            rsel_t0 = Clock::now(); // flush_edits is rcommit, not rsel
            rsel_io0 = g_io->io_s;
        }
    }
    // --lambda-calibrate-first-batch, pass 2: derive lambda from the batch that
    // just ran, then commit only the decisions that clear it.
    if (calibrate_first_batch && !pending_decisions.empty()) {
        std::vector<double> sample;
        sample.reserve(pending_decisions.size());
        for (const auto &decision : pending_decisions) sample.push_back(std::fabs(decision.gain));
        std::sort(sample.begin(), sample.end());
        const size_t n = sample.size();
        const size_t k = (size_t)(g_lambda_noise_floor_quantile * (double)(n - 1));
        const double calibrated_lambda = place_strictly_above(sample, k);
        g_rewire_margin_lambda = calibrated_lambda;   // carried into round 2
        long committed_rows = 0;
        for (const auto &decision : pending_decisions) {
            if (!(std::fabs(decision.gain) > calibrated_lambda)) continue;
            const uint32_t u_id = (uint32_t)decision.u;
            auto edit_it = g.adj_edit.find(u_id);
            std::vector<uint32_t> u_row =
                (edit_it != g.adj_edit.end()) ? edit_it->second : live[u_id];
            remember_restoration_record(g, u_id, u_row);
            for (int v : decision.to_del) { g.remove_edge(u_id, (uint32_t)v, u_row); deleted++; }
            for (int v : decision.to_add) {
                if (g_rewire_unidirectional) {
                    g.add_directed(u_id, (uint32_t)v, u_row);
                } else {
                    std::vector<uint32_t> v_nbrs = g.get_neighbors((uint32_t)v);
                    remember_restoration_record(g, (uint32_t)v, v_nbrs);
                    g.add_edge_bidirectional(u_id, (uint32_t)v, u_row, v_nbrs);
                }
                added++;
            }
            committed_rows++;
            rows_changed++;
            mark_row_rewired(u_id, g.num_points);
            g_admitted_u_rows.insert(u_id);
        }
        fprintf(stderr, "  LAMBDACAL: batch1 two-pass rows=%zu lambda=%.6g q=%.2f "
                        "committed=%ld suppressed=%zu realised_veto=%.4f\n",
                n, calibrated_lambda, g_lambda_noise_floor_quantile, committed_rows,
                n - (size_t)committed_rows,
                (double)(n - (size_t)committed_rows) / (double)n);
        lap(RS_EDIT);
    }
    long sector_writes_before = g_io->writes;
    lap(RS_REPORT);
    g.flush_edits();
    rsel_t0 = Clock::now(); // flush_edits is rcommit, not rsel
    rsel_io0 = g_io->io_s;
    g.edges_added += added;
    g.edges_deleted += deleted;
    // Q_nodes and B_buf_nodes are now the same collocated map's size (see the
    // HotLearnerCache comment above) -- kept as two fields, both reporting
    // learner_hot.tab.size(), so any existing log-parsing script that reads
    // this line by column position keeps working unchanged.
    fprintf(stderr, "  rewire: Q_nodes=%zu B_buf_nodes=%zu gated_in=%zu added=%ld deleted=%ld\n",
            learner_hot.tab.size(), learner_hot.tab.size(), node_vec.size(), added, deleted);
    fprintf(stderr, "  PQDISTCACHE: hits=%ld misses=%ld hit_frac=%.4f\n",
            pq_dist_cache_hits, pq_dist_cache_misses,
            (pq_dist_cache_hits + pq_dist_cache_misses) == 0
                ? 0.0
                : (double)pq_dist_cache_hits / (double)(pq_dist_cache_hits + pq_dist_cache_misses));
    // Write volume is the constraint theta is most likely to move in the wrong
    // direction (every reordered row is a read-modify-write), so it is reported
    // beside the ranking effect rather than left to the end-of-run totals.
    if (price_by_cost || report_cost_coverage)
        fprintf(stderr, "  THETA: theta=%.6g unmeasured=%s rows_changed=%ld/%zu added=%ld deleted=%ld"
                        " sector_writes=%ld\n",
                g_expansion_cost_theta, unmeasured_cost_policy_name(), rows_changed, node_vec.size(),
                added, deleted, g_io->writes - sector_writes_before);
    fprintf(stderr, "  COMMITSPLIT: direct=%d flush_calls=%ld flush_rows=%ld rows_per_call=%.3f\n",
            g_rewire_commit_direct ? 1 : 0, g_commit_flush_calls, g_commit_flush_rows,
            g_commit_flush_calls == 0 ? 0.0 : (double)g_commit_flush_rows / (double)g_commit_flush_calls);
    if (gate_by_margin)
        fprintf(stderr, "  LAMBDAGATE: lambda=%.6g rows_suppressed=%ld committed=%ld realised_veto=%.4f\n",
                g_rewire_margin_lambda, rows_suppressed_by_gate, rows_changed,
                (rows_suppressed_by_gate + rows_changed) > 0
                    ? (double)rows_suppressed_by_gate / (double)(rows_suppressed_by_gate + rows_changed)
                    : 0.0);
    lap(RS_REPORT);
    // Sets the NEXT batch's lambda. Negative margins are UCB reordering the
    // boundary against priced utility, so they are reported rather than dropped;
    // a quantile at or below zero simply leaves the gate disarmed.
    if (lambda_auto && !boundary_gaps.empty()) {
        std::sort(boundary_gaps.begin(), boundary_gaps.end());
        const size_t n = boundary_gaps.size();
        const double q = boundary_gaps[(size_t)(g_lambda_auto_quantile * (double)(n - 1))];
        const double proposed = g_lambda_auto_scale * q;
        const double previous = g_rewire_margin_lambda;
        if (proposed > 0.0)
            g_rewire_margin_lambda = (g_lambda_auto_ewma > 0.0 && previous > 0.0)
                ? (1.0 - g_lambda_auto_ewma) * previous + g_lambda_auto_ewma * proposed
                : proposed;
        const size_t negative = (size_t)(std::lower_bound(boundary_gaps.begin(), boundary_gaps.end(), 0.0) -
                                         boundary_gaps.begin());
        const size_t nonpositive = (size_t)(std::upper_bound(boundary_gaps.begin(), boundary_gaps.end(), 0.0) -
                                            boundary_gaps.begin());
        const double nu_bar = g_io_price_controller.last_nu_bar;
        fprintf(stderr, "  LAMBDAAUTO: gaps=%zu negative=%.4f zero=%.4f q%.2f=%.6g "
                        "lambda=%.6g->%.6g deficit_bound=%.6g%s\n",
                n, (double)negative / (double)n, (double)(nonpositive - negative) / (double)n,
                g_lambda_auto_quantile, q, previous, g_rewire_margin_lambda,
                g_rewire_margin_lambda * nu_bar, nu_bar > 0.0 ? "" : " [nu_bar unmeasured]");
    }
    // Section 10 step (4)'s rule, the same lag-1 structure on a different sample:
    // lambda is the q-th percentile of |g| over THIS batch's changed rows, placed
    // strictly above that value so the veto it buys is the one it names.
    if (lambda_noise_floor && !absolute_gains.empty()) {
        std::sort(absolute_gains.begin(), absolute_gains.end());
        const size_t n = absolute_gains.size();
        const size_t k = (size_t)(g_lambda_noise_floor_quantile * (double)(n - 1));
        const double q = absolute_gains[k];
        const double proposed = place_strictly_above(absolute_gains, k);
        const double previous = g_rewire_margin_lambda;
        // The EWMA can pull lambda BELOW the floor the sample just measured, which
        // is the guarantee this rule exists to buy. It is off by default for that
        // reason; anyone turning it on is trading the floor for stability.
        if (proposed > 0.0)
            g_rewire_margin_lambda = (g_lambda_auto_ewma > 0.0 && previous > 0.0)
                ? (1.0 - g_lambda_auto_ewma) * previous + g_lambda_auto_ewma * proposed
                : proposed;
        size_t distinct = 0;
        for (size_t i = 0; i < n; i++)
            if (i == 0 || absolute_gains[i] != absolute_gains[i - 1]) distinct++;
        const size_t zero = (size_t)(std::upper_bound(absolute_gains.begin(), absolute_gains.end(), 0.0) -
                                     absolute_gains.begin());
        // What the PROPOSAL would veto, not what the applied lambda will: under an
        // EWMA the two differ, and it is the proposal that carries the rule's claim.
        const size_t at_or_below = (size_t)(std::upper_bound(absolute_gains.begin(), absolute_gains.end(), q) -
                                            absolute_gains.begin());
        const double nu_bar_noise = g_io_price_controller.last_nu_bar;
        fprintf(stderr, "  LAMBDANOISE: rows=%zu distinct=%zu zero=%.4f q%.2f=%.6g proposed=%.6g "
                        "lambda=%.6g->%.6g veto_at_proposed=%.4f deficit_bound=%.6g%s\n",
                n, distinct, (double)zero / (double)n, g_lambda_noise_floor_quantile, q, proposed,
                previous, g_rewire_margin_lambda, (double)at_or_below / (double)n,
                g_rewire_margin_lambda * nu_bar_noise,
                nu_bar_noise > 0.0 ? "" : " [nu_bar unmeasured]");
    }
    lap(RS_LAMBDA);
    if (g_recall_prior)
        fprintf(stderr, "  PRIOR: recall_prior=%.6g (Q_0=%.6g, ratio=%.4g)\n",
                rewire_edge_prior, Q_0, Q_0 > 0.0 ? rewire_edge_prior / Q_0 : 0.0);
    if (report_commit_gain) gain_census.report();
    if (report_cost_coverage) cost_coverage.print();
    g_last_rows_changed = rows_changed;
    g_last_hot_nodes = (long)node_vec.size();
    g_last_deleted = deleted;
    if (g_first_batch_deleted == 0 && deleted > 0) g_first_batch_deleted = deleted;
    lap(RS_REPORT);
}

static long layout_step(DiskANNIndex &g, const CoMap &co, double lambda_pi) {
    // kappa < 2: at most one record per sector, so co-location is impossible by
    // construction and every delta is identically zero. True for GIST (kappa=1).
    if (g.nnodes_per_sector < 2) {
        fprintf(stderr, "  layout: skipped, nnodes_per_sector=%llu (<2)\n",
                (unsigned long long)g.nnodes_per_sector);
        return 0;
    }
    // Once per pass, not once per swap_slots(): drains whatever the PRECEDING
    // do_rewire/flush_edits phase queued, so no older async commit can trail a
    // relocation this pass makes. Swaps within the pass need no drain between
    // them -- resident halves are patched through the same buffer + dirty bit
    // the writer already resolves via version/resident (see swap_slots).
    g.hotcache.wait_idle();
    long swaps = 0;
    // How often the null-co-row skip actually fires, and how many pairs reached
    // the delta scan. Without this the skip's value is a guess.
    long null_skips = 0, pairs_seen = 0;
    // --layout-baff-cap accounting: how many u had their partner set truncated,
    // and how many partners survived in total (== pairs_seen when the cap is off).
    long capped_u = 0, partners_kept = 0;
    std::vector<std::pair<long, uint32_t>> cand;
    std::vector<uint32_t> block_u, block_v;
    std::vector<const CoRow *> rows_u;
    // w = argmin over pi(u) of that occupant's total co-visitation weight to its
    // own blockmates -- the resident we lose the least by evicting. u itself is
    // eligible; Algorithm 4 does not exclude it. Depends only on block_u/rows_u,
    // never on v, so it is computed here rather than inside the v-loop below.
    auto select_least_affine = [](const std::vector<uint32_t> &block,
                                  const std::vector<const CoRow *> &rows) {
        uint32_t w = block[0];
        double least_star = std::numeric_limits<double>::max();
        for (size_t ci = 0; ci < block.size(); ci++) {
            double star = 0.0;
            // co_at() is a monotonically-nondecreasing accumulator (log_covisitation
            // only ever increments it), so once the running partial sum reaches
            // least_star this candidate cannot win -- bail without summing the rest.
            for (uint32_t x : block) {
                if (x == block[ci]) continue;
                star += (double)co_at(rows[ci], (int)x);
                if (star >= least_star) break;
            }
            if (star < least_star) { least_star = star; w = block[ci]; }
        }
        return w;
    };
    for (const auto &kv : co) {
        const uint32_t u = (uint32_t)kv.first;
        // block_u/rows_u/w describe u's CURRENT block. Computed once per u and
        // only refreshed when a swap earlier in this u's own v-loop actually
        // changed that block -- swap_slots(v, w) is the only thing that can, and
        // only for the u whose swap it was, so this reproduces exactly what
        // recomputing on every (u,v) pair did, at a fraction of the co_at calls.
        g.block_of(u, block_u);
        rows_u.clear();
        for (uint32_t cand : block_u) rows_u.push_back(co_row(co, (int)cand));
        uint32_t w = select_least_affine(block_u, rows_u);
        // Hoisted with w: co_row(w) was being re-resolved on every (u,v) pair
        // although w changes only when a swap dirties u's block, which is ~640k
        // redundant hash lookups in the first pass alone. Bit-exact.
        const CoRow *row_w = co_row(co, (int)w);
        bool block_dirty = false;
        // Candidate partners of u, capped to the M highest counts when
        // --layout-baff-cap is set. Reused across u to keep this off the
        // allocator: layout_step runs this loop ~11k times in the first pass.
        cand.clear();
        cand.reserve(kv.second.size());
        for (const auto &pv : kv.second) cand.emplace_back((long)pv.second, (uint32_t)pv.first);
        if (g_layout_baff_cap > 0 && (int)cand.size() > g_layout_baff_cap) {
            // count descending, id ascending -- a total order, so the selection
            // does not depend on hash iteration order.
            auto by_count = [](const std::pair<long, uint32_t> &a,
                               const std::pair<long, uint32_t> &b) {
                if (a.first != b.first) return a.first > b.first;
                return a.second < b.second;
            };
            std::nth_element(cand.begin(), cand.begin() + g_layout_baff_cap, cand.end(), by_count);
            cand.resize((size_t)g_layout_baff_cap);
            std::sort(cand.begin(), cand.end(),
                      [](const std::pair<long, uint32_t> &a, const std::pair<long, uint32_t> &b) {
                          return a.second < b.second;
                      });
            capped_u++;
        }
        partners_kept += (long)cand.size();
        for (const auto &pv : cand) {
            const uint32_t v = pv.second;
            // pi(v) is re-read every iteration: a swap made earlier in this same
            // pass may have moved it (including, now, v itself if it was some
            // earlier partner's w -- block_of(v) still resolves through reloc).
            pairs_seen++;
            if (g.node_sector(v) == g.node_sector(u)) continue; // already co-located
            if (block_dirty) {
                g.block_of(u, block_u);
                rows_u.clear();
                for (uint32_t cand : block_u) rows_u.push_back(co_row(co, (int)cand));
                w = select_least_affine(block_u, rows_u);
                row_w = co_row(co, (int)w);
                block_dirty = false;
            }

            // delta = change in total within-block co-visitation weight if v and w
            // trade slots: v's new blockmates are pi(u)\{w}, w's are pi(v)\{v}.
            const CoRow *row_v = co_row(co, (int)v);
            // Neither endpoint has a co-visitation row, so every co_at() below
            // returns 0 and delta is exactly 0 -- it cannot clear a nonnegative
            // lambda_pi. Skipping here avoids block_of(v) and 2*kappa co_at()
            // calls on a pair that was never going to swap. Guarded on the sign
            // of lambda_pi because a NEGATIVE threshold does admit delta == 0.
            if (row_v == nullptr && row_w == nullptr && lambda_pi >= 0.0) { null_skips++; continue; }
            g.block_of(v, block_v);
            double delta = 0.0;
            for (uint32_t x : block_u)
                if (x != w) delta += (double)(co_at(row_v, (int)x) - co_at(row_w, (int)x));
            for (uint32_t y : block_v)
                if (y != v) delta += (double)(co_at(row_w, (int)y) - co_at(row_v, (int)y));

            if (delta > lambda_pi) {
                g.swap_slots(v, w);
                swaps++;
                block_dirty = true; // u's block changed: w left, v entered
            }
        }
    }
    // One write per dirty sector for the whole pass, not two per swap.
    const size_t staged_sectors = g.layout_stage_size();
    g.flush_layout_stage();
    if (staged_sectors > 0)
        fprintf(stderr, "  LAYOUTCOMMIT: batched=1 swaps=%ld halves=%ld staged_sectors=%zu "
                        "coalescing=%.3f\n",
                swaps, 2 * swaps, staged_sectors,
                staged_sectors > 0 ? (double)(2 * swaps) / (double)staged_sectors : 0.0);
    fprintf(stderr, "  LAYOUTSKIP: pairs_seen=%ld null_skips=%ld frac=%.4f scanned=%ld\n",
            pairs_seen, null_skips,
            pairs_seen > 0 ? (double)null_skips / (double)pairs_seen : 0.0,
            pairs_seen - null_skips);
    if (g_layout_baff_cap > 0)
        fprintf(stderr, "  LAYOUTCAP: m=%d capped_u=%ld partners_kept=%ld\n",
                g_layout_baff_cap, capped_u, partners_kept);
    fprintf(stderr,
            "  layout: covisit_scope=%s hot_nodes=%zu pairs=%ld swaps=%ld "
            "relocated=%zu epoch_misses=%zu\n",
            g_covisit_scope_expanded ? "expanded" : "miss",
            co.size(), g_covisit_pairs, swaps, g.reloc.size(),
            g.epoch_missed.size());
    return swaps;
}

static void audit_log_rescue(DiskANNIndex &g, const std::vector<uint32_t> &U,
                             const std::unordered_map<uint32_t, std::vector<uint32_t>> &live_rows,
                             const std::vector<std::vector<float>> &U_coord,
                             const std::vector<float> &aug_y, DiskANNIndex::PQQuery &yq,
                             uint64_t disk_beam_width, HotLearnerCache &learner_hot, int buffer_cap) {
    for (size_t ui = 0; ui < U.size(); ui++) {
        uint32_t u = U[ui];
        // Parity with the reserve test: only nodes remember_restoration_record
        // already accepted (i.e. not counted under g_audit_nobuf) are eligible.
        if (!g_restoration_record.count(u)) continue;
        auto lit = live_rows.find(u);
        if (lit == live_rows.end()) continue;
        const std::vector<uint32_t> &log = get_candidate_log(g, u, disk_beam_width);
        if (log.empty()) continue;
        std::unordered_set<uint32_t> live_set(lit->second.begin(), lit->second.end());
        const float dist_u_y = DiskANNIndex::exact_l2(aug_y.data(), U_coord[ui].data(), g.data_dim);
        const float threshold = dist_u_y / (float)g_alpha - (float)g_m_pq;
        std::vector<std::pair<float, uint32_t>> candidates;
        for (uint32_t v : log) {
            if (live_set.count(v)) continue;
            const float d = yq.dist(v);
            if (d <= threshold && repair_is_admissible(u, v) && repair_has_effect(dist_u_y, d))
                candidates.emplace_back(d, v);
        }
        if (candidates.empty()) continue;
        std::sort(candidates.begin(), candidates.end());
        const size_t n = std::min(candidates.size(), (size_t)g_log_rescue_max);

        for (size_t i = 0; i < n; i++) {
            const uint32_t best = candidates[i].second;
            g_audit_log_rescue++;
            g_repair_volume++;
            if (learner_hot.contains((int)u))
                if (learner_hot.lru((int)u).touch((int)best, buffer_cap)) mark_row_dirty((int)u);
            if (g_protect_direct_repair) queue_audit_direct_repair(u, best);
            const int rep = ++g_rescue_repeats[u][best];
            if (rep >= g_pin_repeat) g_pinned_edges[u].insert(best);
        }
        if (!g_log_rescue_all_path) return; // one rescuable node per audit
    }
}

static bool audit_search_rescue(DiskANNIndex &g, const std::vector<uint32_t> &U,
                                const std::unordered_map<uint32_t, std::vector<uint32_t>> &live_rows,
                                const std::vector<std::vector<float>> &U_coord,
                                const float *y_vec, const std::vector<float> &aug_y,
                                DiskANNIndex::PQQuery &yq, uint64_t disk_beam_width,
                                HotLearnerCache &learner_hot, int buffer_cap) {
    bool proposed = false;
    for (size_t ui = 0; ui < U.size(); ui++) {
        const uint32_t u = U[ui];
        const float threshold = DiskANNIndex::exact_l2(aug_y.data(), U_coord[ui].data(), g.data_dim) /
                                (float)g_alpha - (float)g_m_pq;
        const float dist_u_y = DiskANNIndex::exact_l2(aug_y.data(), U_coord[ui].data(), g.data_dim);
        std::vector<int> result_ids, visited, expanded;
        g_search_ep_override = (int)u;
        g.search(y_vec, /*K=*/1, /*ef=*/50, disk_beam_width, result_ids, visited, &expanded);
        g_search_ep_override = -1;
        std::vector<std::pair<float, uint32_t>> candidates;
        std::unordered_set<uint32_t> seen(live_rows.at(u).begin(), live_rows.at(u).end());
        for (int id : expanded) {
            const uint32_t v = (uint32_t)id;
            if (v != u && seen.insert(v).second) candidates.emplace_back(yq.dist(v), v);
        }
        std::sort(candidates.begin(), candidates.end());
        int taken = 0;
        for (const auto &[d, v] : candidates) {
            if (d > threshold || taken == 10) break;
            if (!repair_is_admissible(u, v) || !repair_has_effect(dist_u_y, d)) continue;
            proposed = true;
            g_audit_search_proposals++;
            g_repair_volume++;
            if (!g_audit_passive && learner_hot.contains((int)u))
                if (learner_hot.lru((int)u).touch((int)v, buffer_cap)) mark_row_dirty((int)u);
            if (!g_audit_passive && g_protect_direct_repair) queue_audit_direct_repair(u, v);
            taken++;
        }
    }
    return proposed;
}

struct AuditCostPart {
    double s = 0, io_s = 0; long dev_reads = 0, dev_writes = 0, calls = 0;
};
static AuditCostPart g_ac_search, g_ac_fix, g_ac_apply, g_ac_prelog;
struct AuditCostScope {
    AuditCostPart &part; const IOCounters &io; double t0, io0; long r0, w0;
    AuditCostScope(AuditCostPart &p, const IOCounters &c)
        : part(p), io(c), t0(now_s()), io0(c.io_s), r0(c.device_reads), w0(c.device_writes) {}
    ~AuditCostScope() {
        part.s += now_s() - t0; part.io_s += io.io_s - io0;
        part.dev_reads += io.device_reads - r0; part.dev_writes += io.device_writes - w0; part.calls++;
    }
    static double now_s() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
};

static void audit_step(DiskANNIndex &g, HotLearnerCache &learner_hot, int buffer_cap, uint64_t ef,
                       uint64_t disk_beam_width) {
    IOPhase ph(g_io_audit);
    // Whole audit_step lands in g_ac_fix; the search part is moved out of it below.
    AuditCostScope whole_cost(g_ac_fix, g_io_audit);
    auto search_cost = std::make_unique<AuditCostScope>(g_ac_search, g_io_audit);
    uint32_t y = g_audit_target_ids.empty()
                     ? (uint32_t)std::min<double>((double)g.num_points - 1, g_unif01(g_nav_rng) * (double)g.num_points)
                     : g_audit_target_ids[g_audit_target_ids_trial_count++ % g_audit_target_ids.size()];

    std::vector<std::vector<float>> ycoord;
    g.read_nodes({y}, &ycoord, nullptr);
    const float *y_vec = ycoord[0].data();

    g_audit_current_y = (int64_t)y;
    std::vector<int> result_ids, visited, expanded;
    const uint64_t audit_ef = g_audit_ef > 0 ? (uint64_t)g_audit_ef : ef;
    if (g_audit_ep_last_result) g_search_ep_override = g_last_serving_ep;
    g.search(y_vec, /*K=*/1, audit_ef, disk_beam_width, result_ids, visited, &expanded);
    g_search_ep_override = -1;
    search_cost.reset();

    g_audit_dump_gqc++;
    auto dump = [&](const char *outcome) {
        if (!g_audit_dump_f) return;
        fprintf(g_audit_dump_f, "%ld,%u,%s\n", g_audit_dump_gqc, y, outcome);
    };

    bool reached = false;
    for (int u : expanded) {
        if (u == (int)y || g.node_sector((uint32_t)u) == g.node_sector(y)) {
            reached = true;
            break;
        }
    }
    if (reached) { g_governor.window.pass++; g_audit_pass++; dump("pass"); return; }
    if (expanded.empty()) { g_audit_degenerate++; dump("degenerate"); return; } // nothing expanded at all
    g_audit_failed_targets.insert(y);
    if (g_protect_repair_path_to_y && g_protect_direct_repair && !g_audit_passive) {
        for (int u : expanded)
            if (u != (int)y) g_direct_repair_queue.emplace_back((uint32_t)u, y);
        if (g_protect_direct_repair_retest) g_direct_repair_retest_targets.emplace_back(g_audit_dump_gqc, y);
    }

    std::vector<float> aug_y = g.augment_query(y_vec);
    DiskANNIndex::PQQuery yq = g.prepare_query(aug_y.data());
    std::vector<uint32_t> U;
    {
        std::vector<std::pair<float, uint32_t>> by_dist;
        std::unordered_set<uint32_t> seen;
        by_dist.reserve(expanded.size());
        for (int id : expanded)
            if (seen.insert((uint32_t)id).second)
                by_dist.emplace_back(yq.dist((uint32_t)id), (uint32_t)id);
        std::sort(by_dist.begin(), by_dist.end());
        size_t u_cap = g_protect_search_repair ? by_dist.size()
                                               : std::min(by_dist.size(), (size_t)std::max<uint64_t>(1, disk_beam_width));
        U.reserve(u_cap);
        for (size_t i = 0; i < u_cap; i++) U.push_back(by_dist[i].second);
    }
    std::vector<int> U_ids;
    U_ids.reserve(U.size());
    for (uint32_t u : U) U_ids.push_back((int)u);
    auto live_rows = g.fetch_neighbors(U_ids);
    // Exact dist(u, x_y) for the progress test below (§5.1): the paper computes
    // this side with full-precision vectors, only dist_PQ(v, x_y) is quantized.
    // yq.dist(u) would quantize both sides and let m_pq's calibration (which
    // only bounds v's quantization error) silently absorb u's error too.
    std::vector<std::vector<float>> U_coord;
    g.read_nodes(U, &U_coord, nullptr);

    if (g_protect_search_repair) {
        const bool proposed = audit_search_rescue(g, U, live_rows, U_coord, y_vec, aug_y, yq,
                                                  disk_beam_width, learner_hot, buffer_cap);
        if (proposed) {
            g_governor.window.attributable++;
            g_audit_attrib++;
            dump("attributable");
        } else {
            g_governor.window.unattributable++;
            g_audit_norescue++;
            dump("norescue");
        }
        return;
    }

    bool attributable = false;
    uint32_t rescue_u = 0, rescuer = 0;
    float rescue_u_y = 0.0f;
    for (size_t ui = 0; ui < U.size(); ui++) {
        uint32_t u = U[ui];
        auto live_edit = g.adj_edit.find(u);
        const std::vector<uint32_t> &live = live_edit != g.adj_edit.end() ? live_edit->second : live_rows[u];
        remember_restoration_record(g, u, live);
        auto rit = g_restoration_record.find(u);
        if (rit == g_restoration_record.end()) {
            g_audit_nobuf++;
            dump("nobuf");
            continue;
        }
        const std::vector<uint32_t> &record = rit->second;
        const size_t begin = std::min(record.size(), (size_t)std::max(0, g_governor.k_p));
        const size_t end = std::min(record.size(), (size_t)std::max(0, g_governor.K_max));
        const float dist_u_y = DiskANNIndex::exact_l2(aug_y.data(), U_coord[ui].data(), g.data_dim);
        const float threshold = dist_u_y / (float)g_alpha - (float)g_m_pq;
        // §5.2's red[k_p]/sw[k_p]: how redundant the CURRENT protected prefix
        // record[0, k_p) is at covering this same target, gathered at zero extra
        // I/O since `record` is already fetched. See governor_demote_gates note
        // near AuditWindow for the operational definition (the paper names these
        // quantities in prose but gives no formula).
        {
            const size_t kp = std::min(record.size(), (size_t)std::max(0, g_governor.k_p));
            int witnesses = 0;
            bool boundary_witness = false;
            for (size_t i = 0; i < kp; i++) {
                if (yq.dist(record[i]) <= threshold) {
                    witnesses++;
                    if (i == kp - 1) boundary_witness = true;
                }
            }
            if (kp > 0) {
                g_governor.window.red_total++;
                if (witnesses >= 2) g_governor.window.red_redundant++;
                g_governor.window.sw_total++;
                if (witnesses == 1 && boundary_witness) g_governor.window.sw_sole++;
            }
        }
        for (size_t i = begin; !attributable && i < end; i++) {
            uint32_t v = record[i];
            if (std::find(live.begin(), live.end(), v) != live.end() || repair_is_pending(u, v)) continue;
            if (yq.dist(v) <= threshold) {
                attributable = true;
                rescue_u = u;
                rescuer = v;
                rescue_u_y = dist_u_y;
            }
        }
        if (attributable) break;
    }

    if (attributable) {
        g_governor.window.attributable++;
        dump("attributable");
        g_audit_attrib++;
        // Classification stays independent of the gate: it is a property of
        // reachability, whereas acceptance is a separate effect-size decision.
        const float rescue_dist = yq.dist(rescuer);
        if (!repair_is_admissible(rescue_u, rescuer) || !repair_has_effect(rescue_u_y, rescue_dist))
            return;
        g_repair_volume++;
        // --protect-audit-passive: classification above still happened, so the
        // governor's miss-rate signal is untouched; only the repair is withheld.
        if (!g_audit_passive && learner_hot.contains((int)rescue_u))
            // Second B_buf writer: the audit rescue. Marked on the same rule as
            // the query loop -- only an actual membership change dirties the row.
            if (learner_hot.lru((int)rescue_u).touch((int)rescuer, buffer_cap)) mark_row_dirty((int)rescue_u);
        // --protect-direct-repair: queue the same (u, v) as a real edge edit.
        // Not gated on learner_hot: the point of the flag is to stop the learner
        // deciding whether the repair survives.
        if (!g_audit_passive && g_protect_direct_repair) {
            record_direct_repair(y, rescue_u, rescuer, rescue_u_y - rescue_dist);
            queue_audit_direct_repair(rescue_u, rescuer);
        }
    } else {
        g_governor.window.unattributable++;
        g_audit_norescue++;
        dump("norescue");
        // Algorithm 2 lines 9-10: no reserve edge clears alpha, so fall back to
        // the retained candidate log and recruit the best progress-making
        // candidate, if any, pinning it once it has recurred.
        // Suppressed under --protect-audit-passive for the same reason as the
        // attributable branch above: no repair may reach the graph.
        if (!g_audit_passive)
            audit_log_rescue(g, U, live_rows, U_coord, aug_y, yq, disk_beam_width, learner_hot, buffer_cap);
    }
}

static void fill_from_y_log(DiskANNIndex &g, uint32_t u, uint32_t y, uint64_t disk_beam_width) {
    if (!g_protect_repair_y_log_fill || !g_audit_failed_targets.count(y)) return;
    g_y_log_fill_calls++;
    std::vector<uint32_t> u_nbrs = g.get_neighbors(u);
    if (u_nbrs.size() >= g.max_degree) {
        // No spare budget: evict the farther half (PQ distance from u) of u's
        // non-pinned edges to make room.
        g_y_log_fill_zero_budget++;
        DiskANNIndex::PQQuery uq = g.node_query(u);
        auto pin_it = g_pinned_edges.find(u);
        std::vector<uint32_t> kept;
        std::vector<std::pair<float, uint32_t>> unpinned;
        for (uint32_t w : u_nbrs) {
            if (pin_it != g_pinned_edges.end() && pin_it->second.count(w)) kept.push_back(w);
            else unpinned.emplace_back(uq.dist(w), w);
        }
        std::sort(unpinned.begin(), unpinned.end());
        const size_t keep_unpinned = unpinned.size() - unpinned.size() / 2;
        g_y_log_fill_evicted += (long)(unpinned.size() - keep_unpinned);
        for (size_t i = 0; i < keep_unpinned; i++) kept.push_back(unpinned[i].second);
        std::vector<uint32_t> &row = g.edit_row(u, u_nbrs);
        row = kept;
        mark_row_dirty((int)u);
        u_nbrs = kept;
        if (u_nbrs.size() >= g.max_degree) return; // row entirely pinned
    }
    const size_t budget = (size_t)g.max_degree - u_nbrs.size();
    const std::vector<uint32_t> &y_log = get_candidate_log(g, y, disk_beam_width);
    std::unordered_set<uint32_t> in_row(u_nbrs.begin(), u_nbrs.end());
    std::vector<uint32_t> additions;
    for (uint32_t w : y_log) {
        if (additions.size() == budget) break;
        if (w != u && !in_row.count(w)) additions.push_back(w);
    }
    if (additions.empty()) return;
    std::vector<uint32_t> &row = g.edit_row(u, u_nbrs);
    for (uint32_t w : additions) {
        row.push_back(w);
        if (g_protect_direct_repair_pin) g_pinned_edges[u].insert(w);
    }
    g_y_log_fill_added += (long)additions.size();
    mark_row_dirty((int)u);
}

static void apply_direct_repairs(DiskANNIndex &g, uint64_t disk_beam_width) {
    if (!g_protect_direct_repair || g_direct_repair_queue.empty()) return;
    for (auto &pr : g_direct_repair_queue) {
        const uint32_t u = pr.first, v = pr.second;
        if (u == v) continue;
        std::vector<uint32_t> u_nbrs = g.get_neighbors(u);
        if (std::find(u_nbrs.begin(), u_nbrs.end(), v) != u_nbrs.end()) {
            g_direct_repair_existing++;
            fill_from_y_log(g, u, v, disk_beam_width);
            continue;
        }
        if (g_protect_direct_repair_force && u_nbrs.size() >= g.max_degree) {
            // Keep v, apply the cap to the other candidates only: rank the
            // incumbents by PQ distance from u and keep the R-1 closest.
            // Pinned edges (earlier forced repairs under --protect-direct-repair-pin)
            // are kept ahead of every unpinned incumbent, so a later repair cannot
            // evict an earlier one.
            DiskANNIndex::PQQuery uq = g.node_query(u);
            auto pin_it = g_pinned_edges.find(u);
            std::vector<std::pair<int, std::pair<float, uint32_t>>> cands; // (unpinned?, (dist, w))
            cands.reserve(u_nbrs.size());
            for (uint32_t w : u_nbrs) {
                const bool pinned = pin_it != g_pinned_edges.end() && pin_it->second.count(w);
                cands.push_back({pinned ? 0 : 1, {uq.dist(w), w}});
            }
            std::sort(cands.begin(), cands.end());
            const size_t pinned_count = (size_t)std::count_if(cands.begin(), cands.end(),
                                                              [](const auto &c) { return c.first == 0; });
            if (pinned_count > (size_t)g.max_degree - 1) {
                // Row is entirely pinned: no room without evicting a repair.
                g_direct_repair_capped++;
                continue;
            }
            cands.resize((size_t)g.max_degree - 1);
            std::vector<uint32_t> &row = g.edit_row(u, u_nbrs);
            row.clear();
            for (auto &c : cands) row.push_back(c.second.second);
            row.push_back(v);
        } else {
            g.add_directed(u, v, u_nbrs);
        }
        // add_directed may have dropped v itself on the cap re-rank; only count
        // a repair that is actually in the row.
        auto eit = g.adj_edit.find(u);
        const bool present = eit != g.adj_edit.end() &&
                             std::find(eit->second.begin(), eit->second.end(), v) != eit->second.end();
        if (present) {
            g_direct_repair_applied++;
            mark_row_dirty((int)u);
            if (g_protect_direct_repair_pin) g_pinned_edges[u].insert(v);
            fill_from_y_log(g, u, v, disk_beam_width);
        } else {
            g_direct_repair_capped++;
        }
    }
    g_direct_repair_queue.clear();
    g.flush_edits();
    if (g_protect_direct_repair_retest) {
        g_retest_n = g_retest_y_top1 = g_retest_testq_n = g_retest_testq_top1 = 0;
        {
            std::unordered_set<uint32_t> seen_y;
            std::vector<std::pair<long, uint32_t>> unique_targets;
            for (const auto &tr : g_direct_repair_retest_targets)
                if (seen_y.insert(tr.second).second) unique_targets.push_back(tr);
            g_direct_repair_retest_targets.swap(unique_targets);
        }
        for (const auto &[trial, y] : g_direct_repair_retest_targets) {
            // Matching held-out query, same entry point, same ef.
            for (size_t ti = 0; ti < g_audit_target_ids.size() && ti < g_audit_retest_queries.size(); ti++) {
                if (g_audit_target_ids[ti] != y) continue;
                std::vector<int> tq_result, tq_visited, tq_expanded;
                g.search(g_audit_retest_queries[ti].data(), /*K=*/1, (uint64_t)g_audit_ef, disk_beam_width,
                         tq_result, tq_visited, &tq_expanded);
                g_retest_testq_n++;
                if (!tq_result.empty() && tq_result[0] == (int)y) g_retest_testq_top1++;
                break;
            }
            std::vector<std::vector<float>> ycoord;
            g.read_nodes({y}, &ycoord, nullptr);
            std::vector<int> result_ids, visited, expanded;
            g.search(ycoord[0].data(), /*K=*/1, (uint64_t)g_audit_ef, disk_beam_width,
                     result_ids, visited, &expanded);
            bool reached = false;
            for (int u : expanded)
                if (u == (int)y || g.node_sector((uint32_t)u) == g.node_sector(y)) {
                    g_direct_repair_retest_reached++;
                    reached = true;
                    break;
                }
            const int top1 = result_ids.empty() ? -1 : result_ids[0];
            g_retest_n++;
            if (top1 == (int)y) g_retest_y_top1++;
            if (g_audit_retest_dump_f)
                fprintf(g_audit_retest_dump_f, "%ld,%u,0,%d,%d,%d\n", trial, y,
                        top1 == (int)y ? 1 : 0, top1, reached ? 1 : 0);
        }
        g_direct_repair_retest_targets.clear();
    }
}

static double measure_build_floor(DiskANNIndex &g, int n_probes, uint64_t serving_ef,
                                  uint64_t disk_beam_width) {
    IOPhase ph(g_io_floor);
    std::vector<uint64_t> widths;
    const uint64_t audit_w = g_audit_ef > 0 ? (uint64_t)g_audit_ef : serving_ef;
    widths.push_back(audit_w);
    if (serving_ef != audit_w) widths.push_back(serving_ef);

    double b0_audit = 0.0;
    for (uint64_t w : widths) {
        std::mt19937_64 rng_floor(g_floor_seed);
        std::uniform_real_distribution<double> u01(0.0, 1.0);
        long fails = 0;
        for (int i = 0; i < n_probes; i++) {
            uint32_t y = (uint32_t)std::min<double>((double)g.num_points - 1,
                                                    u01(rng_floor) * (double)g.num_points);
            std::vector<std::vector<float>> ycoord;
            g.read_nodes({y}, &ycoord, nullptr);
            std::vector<int> ids, visited;
            g.search(ycoord[0].data(), /*K=*/1, w, disk_beam_width, ids, visited);
            if (std::find(visited.begin(), visited.end(), (int)y) == visited.end()) fails++;
        }
        const double b0 = (double)fails / (double)n_probes;
        if (w == audit_w) b0_audit = b0;
        fprintf(stderr, "  BUILDFLOOR: n=%d ef=%llu seed=%llu fails=%ld b0=%.4f "
                        "se=%.4f eps_r_suggested=%.4f floor_io=%ld\n",
                n_probes, (unsigned long long)w, g_floor_seed, fails, b0,
                std::sqrt(b0 * (1.0 - b0) / (double)n_probes), 2.0 * b0 + 1e-3,
                g_io_floor.device_reads + g_io_floor.device_writes);
    }
    return b0_audit;
}

// Algorithm 6, restricted per the file-header note above this section.
// Promotion is fast (any repair this window, or a confidently-breached upper
// bound); demotion is slow (a settled, confidently-safe window after a
// minimum dwell). Called once per do_rewire cycle.
static void governor_step(DiskANNIndex &g, long gqc) {
    AuditWindow &w = g_governor.window;

    // --protect-observe: measure the attributable-miss-rate trajectory without
    // ever moving k_p. See the flag's declaration for how to read the output.
    if (g_gov_observe) {
        static long cum_attrib = 0, cum_unattrib = 0, cum_pass = 0;
        cum_attrib += w.attributable;
        cum_unattrib += w.unattributable;
        cum_pass += w.pass;
        const long bt = w.total();
        const long cum_t = cum_attrib + cum_unattrib + cum_pass;
        const long hot = std::max<long>(1, g_last_hot_nodes);
        fprintf(stderr,
                "  [gov-observe] q=%ld k_p=%d batch[attrib=%ld unattrib=%ld pass=%ld n=%ld "
                "miss_rate=%.4f attrib_rate=%.4f] cum[attrib=%ld unattrib=%ld pass=%ld n=%ld "
                "miss_rate=%.4f] churn=%.4f rows=%ld/%ld deleted=%ld\n",
                gqc, g_governor.k_p, w.attributable, w.unattributable, w.pass, bt,
                bt ? (double)w.attributable / (double)bt : 0.0,
                (w.attributable + w.pass) ? (double)w.attributable / (double)(w.attributable + w.pass) : 0.0,
                cum_attrib, cum_unattrib, cum_pass, cum_t,
                cum_t ? (double)cum_attrib / (double)cum_t : 0.0,
                (double)g_last_rows_changed / (double)hot, g_last_rows_changed, g_last_hot_nodes,
                g_last_deleted);
        w.clear();
        g_repair_volume = 0;
        return;
    }

    // --protect-fix-k-p: k_p never moves; everything else still resets.
    if (g_fix_k_p) {
        w.clear();
        g_repair_volume = 0;
        return;
    }

    // --protect-slo-warmup: calibrate eps_r on the live audit stream. k_p is
    // held while the floor is being measured, so the level the warm-up sees is
    // the one it was initialised with and the floor is not contaminated by the
    // governor's own moves.
    if (g_slo_warmup_n > 0 && !g_slo_frozen) {
        g_slo_warm_attrib += w.attributable;
        g_slo_warm_total += w.total();
        if (g_slo_warm_total < g_slo_warmup_n) {
            fprintf(stderr, "  [governor] q=%ld slo-warmup %ld/%ld (attrib=%ld) k_p=%d held\n",
                    gqc, g_slo_warm_total, g_slo_warmup_n, g_slo_warm_attrib, g_governor.k_p);
            w.clear();
            g_repair_volume = 0;
            return;
        }
        const double b0_live = (double)g_slo_warm_attrib / (double)g_slo_warm_total;
        g_epsilon_r = b0_live + g_slo_margin;
        g_slo_frozen = true;
        fprintf(stderr, "  [governor] q=%ld slo-warmup done: b0_live=%.4f (%ld/%ld) "
                        "eps_r <- b0_live + %.4f = %.4f\n",
                gqc, b0_live, g_slo_warm_attrib, g_slo_warm_total, g_slo_margin, g_epsilon_r);
        // The warm-up window IS evidence, and the paper's confidence sequence is
        // never reset, so it is left in place for this batch's gates to test.
    }

    // Reachability check: eps_r above every UCB the window ever produces means
    // promote can never fire and demote never has to certify anything.
    {
        const long bt = w.total();
        if (bt > 0) {
            g_slo_batches_seen++;
            g_slo_max_ucb_seen = std::max(g_slo_max_ucb_seen, w.ucb());
            if (!g_slo_unreachable_warned && g_slo_batches_seen >= 50 &&
                g_epsilon_r > g_slo_max_ucb_seen) {
                fprintf(stderr, "  [governor] WARNING eps_r=%.4f exceeds the largest UCB observed "
                                "(%.4f) over %ld batches -- promote is unreachable and the demote "
                                "gate is unconditional\n",
                        g_epsilon_r, g_slo_max_ucb_seen, g_slo_batches_seen);
                g_slo_unreachable_warned = true;
            }
        }
    }

    // --protect-schedule: open-loop ladder, no audit feedback. With
    // --protect-schedule-every-batches > 0, the ladder is driven by the count
    // of do_rewire/governor_step cycles (batches) rather than the query
    // count, so "decay by step every batch" is exact regardless of b_rewire
    // or a short final batch.
    if (g_protect_schedule) {
        int drops;
        long period;
        const char *unit;
        if (g_protect_schedule_every_batches > 0) {
            g_protect_schedule_batches_done++;
            period = g_protect_schedule_every_batches;
            drops = (int)(g_protect_schedule_batches_done / period);
            unit = "batches";
        } else {
            period = std::max(1L, g_protect_schedule_every_queries);
            drops = (int)(gqc / period);
            unit = "queries";
        }
        int target = g_k_p_init - drops * g_protect_schedule_step;
        target = std::max(g_protect_schedule_min, std::min(g_governor.K_max, target));
        if (target < g_governor.k_p) {
            fprintf(stderr, "  [governor-schedule] q=%ld demote k_p %d -> %d (period=%ld %s step=%d)\n",
                    gqc, g_governor.k_p, target, period, unit, g_protect_schedule_step);
            g_governor.k_ps = g_governor.k_p;
            g_governor.k_p = target;
            g_governor.demotions++;
            g_governor.dwell = 0;
        } else {
            fprintf(stderr, "  [governor-schedule] q=%ld hold k_p=%d (period=%ld %s step=%d)\n",
                    gqc, g_governor.k_p, period, unit, g_protect_schedule_step);
        }
        w.clear();
        g_repair_volume = 0;
        return;
    }

    // --protect-anneal-linear: deterministic k_p decay to the floor by query N.
    if (g_gov_anneal_linear_n > 0) {
        const int init = std::min(g_governor.K_max, g_k_p_init);
        const double frac = std::max(0.0, 1.0 - (double)gqc / (double)g_gov_anneal_linear_n);
        int k = (int)std::lround((double)init * frac);
        k = std::max(g_gov_anneal_min, std::min(g_governor.K_max, k));
        if (k != g_governor.k_p) {
            fprintf(stderr, "  [governor] q=%ld anneal-linear k_p %d -> %d (frac=%.3f)\n",
                    gqc, g_governor.k_p, k, frac);
            g_governor.k_p = k;
        }
        w.clear();
        g_repair_volume = 0;
        return;
    }

    // --protect-anneal: warm-up schedule. Step k_p down on a converged batch
    // (low selection churn), step it up on a demonstrated audit miss.
    if (g_gov_anneal) {
        const long hot = std::max<long>(1, g_last_hot_nodes);
        const double churn = (double)g_last_rows_changed / (double)hot;
        const double del_ratio = g_first_batch_deleted > 0
            ? (double)g_last_deleted / (double)g_first_batch_deleted : 1.0;
        const bool by_deleted = g_gov_anneal_deleted_frac > 0.0;
        const bool converged = by_deleted ? (del_ratio < g_gov_anneal_deleted_frac)
                                          : (churn < g_gov_anneal_tau);
        const char *act = "hold";
        if (g_repair_volume > 0 && g_governor.k_p < g_governor.K_max) {
            g_governor.k_p = std::min(g_governor.K_max, g_governor.k_p + g_gov_anneal_step);
            if (g_governor.k_p > g_governor.anneal_floor) g_governor.anneal_floor = g_governor.k_p;
            g_governor.promotions++;
            act = "PROMOTE";
        } else if (converged && g_governor.k_p > g_governor.anneal_floor) {
            g_governor.k_p = std::max(g_governor.anneal_floor, g_governor.k_p - g_gov_anneal_step);
            g_governor.demotions++;
            act = "demote";
        }
        fprintf(stderr, "  [governor] q=%ld %s k_p=%d (churn=%.4f del_ratio=%.4f rows=%ld/%ld "
                "del=%ld/%ld floor=%d repairs=%ld)\n",
                gqc, act, g_governor.k_p, churn, del_ratio, g_last_rows_changed, g_last_hot_nodes,
                g_last_deleted, g_first_batch_deleted, g_governor.anneal_floor, g_repair_volume);
        w.clear();
        g_repair_volume = 0;
        return;
    }

    // --protect-symmetric: per-batch hill-climb on an EWMA of the per-batch
    // attributable-miss rate. No dwell, no fixed-m certificate; the varsigma
    // band is the whole stability mechanism.
    if (g_gov_symmetric) {
        const long bt = w.total();
        g_governor.audits_total += bt;
        if (bt >= 1) {
            const double batch_mr = (double)w.attributable / (double)bt;
            g_governor.mr_ewma = (g_governor.mr_ewma < 0.0)
                ? batch_mr
                : (1.0 - g_gov_mr_beta) * g_governor.mr_ewma + g_gov_mr_beta * batch_mr;
        }
        w.clear();
        const double mr = g_governor.mr_ewma;
        const char *act = "hold";
        if (g_governor.audits_total >= g_demote_n && mr >= 0.0) {
            if (mr > g_epsilon_r + g_varsigma && g_governor.k_p < g_governor.K_max) {
                g_governor.k_p++; g_governor.promotions++; act = "PROMOTE";
            } else if (mr < g_epsilon_r - g_varsigma && g_governor.k_p > 0) {
                g_governor.k_ps = g_governor.k_p;
                g_governor.k_p--; g_governor.demotions++; act = "demote";
            }
        }
        fprintf(stderr, "  [governor] q=%ld %s k_p=%d (mr_ewma=%.4f batch_n=%ld "
                "band=[%.4f,%.4f] audits_total=%ld)\n",
                gqc, act, g_governor.k_p, mr < 0 ? 0.0 : mr, bt,
                g_epsilon_r - g_varsigma, g_epsilon_r + g_varsigma, g_governor.audits_total);
        g_repair_volume = 0;
        return;
    }

    bool ucb_breach = w.total() >= 5 &&
                      (g_promote_on_lcb ? w.lcb() : w.ucb()) > g_epsilon_r;
    bool repair_spike = g_repair_volume >= g_chi_rep;

    if (ucb_breach || repair_spike) {
        int new_k_p = std::min(g_governor.K_max, std::max(g_governor.k_ps, g_governor.k_p + 1));
        if (new_k_p != g_governor.k_p) {
            fprintf(stderr, "  [governor] q=%ld PROMOTE k_p %d -> %d (miss_rate=%.4f n=%ld repairs=%ld)\n",
                    gqc, g_governor.k_p, new_k_p, w.miss_rate(), w.total(), g_repair_volume);
            g_governor.k_p = new_k_p;
            g_governor.promotions++;
            // No restoration pass: live-referenced protection has no frozen
            // history to restore FROM (see the section header). Promotion is
            // purely prospective -- it takes effect the moment the next
            // do_rewire re-derives each node's live ranking and pins k_p
            // slots instead of k_p-1; nothing already evicted comes back.
        }
        if (!g_gov_window_accumulate) w.clear();
        g_governor.dwell = 0;
    } else if (g_governor.dwell >= g_T_dwell && w.total() >= g_demote_n && g_governor.k_p > 0 &&
               w.ucb() <= g_epsilon_r - g_varsigma &&
               w.red_total > 0 && w.redundancy() >= 1.0 - g_delta_red &&
               w.sole_witness_rate() <= g_delta_sw) {
        g_governor.k_ps = g_governor.k_p;
        g_governor.k_p--;
        fprintf(stderr, "  [governor] q=%ld demote k_p -> %d (miss_rate=%.4f n=%ld red=%.4f sw=%.4f "
                "repairs=%ld/%ld, settled)\n",
                gqc, g_governor.k_p, w.miss_rate(), w.total(), w.redundancy(), w.sole_witness_rate(),
                g_repair_volume, g_chi_rep);
        g_governor.demotions++;
        // Algorithm 6 line 8 only resets dwell, not the window statistics -- the
        // "reset statistics" instruction (line 5) is in the RAISE branch only.
        // Unlike the raise branch above, this is never gated by
        // --protect-window-accumulate: the confidence sequence is meant to keep
        // accumulating across a demotion, not just optionally.
        g_governor.dwell = 0;
    }
    else if (g_promote_on_lcb && w.total() >= g_demote_n) {
        // Neither branch fired: the paper's hold zone. Logged because a silent
        // governor and a broken one look identical in the logs otherwise.
        fprintf(stderr, "  [governor] q=%ld hold k_p=%d (miss_rate=%.4f n=%ld lcb=%.4f ucb=%.4f "
                "red=%.4f sw=%.4f dwell=%ld need_lcb>%.4f need_ucb<=%.4f repairs=%ld/%ld)\n",
                gqc, g_governor.k_p, w.miss_rate(), w.total(), w.lcb(), w.ucb(),
                w.redundancy(), w.sole_witness_rate(), g_governor.dwell, g_epsilon_r,
                g_epsilon_r - g_varsigma, g_repair_volume, g_chi_rep);
    }
    g_repair_volume = 0;
}

static void report_hot_cache(DiskANNIndex &g, long gqc, bool do_drop) {
    HotNodeCache &hc = g.hotcache;
    if (!hc.enabled) return;
    const long th = g_io_traversal.cblk_hits    - hc.snap_trav_hit;
    const long ta = g_io_traversal.cblk_admits  - hc.snap_trav_admit;
    const long rh = g_io_rewire.cblk_hits       - hc.snap_rew_hit;
    const long ra = g_io_rewire.cblk_admits     - hc.snap_rew_admit;
    const long qh = g_io_traversal.qbuf_hits    - hc.snap_qbuf_hit;
    const long rp = g_io_rewire.rmw_preads      - hc.snap_rmw;
    const long rr = g_io_rewire.rmw_preads_resident - hc.snap_rmw_res;
    const long tf = g_io_traversal.cblk_frozen - hc.snap_trav_frozen;
    const long rf = g_io_rewire.cblk_frozen - hc.snap_rew_frozen;
    // Every miss is in the denominator, whether or not it was admitted. Uncapped,
    // frozen is 0 and this is the old hits/(hits+admits).
    auto rate = [](long h, long a, long f) {
        return (h + a + f) ? (double)h / (double)(h + a + f) : 0.0;
    };
    fprintf(stderr, "  CBLK: batch=%ld q=%ld sectors=%zu resident=%.1fMB cap=%zu "
            "trav[hit=%ld admit=%ld frozen=%ld rate=%.4f] "
            "rewire[hit=%ld admit=%ld frozen=%ld rate=%.4f] "
            "qbuf_hits=%ld rmw_preads=%ld resident_of_those=%ld policy=%s evictions=%ld\n",
            hc.batches + 1, gqc, hc.used, hc.resident_mb(), hc.cap_sectors,
            th, ta, tf, rate(th, ta, tf), rh, ra, rf, rate(rh, ra, rf), qh, rp, rr,
            hc.cap_sectors == 0 ? "uncapped" : (hc.lru ? "lru" : "freeze"), hc.evictions);
    hc.snap_trav_hit = g_io_traversal.cblk_hits;
    hc.snap_trav_admit = g_io_traversal.cblk_admits;
    hc.snap_rew_hit = g_io_rewire.cblk_hits;
    hc.snap_rew_admit = g_io_rewire.cblk_admits;
    hc.snap_qbuf_hit = g_io_traversal.qbuf_hits;
    hc.snap_rmw = g_io_rewire.rmw_preads;
    hc.snap_rmw_res = g_io_rewire.rmw_preads_resident;
    hc.snap_trav_frozen = g_io_traversal.cblk_frozen;
    hc.snap_rew_frozen = g_io_rewire.cblk_frozen;
    // With write-through and lifetime 0 the cache is never dropped: the line above
    // is a progress report on a set that keeps growing, not a batch summary.
    if (do_drop) hc.drop();
    else hc.peak_sectors = std::max(hc.peak_sectors, hc.used);
}

static void diagnose_recall(DiskANNIndex &g, const char *tag, const float *qv, const int64_t *gt,
                            int gt_stride, int n_check, int K, int ef, uint64_t disk_beam_width) {
    const int kshow = 10;
    for (int i = 0; i < n_check; i++) {
        const float *q = qv + (size_t)i * g_query_dim;
        std::vector<int> ids, visited;
        g.search(q, (uint64_t)K, (uint64_t)ef, disk_beam_width, ids, visited);
        std::vector<float> aug = g.augment_query(q);

        std::vector<uint32_t> returned_ids;
        for (int j = 0; j < kshow && j < (int)ids.size(); j++) returned_ids.push_back((uint32_t)ids[j]);
        std::vector<std::vector<float>> returned_coords;
        g.read_nodes(returned_ids, &returned_coords, nullptr);

        std::vector<uint32_t> gt_ids;
        bool gt_out_of_range = false;
        for (int j = 0; j < kshow; j++) {
            const int64_t id = gt[(size_t)i * gt_stride + j];
            if (id < 0 || (uint64_t)id >= g.num_points) { gt_out_of_range = true; continue; }
            gt_ids.push_back((uint32_t)id);
        }
        std::vector<std::vector<float>> gt_coords;
        g.read_nodes(gt_ids, &gt_coords, nullptr);

        std::unordered_set<int> visited_set(visited.begin(), visited.end());
        int gt_in_visited = 0;
        for (int j = 0; j < gt_stride; j++)
            if (visited_set.count((int)gt[(size_t)i * gt_stride + j])) gt_in_visited++;

        auto dist_to = [&](const std::vector<std::vector<float>> &cs, size_t k) {
            return cs.size() > k ? DiskANNIndex::exact_l2(aug.data(), cs[k].data(), g.data_dim) : -1.0f;
        };
        double query_norm = 0.0;
        for (int d = 0; d < g_query_dim; d++) query_norm += (double)q[d] * (double)q[d];
        query_norm = std::sqrt(query_norm);
        auto ip_cos = [&](const std::vector<float> &x, double &ip, double &cos, double &xnorm) {
            ip = 0.0; xnorm = 0.0;
            for (int d = 0; d < g_query_dim; d++) { ip += (double)q[d] * (double)x[d]; xnorm += (double)x[d] * (double)x[d]; }
            xnorm = std::sqrt(xnorm);
            cos = (xnorm > 0 && query_norm > 0) ? ip / (xnorm * query_norm) : 0.0;
        };
        if (!returned_coords.empty() && !gt_coords.empty()) {
            double ip_s, cos_s, n_s, ip_g, cos_g, n_g;
            ip_cos(returned_coords[0], ip_s, cos_s, n_s);
            ip_cos(gt_coords[0], ip_g, cos_g, n_g);
            fprintf(stderr,
                    "      [%s q=%d] search_top1: ip=%.5f cos=%.5f |x|=%.5f | "
                    "cache_gt0: ip=%.5f cos=%.5f |x|=%.5f | |q|=%.5f  ==> gt wins on %s\n",
                    tag, i, ip_s, cos_s, n_s, ip_g, cos_g, n_g, query_norm,
                    cos_g > cos_s ? (ip_g > ip_s ? "both" : "COSINE only")
                                  : (ip_g > ip_s ? "IP only" : "neither"));
        }
        fprintf(stderr,
                "DIAG[%s] q=%d search: top1=%d d=%.5f d@%d=%.5f visited=%zu | "
                "cache_gt: id0=%lld d=%.5f d@%d=%.5f in_visited=%d/%d%s\n",
                tag, i, ids.empty() ? -1 : ids[0], dist_to(returned_coords, 0),
                (int)returned_coords.size(), dist_to(returned_coords, returned_coords.size() - 1),
                visited.size(), (long long)gt[(size_t)i * gt_stride], dist_to(gt_coords, 0),
                (int)gt_coords.size(), dist_to(gt_coords, gt_coords.size() - 1), gt_in_visited,
                gt_stride, gt_out_of_range ? "  [GT ID OUT OF RANGE]" : "");
    }
}

enum QueryOrder { QUERY_ORDER_IID = 0, QUERY_ORDER_BLOCKED = 1, QUERY_ORDER_REPEAT = 2 };

static void make_single_sigma_stream(const float *refs, const int64_t *proxy_gt, int nref, float sigma,
                                     int n_queries, long seed, std::vector<float> &vecs,
                                     std::vector<int64_t> &gts,
                                     int query_cardinality = 0,
                                     int query_order = QUERY_ORDER_IID,
                                     int query_repeat = 1,
                                     int interleave_centers_block = 0,
                                     std::vector<int> *center_of_query = nullptr,
                                     bool sequential_refs = false) {
    npr::Generator rng(seed);
    // Per-dimension noise std scaled by 1/sqrt(dim) so `sigma` is an isotropic
    // total displacement budget independent of g_query_dim.
    double std = (double)sigma / std::sqrt((double)g_query_dim);
    const int n_distinct_queries =
        (query_cardinality > 0 && query_cardinality < n_queries) ? query_cardinality : n_queries;
    const bool interleave = interleave_centers_block > 0;
    if (interleave) {
        if (nref % 2 != 0)
            throw std::runtime_error("--interleave-centers needs an even --n-refs (nref/2 per centre)");
        if (query_cardinality > 0 && query_cardinality < n_queries)
            throw std::runtime_error("--interleave-centers is incompatible with --query-cardinality");
        if (query_order != QUERY_ORDER_IID)
            throw std::runtime_error("--interleave-centers requires --query-order iid");
    }
    const int half = nref / 2;
    if (sequential_refs && (interleave || query_cardinality > 0))
        throw std::runtime_error("--post-test-sequential-refs is incompatible with "
                                  "--interleave-centers / --query-cardinality");
    std::vector<int> reference_of_distinct_query(n_distinct_queries);
    for (int i = 0; i < n_distinct_queries; i++) {
        if (sequential_refs) {
            // Deterministic 1:1 coverage: ref i % nref, in order -- no RNG draw,
            // so every distinct query slot maps to a distinct ref when
            // n_distinct_queries == nref (e.g. testing a hard100 set exactly once).
            reference_of_distinct_query[i] = i % nref;
        } else if (interleave) {
            const int side = (i / interleave_centers_block) % 2;   // 0 -> centre A, 1 -> centre B
            reference_of_distinct_query[i] = side * half + (int)rng.integers(half);
        } else {
            reference_of_distinct_query[i] = (int)rng.integers(nref);
        }
    }
    std::vector<float> distinct_vecs((size_t)n_distinct_queries * g_query_dim);
    std::vector<int64_t> distinct_gts((size_t)n_distinct_queries * g_k);
    for (int i = 0; i < n_distinct_queries; i++) {
        // Synthetic query = a randomly chosen reference vector + Gaussian jitter.
        const int r = reference_of_distinct_query[i];
        for (int d = 0; d < g_query_dim; d++)
            distinct_vecs[(size_t)i * g_query_dim + d] =
                refs[(size_t)r * g_query_dim + d] + (float)rng.normal(0.0, std);
        // GT is reused from the same reference, valid while sigma is small enough
        // that the jitter does not change the true kNN set.
        for (int j = 0; j < g_k; j++)
            distinct_gts[(size_t)i * g_k + j] = proxy_gt[(size_t)r * GT_STRIDE + j];
    }

    // serving_order[i] is which distinct query is served i-th.
    std::vector<int> serving_order(n_queries);
    if (query_order == QUERY_ORDER_REPEAT) {
        // Each distinct query served query_repeat times back to back, cycling.
        const int repeat = query_repeat > 1 ? query_repeat : 1;
        for (int i = 0; i < n_queries; i++)
            serving_order[i] = (i / repeat) % n_distinct_queries;
    } else {
        for (int i = 0; i < n_queries; i++)
            serving_order[i] = (n_distinct_queries == n_queries) ? i
                                                                 : (int)rng.integers(n_distinct_queries);
        if (query_order == QUERY_ORDER_BLOCKED) {
            // Group by the reference each query was jittered from, so consecutive
            // queries walk the same region. Stable, so within a reference the
            // stock relative order survives.
            std::stable_sort(serving_order.begin(), serving_order.end(),
                             [&](int a, int b) {
                                 return reference_of_distinct_query[a] < reference_of_distinct_query[b];
                             });
        }
    }

    vecs.resize((size_t)n_queries * g_query_dim);
    gts.resize((size_t)n_queries * g_k);
    if (center_of_query) center_of_query->assign(n_queries, 0);
    for (int i = 0; i < n_queries; i++) {
        const int picked = serving_order[i];
        if (center_of_query)
            (*center_of_query)[i] = reference_of_distinct_query[picked] >= half ? 1 : 0;
        std::copy(distinct_vecs.begin() + (size_t)picked * g_query_dim,
                  distinct_vecs.begin() + (size_t)(picked + 1) * g_query_dim,
                  vecs.begin() + (size_t)i * g_query_dim);
        std::copy(distinct_gts.begin() + (size_t)picked * g_k,
                  distinct_gts.begin() + (size_t)(picked + 1) * g_k,
                  gts.begin() + (size_t)i * g_k);
    }
}

static void write_query_stream_bin(const std::string &path, const std::vector<float> &vecs, int n_queries,
                                   int dim) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) throw std::runtime_error("cannot open " + path + " for writing");
    const int32_t header[2] = {(int32_t)n_queries, (int32_t)dim};
    if (fwrite(header, sizeof(int32_t), 2, f) != 2 ||
        fwrite(vecs.data(), sizeof(float), (size_t)n_queries * dim, f) != (size_t)n_queries * dim) {
        fclose(f);
        throw std::runtime_error("short write to " + path);
    }
    fclose(f);
    fprintf(stderr, "QUERYDUMP: %s  %d x %d float32\n", path.c_str(), n_queries, dim);
}

static void load_query_stream_bin(const std::string &path, std::vector<float> &vecs, int &n_out, int dim_expected) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t header[2];
    if (fread(header, sizeof(int32_t), 2, f) != 2) throw std::runtime_error("truncated " + path);
    if ((int)header[1] != dim_expected)
        throw std::runtime_error(path + " dim=" + std::to_string(header[1]) + " != index query dim " +
                                 std::to_string(dim_expected));
    n_out = (int)header[0];
    vecs.resize((size_t)n_out * dim_expected);
    if (fread(vecs.data(), sizeof(float), vecs.size(), f) != vecs.size())
        throw std::runtime_error("short read on " + path);
    fclose(f);
    fprintf(stderr, "EXTQUERY: %s  %d x %d float32\n", path.c_str(), n_out, dim_expected);
}

static void load_truthset_ids(const std::string &path, int n_queries, int k, std::vector<int64_t> &gts) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open truthset " + path);
    uint32_t header[2];
    if (fread(header, sizeof(uint32_t), 2, f) != 2) throw std::runtime_error("truncated truthset " + path);
    if ((int)header[0] != n_queries)
        throw std::runtime_error("truthset " + path + " has " + std::to_string(header[0]) +
                                 " rows but the stream has " + std::to_string(n_queries) +
                                 " queries -- it was computed for a different stream");
    if ((int)header[1] < k)
        throw std::runtime_error("truthset " + path + " has K=" + std::to_string(header[1]) + " < " +
                                 std::to_string(k));
    const int file_k = (int)header[1];
    std::vector<uint32_t> ids((size_t)n_queries * file_k);
    if (fread(ids.data(), sizeof(uint32_t), ids.size(), f) != ids.size())
        throw std::runtime_error("truncated truthset ids in " + path);
    fclose(f);
    gts.assign((size_t)n_queries * k, 0);
    for (int i = 0; i < n_queries; i++)
        for (int j = 0; j < k; j++) gts[(size_t)i * k + j] = (int64_t)ids[(size_t)i * file_k + j];
    fprintf(stderr, "TRUEGT: %s  %d x %d (using first %d) -- proxy_gt REPLACED, recall is now exact\n",
            path.c_str(), n_queries, file_k, k);
}

static const npio::NpyArray &find_npz(const std::vector<std::pair<std::string, npio::NpyArray>> &npz,
                                      const std::string &name) {
    for (auto &kv : npz) if (kv.first == name) return kv.second;
    throw std::runtime_error("npz missing member " + name);
}

static bool has_npz(const std::vector<std::pair<std::string, npio::NpyArray>> &npz,
                   const std::string &name) {
    for (auto &kv : npz) if (kv.first == name) return true;
    return false;
}

static std::vector<std::string> split_csv(const std::string &s) {
    std::vector<std::string> out;
    size_t p = 0;
    while (true) {
        size_t q = s.find(',', p);
        out.push_back(s.substr(p, q == std::string::npos ? q : q - p));
        if (q == std::string::npos) break;
        p = q + 1;
    }
    return out;
}

int main(int argc, char **argv) {
    std::string index_prefix = "../diskann_index/gist32";
    std::string cache_npz = "../cache/realq_gt_gist/center_0.npz";
    std::string methods_arg = "beam";
    int sigma_index = 1;
    double sigma_override = -1.0;                 // --sigma: off-grid level, overrides sigma_index
    // --n-refs N: draw the training stream from only the first N reference
    // vectors of the cache (0 = use all of them, the standing behaviour).
    int n_reference_vectors_override = 0;
    int query_cardinality = 0;                    // --query-cardinality
    int query_order = QUERY_ORDER_IID;            // --query-order
    int query_repeat = 1;                         // --query-repeat
    // --interleave-centers B: treat the cache's references as two centres
    // (first half / second half) and alternate the training stream between them
    // in blocks of B queries. 0 = off. --dump-query-centers writes q,center.
    int interleave_centers_block = 0;
    std::string query_centers_csv;
    int K = 100, ef = 100, b_rewire = 50;
    int diagnose_recall_n = 0;              // --diagnose-recall N
    int n_queries = 1000, eval_every = 100;
    long seed = 42;
    uint64_t disk_beam_width = 4; // DiskANN search beam: sectors fetched per hop
    bool writeback = true;        // commit rewires to the on-disk sectors
    std::string work_copy;        // scratch copy of _disk.index to rewire instead of the original
    std::string reuse_work_copy;  // an existing final work copy, for inference-only reload
    bool keep_work_copy = false;
    std::string layout_map_save;  // persisted logical-id -> physical-slot permutation
    std::string layout_map_load;
    std::string query_results_csv; // --dump-query-results
    std::string dump_queries_path;           // --dump-queries
    std::string true_gt_path;                // --true-gt
    std::string dump_post_test_queries_path; // --dump-post-test-queries
    std::string post_test_true_gt_path;      // --post-test-true-gt
    std::string ext_query_file;              // --ext-query-file
    int ext_query_limit = 0;                  // --ext-query-limit
    std::string post_test_ext_query_file;    // --post-test-ext-query-file
    long rewire_until = -1;                  // --rewire-until
    std::string side_eval_a_query, side_eval_a_gt;
    std::string side_eval_b_query, side_eval_b_gt;
    std::string side_eval_csv;               // --side-eval-csv
    std::string side_eval_per_query_csv;     // --side-eval-per-query-csv: one row per
    std::string detector_calibration_csv;    // --detector-calibration-csv
    int detector_calibration_window = 100;   // --detector-calibration-window
    long detector_calibration_freeze_at = -1;// --detector-calibration-freeze-at
    bool detector_control = false;            // --detector-control
    double detector_stop_threshold = 0.21;
    int detector_stop_windows = 3;
    long detector_min_adapt_q = 500;
    double detector_drift_threshold = 0.70;
    bool side_eval_use_hot_cache = false;
    // (checkpoint, query) instead of the aggregate-only --side-eval-csv, so individual
    // queries' recall can be ranked/diffed across arms (e.g. auditon vs auditoff).
    long side_eval_ef = -1;                  // --side-eval-ef: search width for
    // run_side_eval only (defaults to the main --ef); lets a side-eval set be
    // scored at the audit's search width without changing serving/training ef.
    std::vector<int> side_eval_efs;          // --side-eval-ef-list: score every checkpoint at each width
    int post_test_n = 0;
    long post_test_seed = 424242;
    bool post_test_sequential_refs = false; // --post-test-sequential-refs
    int post_test_shard_index = 0, post_test_shard_count = 1;
    std::string post_test_barrier_dir;
    std::string post_test_results_csv;
    std::string checkpoint_test_results_csv;
    std::vector<int> post_test_efs;
    std::string post_test_ef_results_csv;
    std::string post_test_per_query_results_csv;
    bool post_test_use_hot_cache = false;
    std::string ood_cache_npz;
    int ood_n = 0;
    std::string ood_results_csv;
    std::string ood_after_damage_results_csv;
    std::string dump_ood_queries_path;
    bool no_rewire = false;       // pure static-graph traversal, no mutation
    bool heatup = false;          // unrecorded warm-up pass over the same queries
    // The pristine graph always exists on EBS and --work-copy hands each run a fresh
    // copy, so the default is to drop the rewired graph rather than pay to undo it.
    // --restore-dirty re-enables the undo for in-place runs that share one file across
    // methods. --isolated always restores: per-query reset is its definition.
    bool restore_dirty_between = false;
    bool isolated = false;        // reset graph + Q/c/B_buf before EVERY query
    std::string per_query_csv;
    std::string gain_terms_csv;   // --dump-gain-terms
    // §7 layout co-adaptation. lambda_pi is the swap hysteresis: Table 2's default
    // is 2*lambda, but do_rewire here is ungated, so 1.0 stands in -- "a swap must
    // be worth at least one co-visitation". Raise it if swap volume runs away.
    bool hot_cache_lru = false;   // --hot-cache-lru: evict at the cap instead of freezing
    bool layout_adapt = false;
    long layout_adapt_until_query = 0;
    long layout_adapt_stop_below = 0;
    bool layout_adapt_retired = false;
    long warmup_queries = 0;
    int warmup_b_rewire = 0;
    long arm_mechanisms_from = 0;
    bool layout_commit_batched = false;
    long stats_from_query = 0;
    long price_until_query = 0;
    double lambda_pi = 1.0;
    bool qbuf_on = false;         // within-query block dedup (Pagewire eq. (1))
    size_t qbuf_cap = 512;        // blocks; 2 MB at 4KB, ~4x nu_bar on T2I-10M
    size_t learner_hot_cap = 0;   // --learner-hot-cap H; 0 keeps the old unbounded learner state
    // C_blk. Off is an exact no-op: nothing is allocated and find() returns
    // before touching the map, so the control arm runs the identical code path
    // that existed before the cache and must reproduce a prior run byte for byte.
    // Same standard as theta == 0 and lambda == 0.
    bool hot_cache_on = false;    // --hot-node-cache-unbounded
    bool hot_cache_wt = false;    // --hot-cache-writethrough
    bool async_rewire_commit = false; // --async-rewire-commit
    size_t hot_cache_ceiling = 0; // --hot-cache-ceiling S sectors; 0 => whole index
    // --hot-cache-cap S sectors; 0 => uncapped (the original path). A cap is a
    // FREEZE, not an eviction policy: the cache fills to S and then stops
    // admitting, serving whatever it already holds for the rest of the run.
    size_t hot_cache_cap = 0;
    int hot_cache_lifetime_q = -1; // --hot-cache-lifetime Q; 0 => never drop
    int measure_build_floor_n = 0; // --measure-build-floor N: offline b_0 probes on G_0

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nxt = [&] { return std::string(argv[++i]); };
        if (a == "--index-prefix") index_prefix = nxt();
        else if (a == "--data_type") g_data_type = nxt();
        else if (a == "--cache-npz") cache_npz = nxt();
        else if (a == "--methods") methods_arg = nxt();
        else if (a == "--sigma-index") sigma_index = std::stoi(nxt());
        else if (a == "--sigma") sigma_override = std::stod(nxt());
        else if (a == "--q-update-full-pool") g_q_update_full_pool = true;
        else if (a == "--n-refs") n_reference_vectors_override = std::stoi(nxt());
        else if (a == "--query-cardinality") query_cardinality = std::stoi(nxt());
        else if (a == "--query-repeat") query_repeat = std::stoi(nxt());
        else if (a == "--query-order") {
            const std::string v = nxt();
            if (v == "iid") query_order = QUERY_ORDER_IID;
            else if (v == "blocked") query_order = QUERY_ORDER_BLOCKED;
            else if (v == "repeat") query_order = QUERY_ORDER_REPEAT;
            else { fprintf(stderr, "--query-order takes iid | blocked | repeat\n"); return 1; }
        }
        else if (a == "--k") K = std::stoi(nxt());
        else if (a == "--ef") ef = std::stoi(nxt());
        else if (a == "--beam-width") disk_beam_width = (uint64_t)std::stoi(nxt());
        else if (a == "--b-rewire") b_rewire = std::stoi(nxt());
        else if (a == "--n-queries") n_queries = std::stoi(nxt());
        else if (a == "--eval-every") eval_every = std::stoi(nxt());
        else if (a == "--seed") seed = std::stol(nxt());
        else if (a == "--ucb-gate-rewire") g_ucb_gate_rewire = true;
        else if (a == "--protect-direct-repair") g_protect_direct_repair = true;
        else if (a == "--protect-audit-ep-last-result") g_audit_ep_last_result = true;
        else if (a == "--protect-direct-repair-force") { g_protect_direct_repair = true; g_protect_direct_repair_force = true; }
        else if (a == "--protect-direct-repair-to-y") { g_protect_direct_repair = true; g_protect_direct_repair_to_y = true; }
        else if (a == "--protect-repair-path-to-y") { g_protect_direct_repair = true; g_protect_repair_path_to_y = true; }
        else if (a == "--protect-repair-y-log-fill") g_protect_repair_y_log_fill = true;
        else if (a == "--y-log-predamage") g_y_log_predamage = true;
        else if (a == "--y-log-overlap-gt") {
            std::ifstream gf(nxt(), std::ios::binary);
            if (!gf) throw std::runtime_error("cannot open --y-log-overlap-gt file");
            uint32_t n_rows = 0, k = 0;
            gf.read((char *)&n_rows, 4); gf.read((char *)&k, 4);
            g_y_log_overlap_gt.assign(n_rows, std::vector<uint32_t>(k));
            for (auto &row : g_y_log_overlap_gt) gf.read((char *)row.data(), (std::streamsize)k * 4);
        }
        else if (a == "--audit-retest-queries") {
            std::ifstream qf(nxt(), std::ios::binary);
            if (!qf) throw std::runtime_error("cannot open --audit-retest-queries file");
            uint32_t n_rows = 0, dim = 0;
            qf.read((char *)&n_rows, 4); qf.read((char *)&dim, 4);
            g_audit_retest_queries.assign(n_rows, std::vector<float>(dim));
            for (auto &row : g_audit_retest_queries) qf.read((char *)row.data(), (std::streamsize)dim * 4);
        }
        else if (a == "--nav-damage-in-edges") g_nav_damage_in_edges = true;
        else if (a == "--nav-damage-target-frac") g_nav_damage_target_frac = std::stod(nxt());
        else if (a == "--protect-direct-repair-pin") { g_protect_direct_repair = true; g_protect_direct_repair_pin = true; }
        else if (a == "--protect-direct-repair-retest") g_protect_direct_repair_retest = true;
        else if (a == "--audit-repair-dump-csv") g_audit_repair_dump_csv = nxt();
        else if (a == "--audit-retest-dump-csv") g_audit_retest_dump_csv = nxt();
        else if (a == "--protect-repair-min-improvement") g_protect_repair_min_improvement = std::stof(nxt());
        else if (a == "--protect-repair-known-damage-only") g_protect_repair_known_damage_only = true;
        else if (a == "--no-cost-coverage") g_no_cost_coverage = true;
        else if (a == "--rewire-dirty-rows") g_rewire_dirty_rows = true;
        // OOD eval is gone with every other eval pass. Still accepted, and
        // ignored, so the runner scripts that pass them keep working.
        // --ood-labels is a no-op left so old scripts still parse; the OOD eval
        // path it belonged to was removed with the eval passes at 76ad5eab.
        // NOTE: there used to be an identical stub for --ood-cache-npz here. It
        // silently shadowed the real branch below, so the OOD post-test guard saw
        // an empty path and the whole pass was skipped with no error.
        else if (a == "--ood-labels") { (void)nxt(); }
        else if (a == "--no-ood") { }
        else if (a == "--no-writeback") writeback = false;
        else if (a == "--work-copy") work_copy = nxt();
        else if (a == "--reuse-work-copy") reuse_work_copy = nxt();
        else if (a == "--keep-work-copy") keep_work_copy = true;
        else if (a == "--layout-map-save") layout_map_save = nxt();
        else if (a == "--layout-map-load") layout_map_load = nxt();
        else if (a == "--ext-query-file") ext_query_file = nxt();
        else if (a == "--ext-query-limit") ext_query_limit = std::stoi(nxt());
        else if (a == "--post-test-ext-query-file") post_test_ext_query_file = nxt();
        else if (a == "--rewire-until") rewire_until = std::stol(nxt());
        else if (a == "--side-eval-a-query") side_eval_a_query = nxt();
        else if (a == "--side-eval-a-gt") side_eval_a_gt = nxt();
        else if (a == "--side-eval-b-query") side_eval_b_query = nxt();
        else if (a == "--side-eval-b-gt") side_eval_b_gt = nxt();
        else if (a == "--side-eval-csv") side_eval_csv = nxt();
        else if (a == "--side-eval-per-query-csv") side_eval_per_query_csv = nxt();
        else if (a == "--detector-calibration-csv") detector_calibration_csv = nxt();
        else if (a == "--detector-calibration-window") detector_calibration_window = std::stoi(nxt());
        else if (a == "--detector-calibration-freeze-at") detector_calibration_freeze_at = std::stol(nxt());
        else if (a == "--detector-control") detector_control = true;
        else if (a == "--detector-stop-threshold") detector_stop_threshold = std::stod(nxt());
        else if (a == "--detector-stop-windows") detector_stop_windows = std::stoi(nxt());
        else if (a == "--detector-min-adapt") detector_min_adapt_q = std::stol(nxt());
        else if (a == "--detector-drift-threshold") detector_drift_threshold = std::stod(nxt());
        else if (a == "--detector-sample-every") g_detector_sample_every = std::stoi(nxt());
        else if (a == "--side-eval-use-hot-cache") side_eval_use_hot_cache = true;
        else if (a == "--side-eval-ef") side_eval_ef = std::stol(nxt());
        else if (a == "--side-eval-ef-list") {
            std::string s = nxt();
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                side_eval_efs.push_back(std::stoi(s.substr(pos, comma - pos)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        // Eval passes no longer exist, so --skip-eval is the only behaviour;
        // accepted and ignored for the runner scripts that pass it.
        else if (a == "--skip-eval") { }
        else if (a == "--dump-query-results") query_results_csv = nxt();
        else if (a == "--dump-queries") dump_queries_path = nxt();
        else if (a == "--true-gt") true_gt_path = nxt();
        else if (a == "--dump-post-test-queries") dump_post_test_queries_path = nxt();
        else if (a == "--post-test-true-gt") post_test_true_gt_path = nxt();
        else if (a == "--interleave-centers") interleave_centers_block = std::stoi(nxt());
        else if (a == "--dump-query-centers") query_centers_csv = nxt();
        else if (a == "--post-test-n") post_test_n = std::stoi(nxt());
        else if (a == "--post-test-sequential-refs") post_test_sequential_refs = true;
        else if (a == "--post-test-seed") post_test_seed = std::stol(nxt());
        else if (a == "--post-test-shard-index") post_test_shard_index = std::stoi(nxt());
        else if (a == "--post-test-shard-count") post_test_shard_count = std::stoi(nxt());
        else if (a == "--post-test-barrier-dir") post_test_barrier_dir = nxt();
        else if (a == "--post-test-ef-list") {
            std::string s = nxt();
            size_t pos = 0;
            while (pos < s.size()) {
                size_t comma = s.find(',', pos);
                post_test_efs.push_back(std::stoi(s.substr(pos, comma - pos)));
                if (comma == std::string::npos) break;
                pos = comma + 1;
            }
        }
        else if (a == "--post-test-ef-results") post_test_ef_results_csv = nxt();
        else if (a == "--post-test-per-query-results") post_test_per_query_results_csv = nxt();
        else if (a == "--post-test-use-hot-cache") post_test_use_hot_cache = true;
        else if (a == "--post-test-results") post_test_results_csv = nxt();
        else if (a == "--checkpoint-test-results") checkpoint_test_results_csv = nxt();
        else if (a == "--ood-cache-npz") ood_cache_npz = nxt();
        else if (a == "--ood-n") ood_n = std::stoi(nxt());
        else if (a == "--ood-results") ood_results_csv = nxt();
        else if (a == "--ood-after-damage-results") ood_after_damage_results_csv = nxt();
        else if (a == "--dump-ood-queries") dump_ood_queries_path = nxt();
        else if (a == "--no-rewire") no_rewire = true;
        else if (a == "--skip-rewire-commit") g_skip_rewire_commit = true;
        else if (a == "--search-only") g_search_only = true;
        else if (a == "--heatup") heatup = true;
        else if (a == "--restore-dirty") restore_dirty_between = true;
        else if (a == "--isolated") isolated = true;
        else if (a == "--layout-adapt") layout_adapt = true;
        else if (a == "--layout-adapt-until") layout_adapt_until_query = std::stol(nxt());
        else if (a == "--layout-adapt-stop-below") layout_adapt_stop_below = std::stol(nxt());
        else if (a == "--layout-baff-cap") g_layout_baff_cap = std::stoi(nxt());
        else if (a == "--warmup-queries") warmup_queries = std::stol(nxt());
        else if (a == "--warmup-b-rewire") warmup_b_rewire = std::stoi(nxt());
        else if (a == "--arm-mechanisms-from") arm_mechanisms_from = std::stol(nxt());
        else if (a == "--price-until-query") price_until_query = std::stol(nxt());
        else if (a == "--lambda-pi") lambda_pi = std::stod(nxt());
        else if (a == "--layout-commit-batched") layout_commit_batched = true;
        else if (a == "--stats-from-query") stats_from_query = std::stol(nxt());
        else if (a == "--per-query-csv") per_query_csv = nxt();
        else if (a == "--query-block-buffer") qbuf_on = true;
        else if (a == "--qbuf-cap") qbuf_cap = (size_t)std::stoul(nxt());
        else if (a == "--learner-hot-cap") learner_hot_cap = (size_t)std::stoul(nxt());
        else if (a == "--hot-node-cache-unbounded") hot_cache_on = true;
        else if (a == "--hot-cache-writethrough") hot_cache_wt = true;
        else if (a == "--async-rewire-commit") async_rewire_commit = true;
        else if (a == "--early-rewire-commit") g_early_rewire_commit = true;
        else if (a == "--rewire-commit-direct") g_rewire_commit_direct = true;
        else if (a == "--hot-cache-ceiling") hot_cache_ceiling = (size_t)std::stoul(nxt());
        else if (a == "--hot-cache-cap") hot_cache_cap = (size_t)std::stoul(nxt());
        else if (a == "--hot-cache-lru") hot_cache_lru = true;
        else if (a == "--io-uring") g_use_io_uring = true;
        else if (a == "--in-memory") g_in_memory = true;
        else if (a == "--nav-damage-at") g_nav_damage_at = std::stol(nxt());
        else if (a == "--nav-damage-frac") g_nav_damage_frac = std::stod(nxt());
        else if (a == "--nav-damage-seed") g_nav_damage_seed = (unsigned)std::stoul(nxt());
        else if (a == "--nav-damage-strip-all") g_nav_damage_strip_all = true;
        else if (a == "--nav-damage-longest") g_nav_damage_longest = std::stoi(nxt());
        else if (a == "--nav-damage-dump-csv") g_nav_damage_dump_csv = nxt();
else if (a == "--nav-damage-target-ids") {
    std::ifstream tf(nxt());
    if (!tf) throw std::runtime_error("cannot open --nav-damage-target-ids file");
    long id;
    while (tf >> id) g_nav_damage_target_ids.push_back((int)id);
}
else if (a == "--nav-damage-target-n") g_nav_damage_target_n = std::stoi(nxt());
else if (a == "--audit-target-ids") {
    std::ifstream tf(nxt());
    if (!tf) throw std::runtime_error("cannot open --audit-target-ids file");
    long id;
    while (tf >> id) g_audit_target_ids.push_back((uint32_t)id);
}
        else if (a == "--nav-damage-from-prescan-ef") g_nav_damage_prescan_ef = std::stoi(nxt());
        else if (a == "--nav-damage-prescan-use-side-eval-a") g_nav_damage_prescan_use_side_eval_a = true;
        else if (a == "--nav-damage-prescan-force-audit-target") g_nav_damage_prescan_force_audit_target = true;
        else if (a == "--nav-damage-prescan-target-cap") g_nav_damage_prescan_target_cap = std::stoi(nxt());
        else if (a == "--audit-dump-csv") g_audit_dump_csv = nxt();
        else if (a == "--protect-schedule") g_protect_schedule = true;
        else if (a == "--protect-schedule-every-queries") g_protect_schedule_every_queries = std::stol(nxt());
        else if (a == "--protect-schedule-every-batches") g_protect_schedule_every_batches = std::stol(nxt());
        else if (a == "--protect-schedule-step") g_protect_schedule_step = std::stoi(nxt());
        else if (a == "--protect-schedule-min") g_protect_schedule_min = std::stoi(nxt());
        else if (a == "--protect-fix-k-p") g_fix_k_p = true;
        else if (a == "--protect-audit-passive") g_audit_passive = true;
        else if (a == "--protect-search-repair") g_protect_search_repair = true;
        else if (a == "--hot-cache-lifetime") hot_cache_lifetime_q = std::stoi(nxt());
        else if (a == "--track-expansion-cost") g_track_expansion_cost = true;
        else if (a == "--harvest-cosector") g_harvest_cosector = true;
        else if (a == "--harvest-ratio") g_harvest_ratio = std::stod(nxt());
        else if (a == "--expansion-cost-beta") g_expansion_cost_ewma_rate = std::stod(nxt());
        else if (a == "--diagnose-recall") diagnose_recall_n = std::stoi(nxt());
        else if (a == "--expansion-cost-theta") g_expansion_cost_theta = std::stod(nxt());
        else if (a == "--uncacheable-node-fraction") g_uncacheable_node_fraction = std::stod(nxt());
        else if (a == "--uncacheable-node-seed") g_uncacheable_node_seed = std::stoull(nxt());
        else if (a == "--uncacheable-node-list") {
            const std::string list_path = nxt();
            FILE *list_file = fopen(list_path.c_str(), "r");
            if (!list_file) { fprintf(stderr, "cannot open --uncacheable-node-list %s\n", list_path.c_str()); return 1; }
            unsigned long node_id = 0;
            while (fscanf(list_file, "%lu", &node_id) == 1) g_uncacheable_node_set.insert((uint32_t)node_id);
            fclose(list_file);
            fprintf(stderr, "uncacheable-node-list: %zu nodes from %s\n", g_uncacheable_node_set.size(), list_path.c_str());
        }
        else if (a == "--dump-node-fetch-counts") g_node_fetch_count_path = nxt();
        else if (a == "--io-target-per-query") {
            g_io_price_controller.enabled = true;
            g_io_price_controller.io_target_per_query = std::stod(nxt());
        }
        else if (a == "--dual-ascent-gamma") g_io_price_controller.gamma_override = std::stod(nxt());
        else if (a == "--io-target-deadband")
            g_io_price_controller.deadband_sectors_per_query = std::stod(nxt());
        else if (a == "--io-target-one-sided") g_io_price_controller.one_sided = true;
        else if (a == "--rewire-margin-lambda") g_rewire_margin_lambda = std::stod(nxt());
        else if (a == "--lambda-auto") g_lambda_auto_quantile = 0.10;
        else if (a == "--lambda-auto-quantile") g_lambda_auto_quantile = std::stod(nxt());
        else if (a == "--lambda-auto-scale") g_lambda_auto_scale = std::stod(nxt());
        else if (a == "--lambda-auto-ewma") g_lambda_auto_ewma = std::stod(nxt());
        else if (a == "--lambda-noise-floor") g_lambda_noise_floor_quantile = 0.90;
        else if (a == "--lambda-noise-floor-quantile") g_lambda_noise_floor_quantile = std::stod(nxt());
        else if (a == "--lambda-calibrate-first-batch") g_lambda_calibrate_first_batch = true;
        else if (a == "--lambda-calibrate-every-batch") g_lambda_calibrate_every_batch = true;
        else if (a == "--rewire-unidirectional") g_rewire_unidirectional = true;
        else if (a == "--write-delay-us") g_write_delay_us = atof(nxt().c_str());
        else if (a == "--write-delay-prob") g_write_delay_prob = atof(nxt().c_str());
        else if (a == "--read-delay-us") g_read_delay_us = atof(nxt().c_str());
        else if (a == "--read-delay-prob") g_read_delay_prob = atof(nxt().c_str());
        else if (a == "--read-delay-start-query") g_read_delay_start_query = std::stol(nxt());
        else if (a == "--dump-gain-terms") gain_terms_csv = nxt();
        else if (a == "--recall-prior") g_recall_prior = true;
        else if (a == "--label-gt") g_label_from_gt = true;   // oracle, not the default -- see g_label_from_gt's comment
        else if (a == "--self-label") g_label_from_gt = false;
        else if (a == "--quick-gt") g_quick_gt = true;   // alternative GT source, off by default -- see g_quick_gt's comment
        else if (a == "--gt-label-rate") g_gt_label_rate = std::stod(nxt());
        else if (a == "--gt-label-seed") { g_gt_label_seed = (unsigned)std::stoul(nxt()); g_gt_label_rng.seed(g_gt_label_seed); }
        else if (a == "--buffer-cap-mult") g_buffer_cap_mult = atof(nxt().c_str());
        else if (a == "--q-update-traversed") g_q_update_traversed_only = true;
        else if (a == "--covisit-scope") {
            const std::string p = nxt();
            if (p == "expanded") g_covisit_scope_expanded = true;
            else if (p == "miss") g_covisit_scope_expanded = false;
            else { fprintf(stderr, "--covisit-scope must be miss or expanded\n"); return 1; }
        }
        else if (a == "--unmeasured-cost") {
            const std::string p = nxt();
            if (p == "pool_mean") g_unmeasured_cost_policy = UnmeasuredCostPolicy::PoolMean;
            else if (p == "free") g_unmeasured_cost_policy = UnmeasuredCostPolicy::Free;
            else if (p == "full") g_unmeasured_cost_policy = UnmeasuredCostPolicy::Full;
            else { fprintf(stderr, "--unmeasured-cost must be pool_mean|free|full\n"); return 1; }
        }
        // --- protection arms (edge_protection branch) -------------------
        else if (a == "--edge-protection") g_edge_protection = true;
        else if (a == "--protect-k-p-init") g_k_p_init = std::stoi(nxt());
        else if (a == "--protect-k-p-max") g_K_max = std::stoi(nxt());
        else if (a == "--protect-audit-rate") g_p_a = std::stod(nxt());
        else if (a == "--protect-audit-ef") g_audit_ef = std::stoi(nxt());
        else if (a == "--dump-protect-mask") g_dump_protect_mask = true;
        else if (a == "--measure-build-floor") measure_build_floor_n = std::stoi(nxt());
        else if (a == "--build-floor-seed") g_floor_seed = std::stoull(nxt());
        else if (a == "--protect-slo") g_epsilon_r = std::stod(nxt());
        else if (a == "--protect-slack") g_varsigma = std::stod(nxt());
        else if (a == "--protect-dwell") g_T_dwell = std::stol(nxt());
        else if (a == "--protect-alpha") g_alpha = std::stod(nxt());
        else if (a == "--protect-pq-margin") g_m_pq = std::stod(nxt());
        else if (a == "--protect-chi-rep") g_chi_rep = std::stol(nxt());
        else if (a == "--protect-delta") g_delta = std::stod(nxt());
        else if (a == "--protect-promote-lcb") g_promote_on_lcb = true;   // now the default; kept as a no-op
        else if (a == "--protect-promote-ucb") g_promote_on_lcb = false;
        else if (a == "--protect-delta-red") g_delta_red = std::stod(nxt());
        else if (a == "--protect-delta-sw") g_delta_sw = std::stod(nxt());
        else if (a == "--protect-candlog-ef") g_candlog_ef = std::stoi(nxt());
        else if (a == "--protect-log-rescue-max") g_log_rescue_max = std::stoi(nxt());
        else if (a == "--protect-log-rescue-all-path") g_log_rescue_all_path = true;
        else if (a == "--protect-pin-repeat") g_pin_repeat = std::stoi(nxt());
        else if (a == "--protect-window-accumulate") g_gov_window_accumulate = true;
        else if (a == "--protect-symmetric") g_gov_symmetric = true;
        else if (a == "--protect-observe") g_gov_observe = true;
        else if (a == "--protect-mr-beta") g_gov_mr_beta = std::stod(nxt());
        else if (a == "--protect-auto-slo") g_protect_auto_slo = true;
        else if (a == "--protect-slo-margin") g_slo_margin = std::stod(nxt());
        else if (a == "--protect-slo-warmup") g_slo_warmup_n = std::stol(nxt());
        else if (a == "--protect-anneal") g_gov_anneal = true;
        else if (a == "--protect-anneal-deleted") { g_gov_anneal = true; g_gov_anneal_deleted_frac = std::stod(nxt()); }
        else if (a == "--protect-anneal-tau") g_gov_anneal_tau = std::stod(nxt());
        else if (a == "--protect-anneal-step") g_gov_anneal_step = std::stoi(nxt());
        else if (a == "--protect-anneal-min") g_gov_anneal_min = std::stoi(nxt());
        else if (a == "--protect-anneal-linear") g_gov_anneal_linear_n = std::stol(nxt());
        else if (a == "--protect-warmup-freeze") g_protect_warmup_freeze = std::stol(nxt());
        else if (a == "--protect-demote-n") g_demote_n = std::stol(nxt());
        else if (a == "--pct-protect-mode") {
            std::string mode = nxt();
            g_pct_protect = true;
            if (mode == "longest") g_pct_protect_longest = true;
            else if (mode == "shortest") g_pct_protect_longest = false;
            else { fprintf(stderr, "--pct-protect-mode must be longest|shortest\n"); return 1; }
        }
        else if (a == "--pct-protect-frac") g_pct_protect_frac = std::stod(nxt());
        else if (a == "--skeleton-protect") g_skeleton_protect = true;
        else if (a == "--skeleton-alpha") g_skeleton_alpha = std::stod(nxt());
        else if (a == "--skeleton-rp") g_skeleton_rp = std::stoi(nxt());
        else if (a == "--skeleton-profile-queries") g_skeleton_profile_queries = std::stoi(nxt());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
    }

    int protect_modes_on = (int)g_edge_protection + (int)g_pct_protect + (int)g_skeleton_protect;
    if (protect_modes_on > 1) {
        fprintf(stderr, "--edge-protection, --pct-protect-mode and --skeleton-protect are exclusive "
                        "comparison arms; pass only one\n");
        return 1;
    }
    if (!work_copy.empty() && !reuse_work_copy.empty())
        throw std::runtime_error("--work-copy and --reuse-work-copy are mutually exclusive");
    if (!layout_map_load.empty() && reuse_work_copy.empty())
        throw std::runtime_error("--layout-map-load requires --reuse-work-copy");
    if (!post_test_ef_results_csv.empty() && post_test_efs.empty())
        throw std::runtime_error("--post-test-ef-results requires --post-test-ef-list");
    if (post_test_shard_count < 1 || post_test_shard_index < 0 ||
        post_test_shard_index >= post_test_shard_count)
        throw std::runtime_error("invalid post-test shard index/count");

    DiskANNIndex g;
    g.keep_work_copy = keep_work_copy;
    // Capturing pristine rows costs one adjacency copy per touched node and is only
    // ever read by restore_dirty(), so skip it entirely when nothing will restore.
    // (edge_protection no longer needs this: live_rank_of ranks the current row,
    // never the pristine one -- see the mechanism's section header.)
    g.track_orig = isolated || restore_dirty_between;
    g.collect_epoch_misses = layout_adapt;
    g.qbuf.enabled = qbuf_on;
    g.qbuf.cap_blocks = qbuf_cap;
    // C_blk caches SECTORS against a device. With the index already resident
    // there is nothing for it to cache and its hit/admit counters would be
    // meaningless, so --in-memory forces it off rather than silently reporting
    // a cache over RAM. Same for the write/read delay rig and io_uring: both
    // price a device this mode does not have.
    if (g_in_memory) {
        if (hot_cache_on)
            fprintf(stderr, "IN-MEMORY: ignoring C_blk flags (no device to cache)\n");
        hot_cache_on = false;
        if (!work_copy.empty()) {
            fprintf(stderr, "IN-MEMORY: ignoring --work-copy (index is read into RAM)\n");
            work_copy.clear();
        }
        if (g_use_io_uring)
            throw std::runtime_error("--in-memory and --io-uring are mutually exclusive");
        if (g_write_delay_us > 0.0 || g_read_delay_us > 0.0)
            throw std::runtime_error("--in-memory has no device; --write-delay-us/--read-delay-us "
                                     "would price I/O the run never performs");
    }
    g.hotcache.enabled = hot_cache_on;
    g.hotcache.writethrough = hot_cache_wt;
    g.hotcache.async_writeback = async_rewire_commit;
    g.hotcache.cap_sectors = hot_cache_cap;
    g.hotcache.lru = hot_cache_lru;
    // The lifetime is a query count in both loop modes, so --isolated is no longer
    // a special case: there is no rebuild to pay for, and a batch of one query is
    // as meaningful as any other length.
    if (hot_cache_lifetime_q < 0) hot_cache_lifetime_q = (isolated && !hot_cache_wt) ? 1 : b_rewire;
    // Lifetime 0 == never drop. Without write-through, flush_edits leaves every
    // committed sector's resident copy holding pre-write bytes and the drop is the
    // only thing that discharges it, so a never-dropped read-only cache would serve
    // stale adjacency. Refuse rather than produce quietly wrong recall.
    if (hot_cache_on && hot_cache_lifetime_q == 0 && !hot_cache_wt) {
        fprintf(stderr, "--hot-cache-lifetime 0 requires --hot-cache-writethrough "
                        "(without it the drop is the only thing keeping C_blk coherent "
                        "across flush_edits)\n");
        return 1;
    }
    if (async_rewire_commit &&
        (!hot_cache_on || !hot_cache_wt || hot_cache_lifetime_q != 0)) {
        fprintf(stderr, "--async-rewire-commit uses C_blk as the dirty overlay, so it requires "
                        "--hot-node-cache-unbounded --hot-cache-writethrough --hot-cache-lifetime 0\n");
        return 1;
    }
    if (hot_cache_cap != 0 && !hot_cache_on) {
        fprintf(stderr, "--hot-cache-cap caps C_blk, so it requires --hot-node-cache-unbounded "
                        "(cap 0 is the uncapped cache; there is no cache to cap without it)\n");
        return 1;
    }
    if (async_rewire_commit && !writeback) {
        fprintf(stderr, "--async-rewire-commit needs --writeback enabled so the worker has a disk target\n");
        return 1;
    }
    // --isolated calls restore_dirty() before every query, which commits pristine
    // rows through flush_edits. With write-through that path patches the resident
    // copies too, so the cache stays coherent at any lifetime; without it, a cache
    // surviving into the next query would hold pre-restore bytes.
    if (isolated && hot_cache_on && !hot_cache_wt && hot_cache_lifetime_q != 1) {
        fprintf(stderr, "--isolated without --hot-cache-writethrough forces "
                        "--hot-cache-lifetime 1 (restore_dirty writes before every "
                        "query); requested %d\n", hot_cache_lifetime_q);
        hot_cache_lifetime_q = 1;
    }
    // eq. (7) rewrites theta every batch, so a fixed one is the other arm.
    if (g_io_price_controller.enabled && g_expansion_cost_theta != 0.0) {
        fprintf(stderr, "--io-target-per-query drives theta by dual ascent (eq. 7) and "
                        "overwrites --expansion-cost-theta at the first batch boundary; "
                        "pass one or the other\n");
        return 1;
    }
    // Thm 5.2's rule rewrites lambda every batch, so a fixed one is the other arm.
    if (g_lambda_auto_quantile > 0.0 && g_rewire_margin_lambda != 0.0) {
        fprintf(stderr, "--lambda-auto sets lambda from the measured boundary gap and "
                        "overwrites --rewire-margin-lambda after the first batch; "
                        "pass one or the other\n");
        return 1;
    }
    // Section 10 step (4)'s rule, same story on a different sample.
    if (g_lambda_noise_floor_quantile > 0.0 && g_rewire_margin_lambda != 0.0) {
        fprintf(stderr, "--lambda-noise-floor sets lambda from the measured |g| noise "
                        "floor and overwrites --rewire-margin-lambda after the first "
                        "batch; pass one or the other\n");
        return 1;
    }
    // Two rules on two different quantities. Section 10 step (7) runs them together
    // as max(half the boundary-gap p10, the noise floor); that is a third rule and
    // neither flag implements it, so refuse the pair rather than pick a winner.
    if (g_lambda_auto_quantile > 0.0 && g_lambda_noise_floor_quantile > 0.0) {
        fprintf(stderr, "--lambda-auto (boundary gap, Thm 5.2) and --lambda-noise-floor "
                        "(|g| noise floor, section 10 step 4) are separate rules and would "
                        "each rewrite lambda every batch; pass one or the other\n");
        return 1;
    }
    if (g_lambda_noise_floor_quantile < 0.0 || g_lambda_noise_floor_quantile > 1.0) {
        fprintf(stderr, "--lambda-noise-floor-quantile must lie in (0, 1]; %g indexes "
                        "outside the sample\n", g_lambda_noise_floor_quantile);
        return 1;
    }
    if (g_lambda_calibrate_every_batch && g_rewire_commit_direct) {
        fprintf(stderr, "--lambda-calibrate-every-batch requires batched overlay commits; "
                        "remove --rewire-commit-direct\n");
        return 1;
    }
    if (g_write_delay_prob < 0.0 || g_write_delay_prob > 1.0) {
        fprintf(stderr, "--write-delay-prob must lie in [0, 1]; %g is not a probability\n",
                g_write_delay_prob);
        return 1;
    }
    if (g_read_delay_us < 0.0) {
        fprintf(stderr, "--read-delay-us must be nonnegative: %g\n", g_read_delay_us);
        return 1;
    }
    if (g_read_delay_prob < 0.0 || g_read_delay_prob > 1.0) {
        fprintf(stderr, "--read-delay-prob must lie in [0, 1]; %g is not a probability\n",
                g_read_delay_prob);
        return 1;
    }
    if (g_read_delay_start_query < 0) {
        fprintf(stderr, "--read-delay-start-query must be nonnegative: %ld\n",
                g_read_delay_start_query);
        return 1;
    }
    if (!detector_calibration_csv.empty() &&
        (detector_calibration_window <= 0 || (!detector_control && detector_calibration_freeze_at < 0))) {
        fprintf(stderr, "detector calibration needs a positive window and, without control, a nonnegative freeze point\n");
        return 1;
    }
    if (detector_control && (detector_calibration_csv.empty() || detector_stop_windows <= 0 ||
        detector_min_adapt_q < 0 || detector_stop_threshold < 0.0 ||
        detector_drift_threshold < 0.0 || detector_drift_threshold > 1.0 ||
        g_detector_sample_every <= 0 || g_search_only)) {
        fprintf(stderr, "--detector-control needs its CSV, valid thresholds/windows, and learning enabled\n");
        return 1;
    }
    // The charge blocks the calling thread, which only stands in for a slower
    // device while the write is ON the critical path. With async write-back it is
    // not, so say so rather than let the arm be read as a measurement.
    if (g_write_delay_us > 0.0 && async_rewire_commit)
        fprintf(stderr, "NOTE: --write-delay-us with --async-rewire-commit charges the "
                        "writer thread, not the critical path; the run models a device "
                        "rather than measuring one\n");
    // Both restore or freeze the graph, so I_hat can never respond to the price.
    if (g_io_price_controller.enabled && (isolated || no_rewire)) {
        fprintf(stderr, "--io-target-per-query needs a graph that persists across "
                        "batches; --isolated and --no-rewire both remove one\n");
        return 1;
    }
    // c(v) is the price's only input and is measured only under the census.
    if (g_io_price_controller.enabled && !g_track_expansion_cost) {
        g_track_expansion_cost = true;
        fprintf(stderr, "--io-target-per-query implies --track-expansion-cost "
                        "(c(v) is the price's only input)\n");
    }
    if (!gain_terms_csv.empty()) {
        g_gain_terms_file = fopen(gain_terms_csv.c_str(), "w");
        if (!g_gain_terms_file) { fprintf(stderr, "cannot open %s\n", gain_terms_csv.c_str()); return 1; }
        fprintf(g_gain_terms_file,
                "gqc,u,deg_u,R,n_add,n_del,sum_new,sum_old,g,n_add_fallback,n_del_fallback\n");
    }
    if (!detector_calibration_csv.empty()) {
        g_detector_calibration.file = fopen(detector_calibration_csv.c_str(), "w");
        if (!g_detector_calibration.file) {
            fprintf(stderr, "cannot open %s\n", detector_calibration_csv.c_str());
            return 1;
        }
        g_detector_calibration.window_q = detector_calibration_window;
        g_detector_calibration.freeze_q = detector_calibration_freeze_at;
        g_detector_calibration.control = detector_control;
        g_detector_calibration.stop_threshold = detector_stop_threshold;
        g_detector_calibration.stop_windows = detector_stop_windows;
        g_detector_calibration.min_adapt_q = detector_min_adapt_q;
        g_detector_calibration.drift_threshold = detector_drift_threshold;
        fprintf(g_detector_calibration.file,
                "q,mode,event,stable_windows,q_updates,q_delta,q_innovation,buffer_unique,buffer_novel,buffer_novel_frac,signature_size,novelty_sample_queries\n");
        fprintf(stderr, "detector frozen-mode sampling: every %d quer%s\n",
                g_detector_sample_every, g_detector_sample_every == 1 ? "y" : "ies");
    }

    if (!g_audit_dump_csv.empty()) {
        g_audit_dump_f = fopen(g_audit_dump_csv.c_str(), "w");
        if (!g_audit_dump_f) { fprintf(stderr, "cannot open %s\n", g_audit_dump_csv.c_str()); return 1; }
        fprintf(g_audit_dump_f, "gqc,y,outcome\n");
    }
    if (!g_audit_repair_dump_csv.empty()) {
        g_audit_repair_dump_f = fopen(g_audit_repair_dump_csv.c_str(), "w");
        if (!g_audit_repair_dump_f) { fprintf(stderr, "cannot open %s\n", g_audit_repair_dump_csv.c_str()); return 1; }
        fprintf(g_audit_repair_dump_f, "audit_trial,y,u,v,improvement\n");
    }
    if (!g_audit_retest_dump_csv.empty()) {
        g_audit_retest_dump_f = fopen(g_audit_retest_dump_csv.c_str(), "w");
        if (!g_audit_retest_dump_f) { fprintf(stderr, "cannot open %s\n", g_audit_retest_dump_csv.c_str()); return 1; }
        fprintf(g_audit_retest_dump_f, "audit_trial,y,pre_recall_at1,post_recall_at1,post_top1,post_sector_reached\n");
    }
    if (!g_nav_damage_dump_csv.empty()) {
        g_nav_damage_dump_f = fopen(g_nav_damage_dump_csv.c_str(), "w");
        if (!g_nav_damage_dump_f) { fprintf(stderr, "cannot open %s\n", g_nav_damage_dump_csv.c_str()); return 1; }
        fprintf(g_nav_damage_dump_f, "u,v,dist_removed\n");
    }
    // Damage is injected at a query ordinal, so a run that asks for it without
    // saying when would silently never fire.
    // The ef=1 probe fails ~25% of trials on a PRISTINE index, so any eps_r
    // anchored on that floor is mis-scaled no matter what the arithmetic is.
    // --protect-slo-warmup calibrates on the live audit stream instead, which is
    // the combination that works at ef=1; a hand-set --protect-slo also stands.
    if (g_protect_auto_slo && g_audit_ef == 1 && g_slo_warmup_n <= 0)
        throw std::runtime_error("--protect-auto-slo with --protect-audit-ef 1 is mis-scaled "
                                 "(the pristine-graph floor at ef=1 is ~0.25, so eps_r lands "
                                 "above any rate the audit can report); use "
                                 "--protect-slo-warmup N or set --protect-slo explicitly");
    if ((g_nav_damage_frac > 0.0 || g_nav_damage_longest > 0 || !g_nav_damage_target_ids.empty() ||
         g_nav_damage_prescan_ef > 0) &&
        g_nav_damage_at < 0)
        throw std::runtime_error("--nav-damage-frac/--nav-damage-longest/--nav-damage-target-ids/"
                                 "--nav-damage-from-prescan-ef needs --nav-damage-at <query> to fire at");
    if (g_nav_damage_prescan_ef > 0 && !g_nav_damage_target_ids.empty())
        throw std::runtime_error("--nav-damage-from-prescan-ef and --nav-damage-target-ids "
                                 "are mutually exclusive (the prescan derives its own target-ids list)");
    if (g_nav_damage_prescan_force_audit_target && !g_audit_target_ids.empty())
        throw std::runtime_error("--nav-damage-prescan-force-audit-target and --audit-target-ids "
                                 "are mutually exclusive (the prescan derives its own audit-target list)");
    if (g_nav_damage_prescan_force_audit_target && g_nav_damage_prescan_ef == 0)
        throw std::runtime_error("--nav-damage-prescan-force-audit-target requires "
                                 "--nav-damage-from-prescan-ef");
    if (g_nav_damage_prescan_use_side_eval_a && g_nav_damage_prescan_ef == 0)
        throw std::runtime_error("--nav-damage-prescan-use-side-eval-a requires "
                                 "--nav-damage-from-prescan-ef");
    if (g_protect_repair_min_improvement < 0.0f)
        throw std::runtime_error("--protect-repair-min-improvement must be nonnegative");
    if (g_log_rescue_max <= 0)
        throw std::runtime_error("--protect-log-rescue-max must be positive");
    if (g_nav_damage_prescan_target_cap < 0)
        throw std::runtime_error("--nav-damage-prescan-target-cap must be nonnegative");

    g.layout_commit_batched = layout_commit_batched;
    g.load(index_prefix, writeback, work_copy, reuse_work_copy);
    if (!layout_map_load.empty()) g.load_layout_map(layout_map_load);
    if (g.hotcache.async_writeback)
        g.hotcache.start_async_writer(g.write_fd, g.num_sectors_per_node * diskann::defaults::SECTOR_LEN);
    if (g.hotcache.enabled) {
        // Default ceiling is the whole index: one slot per sector the file has, so
        // "unbounded" cannot silently mean "until it wasn't". At 3 nodes per 4KB
        // sector this is ~13.7 GB on T2I-10M, which a 1k-query batch will not
        // approach -- the ceiling is a tripwire, not a budget.
        const uint64_t sectors_in_index =
            g.nnodes_per_sector > 0 ? (g.num_points + g.nnodes_per_sector - 1) / g.nnodes_per_sector
                                    : g.num_points * g.num_sectors_per_node;
        if (hot_cache_ceiling == 0) hot_cache_ceiling = (size_t)sectors_in_index;
        g.hotcache.ceiling_sectors = hot_cache_ceiling;
        if (hot_cache_cap != 0)
            fprintf(stderr, "hot-node cache: CAPPED at %zu sectors (%.1f MB), FROZEN on arrival "
                            "(no eviction, admission simply stops), %s%s, lifetime=%dq. The hit "
                            "rate is the truncated ceiling, still not a deployable policy.\n",
                    hot_cache_cap,
                    (double)(hot_cache_cap * g.num_sectors_per_node * diskann::defaults::SECTOR_LEN) / 1048576.0,
                hot_cache_wt ? "WRITE-THROUGH (flush_edits patches resident sectors in place)"
                             : "read-only (flush_edits ignores it)",
                (async_rewire_commit && g_rewire_commit_direct ? " + ASYNC-COMMIT + DIRECT-COMMIT" :
                 async_rewire_commit ? " + ASYNC-COMMIT (non-resident dirty sectors take a sync RMW)" :
                 g_rewire_commit_direct ? " + DIRECT-COMMIT" : ""),
                hot_cache_lifetime_q);
        else
        fprintf(stderr, "hot-node cache: UNBOUNDED, no eviction, %s%s, lifetime=%s, "
                        "ceiling=%zu sectors (%.1f GB). Admission is unconditional, "
                        "so the hit rate is a CEILING, not a deployable policy.\n",
                hot_cache_wt ? "WRITE-THROUGH (flush_edits patches in place)"
                             : "read-only (flush_edits ignores it)",
                (async_rewire_commit && g_rewire_commit_direct ? " + ASYNC-COMMIT + DIRECT-COMMIT" :
                 async_rewire_commit ? " + ASYNC-COMMIT" :
                 g_rewire_commit_direct ? " + DIRECT-COMMIT" : ""),
                (hot_cache_lifetime_q == 0 ? std::string("never dropped")
                                           : std::to_string(hot_cache_lifetime_q) + " queries").c_str(),
                hot_cache_ceiling,
                (double)hot_cache_ceiling * g.num_sectors_per_node *
                    diskann::defaults::SECTOR_LEN / 1073741824.0);
    }
    if (learner_hot_cap > 0)
        fprintf(stderr, "learner hot-state cache: H=%zu source nodes (LFU, evicts Q/B_buf only)\n",
                learner_hot_cap);
    if (g_edge_protection && g_K_max <= 0) g_K_max = std::max(1, (int)g.max_degree / 2);

    auto npz = npio::load_npz(cache_npz);
    auto &refs_a = find_npz(npz, "refs");
    auto &proxy_gt_a = find_npz(npz, "proxy_gt");
    auto &sigma_levels_a = find_npz(npz, "sigma_levels");
    int nref = (int)refs_a.shape[0];
    const int n_reference_vectors_in_cache = nref;
    if (n_reference_vectors_override > 0) {
        if (n_reference_vectors_override > nref) {
            fprintf(stderr, "--n-refs %d exceeds the cache's %d reference vectors\n",
                    n_reference_vectors_override, nref);
            return 1;
        }
        nref = n_reference_vectors_override;
    }
    int n_sig = (int)sigma_levels_a.shape[0];
    if (sigma_index < 0 || sigma_index >= n_sig) { fprintf(stderr, "bad sigma-index\n"); return 1; }
    if ((int)refs_a.shape[1] != g_query_dim) {
        fprintf(stderr, "cache dim %d != index query dim %d%s\n", (int)refs_a.shape[1], g_query_dim,
                g_mips ? " (MIPS index: query dim is index dim - 1)" : "");
        return 1;
    }
    // --sigma overrides the cache's grid so off-grid levels can be tested without
    // rebuilding the npz. Safe now only because --true-gt exists: the cache's
    // proxy_gt is the reference's top-k and its validity is a function of sigma,
    // so an off-grid sigma MUST be scored against a recomputed exact truthset.
    float sigma = sigma_override >= 0.0 ? (float)sigma_override : sigma_levels_a.f32()[sigma_index];
    if (sigma_override >= 0.0)
        fprintf(stderr, "SIGMA_OVERRIDE: %.5f (grid level %d was %.5f) -- proxy_gt is off its "
                "calibration, use --true-gt\n", sigma, sigma_index, sigma_levels_a.f32()[sigma_index]);

    fprintf(stderr, "DiskANN index: N=%llu dim=%d R=%llu medoid=%u  nodes/sector=%llu "
            "sectors/node=%llu  pq_chunks=%llu (%.2f GB codes in RAM)\n",
            (unsigned long long)g.num_points, g_dim, (unsigned long long)g.max_degree, g.medoid,
            (unsigned long long)g.nnodes_per_sector, (unsigned long long)g.num_sectors_per_node,
            (unsigned long long)g.n_chunks, (double)g.pq_codes.size() / 1e9);
    fprintf(stderr, "sigma_index=%d sigma=%.5f  train_n=%d disk_beam=%llu writeback=%d\n",
            sigma_index, sigma, n_queries, (unsigned long long)disk_beam_width, (int)writeback);

    std::vector<float> train_v; std::vector<int64_t> train_g;
    std::vector<int> train_center;
    if (!ext_query_file.empty()) {
        if (true_gt_path.empty())
            throw std::runtime_error("--ext-query-file requires --true-gt (no cache proxy_gt exists for a "
                                     "real query stream)");
        load_query_stream_bin(ext_query_file, train_v, n_queries, g_query_dim);
        train_g.assign((size_t)n_queries * g_k, 0); // filled by load_truthset_ids below
    } else {
        make_single_sigma_stream(refs_a.f32(), proxy_gt_a.i64(), nref, sigma, n_queries, seed, train_v, train_g,
                                 query_cardinality, query_order, query_repeat,
                                 interleave_centers_block, &train_center);
    }
    // The label source is now a default rather than a flag, so it has to appear
    // in every log -- otherwise an oracle run and an honest one are
    // indistinguishable after the fact.
    fprintf(stderr, "LABEL: T_t = %s%s\n",
            g_label_from_gt ? "GROUND TRUTH (oracle -- not deployable; pass --self-label for the honest arm)"
                            : "self-label (Algorithm 1 line 8)",
            g_label_from_gt && true_gt_path.empty()
                ? "  [source: cache proxy_gt, the reference vector top-k; pass --true-gt for the exact one]" : "");

    if (!dump_queries_path.empty()) write_query_stream_bin(dump_queries_path, train_v, n_queries, g_query_dim);
    if (!true_gt_path.empty()) load_truthset_ids(true_gt_path, n_queries, g_k, train_g);
    if (!ext_query_file.empty() && ext_query_limit > 0 && n_queries > ext_query_limit) {
        n_queries = ext_query_limit;
        train_v.resize((size_t)n_queries * g_query_dim);
        train_g.resize((size_t)n_queries * g_k);
        fprintf(stderr, "EXTQUERY: clamped to first %d queries by --ext-query-limit\n", n_queries);
    }
    // The truthset is not only scored against: under the default oracle label
    // T_t IS the truthset row, and train_batch pushes it straight into
    // B_buf[u].order, which do_rewire later hands to codeword_pair_distance as
    // a node id. An id outside [0, N) is therefore an out-of-bounds PQ-code
    // read in the rewire path, not a recall miss -- check once, here.
    for (size_t gi = 0; gi < train_g.size(); gi++) {
        const int64_t id = train_g[gi];
        if (id < 0 || (uint64_t)id >= g.num_points) {
            fprintf(stderr, "truthset id out of range: train_g[%zu]=%lld, index has %llu points\n",
                    gi, (long long)id, (unsigned long long)g.num_points);
            return 1;
        }
    }
    fprintf(stderr, "query stream: n=%d distinct=%d order=%s repeat=%d refs=%d/%d interleave_centers=%d\n", n_queries,
            (query_cardinality > 0 && query_cardinality < n_queries) ? query_cardinality : n_queries,
            query_order == QUERY_ORDER_BLOCKED ? "blocked"
            : query_order == QUERY_ORDER_REPEAT ? "repeat" : "iid",
            query_repeat, nref, n_reference_vectors_in_cache, interleave_centers_block);
    if (interleave_centers_block > 0) {
        long c0 = 0, c1 = 0;
        for (int cc : train_center) (cc == 0 ? c0 : c1)++;
        fprintf(stderr, "CENTER_SPLIT: block=%d center0=%ld center1=%ld\n",
                interleave_centers_block, c0, c1);
    }
    if (!query_centers_csv.empty()) {
        FILE *qcf = fopen(query_centers_csv.c_str(), "w");
        if (!qcf) { fprintf(stderr, "cannot open %s\n", query_centers_csv.c_str()); return 1; }
        fprintf(qcf, "q,center\n");
        for (int i = 0; i < n_queries; i++) fprintf(qcf, "%d,%d\n", i + 1, train_center[i]);
        fclose(qcf);
        fprintf(stderr, "  wrote %s\n", query_centers_csv.c_str());
    }

    if (diagnose_recall_n > 0) {
        diagnose_recall(g, "train", train_v.data(), train_g.data(), g_k,
                        std::min(diagnose_recall_n, n_queries), K, ef, disk_beam_width);
        return 0; // read-only probe: never trains, never writes back
    }

    std::vector<float> post_v;
    std::vector<int64_t> post_g;
    std::vector<int> post_center;
    const bool need_post_test_stream =
        (post_test_n > 0 || !post_test_ext_query_file.empty()) &&
        (!post_test_results_csv.empty() || !checkpoint_test_results_csv.empty() ||
         !dump_post_test_queries_path.empty() || !post_test_efs.empty() ||
         !post_test_per_query_results_csv.empty());
    if (need_post_test_stream) {
        if (!post_test_ext_query_file.empty()) {
            if (post_test_true_gt_path.empty())
                throw std::runtime_error("--post-test-ext-query-file requires --post-test-true-gt");
            load_query_stream_bin(post_test_ext_query_file, post_v, post_test_n, g_query_dim);
            load_truthset_ids(post_test_true_gt_path, post_test_n, g_k, post_g);
            fprintf(stderr, "post-test stream: n=%d [source: --post-test-ext-query-file]\n", post_test_n);
        } else {
        make_single_sigma_stream(refs_a.f32(), proxy_gt_a.i64(), nref, sigma, post_test_n,
                                 post_test_seed, post_v, post_g,
                                 0, QUERY_ORDER_IID, 1, interleave_centers_block, &post_center,
                                 post_test_sequential_refs);
        if (!dump_post_test_queries_path.empty())
            write_query_stream_bin(dump_post_test_queries_path, post_v, post_test_n, g_query_dim);
        if (!post_test_true_gt_path.empty())
            load_truthset_ids(post_test_true_gt_path, post_test_n, g_k, post_g);
        fprintf(stderr, "post-test stream: n=%d seed=%ld%s\n", post_test_n, post_test_seed,
                post_test_true_gt_path.empty() ? " [source: cache proxy_gt]" : " [source: --post-test-true-gt]");
        } // end else (sigma-generated stream)
    } else if (post_test_n == 0 && !checkpoint_test_results_csv.empty()) {
        auto &eval_vecs_a = find_npz(npz, "eval_vecs");
        auto &eval_gt_a = find_npz(npz, "eval_gt");
        post_test_n = (int)eval_vecs_a.shape[1];
        const size_t vec_offset = (size_t)sigma_index * post_test_n * g_query_dim;
        const size_t gt_offset = (size_t)sigma_index * post_test_n * g_k;
        post_v.assign(eval_vecs_a.f32() + vec_offset,
                      eval_vecs_a.f32() + vec_offset + (size_t)post_test_n * g_query_dim);
        post_g.assign(eval_gt_a.i64() + gt_offset,
                      eval_gt_a.i64() + gt_offset + (size_t)post_test_n * g_k);
        fprintf(stderr, "checkpoint-test stream: n=%d sigma_index=%d [source: cache eval_vecs/eval_gt]\n",
                post_test_n, sigma_index);
    }

    if (g_skeleton_protect) {
        int n_profile = g_skeleton_profile_queries > 0 ? g_skeleton_profile_queries : n_queries;
        if (n_profile > n_queries) {
            fprintf(stderr, "  [skeleton] --skeleton-profile-queries %d exceeds --n-queries %d "
                            "(train_v only holds %d vectors); clamping.\n",
                    n_profile, n_queries, n_queries);
            n_profile = n_queries;
        }
        extract_skeleton(g, train_v.data(), n_profile, K, ef, disk_beam_width);
    }

    // Before any method runs, so the graph probed really is G_0.
    if (measure_build_floor_n > 0) {
        const double b0_audit = measure_build_floor(g, measure_build_floor_n, (uint64_t)ef, disk_beam_width);
        if (g_protect_auto_slo) {
            g_epsilon_r = b0_audit + g_slo_margin;
            fprintf(stderr, "  [governor] --protect-auto-slo: eps_r <- b0_audit + margin = %.4f + %.4f = %.4f\n",
                    b0_audit, g_slo_margin, g_epsilon_r);
        }
        // The probes ran with collect_epoch_misses on, so they left their fetched
        // ids in epoch_missed. do_rewire only reads the growth past a per-query
        // watermark (miss_mark), so this is inert -- but it inflates the
        // `epoch_misses=` figure the layout line prints, so drop it.
        g.epoch_missed.clear();
    }

    std::vector<std::pair<std::string, MethodCfg>> all_methods = {
        {"greedy",          {"QL-Greedy",          "binary"}},  // alias of beam_ucb
        {"greedy_eps",      {"QL-Greedy-Eps",      "binary"}},  // alias of beam_ucb
        {"beam",            {"QL-Beam",            "binary"}},  // alias of beam_ucb
        {"beam_ucb",        {"QL-Beam-UCB",        "binary"}},
        {"beam_ucb_recall", {"QL-Beam-UCB-Recall", "recall"}},
        {"beam_ucb_one",    {"QL-Beam-UCB-One",    "one"}},
    };
    std::vector<std::string> want = split_csv(methods_arg);

    // No recall column: recall is scored offline from --dump-query-results.
    printf("method,q,net_edge_delta,touched_nodes,"
           "reads_traversal,reads_rewire,writes_rewire,io_s_total,k_p,"
           "lambda,theta,i_hat\n");
    fflush(stdout);

    if (!query_results_csv.empty()) {
        g_query_results_file = fopen(query_results_csv.c_str(), "w");
        if (!g_query_results_file) { fprintf(stderr, "cannot open %s\n", query_results_csv.c_str()); return 1; }
        fprintf(g_query_results_file, "q");
        for (int j = 0; j < K; j++) fprintf(g_query_results_file, ",res%d", j);
        for (int j = 0; j < g_k; j++) fprintf(g_query_results_file, ",gt%d", j);
        fputc('\n', g_query_results_file);
    }

    FILE *pqf = nullptr;
    if (!per_query_csv.empty()) {
        pqf = fopen(per_query_csv.c_str(), "w");
        if (!pqf) fprintf(stderr, "cannot open --per-query-csv %s\n", per_query_csv.c_str());
        else fprintf(pqf, "method,q_idx,recall,reset_time_s,reads_reset,writes_reset,"
                          "query_time_s,reads_traversal,io_s_traversal,"
                          "rewire_time_s,reads_rewire,writes_rewire,io_s_rewire\n");
    }

    FILE *ctf = nullptr;
    if (!checkpoint_test_results_csv.empty()) {
        ctf = fopen(checkpoint_test_results_csv.c_str(), "w");
        if (!ctf) { fprintf(stderr, "cannot open %s\n", checkpoint_test_results_csv.c_str()); return 1; }
        fprintf(ctf, "method,q,n,recall\n");
    }

    struct PostTestMetrics {
        double recall = 0.0, qps = 0.0, mean_latency_us = 0.0, wall_qps = 0.0;
        double p99_us = 0.0, io_s = 0.0, wall_s = 0.0;
        long reads = 0, device_reads = 0, sector_reqs = 0;
    };
    FILE *post_per_query_file = nullptr;
    if (!post_test_per_query_results_csv.empty()) {
        post_per_query_file = fopen(post_test_per_query_results_csv.c_str(), "w");
        if (!post_per_query_file)
            throw std::runtime_error("cannot open " + post_test_per_query_results_csv);
        fprintf(post_per_query_file, "stage,q,ef,query_idx,recall,latency_us,qps\n");
    }
    auto run_post_test = [&](const MethodCfg &m, long q, FILE *summary_file, FILE *results_file) {
        if (post_test_n <= 0) return PostTestMetrics{};
        g.hotcache.wait_idle();
        const bool saved_hotcache_enabled = g.hotcache.enabled;
        // An EF point is one cold serving pass at the configured cache capacity.
        // Reset before it so no later EF inherits the prior point's resident set.
        if (post_test_use_hot_cache && g.hotcache.enabled) g.hotcache.drop();
        if (!post_test_use_hot_cache) g.hotcache.enabled = false;
        if (!post_test_barrier_dir.empty()) {
            std::filesystem::create_directories(post_test_barrier_dir);
            const std::string marker = post_test_barrier_dir + "/ef" + std::to_string(ef) +
                                       ".worker" + std::to_string(post_test_shard_index);
            std::ofstream ready(marker);
            ready.close();
            for (int worker = 0; worker < post_test_shard_count; worker++) {
                const std::string peer = post_test_barrier_dir + "/ef" + std::to_string(ef) +
                                         ".worker" + std::to_string(worker);
                while (!std::filesystem::exists(peer))
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        }
        IOCounters eval_io;
        int64_t hits = 0;
        std::vector<double> latency_us;
        const int shard_n = (post_test_n - post_test_shard_index + post_test_shard_count - 1) /
                            post_test_shard_count;
        latency_us.reserve(std::max(0, shard_n));
        const Clock::time_point post_test_t0 = Clock::now();
        {
            IOPhase ph(eval_io);
            for (int i = post_test_shard_index; i < post_test_n; i += post_test_shard_count) {
                std::vector<int> result_ids, visited;
                const auto query_t0 = Clock::now();
                g.search(post_v.data() + (size_t)i * g_query_dim, (uint64_t)K, (uint64_t)ef,
                         disk_beam_width, result_ids, visited);
                latency_us.push_back(1e6 * secs_since(query_t0));
                std::unordered_set<int> gtset;
                for (int j = 0; j < K; j++) gtset.insert((int)post_g[(size_t)i * g_k + j]);
                int hits_i = 0;
                for (int id : result_ids) if (gtset.count(id)) hits_i++;
                hits += hits_i;
                if (post_per_query_file) {
                    const double latency = latency_us.back();
                    fprintf(post_per_query_file, "%s,%ld,%d,%d,%.6f,%.3f,%.6f\n",
                            q == 0 ? "initial" : "post_train", q, ef, i,
                            (double)hits_i / (double)K, latency,
                            latency > 0.0 ? 1e6 / latency : 0.0);
                }
                if (results_file) {
                    fprintf(results_file, "%d", i + 1);
                    for (int j = 0; j < K; j++)
                        fprintf(results_file, ",%d", j < (int)result_ids.size() ? result_ids[j] : -1);
                    for (int j = 0; j < g_k; j++)
                        fprintf(results_file, ",%lld", (long long)post_g[(size_t)i * g_k + j]);
                    fputc('\n', results_file);
                }
            }
        }
        g.hotcache.enabled = saved_hotcache_enabled;
        if (post_per_query_file) fflush(post_per_query_file);
        const double post_test_wall_s = secs_since(post_test_t0);
        const double recall = (double)hits / ((double)latency_us.size() * K);
        std::sort(latency_us.begin(), latency_us.end());
        double latency_sum_us = 0.0;
        for (double latency : latency_us) latency_sum_us += latency;
        const double mean_latency_us = latency_sum_us / latency_us.size();
        const double p99_us = latency_us[(size_t)std::ceil(0.99 * latency_us.size()) - 1];
        if (summary_file) {
            fprintf(summary_file, "%s,%ld,%zu,%.6f\n", m.label.c_str(), q, latency_us.size(), recall);
            fflush(summary_file);
        }
        // qps is the reciprocal of mean single-query search latency. wall_qps
        // separately captures aggregate scored-loop throughput, including
        // evaluation and output overhead.
        const PostTestMetrics metrics{recall,
            mean_latency_us > 0.0 ? 1e6 / mean_latency_us : 0.0,
            mean_latency_us,
            post_test_wall_s > 0 ? (double)latency_us.size() / post_test_wall_s : 0.0,
            p99_us, eval_io.io_s, post_test_wall_s, eval_io.reads, eval_io.device_reads,
            eval_io.sector_reqs};
        fprintf(stderr, "  TESTRECALL: q=%ld n=%zu/%d worker=%d/%d recall@%d=%.6f reads=%ld device_reads=%ld sector_reqs=%ld io_s=%.3f wall_s=%.3f qps=%.2f mean_latency_us=%.1f wall_qps=%.2f p99_us=%.1f\n",
                q, latency_us.size(), post_test_n, post_test_shard_index, post_test_shard_count,
                K, recall, eval_io.reads, eval_io.device_reads,
                eval_io.sector_reqs, eval_io.io_s, post_test_wall_s,
                metrics.qps, mean_latency_us, metrics.wall_qps, p99_us);
        fprintf(stderr, "  TESTREWIRED: q=%ld rewired_rows=%ld ids=%ld ids_on_rewired=%ld "
                        "sector_reqs=%ld sector_reqs_on_rewired=%ld\n",
                q, g_rewired_row_count, eval_io.reads, eval_io.ids_on_rewired,
                eval_io.sector_reqs, eval_io.sector_reqs_on_rewired);
        return metrics;
    };

    std::vector<float> side_eval_a_v, side_eval_b_v;
    std::vector<int64_t> side_eval_a_g, side_eval_b_g;
    int side_eval_a_n = 0, side_eval_b_n = 0;
    if (!side_eval_a_query.empty()) {
        load_query_stream_bin(side_eval_a_query, side_eval_a_v, side_eval_a_n, g_query_dim);
        load_truthset_ids(side_eval_a_gt, side_eval_a_n, g_k, side_eval_a_g);
    }
    if (!side_eval_b_query.empty()) {
        load_query_stream_bin(side_eval_b_query, side_eval_b_v, side_eval_b_n, g_query_dim);
        load_truthset_ids(side_eval_b_gt, side_eval_b_n, g_k, side_eval_b_g);
    }
    FILE *sef = nullptr;
    if (!side_eval_csv.empty()) {
        sef = fopen(side_eval_csv.c_str(), "w");
        if (!sef) { fprintf(stderr, "cannot open %s\n", side_eval_csv.c_str()); return 1; }
        fprintf(sef, "method,q,set,ef,n,recall,qps,mean_latency_us,p99_us,wall_qps,reads,device_reads,sector_reqs,io_s,wall_s\n");
    }
    FILE *sepf = nullptr;
    if (!side_eval_per_query_csv.empty()) {
        sepf = fopen(side_eval_per_query_csv.c_str(), "w");
        if (!sepf) { fprintf(stderr, "cannot open %s\n", side_eval_per_query_csv.c_str()); return 1; }
        fprintf(sepf, "method,q,set,ef,query_idx,recall,latency_us,qps\n");
    }
    if (side_eval_efs.empty()) side_eval_efs.push_back(side_eval_ef > 0 ? side_eval_ef : ef);
    auto run_side_eval = [&](const MethodCfg &m, long q, const char *label, const float *sv,
                             const int64_t *sg, int sn, uint64_t search_ef) {
        if (sn <= 0) return;
        g.hotcache.wait_idle();
        const bool saved_hotcache_enabled = g.hotcache.enabled;
        if (!side_eval_use_hot_cache) g.hotcache.enabled = false;
        IOCounters eval_io;
        int64_t hits = 0;
        std::vector<double> latency_us;
        latency_us.reserve(sn);
        const auto wall_t0 = Clock::now();
        {
            IOPhase ph(eval_io);
            for (int i = 0; i < sn; i++) {
                std::vector<int> result_ids, visited;
                const auto query_t0 = Clock::now();
                g.search(sv + (size_t)i * g_query_dim, (uint64_t)K, search_ef,
                         disk_beam_width, result_ids, visited);
                const double latency = 1e6 * secs_since(query_t0);
                latency_us.push_back(latency);
                std::unordered_set<int> gtset;
                for (int j = 0; j < K; j++) gtset.insert((int)sg[(size_t)i * g_k + j]);
                int64_t hits_i = 0;
                for (int id : result_ids) if (gtset.count(id)) hits_i++;
                hits += hits_i;
                if (sepf) fprintf(sepf, "%s,%ld,%s,%lu,%d,%.6f,%.3f,%.6f\n",
                                  m.label.c_str(), q, label, search_ef, i,
                                  (double)hits_i / (double)K, latency,
                                  latency > 0.0 ? 1e6 / latency : 0.0);
            }
        }
        const double wall_s = secs_since(wall_t0);
        g.hotcache.enabled = saved_hotcache_enabled;
        if (sepf) fflush(sepf);
        const double recall = (double)hits / ((double)sn * K);
        double latency_sum = 0.0;
        for (double latency : latency_us) latency_sum += latency;
        const double mean_latency = latency_sum / latency_us.size();
        std::sort(latency_us.begin(), latency_us.end());
        const double p99 = latency_us[(size_t)std::ceil(0.99 * latency_us.size()) - 1];
        const double qps = mean_latency > 0.0 ? 1e6 / mean_latency : 0.0;
        const double wall_qps = wall_s > 0.0 ? (double)sn / wall_s : 0.0;
        fprintf(stderr, "  SIDEEVAL: %s q=%ld ef=%lu n=%d recall@%d=%.6f qps=%.2f mean_latency_us=%.1f p99_us=%.1f wall_qps=%.2f reads=%ld device_reads=%ld sector_reqs=%ld io_s=%.3f wall_s=%.3f cache=%s\n",
                label, q, search_ef, sn, K, recall, qps, mean_latency, p99,
                wall_qps, eval_io.reads, eval_io.device_reads, eval_io.sector_reqs,
                eval_io.io_s, wall_s, side_eval_use_hot_cache ? "live" : "off");
        if (sef) {
            fprintf(sef, "%s,%ld,%s,%lu,%d,%.6f,%.6f,%.3f,%.3f,%.6f,%ld,%ld,%ld,%.6f,%.6f\n",
                    m.label.c_str(), q, label, search_ef, sn, recall, qps,
                    mean_latency, p99, wall_qps, eval_io.reads, eval_io.device_reads,
                    eval_io.sector_reqs, eval_io.io_s, wall_s);
            fflush(sef);
        }
    };
    auto run_side_evals = [&](const MethodCfg &m, long q) {
        for (int search_ef : side_eval_efs) {
            run_side_eval(m, q, "A", side_eval_a_v.data(), side_eval_a_g.data(), side_eval_a_n, search_ef);
            run_side_eval(m, q, "B", side_eval_b_v.data(), side_eval_b_g.data(), side_eval_b_n, search_ef);
        }
    };

    for (auto &name : want) {
        MethodCfg m;
        bool found = false;
        for (auto &kv : all_methods) if (kv.first == name) { m = kv.second; found = true; break; }
        if (!found) { fprintf(stderr, "unknown method %s\n", name.c_str()); continue; }
        g_detector_calibration.reset();

        // Reset I/O counters per method so each method's attribution is clean.
        // Must precede the restore below, or the restore's I/O is silently discarded.
        g_io_traversal.clear(); g_io_rewire.clear(); g_io_reset.clear();
        g_io_layout.clear(); g_io_audit.clear();
        // g_io_skeleton is NOT cleared: extraction runs once, before any
        // method, so per-method clearing would erase its only record.

        // The c(v) marginal is per method too: it is measured against one
        // method's traversal pattern and means nothing carried into the next.
        g_expansion_cost_by_node.clear();

        // Governor state is per method too: the governed level is a property
        // of one method's training run, not something later methods should
        // inherit. (No ranking cache to clear -- live_rank_of never caches.)
        g_repair_volume = 0;
        g_last_rows_changed = g_last_hot_nodes = g_last_deleted = g_first_batch_deleted = 0;
        g_audit_pass = g_audit_degenerate = g_audit_nobuf = g_audit_norescue = g_audit_attrib = 0;
        g_audit_log_rescue = g_audit_search_proposals = 0;
        g_direct_repair_queue.clear();
        g_direct_repair_retest_targets.clear();
        g_direct_repair_applied = g_direct_repair_existing = g_direct_repair_capped = 0;
        g_direct_repair_retest_reached = 0;
        g_repair_margin_rejected = g_repair_margin_accepted = 0;
        g_repair_margin_accepted_sum = g_repair_margin_rejected_sum = 0.0;
        g_nav_damage_removed.clear();
        g_flagged_count = 0;
        g_restoration_record.clear();
        g_candidate_log.clear();
        g_rescue_repeats.clear();
        g_pinned_edges.clear();
        g_flagged_nodes.clear();
        g_governor = Governor{};
        if (g_edge_protection) {
            g_governor.K_max = g_K_max;
            // Bare --protect-anneal defaults to the DETERMINISTIC linear schedule
            // over 0.6*n_queries: the convergence-paced variants (churn, deleted
            // fraction) are inert on this workload -- the learner never stops
            // evicting, it only decays to a floor rate -- so a fixed schedule is
            // both the only thing that works and the best config measured
            // (figures/governor_ramp / governor_anneal). Explicit
            // --protect-anneal-linear / --protect-anneal-deleted override.
            if (g_gov_anneal && g_gov_anneal_linear_n == 0 && g_gov_anneal_deleted_frac == 0.0)
                g_gov_anneal_linear_n = std::max(1L, (long)(0.6 * (double)n_queries));
            // --protect-anneal / --protect-anneal-linear start at the ceiling
            // unless the user pinned an explicit init: the anneal only ever
            // moves DOWN, so a low start would just be a fixed low level.
            const bool is_anneal = g_gov_anneal || g_gov_anneal_linear_n > 0;
            if (is_anneal && g_k_p_init == 4) g_k_p_init = g_K_max;
            g_governor.k_p = std::min(g_K_max, g_k_p_init);
            g_governor.k_ps = g_governor.k_p;
            g_governor.anneal_floor = std::max(0, std::min(g_K_max, g_gov_anneal_min));
            // Same shape as the in-memory harness's line, so a log from either
            // binary states its own governor config and a run can be
            // reconstructed from the log alone rather than from a launcher
            // script that may not have been kept.
            fprintf(stderr,
                    "  [governor] init k_p=%d K_max=%d schedule=%d fix_k_p=%d "
                    "audit_passive=%d audit_rate=%.3f audit_ef=%d dwell=%ld T_dwell=%ld "
                    "schedule_period=%ld schedule_step=%d schedule_min=%d "
                    "pin_repeat=%d log_rescue_max=%d log_rescue_all_path=%d direct_repair_force=%d "
                    "direct_repair_pin=%d\n",
                    g_governor.k_p, g_governor.K_max, (int)g_protect_schedule,
                    (int)g_fix_k_p, (int)g_audit_passive, g_p_a, g_audit_ef,
                    g_governor.dwell, g_T_dwell,
                    g_protect_schedule_every_queries, g_protect_schedule_step,
                    g_protect_schedule_min, g_pin_repeat, g_log_rescue_max,
                    (int)g_log_rescue_all_path, (int)g_protect_direct_repair_force,
                    (int)g_protect_direct_repair_pin);
        }

        // NOTE: unlike the Vamana harness, the graph cannot be cheaply rebuilt per
        // method -- it lives on disk. With --restore-dirty, methods are separated by
        // undoing their edits rather than by a fresh build; without it (the default)
        // each method inherits the previous one's graph, which is only sound when a
        // single method runs per invocation against a fresh --work-copy.
        if (restore_dirty_between) {
            IOPhase ph(g_io_reset);
            g.restore_dirty();
        }
        g.edges_added = g.edges_deleted = 0;

        int buffer_cap = (int)(g_buffer_cap_mult * (double)g.max_degree); // B_buf slack above the degree cap
        // Optimistic prior: the chance a uniformly-random node is in the top-K.
        double Q_0 = (double)K / (double)g.num_points;
        // Q and B_buf now live inside learner_hot.tab (see HotLearnerCache);
        // there is no separate Qmap/Bmap local to declare here any more.
        HotLearnerCache learner_hot;
        learner_hot.cap = learner_hot_cap;
        learner_hot.reserve();
        CoMap co; // §7 co-visitation, batch-local: cleared after every LayoutStep
        long gqc = 0;
        // C_blk lifetime is counted in queries from the last drop. A method
        // boundary clears the IOCounters, so the per-batch deltas must be rebased
        // here or the first batch of method 2 reports a negative hit count.
        long last_cblk_drop_q = 0;
        g.hotcache.reset_for_method();

        auto score_ood_after_damage = [&]() {
            if (ood_after_damage_results_csv.empty()) return;
            auto npz = npio::load_npz(ood_cache_npz);
            auto &ev = find_npz(npz, "eval_vecs");
            auto &eg = find_npz(npz, "eval_gt");
            const int cache_n = (int)ev.shape[1];
            const int n_use = std::min(ood_n, cache_n);
            const int ood_dim = (int)ev.shape[2];
            const size_t voff = (size_t)sigma_index * cache_n * ood_dim;
            const size_t goff = (size_t)sigma_index * cache_n * g_k;
            FILE *of = fopen(ood_after_damage_results_csv.c_str(), "w");
            int64_t hits = 0;
            fprintf(of, "q");
            for (int j = 0; j < K; j++) fprintf(of, ",res%d", j);
            for (int j = 0; j < g_k; j++) fprintf(of, ",gt%d", j);
            fputc('\n', of);
            for (int i = 0; i < n_use; i++) {
                std::vector<int> ids, visited;
                g.search(ev.f32() + voff + (size_t)i * ood_dim, (uint64_t)K, (uint64_t)ef,
                         disk_beam_width, ids, visited);
                std::unordered_set<int> gt;
                for (int j = 0; j < g_k; j++) gt.insert((int)eg.i64()[goff + (size_t)i * g_k + j]);
                for (int id : ids) if (gt.count(id)) hits++;
                fprintf(of, "%d", i + 1);
                for (int j = 0; j < K; j++) fprintf(of, ",%d", j < (int)ids.size() ? ids[j] : -1);
                for (int j = 0; j < g_k; j++) fprintf(of, ",%lld", (long long)eg.i64()[goff + (size_t)i * g_k + j]);
                fputc('\n', of);
            }
            fclose(of);
            fprintf(stderr, "  OOD_AFTER_DAMAGE: n=%d recall@%d=%.6f -> %s\n", n_use, K,
                    (double)hits / ((double)n_use * K), ood_after_damage_results_csv.c_str());
        };

        // --nav-damage-from-prescan-ef: derive the damage locality from the
        // real training-query distribution instead of a caller-supplied
        // target-ids file. Read-only replay: g.search() alone touches no
        // Q/c/B_buf/governor/gqc state (unlike train_batch, used by --heatup
        // below), so this cannot leak into the measured run that follows.
        if (!g_nav_damage_done && g_nav_damage_prescan_ef > 0 && g_nav_damage_target_ids.empty()) {
            const bool use_side_eval_a = g_nav_damage_prescan_use_side_eval_a && !side_eval_a_v.empty();
            if (g_nav_damage_prescan_use_side_eval_a && side_eval_a_v.empty()) {
                throw std::runtime_error("--nav-damage-prescan-use-side-eval-a requires "
                                          "--side-eval-a-query/-gt to be set");
            }
            const float *prescan_v = use_side_eval_a ? side_eval_a_v.data() : train_v.data();
            const int prescan_n = use_side_eval_a ? side_eval_a_n : n_queries;
            std::unordered_map<int, int> touched;
            for (int qi = 0; qi < prescan_n; qi++) {
                std::vector<int> result_ids, visited, expanded;
                g.search(prescan_v + (size_t)qi * g_query_dim, (uint64_t)K,
                         (uint64_t)g_nav_damage_prescan_ef, disk_beam_width, result_ids, visited,
                         &expanded);
                for (int u : expanded) touched[u]++;
            }
            std::vector<std::pair<int, int>> ranked(touched.begin(), touched.end());
            std::sort(ranked.begin(), ranked.end(), [](const auto &a, const auto &b) {
                return a.second != b.second ? a.second > b.second : a.first < b.first;
            });
            if (g_nav_damage_prescan_target_cap > 0 &&
                ranked.size() > (size_t)g_nav_damage_prescan_target_cap)
                ranked.resize((size_t)g_nav_damage_prescan_target_cap);
            g_nav_damage_target_ids.reserve(ranked.size());
            for (const auto &[u, count] : ranked) g_nav_damage_target_ids.push_back(u);
            fprintf(stderr,
                    "  [nav-damage-prescan] ef=%d over %d %s queries: %zu unique nodes retained (cap=%d)\n",
                    g_nav_damage_prescan_ef, prescan_n,
                    use_side_eval_a ? "side-eval-A" : "training", g_nav_damage_target_ids.size(),
                    g_nav_damage_prescan_target_cap);
            if (g_nav_damage_prescan_force_audit_target) {
                g_audit_target_ids.assign(g_nav_damage_target_ids.begin(), g_nav_damage_target_ids.end());
                fprintf(stderr,
                        "  [nav-damage-prescan] audit target forced onto the same %zu prescanned nodes\n",
                        g_audit_target_ids.size());
            }
        }

        if (!g_nav_damage_done &&
            (g_nav_damage_frac > 0.0 || g_nav_damage_longest > 0 || !g_nav_damage_target_ids.empty()) &&
            g_nav_damage_at == 0) {
            std::unordered_map<uint32_t, std::vector<uint32_t>> predamage_y_logs;
            {
                IOCounters prelog_io;
                IOPhase ph(prelog_io);
                AuditCostScope c(g_ac_prelog, prelog_io);
                build_predamage_y_logs(g, disk_beam_width, predamage_y_logs);
            }
            auto nav_damage_t0 = std::chrono::steady_clock::now();
            NavDamageStats nds = inject_navigability_damage(
                g, g_nav_damage_frac, g_nav_damage_seed, g_nav_damage_strip_all,
                g_nav_damage_target_ids.empty() ? g_nav_damage_longest : g_nav_damage_target_n,
                g_nav_damage_target_ids.empty() ? nullptr : &g_nav_damage_target_ids);
            double nav_damage_s = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - nav_damage_t0).count();
            fprintf(stderr, "  [nav-damage] q=0 sampled=%ld damaged=%ld edges_removed=%ld (%.3fs)\n",
                    nds.nodes_sampled, nds.nodes_damaged, nds.edges_removed, nav_damage_s);
            g_nav_damage_done = true;
            use_predamage_y_logs(g, disk_beam_width, predamage_y_logs);
            score_ood_after_damage();
        }

        // Unrecorded warm-up: replay the same n_queries once (never rewiring) to
        // populate Q/c/B_buf and warm the device queue, before the measured pass
        // replays the identical stream with counters reset.
        if (heatup) {
            long hqc = 0;
            while (hqc < n_queries) {
                g_read_delay_query = hqc;
                hqc = train_batch(g, train_v.data() + (size_t)hqc * g_query_dim, train_g.data() + (size_t)hqc * g_k, 1,
                                  learner_hot, Q_0, m.reward, hqc, K, ef, buffer_cap,
                                  disk_beam_width, nullptr, nullptr);
            }
            fprintf(stderr, "  [heatup] %s: %ld queries warmed (no rewire, not recorded)\n", m.label.c_str(), hqc);
        }

        if (post_per_query_file && post_test_n > 0) {
            const int saved_ef = ef;
            const std::vector<int> initial_efs = post_test_efs.empty()
                ? std::vector<int>{ef} : post_test_efs;
            for (int e : initial_efs) {
                ef = e;
                fprintf(stderr, "  INITIAL_EFSWEEP: ef=%d\n", e);
                run_post_test(m, 0, nullptr, nullptr);
            }
            ef = saved_ef;
        }

        // The round-0 pass above is cold and read-only: it neither mutates the
        // graph nor leaves C_blk enabled for training.
        fprintf(stderr, "\n=== %s === (train-query results only; recall is scored offline)\n",
                m.label.c_str());
        run_side_evals(m, 0);

        double reset_time_total = 0.0;
        g_tp = TimeProfile();
        auto t0 = Clock::now();
        if (isolated) {
            // Isolated record stage: restore the pristine graph AND start from empty
            // Q/c/B_buf before every query, so each query's own rewire effect is
            // measured with zero carryover.
            fprintf(stderr, "  [isolated] record stage: graph + Q/c/B_buf reset before each of %d queries\n", n_queries);
            for (int qi = 0; qi < n_queries; qi++) {
                g_read_delay_query = qi;
                long r0 = g_io_reset.reads, w0 = g_io_reset.writes;
                auto b_t0 = Clock::now();
                { IOPhase ph(g_io_reset); g.restore_dirty(); }
                double b_time = secs_since(b_t0);
                reset_time_total += b_time;
                // Qi/Bi used to be separate per-query maps; Hi's tab now holds
                // that state directly (see HotLearnerCache).
                HotLearnerCache Hi;
                Hi.cap = learner_hot_cap;
                Hi.reserve();

                long tr0 = g_io_traversal.reads;
                double tio0 = g_io_traversal.io_s;
                auto q_t0 = Clock::now();
                std::vector<double> qrecalls;
                {
                    IOPhase ph(g_io_traversal);
                    train_batch(g, train_v.data() + (size_t)qi * g_query_dim, train_g.data() + (size_t)qi * g_k, 1,
                                Hi, Q_0, m.reward, qi, K, ef, buffer_cap,
                                disk_beam_width, &qrecalls, nullptr);
                }
                double q_time = secs_since(q_t0);

                long rr0 = g_io_rewire.reads, rw0 = g_io_rewire.writes;
                double rio0 = g_io_rewire.io_s;
                double r_time = 0.0;
                // take == 1 here, so this IS the current query's recall.
                const double rewire_edge_prior = recall_prior_for_batch(qrecalls, Q_0, m.reward);
                fprintf(stderr, "  TRAINRECALL: batch_mean=%.6f n=%zu\n",
                        mean_or(qrecalls, -1.0), qrecalls.size());
                if (!no_rewire && (rewire_until < 0 || (long)(qi + 1) <= rewire_until)) {
                    auto r_t0 = Clock::now();
                    const double commit_before = g_tp.rewire_commit;
                    const double detail_before = g_tp.rsel_sum(false);
                    const double commit_io_before = g_tp.rewire_commit_io;
                    const double io_detail_before = g_tp.rsel_io_sum(false);
                    const double rewire_io_before = g_io_rewire.io_s;
                    // gqc is not advanced yet at this point in the isolated loop
                    { IOPhase ph(g_io_rewire); do_rewire(g, Hi, Q_0, rewire_edge_prior, (long)(qi + 1)); }
                    r_time = secs_since(r_t0);
                    record_rewire_select(r_time - (g_tp.rewire_commit - commit_before), detail_before,
                                         (g_io_rewire.io_s - rewire_io_before) -
                                             (g_tp.rewire_commit_io - commit_io_before),
                                         io_detail_before);
                }
                if (pqf) {
                    fprintf(pqf, "%s,%d,%.6f,%.6f,%ld,%ld,%.6f,%ld,%.6f,%.6f,%ld,%ld,%.6f\n", m.label.c_str(), qi + 1,
                            qrecalls.empty() ? 0.0 : qrecalls[0],
                            b_time, g_io_reset.reads - r0, g_io_reset.writes - w0,
                            q_time, g_io_traversal.reads - tr0, g_io_traversal.io_s - tio0,
                            r_time, g_io_rewire.reads - rr0, g_io_rewire.writes - rw0, g_io_rewire.io_s - rio0);
                }
                gqc = qi + 1;

                // C_blk lifetime: a query count, not a rewire count. Tying it to the
                // rewire would give --no-rewire (which takes one query per iteration)
                // a whole-run cache and the rewiring arm a per-batch one -- a lifetime
                // difference sitting exactly on the comparison under test. do_rewire's
                // flush_edits is the batch's last disk write, so dropping after it
                // leaves no window in which a stale sector could be served.
                if (g.hotcache.enabled && hot_cache_lifetime_q > 0 &&
                    gqc - last_cblk_drop_q >= hot_cache_lifetime_q) {
                    report_hot_cache(g, gqc, true);
                    last_cblk_drop_q = gqc;
                } else if (g.hotcache.enabled && hot_cache_lifetime_q == 0 &&
                       gqc - last_cblk_drop_q >= b_rewire) {
                    report_hot_cache(g, gqc, false); // never-drop: progress only
                    last_cblk_drop_q = gqc;
                }
                if (gqc % eval_every == 0 || gqc == n_queries) {
                    fprintf(stderr, "  q=%5ld  reads[traversal=%ld rewire=%ld] writes[rewire=%ld]\n",
                            gqc, g_io_traversal.reads, g_io_rewire.reads, g_io_rewire.writes);
                    if (ctf) run_post_test(m, gqc, ctf, nullptr);
                    run_side_evals(m, gqc);
                    if (pqf) fflush(pqf);
                }
            }
        } else {
        // --warmup-queries: hold the requested pricing/gating aside and run the
        // warm-up phase without them.  Saved per method, since gqc restarts at
        // each method boundary and the second method must warm up too.
        const double requested_lambda_nf_quantile = g_lambda_noise_floor_quantile;
        const bool requested_io_price_enabled = g_io_price_controller.enabled;
        const double requested_expansion_cost_theta = g_expansion_cost_theta;
        bool priced_phase_armed = (arm_mechanisms_from <= 0);
        bool stats_reset_done = false;
        bool price_retired = false;
        if (arm_mechanisms_from > 0) {
            g_lambda_noise_floor_quantile = 0.0;
            g_io_price_controller.enabled = false;
            g_expansion_cost_theta = 0.0;
            fprintf(stderr, "  PHASE: warm-up q=0..%ld -- cost utility and lambda gate OFF"
                            " (b_rewire=%d), armed from q=%ld\n",
                    arm_mechanisms_from, warmup_b_rewire > 0 ? warmup_b_rewire : b_rewire,
                    arm_mechanisms_from);
        }

        // Cumulative loop: Q/c/B_buf and the graph both persist and compound across
        // the run -- the normal "learn as you go" mode.
        //
        long next_eval_q = eval_every;
        int queries_since_rewire = 0;
        std::vector<double> rewire_recalls_accum;
        long price_sector_reqs_accum = 0;
        long price_expansions_accum = 0;
        while (gqc < n_queries) {
            // --stats-from-query: everything measured before N is adaptation cost
            // the caller has asked not to be billed for. Zeroed here, at a batch
            // boundary, so no partial batch straddles the reset.
            if (stats_from_query > 0 && !stats_reset_done && gqc >= stats_from_query) {
                for (IOCounters *c : {&g_io_traversal, &g_io_rewire, &g_io_reset,
                                      &g_io_layout, &g_io_audit, &g_io_floor})
                    c->clear();
                g_tp = TimeProfile();
                g_write_delay_offered = g_write_delay_charged = 0;
                g_write_delay_s = 0.0;
                g_read_delay_offered = g_read_delay_charged = 0;
                g_read_delay_s = 0.0;
                g_coalesced_reads = 0;
                // train_s is measured by the served loop's own clock, not by
                // TimeProfile, so it has to be restarted here too or the window's
                // "total" would still be the whole run's.
                t0 = Clock::now();
                stats_reset_done = true;
                fprintf(stderr, "  PHASE: statistics ZEROED at q=%ld -- every wall-clock and"
                                " device counter below describes q=%ld..%d only\n",
                        gqc, gqc, n_queries);
            }
            const bool in_warmup_phase = warmup_queries > 0 && gqc < warmup_queries;
            // --price-until-query: retire the cost utility, leaving lambda alone.
            if (price_until_query > 0 && !price_retired && gqc >= price_until_query) {
                g_io_price_controller.enabled = false;
                g_expansion_cost_theta = 0.0;
                price_retired = true;
                fprintf(stderr, "  PHASE: cost utility RETIRED at q=%ld"
                                " (lambda_nf_q held at %.3g)\n",
                        gqc, g_lambda_noise_floor_quantile);
            }
            if (arm_mechanisms_from > 0 && gqc >= arm_mechanisms_from && !priced_phase_armed) {
                g_lambda_noise_floor_quantile = requested_lambda_nf_quantile;
                g_io_price_controller.enabled = requested_io_price_enabled;
                g_expansion_cost_theta = requested_expansion_cost_theta;
                priced_phase_armed = true;
                fprintf(stderr, "  PHASE: pricing+gating ARMED at q=%ld"
                                " (theta=%.6g io_target=%s lambda_nf_q=%.3g b_rewire=%d)\n",
                        gqc, g_expansion_cost_theta,
                        g_io_price_controller.enabled ? "on" : "off",
                        g_lambda_noise_floor_quantile, b_rewire);
            }
            // The static reference is one query at a time so timing/I/O stay
            // attributable. A layout-only arm still needs the configured epoch:
            // its co-visitation evidence and LayoutStep cadence must match the
            // rewiring arms even though topology remains static.
            const bool batch_learning = g_detector_calibration.learning();
            const int phase_b_rewire =
                (in_warmup_phase && warmup_b_rewire > 0) ? warmup_b_rewire : b_rewire;
            int take = (no_rewire && !layout_adapt) ? 1
                                                    : std::min(batch_learning
                                                        ? phase_b_rewire - queries_since_rewire
                                                        : phase_b_rewire,
                                                        n_queries - (int)gqc);
            take = std::min(take, n_queries - (int)gqc);
            if ((!side_eval_a_query.empty() || !side_eval_b_query.empty()) && next_eval_q > gqc)
                take = std::min(take, (int)(next_eval_q - gqc));
            // A batch must not straddle the phase boundary, or one rewire would
            // be selected under a mixture of the two configurations.
            if (in_warmup_phase) take = std::min(take, (int)(warmup_queries - gqc));
            // A batch must not straddle a mechanism boundary either.
            if (arm_mechanisms_from > 0 && gqc < arm_mechanisms_from)
                take = std::min(take, (int)(arm_mechanisms_from - gqc));
            if (price_until_query > 0 && gqc < price_until_query)
                take = std::min(take, (int)(price_until_query - gqc));
            if (stats_from_query > 0 && gqc < stats_from_query)
                take = std::min(take, (int)(stats_from_query - gqc));
            if (g_read_delay_start_query > 0 && gqc < g_read_delay_start_query)
                take = std::min(take, (int)(g_read_delay_start_query - gqc));

            // Co-location is decided per batch from here on, so
            // --layout-adapt-until can retire it mid-run. Every cost
            // co-location carries hangs off this one flag: the co-visitation
            // log (co_out below), the C_blk-miss set it is keyed on, and
            // LayoutStep itself.
            const bool layout_active =
                batch_learning && layout_adapt && !layout_adapt_retired &&
                (layout_adapt_until_query <= 0 || gqc < layout_adapt_until_query);
            if (g.collect_epoch_misses != layout_active) {
                g.collect_epoch_misses = layout_active;
                if (!layout_active) {
                    // Retiring co-location: drop the evidence accumulated for a
                    // LayoutStep that will now never run, so no stale epoch
                    // survives into the post-Q window.
                    co.clear();
                    g_covisit_pairs = 0;
                    g.epoch_missed.clear();
                }
            }

            long tr0 = g_io_traversal.reads;
            // Dual-ascent feed: device reads, not node requests (the cache serves
            // those for free).
            long trav_sector_reqs_before_batch = g_io_traversal.sector_reqs;
            double tio0 = g_io_traversal.io_s;
            auto q_t0 = Clock::now();
            std::vector<double> qrecalls;
            {
                g_read_delay_query = gqc;
                IOPhase ph(g_io_traversal);
                gqc = train_batch(g, train_v.data() + (size_t)gqc * g_query_dim, train_g.data() + (size_t)gqc * g_k, take,
                                  learner_hot, Q_0, m.reward, gqc, K, ef, buffer_cap,
                                  disk_beam_width, &qrecalls,
                                  layout_active ? &co : nullptr);
            }
            const int actual_take = (int)qrecalls.size();
            const DetectorCalibration::Event detector_event = g_detector_calibration.take_event();
            if (batch_learning) {
                queries_since_rewire += actual_take;
                rewire_recalls_accum.insert(rewire_recalls_accum.end(), qrecalls.begin(), qrecalls.end());
                price_sector_reqs_accum +=
                    g_io_traversal.sector_reqs - trav_sector_reqs_before_batch;
                price_expansions_accum += g_io_traversal.reads - tr0;
            }
            const bool adaptation_batch_due = batch_learning &&
                (queries_since_rewire >= phase_b_rewire ||
                 detector_event == DetectorCalibration::ADAPT_STOP || gqc == n_queries);
            const bool rewire_due = adaptation_batch_due && !no_rewire;
            const int commit_queries = queries_since_rewire;
            double q_time = secs_since(q_t0);

            // Stress-test damage injection: fires exactly once, the first time
            // gqc crosses g_nav_damage_at, so this batch's own audit trials
            // (right below) already see the post-injection graph -- injection
            // event and first detectable effect land in the same checkpoint.
            if (!g_nav_damage_done &&
                (g_nav_damage_frac > 0.0 || g_nav_damage_longest > 0 || !g_nav_damage_target_ids.empty()) &&
                g_nav_damage_at >= 0 && gqc >= g_nav_damage_at) {
                NavDamageStats nds = inject_navigability_damage(
                    g, g_nav_damage_frac, g_nav_damage_seed, g_nav_damage_strip_all,
                    g_nav_damage_target_ids.empty() ? g_nav_damage_longest : g_nav_damage_target_n,
                    g_nav_damage_target_ids.empty() ? nullptr : &g_nav_damage_target_ids);
                fprintf(stderr, "  [nav-damage] q=%ld sampled=%ld damaged=%ld (%.4f%% of N) "
                                "edges_removed=%ld skipped_fully_protected=%ld k_p=%d frac=%.4f "
                                "seed=%u strip_all=%d longest=%d\n",
                        gqc, nds.nodes_sampled, nds.nodes_damaged,
                        100.0 * (double)nds.nodes_damaged / (double)g.num_points, nds.edges_removed,
                        nds.nodes_skipped_fully_protected, g_governor.k_p,
                        g_nav_damage_frac, g_nav_damage_seed, (int)g_nav_damage_strip_all,
                        g_nav_damage_longest);
                g_nav_damage_done = true;
            }

            // Background reachability audits (Algorithm 5): fire independently
            // at rate p_a for each query just served, before the rewire that
            // will act on this batch. Not wired into --isolated (a per-query
            // graph reset defeats the point of a level that persists across
            // queries) or --heatup (unrecorded warm-up).
            if (batch_learning && g_edge_protection) {
                TimeSpan audit_ts(g_tp.audit);
                for (int qi = 0; qi < actual_take; qi++)
                    if (g_unif01(g_nav_rng) < g_p_a)
                        audit_step(g, learner_hot, buffer_cap, (uint64_t)ef, disk_beam_width);
            }

            long rr0 = g_io_rewire.reads, rw0 = g_io_rewire.writes;
            double rio0 = g_io_rewire.io_s;
            double r_time = 0.0;
            // Mean recall over the b_rewire queries whose evidence this rewire acts on.
            const double rewire_edge_prior = recall_prior_for_batch(rewire_recalls_accum, Q_0, m.reward);
            fprintf(stderr, "  TRAINRECALL: batch_mean=%.6f n=%zu\n",
                    mean_or(qrecalls, -1.0), qrecalls.size());
            // Alg. 3 line 1, before do_rewire selects.
            if (adaptation_batch_due && g_io_price_controller.enabled && commit_queries > 0) {
                g_io_price_controller.refresh_price(
                    (double)price_sector_reqs_accum / (double)commit_queries,
                    (double)price_expansions_accum / (double)commit_queries, gqc);
            }
            if (rewire_due && !g_search_only &&
                (rewire_until < 0 || gqc <= rewire_until)) {
                auto r_t0 = Clock::now();
                const double commit_before = g_tp.rewire_commit;
                const double detail_before = g_tp.rsel_sum(false);
                const double commit_io_before = g_tp.rewire_commit_io;
                const double io_detail_before = g_tp.rsel_io_sum(false);
                const double rewire_io_before = g_io_rewire.io_s;
                if (!g_skip_rewire_commit) {
                    IOPhase ph(g_io_rewire);
                    do_rewire(g, learner_hot, Q_0, rewire_edge_prior, gqc);
                }
                // --protect-direct-repair: this batch's audit repairs, committed
                // as real edges. Inside the same IOPhase so the extra writes land
                // on the rewire I/O axis rather than appearing free.
                { IOPhase ph(g_io_rewire); AuditCostScope c(g_ac_apply, g_io_rewire); apply_direct_repairs(g, disk_beam_width); }
                r_time = secs_since(r_t0);
                // flush_edits runs inside do_rewire and books itself; the rest of
                // the call is selection, so subtract rather than double-count.
                record_rewire_select(r_time - (g_tp.rewire_commit - commit_before), detail_before,
                                     (g_io_rewire.io_s - rewire_io_before) -
                                         (g_tp.rewire_commit_io - commit_io_before),
                                     io_detail_before);
                // Governor (Algorithm 6): same cadence as the rewire it follows.
                if (g_edge_protection) {
                    TimeSpan gov_ts(g_tp.governor);
                    g_governor.dwell += actual_take;
                    governor_step(g, gqc);
                }
            }
            if (detector_event == DetectorCalibration::ADAPT_STOP)
                fprintf(stderr, "  ADAPT_STOP_COMMIT,q=%ld,batch_queries=%d,async=%d\n",
                        gqc, commit_queries, async_rewire_commit ? 1 : 0);
            if (detector_event == DetectorCalibration::DRIFT_START)
                fprintf(stderr, "  ADAPT_RESTART,q=%ld,next_query=%ld\n", gqc, gqc + 1);
            // Topology then layout at the same batch boundary (Prop. F.2): the
            // alternation both steps' convergence argument assumes.
            if (adaptation_batch_due && layout_active) {
                TimeSpan layout_ts(g_tp.layout);
                IOPhase ph(g_io_layout);
                const long layout_swaps = layout_step(g, co, lambda_pi);
                if (layout_adapt_stop_below > 0 && layout_swaps < layout_adapt_stop_below) {
                    layout_adapt_retired = true;
                    fprintf(stderr, "  LAYOUTSTOP: q=%ld swaps=%ld below=%ld -- co-location retired\n",
                            gqc, layout_swaps, layout_adapt_stop_below);
                }
                co.clear();
                g_covisit_pairs = 0;
                // The epoch is the batch: pairs are only deduplicated within the
                // window the miss set covers, so it has to end where co does.
                g.epoch_missed.clear();
            }
            if (adaptation_batch_due) {
                queries_since_rewire = 0;
                rewire_recalls_accum.clear();
                price_sector_reqs_accum = 0;
                price_expansions_accum = 0;
            }

            // C_blk lifetime: a query count, not a rewire count. Tying it to the
            // rewire would give --no-rewire (which takes one query per iteration)
            // a whole-run cache and the rewiring arm a per-batch one -- a lifetime
            // difference sitting exactly on the comparison under test. do_rewire's
            // flush_edits is the batch's last disk write, so dropping after it
            // leaves no window in which a stale sector could be served.
            if (g.hotcache.enabled && hot_cache_lifetime_q > 0 &&
                gqc - last_cblk_drop_q >= hot_cache_lifetime_q) {
                report_hot_cache(g, gqc, true);
                last_cblk_drop_q = gqc;
            } else if (g.hotcache.enabled && hot_cache_lifetime_q == 0 &&
                       gqc - last_cblk_drop_q >= b_rewire) {
                report_hot_cache(g, gqc, false); // never-drop: progress only
                last_cblk_drop_q = gqc;
            }
            // Per-query attribution only means something when take==1 (always true
            // under --no-rewire; under rewire, iff --b-rewire 1).
            if (pqf && actual_take == 1) {
                fprintf(pqf, "%s,%ld,%.6f,0.000000,0,0,%.6f,%ld,%.6f,%.6f,%ld,%ld,%.6f\n", m.label.c_str(), gqc,
                        qrecalls.empty() ? 0.0 : qrecalls[0],
                        q_time, g_io_traversal.reads - tr0, g_io_traversal.io_s - tio0,
                        r_time, g_io_rewire.reads - rr0, g_io_rewire.writes - rw0, g_io_rewire.io_s - rio0);
            }

            // Reporting cadence only: --eval-every no longer runs any query.
            if (gqc >= next_eval_q || detector_event != DetectorCalibration::NONE) {
                // Used to be the union of Q's and B_buf's keys via a std::set
                // (two maps, so a key could be in only one). They are now one
                // collocated map, so learner_hot.tab.size() alone is the same
                // count -- no set construction needed. `touched` kept as a
                // name only where its .size() still feeds the CSV row below.
                const size_t touched_size = learner_hot.tab.size();
                fprintf(stderr, "  q=%5ld  net_edges=%+ld  reads[traversal=%ld rewire=%ld] "
                        "writes[rewire=%ld]\n", gqc, g.edges_added - g.edges_deleted,
                        g_io_traversal.reads, g_io_rewire.reads, g_io_rewire.writes);
                report_device_io("checkpoint", gqc);
                report_time_profile("checkpoint", gqc, secs_since(t0));
                if (g_edge_protection)
                    fprintf(stderr, "    [governor] k_p=%d promotions=%ld demotions=%ld "
                            "miss_rate=%.4f red=%.4f sw=%.4f window_n=%ld audit_reads=%ld\n"
                            "    [audit] fired=%ld pass=%ld degenerate=%ld nobuf=%ld norescue=%ld attributable=%ld "
                            "log_rescue=%ld search_proposals=%ld pinned_nodes=%zu flagged_nodes=%ld\n"
                            "    [direct-repair] applied=%ld already_edge=%ld capped_out=%ld "
                            "margin=%.6g accepted=%ld(mean=%.6g) rejected=%ld(mean=%.6g) retest_reached=%ld\n"
                            "    [retest] y_vec_top1=%ld/%ld testq_top1=%ld/%ld\n"
                            "    [y-log-fill] calls=%ld zero_budget=%ld evicted=%ld added=%ld\n"
                            "    [auditcost] search[s=%.3f io_s=%.3f dev_r=%ld n=%ld] fix_in_audit[s=%.3f io_s=%.3f dev_r=%ld] "
                            "apply[s=%.3f io_s=%.3f dev_r=%ld dev_w=%ld n=%ld] prelog[s=%.3f dev_r=%ld]\n",
                            g_governor.k_p, g_governor.promotions, g_governor.demotions,
                            g_governor.window.miss_rate(), g_governor.window.redundancy(),
                            g_governor.window.sole_witness_rate(), g_governor.window.total(), g_io_audit.reads,
                            g_audit_pass + g_audit_degenerate + g_audit_nobuf + g_audit_norescue + g_audit_attrib,
                            g_audit_pass, g_audit_degenerate, g_audit_nobuf, g_audit_norescue, g_audit_attrib,
                            g_audit_log_rescue, g_audit_search_proposals, g_pinned_edges.size(), g_flagged_count,
                            g_direct_repair_applied, g_direct_repair_existing, g_direct_repair_capped,
                            g_protect_repair_min_improvement,
                            g_repair_margin_accepted,
                            g_repair_margin_accepted ? g_repair_margin_accepted_sum / g_repair_margin_accepted : 0.0,
                            g_repair_margin_rejected,
                            g_repair_margin_rejected ? g_repair_margin_rejected_sum / g_repair_margin_rejected : 0.0,
                            g_direct_repair_retest_reached, g_retest_y_top1, g_retest_n,
                            g_retest_testq_top1, g_retest_testq_n,
                            g_y_log_fill_calls, g_y_log_fill_zero_budget, g_y_log_fill_evicted, g_y_log_fill_added,
                            g_ac_search.s, g_ac_search.io_s, g_ac_search.dev_reads, g_ac_search.calls,
                            g_ac_fix.s - g_ac_search.s, g_ac_fix.io_s - g_ac_search.io_s,
                            g_ac_fix.dev_reads - g_ac_search.dev_reads,
                            g_ac_apply.s, g_ac_apply.io_s, g_ac_apply.dev_reads, g_ac_apply.dev_writes, g_ac_apply.calls,
                            g_ac_prelog.s, g_ac_prelog.dev_reads);
                printf("%s,%ld,%ld,%zu,%ld,%ld,%ld,%.6f,%d,%.6g,%.6g,%.6f\n", m.label.c_str(), gqc,
                       g.edges_added - g.edges_deleted, touched_size,
                       g_io_traversal.reads, g_io_rewire.reads, g_io_rewire.writes,
                       g_io_traversal.io_s + g_io_rewire.io_s,
                       g_edge_protection ? g_governor.k_p : -1,
                       g_rewire_margin_lambda, g_expansion_cost_theta,
                       (double)g_io_traversal.sector_reqs / (double)gqc);
                fflush(stdout);
                if (ctf) run_post_test(m, gqc, ctf, nullptr);
                run_side_evals(m, gqc);
                while (next_eval_q <= gqc) next_eval_q += eval_every;
            }
            if (pqf && gqc % eval_every == 0) fflush(pqf);
        }
        }
        // Tail batch: n_queries need not be a multiple of the lifetime, and an
        // unreported final batch would drop out of both the per-batch series and
        // peak_sectors.
        if (g.hotcache.enabled && gqc > last_cblk_drop_q)
            report_hot_cache(g, gqc, hot_cache_lifetime_q > 0);
        double train_s = secs_since(t0);
        report_time_profile("final", gqc, train_s);
        // Layout I/O is reported by sectors, not `reads`: swap_slots does its own
        // pread/pwrite pairs rather than going through the reader, so it charges
        // read_sectors/writes exactly as flush_edits does.
        fprintf(stderr, "  TOTAL_TRAIN_TIME_S=%.3f  RESET_TIME_S=%.3f  READS[reset=%ld traversal=%ld rewire=%ld]  "
                "WRITES[reset=%ld rewire=%ld]  IO_S[reset=%.3f traversal=%.3f rewire=%.3f]  "
                "LAYOUT[swaps=%ld relocated=%zu read_sectors=%ld writes=%ld io_s=%.3f "
                "preads=%ld resident=%ld]  "
                "I_PI[distinct_sectors=%ld expanded=%ld ratio=%.4f coalesced=%ld]\n",
                train_s, reset_time_total, g_io_reset.reads, g_io_traversal.reads, g_io_rewire.reads,
                g_io_reset.writes, g_io_rewire.writes, g_io_reset.io_s, g_io_traversal.io_s, g_io_rewire.io_s,
                g.layout_swaps, g.reloc.size(), g_io_layout.read_sectors, g_io_layout.writes, g_io_layout.io_s,
                g_io_layout.rmw_preads, g_io_layout.rmw_preads_resident,
                g_distinct_sectors, g_expanded_nodes,
                g_expanded_nodes ? (double)g_distinct_sectors / (double)g_expanded_nodes : 0.0,
                g_coalesced_reads);
        if (g_dump_protect_mask && g_pm_rewires) {
        fprintf(stderr, "  PROTMASK: rewires=%ld pinned=%ld bite=%ld(%.4f) fresh=%ld(%.4f) "
                "churn_jaccard=%.4f len_ratio=%.3f\n",
                g_pm_rewires, g_pm_pinned,
                g_pm_bite, (double)g_pm_bite / (double)std::max(g_pm_pinned, 1L),
                g_pm_fresh, (double)g_pm_fresh / (double)std::max(g_pm_pinned, 1L),
                g_pm_churn_n ? g_pm_churn_sum / (double)g_pm_churn_n : 0.0,
                g_pm_len_n ? g_pm_len_ratio_sum / (double)g_pm_len_n : 0.0);
    }
    fprintf(stderr, "  QUPDATE[full_pool=%d bbuf_credited=%ld updates_per_query=%.1f "
                "traversed_only=%d nout_credited=%ld nout_skipped=%ld frac_credited=%.4f]\n",
                g_q_update_full_pool ? 1 : 0, g_qupdate_bbuf_credited,
                n_queries ? (double)(g_qupdate_nout_credited + g_qupdate_bbuf_credited) / (double)n_queries : 0.0,
                g_q_update_traversed_only ? 1 : 0, g_qupdate_nout_credited, g_qupdate_nout_skipped,
                (g_qupdate_nout_credited + g_qupdate_nout_skipped)
                    ? (double)g_qupdate_nout_credited
                          / (double)(g_qupdate_nout_credited + g_qupdate_nout_skipped)
                    : 0.0);
        // sector_reqs is eq. (1) summed over passes; train_batch runs two passes
        // per query (the eval-facing search and the Q-ranked walk), so I_hat is
        // per pass, not per served query.
        // PASSSPLIT: sector_reqs above is eq. (1) summed over BOTH passes
        // train_batch runs. g_search_sector_reqs is the serving beam search's
        // half alone, so the Q-ranked walk's half is the difference. Printed
        // because "is the walk more expensive than the search?" is a question
        // about work, and the answer has to separate I/O from CPU.
        fprintf(stderr, "  PASSSPLIT: sector_reqs_total=%ld search=%ld qwalk=%ld\n",
                g_io_traversal.sector_reqs, g_search_sector_reqs,
                g_io_traversal.sector_reqs - g_search_sector_reqs);
        fprintf(stderr, "  QBUF[%s cap=%zu]: sector_reqs=%ld node_fetches=%ld "
                "saved[cblk=%ld cross_round=%ld same_batch=%ld]  I_hat=%.2f blocks/pass  check=%s\n",
                g.qbuf.enabled ? "on" : "off", g.qbuf.cap_blocks,
                g_io_traversal.sector_reqs, g_io_traversal.reads,
                g_io_traversal.cblk_hits, g_io_traversal.qbuf_hits, g_io_traversal.call_coalesced,
                (double)g_io_traversal.sector_reqs / std::max(1.0, 2.0 * (double)n_queries),
                // Every requested id lands in exactly one bucket. If this ever
                // fails, the four outcomes in read_nodes have stopped being
                // disjoint and the eq. (1) decomposition is not additive.
                (g_io_traversal.cblk_hits + g_io_traversal.qbuf_hits + g_io_traversal.call_coalesced +
                 g_io_traversal.sector_reqs) == g_io_traversal.reads ? "OK" : "MISMATCH");
        // C_blk totals. There is no maintenance cost to report -- admission rides
        // the read that missed -- so what matters is how much traffic the ceiling
        // removed and how big the resident set had to get to remove it.
        if (g.hotcache.enabled) {
            const long th = g_io_traversal.cblk_hits, ta = g_io_traversal.cblk_admits;
            const long rh = g_io_rewire.cblk_hits, ra = g_io_rewire.cblk_admits;
            const long tf = g_io_traversal.cblk_frozen, rf = g_io_rewire.cblk_frozen;
            const char *mode = g.hotcache.async_writeback ? "async-writethrough"
                              : (g.hotcache.writethrough ? "writethrough" : "readonly");
            fprintf(stderr, "  CBLK_TOTAL: mode=%s batches=%ld lifetime=%dq cap=%zu final_sectors=%zu "
                    "final=%.1fMB frozen_misses=%ld peak_sectors=%zu peak=%.1fMB "
                    "trav[hit=%ld admit=%ld frozen=%ld rate=%.4f] "
                    "rewire[hit=%ld admit=%ld frozen=%ld rate=%.4f] "
                    "rmw_preads_issued=%ld rmw_resident=%ld policy=%s evictions=%ld "
                    "max_dirty_at_claim=%ld\n",
                    mode, g.hotcache.batches, hot_cache_lifetime_q, g.hotcache.cap_sectors,
                    g.hotcache.used, g.hotcache.resident_mb(), g.hotcache.frozen_misses,
                    g.hotcache.peak_sectors, g.hotcache.peak_mb(),
                    th, ta, tf, (th + ta + tf) ? (double)th / (th + ta + tf) : 0.0,
                    rh, ra, rf, (rh + ra + rf) ? (double)rh / (rh + ra + rf) : 0.0,
                    g_io_rewire.rmw_preads, g_io_rewire.rmw_preads_resident,
                    g.hotcache.cap_sectors == 0 ? "uncapped"
                        : (g.hotcache.lru ? "lru" : "freeze"), g.hotcache.evictions,
                    g.hotcache.max_dirty_at_claim);
            if (g.hotcache.async_writeback) {
                std::lock_guard<std::mutex> lk(g.hotcache.mu);
                fprintf(stderr, "  CBLK_ASYNC: enqueued=%ld writes=%ld rewrites=%ld pending=%zu "
                                "inflight=%ld io_s=%.3f wait_s=%.3f dirty=%zu\n",
                        g.hotcache.async_enqueued, g.hotcache.async_writes, g.hotcache.async_rewrites,
                        g.hotcache.writeq.size(), g.hotcache.inflight,
                        g.hotcache.async_io_s, g.hotcache.async_wait_s, g.hotcache.dirty.size());
            }
        }
        // The final theta is not a setting anyone chose, so it has to be reported.
        if (g_io_price_controller.enabled) {
            fprintf(stderr, "  IOPRICE_TOTAL: target=%.2f steps=%ld final_theta=%.6g "
                    "final_gamma=%.6g run_io_per_query=%.2f\n",
                    g_io_price_controller.io_target_per_query, g_io_price_controller.steps_taken,
                    g_expansion_cost_theta, g_io_price_controller.dual_ascent_step_gamma,
                    (double)g_io_traversal.sector_reqs / std::max(1, n_queries));
        }
        printf("SUMMARY,%s,%d,%.6f,%.6f,%ld,%ld,%ld,%ld,%ld,%.6f,%.6f,%.6f,%ld,%ld,%ld,%.6f,%ld,%ld,%ld\n",
               m.label.c_str(), n_queries, train_s,
               reset_time_total, g_io_reset.reads, g_io_traversal.reads, g_io_rewire.reads, g_io_reset.writes,
               g_io_rewire.writes, g_io_reset.io_s, g_io_traversal.io_s, g_io_rewire.io_s,
               g.layout_swaps, g_io_layout.read_sectors, g_io_layout.writes, g_io_layout.io_s,
               g_distinct_sectors, g_expanded_nodes, g_coalesced_reads);
        fflush(stdout);

        report_device_io("final", gqc);
        fprintf(stderr, "  FINAL train_queries=%ld train_time=%.1fs  (recall: score %s offline)\n",
                gqc, train_s, query_results_csv.empty() ? "--dump-query-results" : query_results_csv.c_str());
        if (!g_node_fetch_count_path.empty()) {
            FILE *count_file = fopen(g_node_fetch_count_path.c_str(), "w");
            if (count_file) {
                for (const auto &kv : g_node_fetch_count)
                    fprintf(count_file, "%u %llu\n", kv.first, (unsigned long long)kv.second);
                fclose(count_file);
            }
            fprintf(stderr, "  NODEFETCHCOUNTS: nodes=%zu path=%s\n", g_node_fetch_count.size(),
                    g_node_fetch_count_path.c_str());
        }

        // Post-test: same held-out set every time (fixed seed), scored against
        // whatever this run just trained -- still this process, so g.reloc
        // reflects every swap_slots call, exactly as the served system would.
        if (post_test_n > 0 && !post_test_results_csv.empty()) {
            if (interleave_centers_block > 0) {
                FILE *pcf = fopen((post_test_results_csv + ".centers").c_str(), "w");
                if (pcf) {
                    fprintf(pcf, "q,center\n");
                    for (int i = 0; i < post_test_n; i++) fprintf(pcf, "%d,%d\n", i + 1, post_center[i]);
                    fclose(pcf);
                }
            }
            FILE *ptf = fopen(post_test_results_csv.c_str(), "w");
            if (!ptf) {
                fprintf(stderr, "cannot open %s\n", post_test_results_csv.c_str());
            } else {
                fprintf(ptf, "q");
                for (int j = 0; j < K; j++) fprintf(ptf, ",res%d", j);
                for (int j = 0; j < g_k; j++) fprintf(ptf, ",gt%d", j);
                fputc('\n', ptf);
                run_post_test(m, gqc, nullptr, ptf);
                fclose(ptf);
                fprintf(stderr, "  POSTTEST: n=%d -> %s\n", post_test_n, post_test_results_csv.c_str());
            }
        }

        // ef sweep: same held-out set, same live graph/reloc, scored again at
        // each requested ef -- one training pass buys the whole recall-vs-ef
        // curve instead of one retrain per point.
        if (!post_test_efs.empty() && post_test_n > 0) {
            const int saved_ef = ef;
            FILE *sweepf = nullptr;
            if (!post_test_ef_results_csv.empty()) {
                sweepf = fopen(post_test_ef_results_csv.c_str(), "w");
                if (!sweepf) throw std::runtime_error("cannot open " + post_test_ef_results_csv);
                fprintf(sweepf, "ef,recall,qps,mean_latency_us,wall_qps,p99_us,device_reads,sector_reqs,node_reads,io_s,wall_s\n");
            }
            for (int e : post_test_efs) {
                ef = e;
                fprintf(stderr, "  EFSWEEP: ef=%d\n", e);
                // Per-query results at this ef, same layout as --post-test-results.
                FILE *per_query_ef_file = nullptr;
                if (!post_test_ef_results_csv.empty()) {
                    const std::string per_query_ef_path =
                        post_test_ef_results_csv + ".ef" + std::to_string(e) + ".csv";
                    per_query_ef_file = fopen(per_query_ef_path.c_str(), "w");
                    if (!per_query_ef_file) throw std::runtime_error("cannot open " + per_query_ef_path);
                    fprintf(per_query_ef_file, "q");
                    for (int j = 0; j < K; j++) fprintf(per_query_ef_file, ",res%d", j);
                    for (int j = 0; j < g_k; j++) fprintf(per_query_ef_file, ",gt%d", j);
                    fputc('\n', per_query_ef_file);
                }
                const PostTestMetrics x = run_post_test(m, gqc, nullptr, per_query_ef_file);
                if (per_query_ef_file) fclose(per_query_ef_file);
                if (sweepf)
                    fprintf(sweepf, "%d,%.9f,%.6f,%.3f,%.6f,%.3f,%ld,%ld,%ld,%.6f,%.6f\n",
                            e, x.recall, x.qps, x.mean_latency_us, x.wall_qps, x.p99_us,
                            x.device_reads, x.sector_reqs,
                            x.reads, x.io_s, x.wall_s);
            }
            if (sweepf) {
                fclose(sweepf);
                fprintf(stderr, "  EFSWEEP: wrote %s\n", post_test_ef_results_csv.c_str());
            }
            ef = saved_ef;
        }

        // Out-of-distribution post-test: same trained graph, a stream from a
        // DIFFERENT cluster's cache. Its own refs/proxy_gt/sigma_levels are used,
        // so the sigma is that cache's grid entry at sigma_index -- the two
        // caches are built the same way, so the noise level matches even though
        // the centre does not. Recall is scored offline from the dump against an
        // exact truthset; the embedded gt columns are this cache's proxy_gt.
        if (ood_n > 0 && !ood_cache_npz.empty() && !ood_results_csv.empty()) {
            auto ood_npz = npio::load_npz(ood_cache_npz);
            // Two OOD cache shapes are supported. A cluster cache carries
            // refs/proxy_gt/sigma_levels and a jittered stream is generated from
            // it below. A fixed OOD eval set (e.g. deep10m_sparse_ood_cache.npz)
            // carries only eval_vecs/eval_gt -- those ARE the queries, so they
            // are used as-is, exactly like the checkpoint-test stream.
            if (!has_npz(ood_npz, "refs") && has_npz(ood_npz, "eval_vecs")) {
                auto &ev = find_npz(ood_npz, "eval_vecs");
                auto &eg = find_npz(ood_npz, "eval_gt");
                const int cache_n = (int)ev.shape[1];
                const int ood_dim = (int)ev.shape[2];
                if (ood_dim != g_query_dim) {
                    fprintf(stderr, "OOD cache dim %d != index query dim %d\n", ood_dim, g_query_dim);
                } else {
                    const int n_use = ood_n < cache_n ? ood_n : cache_n;
                    const size_t voff = (size_t)sigma_index * cache_n * ood_dim;
                    const size_t goff = (size_t)sigma_index * cache_n * g_k;
                    FILE *of = fopen(ood_results_csv.c_str(), "w");
                    if (!of) {
                        fprintf(stderr, "cannot open %s\n", ood_results_csv.c_str());
                    } else {
                        fprintf(of, "q");
                        for (int j = 0; j < K; j++) fprintf(of, ",res%d", j);
                        for (int j = 0; j < g_k; j++) fprintf(of, ",gt%d", j);
                        fputc('\n', of);
                        int64_t hits = 0;
                        for (int i = 0; i < n_use; i++) {
                            std::vector<int> result_ids, visited;
                            g.search(ev.f32() + voff + (size_t)i * ood_dim, (uint64_t)K, (uint64_t)ef,
                                     disk_beam_width, result_ids, visited);
                            std::unordered_set<int> gtset;
                            for (int j = 0; j < K; j++) gtset.insert((int)eg.i64()[goff + (size_t)i * g_k + j]);
                            for (int id : result_ids) if (gtset.count(id)) hits++;
                            fprintf(of, "%d", i + 1);
                            for (int j = 0; j < K; j++)
                                fprintf(of, ",%d", j < (int)result_ids.size() ? result_ids[j] : -1);
                            for (int j = 0; j < g_k; j++)
                                fprintf(of, ",%lld", (long long)eg.i64()[goff + (size_t)i * g_k + j]);
                            fputc('\n', of);
                        }
                        fclose(of);
                        fprintf(stderr, "  OODTEST: n=%d recall@%d=%.6f (fixed eval set) -> %s\n",
                                n_use, K, (double)hits / ((double)n_use * K), ood_results_csv.c_str());
                    }
                }
            } else {
            auto &ood_refs = find_npz(ood_npz, "refs");
            auto &ood_proxy_gt = find_npz(ood_npz, "proxy_gt");
            auto &ood_sigmas = find_npz(ood_npz, "sigma_levels");
            const int ood_nref = (int)ood_refs.shape[0];
            const float ood_sigma = sigma_override >= 0.0 ? (float)sigma_override
                                                          : ood_sigmas.f32()[sigma_index];
            if ((int)ood_refs.shape[1] != g_query_dim) {
                fprintf(stderr, "OOD cache dim %d != index query dim %d\n",
                        (int)ood_refs.shape[1], g_query_dim);
            } else {
                std::vector<float> ood_v; std::vector<int64_t> ood_g;
                make_single_sigma_stream(ood_refs.f32(), ood_proxy_gt.i64(), ood_nref, ood_sigma,
                                         ood_n, post_test_seed, ood_v, ood_g,
                                         0, QUERY_ORDER_IID, 1, 0, nullptr);
                fprintf(stderr, "  OOD: cache=%s refs=%d sigma=%.5f n=%d seed=%ld\n",
                        ood_cache_npz.c_str(), ood_nref, ood_sigma, ood_n, post_test_seed);
                if (!dump_ood_queries_path.empty())
                    write_query_stream_bin(dump_ood_queries_path, ood_v, ood_n, g_query_dim);
                FILE *of = fopen(ood_results_csv.c_str(), "w");
                if (!of) {
                    fprintf(stderr, "cannot open %s\n", ood_results_csv.c_str());
                } else {
                    fprintf(of, "q");
                    for (int j = 0; j < K; j++) fprintf(of, ",res%d", j);
                    for (int j = 0; j < g_k; j++) fprintf(of, ",gt%d", j);
                    fputc('\n', of);
                    int64_t ood_hits = 0;
                    for (int i = 0; i < ood_n; i++) {
                        std::vector<int> result_ids, visited;
                        g.search(ood_v.data() + (size_t)i * g_query_dim, (uint64_t)K, (uint64_t)ef,
                                 disk_beam_width, result_ids, visited);
                        std::unordered_set<int> gtset;
                        for (int j = 0; j < K; j++) gtset.insert((int)ood_g[(size_t)i * g_k + j]);
                        for (int id : result_ids) if (gtset.count(id)) ood_hits++;
                        fprintf(of, "%d", i + 1);
                        for (int j = 0; j < K; j++)
                            fprintf(of, ",%d", j < (int)result_ids.size() ? result_ids[j] : -1);
                        for (int j = 0; j < g_k; j++)
                            fprintf(of, ",%lld", (long long)ood_g[(size_t)i * g_k + j]);
                        fputc('\n', of);
                    }
                    fclose(of);
                    fprintf(stderr, "  OODTEST: n=%d recall@%d=%.6f (vs this cache proxy_gt) -> %s\n",
                            ood_n, K, (double)ood_hits / ((double)ood_n * K), ood_results_csv.c_str());
                }
            }
            }
        }
        if (g.hotcache.async_writeback) {
            g.hotcache.wait_idle();
            std::lock_guard<std::mutex> lk(g.hotcache.mu);
            fprintf(stderr, "  CBLK_ASYNC_DRAIN: writes=%ld rewrites=%ld pending=%zu "
                            "inflight=%ld io_s=%.3f wait_s=%.3f dirty=%zu\n",
                    g.hotcache.async_writes, g.hotcache.async_rewrites,
                    g.hotcache.writeq.size(), g.hotcache.inflight,
                    g.hotcache.async_io_s, g.hotcache.async_wait_s, g.hotcache.dirty.size());
        }
    }
    if (pqf) fclose(pqf);
    if (ctf) fclose(ctf);
    if (post_per_query_file) fclose(post_per_query_file);
    if (g_query_results_file) { fclose(g_query_results_file); g_query_results_file = nullptr; }
    if (g_audit_repair_dump_f) { fclose(g_audit_repair_dump_f); g_audit_repair_dump_f = nullptr; }
    if (g_audit_retest_dump_f) { fclose(g_audit_retest_dump_f); g_audit_retest_dump_f = nullptr; }
    if (g_gain_terms_file) { fclose(g_gain_terms_file); g_gain_terms_file = nullptr; }
    if (g_detector_calibration.file) {
        fclose(g_detector_calibration.file);
        g_detector_calibration.file = nullptr;
    }
    // Only meaningful for an in-place run: with --work-copy the file is about to be
    // deleted, so undoing the edits would be pure write amplification.
    if (restore_dirty_between) g.restore_dirty();
    if (!layout_map_save.empty()) g.save_layout_map(layout_map_save);
    return 0;
}
