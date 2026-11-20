from pathlib import Path
import difflib,hashlib,json
root=Path('/Users/fu/project/HAI/Strata-decode-head'); out=Path('/Users/fu/project/HAI/Strata/bench/results/2026-11-08-prefill-interleave')
modified={}
def change(p,a,b):
 t=modified.get(p,(root/p).read_text())
 assert t.count(a)==1,(p,t.count(a),a[:60])
 modified[p]=t.replace(a,b)
p='include/strata/program/batch_read.hpp'
change(p,'/// Tokens of the next piece of a prompt read on the pipelined slot groups', '''/// Optional reader-only spacing of successful stage-0 submissions. The clock is
/// monotonic microseconds supplied by the caller; no sleeps or GPU waits here.
/// Only submitted() consumes a budget, so an unready group never delays others.
struct ReaderAdmissionCadence {
    static constexpr int64_t max_interval_us = 20000;
    explicit ReaderAdmissionCadence(int64_t us = 0)
        : interval_us(std::clamp<int64_t>(us, 0, max_interval_us)) {}
    int64_t interval_us = 0, last_us = 0, min_gap_us = 0, max_gap_us = 0;
    uint64_t checks = 0, deferred_checks = 0, submissions = 0, bypass_submissions = 0;
    bool have_last = false, bypass_permit = false;
    bool permit(int64_t now_us, bool cancel_or_yield = false) {
        ++checks;
        const bool due = interval_us == 0 || !have_last ||
                         (now_us >= last_us && now_us - last_us >= interval_us);
        bypass_permit = !due && cancel_or_yield;
        if (due || cancel_or_yield) return true;
        ++deferred_checks;
        return false;
    }
    void submitted(int64_t now_us) {
        if (have_last) {
            const int64_t gap = std::max<int64_t>(0, now_us - last_us);
            if (submissions == 1 || gap < min_gap_us) min_gap_us = gap;
            max_gap_us = std::max(max_gap_us, gap);
        }
        last_us = now_us; have_last = true; ++submissions;
        if (bypass_permit) ++bypass_submissions;
    }
};

/// Tokens of the next piece of a prompt read on the pipelined slot groups''')
p='src/program/generate.cpp'
change(p,'        // one turn: serve every running stage, hand finished groups on, start what can start\n        auto pump =', '''        // Only the strict concurrent reader installs this main-thread-owned budget.
        // In-flight stages, commits, sampling and draft polls always run normally.
        strata::program::ReaderAdmissionCadence* reader_admission_cadence = nullptr;
        auto cadence_now_us = [] { return std::chrono::duration_cast<std::chrono::microseconds>(
            Clock::now().time_since_epoch()).count(); };
        // one turn: serve every running stage, hand finished groups on, start what can start
        auto pump =''')
change(p,'                p4.launched(k, pick, p4_l0);\n                stage_group', '''                if (k == 0 && reader_admission_cadence)
                    reader_admission_cadence->submitted(cadence_now_us());   // successful launch RETURN, not GPU completion
                p4.launched(k, pick, p4_l0);
                stage_group''')
change(p,'                        const uint64_t read_id = resident_read ? ++concurrent_mtp_read_id : 0;\n                        if (resident_read)', '''                        const uint64_t read_id = resident_read ? ++concurrent_mtp_read_id : 0;
                        static const int64_t cadence_us = [] {
                            const char* v = std::getenv("STRATA_PIPED_CONCURRENT_ADMIT_US");
                            return v ? std::clamp<int64_t>(std::strtoll(v, nullptr, 10), 0,
                                strata::program::ReaderAdmissionCadence::max_interval_us) : 0;
                        }();
                        strata::program::ReaderAdmissionCadence cadence(cadence_us);
                        struct CadenceScope {
                            strata::program::ReaderAdmissionCadence*& active;
                            ~CadenceScope() { active = nullptr; }
                        } cadence_scope{reader_admission_cadence};
                        if (resident_read && cadence.interval_us > 0) reader_admission_cadence = &cadence;
                        if (resident_read)''')
change(p,'                            if (decoding) pump_ok = pump(true);\n                            // A conservative output count:', '''                            if (decoding) {
                                bool may_start = true;
                                if (reader_admission_cadence) {
                                    const bool slot_stopped = std::any_of(bs.begin(), bs.end(),
                                        [](const BSlot& s) { return s.active && s.stop; });
                                    const bool control = stop_req.load() || slot_stopped ||
                                                         want_yield.load(std::memory_order_acquire) >= 0;
                                    may_start = cadence.permit(cadence_now_us(), control);
                                }
                                pump_ok = pump(may_start);
                            }
                            // A conservative output count:''')
change(p,'                        reader.join();   // (a failed window leaves the read to finish: it is not stopped mid-run)\n                        sp.stop_at_chunk', '''                        reader.join();   // (a failed window leaves the read to finish: it is not stopped mid-run)
                        reader_admission_cadence = nullptr;   // no budget in the final drain or subsequent admission
                        if (resident_read && cadence.interval_us > 0)
                            std::fprintf(stderr, "strata concurrent admission cadence: read_id=%llu interval_us=%lld "
                                                 "checks=%llu deferred_checks=%llu stage0_submissions_loop=%llu "
                                                 "bypass_submissions=%llu min_gap_us=%lld max_gap_us=%lld\\n",
                                         (unsigned long long) read_id, (long long) cadence.interval_us,
                                         (unsigned long long) cadence.checks, (unsigned long long) cadence.deferred_checks,
                                         (unsigned long long) cadence.submissions, (unsigned long long) cadence.bypass_submissions,
                                         (long long) cadence.min_gap_us, (long long) cadence.max_gap_us);
                        sp.stop_at_chunk''')
p='tests/program/batch_read_test.cpp'
change(p,'int main() {\n    // no --batch', '''int main() {
    {
        using strata::program::ReaderAdmissionCadence;
        ReaderAdmissionCadence off;
        CHECK(off.permit(100)); off.submitted(100); CHECK(off.permit(100));
        ReaderAdmissionCadence c(10000);
        CHECK(c.permit(50000));   // first group is immediate
        CHECK(c.permit(50001));   // an unsuccessful eligibility check consumes no budget
        c.submitted(51000);
        CHECK(!c.permit(60999)); CHECK(c.permit(61000));
        CHECK(c.submissions == 1 && c.deferred_checks == 1);
        c.submitted(62000);       // use successful launch return, not the earlier check
        CHECK(!c.permit(71999)); CHECK(c.permit(72000));
        CHECK(c.permit(62001, true)); c.submitted(62002);   // cancel/BYIELD can bypass
        CHECK(c.bypass_submissions == 1 && c.min_gap_us == 2 && c.max_gap_us == 11000);
        CHECK(!c.permit(62003));
        CHECK(ReaderAdmissionCadence(-1).interval_us == 0);
        CHECK(ReaderAdmissionCadence(999999).interval_us == 20000);
        ReaderAdmissionCadence next(10000);
        CHECK(next.permit(62003));   // no pacing debt survives a read, error or BYIELD
    }
    // no --batch''')
parts=[]
for p,t in modified.items():
 parts.append('diff --git a/'+p+' b/'+p+'\n');parts.extend(difflib.unified_diff((root/p).read_text().splitlines(True),t.splitlines(True),fromfile='a/'+p,tofile='b/'+p))
patch=''.join(parts);(out/'reader-admission-cadence-proposal.patch').write_text(patch)
meta={'base_commit':'bf5a8e5b24b1973efbae747a109f5a970b25e09d','applied':False,'priority':'deferred; modest mixed-prefill cost now accepted','patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'files':list(modified)}
(out/'reader-admission-cadence-proposal.json').write_text(json.dumps(meta,indent=2)+'\n')
for p,t in modified.items():
 dst=out/'reader-cadence-review-src'/p;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(t)
print(json.dumps(meta,indent=2))
