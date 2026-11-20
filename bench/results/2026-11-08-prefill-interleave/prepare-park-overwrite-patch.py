from pathlib import Path
import difflib, hashlib, json
root=Path('/Users/fu/project/HAI/Strata-decode-head')
out=Path('/Users/fu/project/HAI/Strata/bench/results/2026-11-08-prefill-interleave')
modified={}
def change(path, before, after):
    text=modified.get(path,(root/path).read_text())
    if text.count(before)!=1:
        raise RuntimeError(f'expected one match: {path}: {before[:60]!r} ({text.count(before)})')
    modified[path]=text.replace(before,after)
p='src/core/conversation_snapshot.cpp'
change(p,'struct KvSaveTrace { double attrs = 0, resize = 0, copy = 0; };',
         'struct KvSaveTrace { double attrs = 0, resize = 0, copy = 0; size_t overwrite_added = 0; };')
change(p,'bool transfer(void* dst, const void* src, size_t n, std::string& error) {', '''// Opt-in A/B: synchronous saves overwrite every added byte, too. The caller has
// checked keep <= the old size; only [keep, n) is copied, preserving its prefix.
// On transfer failure snapshot_save discards its local captured image.
bool park_overwrite_on() {
    static const bool on = [] { const char* v = std::getenv("STRATA_PARK_OVERWRITE"); return v != nullptr && v[0] == '1'; }();
    return on;
}
void resize_snapshot_part(ConversationBuffer& out, size_t n) {
    if (!park_overwrite_on()) { out.resize(n); return; }
    const size_t added = n > out.size() ? n - out.size() : 0;
    out.resize_for_overwrite(n);
    if (park_trace_on()) kv_save_trace().overwrite_added += added;
}

bool transfer(void* dst, const void* src, size_t n, std::string& error) {''')
change(p,'                    dst[i]->resize(sizes[i]);\n                    if (!dst[i]->visit(keep, sizes[i] - keep,',
         '                    resize_snapshot_part(*dst[i], sizes[i]);\n                    if (!dst[i]->visit(keep, sizes[i] - keep,')
change(p,'        else dst[i]->resize(sizes[i]);', '        else resize_snapshot_part(*dst[i], sizes[i]);')
change(p,'void conversation_kv_save_trace(double& attrs_ms, double& resize_ms, double& copy_ms) {\n    const KvSaveTrace& t = kv_save_trace();\n    attrs_ms = t.attrs; resize_ms = t.resize; copy_ms = t.copy;\n}',
'''void conversation_kv_save_trace(double& attrs_ms, double& resize_ms, double& copy_ms, size_t& overwrite_added) {
    const KvSaveTrace& t = kv_save_trace();
    attrs_ms = t.attrs; resize_ms = t.resize; copy_ms = t.copy;
    overwrite_added = t.overwrite_added;
}''')
p='src/core/conversation_state.cpp'
change(p,'void conversation_kv_save_trace(double& attrs_ms, double& resize_ms, double& copy_ms);',
         'void conversation_kv_save_trace(double& attrs_ms, double& resize_ms, double& copy_ms, size_t& overwrite_added);')
change(p,'        conversation_kv_save_trace(attrs_ms, resize_ms, copy_ms);   // (the main layers\' saves above)',
'''        size_t overwrite_added = 0;
        conversation_kv_save_trace(attrs_ms, resize_ms, copy_ms, overwrite_added);   // (the main layers' saves above)''')
change(p,'"begin %.1f ms, K/V %zu MB %.1f ms (layer max %.2f, sum %.2f; attrs %.2f, resize %.2f, copy %.2f)\\n",',
         '"begin %.1f ms, K/V %zu MB %.1f ms (layer max %.2f, sum %.2f; attrs %.2f, resize %.2f, copy %.2f; overwrite_added_bytes %zu)\\n",')
change(p,'                     ms(p4, now()), kv_max, kv_sum, attrs_ms, resize_ms, copy_ms);',
         '                     ms(p4, now()), kv_max, kv_sum, attrs_ms, resize_ms, copy_ms, overwrite_added);')
p='src/core/conversation_validation_test.cpp'
change(p,'    const size_t full_copies = copied_bytes;',
         '    const size_t full_copies = copied_bytes;\n    const int full_copy_calls = copy_calls;')
change(p,'    check(unpublished.kv.empty() && unpublished.live.ids.empty(), "failed incremental capture cannot publish a partial image");',
'''    check(unpublished.kv.empty() && unpublished.live.ids.empty(), "failed incremental capture cannot publish a partial image");
    // No-zero-fill capture must preserve a kept prefix and overwrite every new
    // byte, whether an existing image shrinks, stays the same size, or grows.
    reset();
    const std::vector<ConversationImageKey> no_images;
    const std::vector<ConversationCheckpoint> no_checkpoints;
    for (int64_t tokens : {1,5,9,17,65}) {
        std::vector<int32_t> ids((size_t) tokens);
        for (int64_t i=0;i<tokens;++i) ids[(size_t) i]=(int32_t) i+1;
        const ConversationView changed{ids,no_images,no_checkpoints,true};
        SavedConversation fresh,kept;
        check(conversation_snapshot_save(fresh,changed,ss,g,draft.st,error), "fresh extent capture");
        ConversationKvReuse retained{full.kv,9,std::min<int64_t>(9,tokens)};
        check(conversation_snapshot_save(kept,changed,ss,g,draft.st,error,std::move(retained)),
              "retained capture after shrink, same extent or growth");
        check(kept.kv.size()==fresh.kv.size(), "retained extent has every K/V layer");
        for (size_t i=0;i<fresh.kv.size();++i)
            check(kept.kv[i].k==fresh.kv[i].k && kept.kv[i].v==fresh.kv[i].v &&
                  kept.kv[i].k_scale==fresh.kv[i].k_scale && kept.kv[i].v_scale==fresh.kv[i].v_scale &&
                  kept.kv[i].pooled==fresh.kv[i].pooled, "every retained/new byte equals a fresh capture");
    }
    // Sweep past the running-state copies into every K/V transfer. A failed
    // capture must leave even an existing output object entirely unpublished.
    for (int at=1;at<=full_copy_calls;++at) {
        reset(); fail_copy=at;
        SavedConversation sentinel;
        sentinel.live.ids={-7};
        check(!conversation_snapshot_save(sentinel,view,ss,g,draft.st,error), "each capture transfer failure is reported");
        check(sentinel.live.ids==std::vector<int32_t>{-7} && sentinel.kv.empty() && sentinel.live.gdn.empty(),
              "partial copy never replaces the caller's image");
    }
    reset();
    QsaState missing=draft.st;
    missing.k_pool=nullptr; missing.k_q=nullptr; missing.k_q4=nullptr;
    SavedConversation sentinel;
    sentinel.live.ids={-7};
    check(!conversation_snapshot_save(sentinel,view,ss,g,missing,error), "missing source fails after local allocation");
    check(sentinel.live.ids==std::vector<int32_t>{-7} && sentinel.kv.empty() && sentinel.live.gdn.empty(),
          "missing source cannot publish an unwritten K/V buffer");''')
p='CMakeLists.txt'
change(p,'    set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)',
'''    set_tests_properties(conversation_snapshot_test PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 120)
    add_test(NAME conversation_snapshot_test_overwrite COMMAND conversation_snapshot_test)
    set_tests_properties(conversation_snapshot_test_overwrite PROPERTIES
      ENVIRONMENT "STRATA_PARK_OVERWRITE=1;STRATA_PARK_ASYNC=0;STRATA_CKPT_ASYNC=0" SKIP_RETURN_CODE 77 TIMEOUT 120)''')
change(p,'      set_tests_properties(conversation_transfer_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)',
'''      set_tests_properties(conversation_transfer_test PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1" TIMEOUT 120)
      add_test(NAME conversation_transfer_test_overwrite COMMAND conversation_transfer_test)
      set_tests_properties(conversation_transfer_test_overwrite PROPERTIES
        ENVIRONMENT "CUDA_VISIBLE_DEVICES=-1;STRATA_PARK_OVERWRITE=1;STRATA_PARK_ASYNC=0;STRATA_CKPT_ASYNC=0" TIMEOUT 120)''')
parts=[]
for p,text in modified.items():
    old=(root/p).read_text()
    parts.append('diff --git a/'+p+' b/'+p+'\n')
    parts.extend(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile='a/'+p,tofile='b/'+p))
patch=''.join(parts)
path=out/'park-overwrite-candidate.patch'
path.write_text(patch)
meta={'base_commit':'bf5a8e5b24b1973efbae747a109f5a970b25e09d','applied':False,
      'patch_sha256':hashlib.sha256(patch.encode()).hexdigest(),'files':list(modified),
      'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in modified}}
(out/'park-overwrite-candidate.json').write_text(json.dumps(meta,indent=2)+'\n')
print(json.dumps(meta,indent=2))
