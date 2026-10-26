# Pre-release checklist

Use this checklist for the release candidate. Keep detailed procedures and test logs in their existing locations; link only material approved for the checklist's audience.

- Release version: v2.17.0
- Candidate commit:
- Dependency versions (including `transformers`):
- Release owner:
- Planned release date (listed in tracker): Thursday, 2026-07-30

## Schedule basis

The release tracker lists v2.17.0 on 2026-07-30 as the next planned release, with checks pending and publication planned. This date is in the past as of 2026-10-26; it is used here at the owner's request, not presented as a newly confirmed upcoming release. Section targets below are planning dates derived from that entry, not completed work or deadlines recorded in the tracker.

## 1. Weight-conversion checks

Target: Tuesday, 2026-07-28 (two days before release), leaving time to resolve parity failures.

For each new or changed adapter, apply the silent-failure checks from `adapter_authoring_checklist.md`. Record non-applicable items with a reason.

- [ ] Read `head_dim` from the model config; do not infer it from `d_model / n_heads`.
- [ ] Map QK-normalization and extra normalization tensors explicitly.
- [ ] Verify norm-epsilon placement against the reference implementation. For Gemma-3, epsilon belongs inside the reciprocal square root.
- [ ] Preserve input-embedding scaling.
- [ ] Preserve MoE routing where supported; do not treat an MoE as dense.
- [ ] Run HF-versus-HookedTransformer parity checks in float32 on the smallest public checkpoint; require maximum absolute logit error ≤ `1e-4`.
- [ ] For Gemma-3, run the strengthened #1121 regression test, including the longer prompt and intermediate-activation checks.

Record the checkpoint, actual maximum error, and an approved acceptance-log link. The #1121 test originally passed on a shorter prompt; a longer prompt exposed the mismatch.

## 2. CI readiness and runtime

Target: Wednesday, 2026-07-29 for the candidate run and runtime record; final confirmation from Yusuke on Thursday, 2026-07-30 before tagging. Sharding remains a separate follow-up, without a release deadline here.

- [ ] Record CI and acceptance results for the candidate commit, including the actual acceptance-matrix duration and what that duration measures.
- [ ] Confirm required tests ran; identify skipped coverage explicitly.
- [ ] Obtain Yusuke's green-CI and acceptance sign-off before tagging.
- [ ] Track sharding separately as a follow-up improvement; do not weaken release checks to reduce runtime.

## 3. Demo notebooks and dependency compatibility

Target: Wednesday evening, 2026-07-29, following the existing pre-release sweep cadence.

- [ ] Run the pre-release demo-notebook sweep in the intended release environment, recording dependency versions including `transformers`.
- [ ] Verify the demo cell reported in #1113 when identifiable; record any unresolved identification or coverage gap.
- [ ] Block the release on demo failures.

## 4. Security and disclosure coordination

Target: Tuesday, 2026-07-28 for private coordination; reconfirm clearance before tagging on Thursday, 2026-07-30. Any coordinated disclosure follows its agreed timing.

- [ ] Confirm privately with Yusuke whether any advisory affects the release.
- [ ] Record only clearance status and any approved public reference in this shareable checklist.
- [ ] Follow the existing coordinated-disclosure procedure. Exclude private writeups, reproducers, and nonpublic technical details from this checklist and its shared attachments or links.

## 5. Release decision and monitoring

Target: Thursday, 2026-07-30 for the go/no-go decision and monitoring-owner assignment. Monitor for 48 hours after actual publication (through Saturday, 2026-08-01 if published on the planned date); record any incident as it occurs.

- [ ] Link the approved results and record the go/no-go decision; do not tag on failed required checks.
- [ ] Assign the documented 48-hour post-release monitoring.
- [ ] Record regressions and hotfix outcomes accurately in the release tracker.

Decision:
Sign-off:
Monitoring owner:

## Existing procedures

Consult the existing Drive documents: `adapter_authoring_checklist.md`, `acceptance_test_notes.md`, `release_checklist.md`, and `security_disclosure_process.md`. Keep restricted security material in its private workflow.

Candidate-specific evidence and runtime recording are additions to the existing process. The release date comes from the tracker's planned entry; no current CI status or completion is assumed.
