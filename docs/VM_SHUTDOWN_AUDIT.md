# VM shutdown audit

Audit date: 2026-08-23 UTC.

## Decision

**GO after the evidence release below is remotely verified.** All final model
checkpoints are on Hugging Face, and all portable experiment evidence is
packaged for GitHub. No training or evaluation process was active during the
audit. The VM may be stopped or destroyed after the GitHub assets pass their
downloaded checksum check.

## Remote checkpoint coverage

| Experiment family | GitHub | Hugging Face | Verification |
|---|---:|---:|---|
| Frozen seen expert, 100k | yes | 1/1 | local SHA equals remote LFS SHA |
| Naive FT, N=1/2/5/10/25 | yes | 30/30 | all local and remote LFS SHA values match |
| Target-LoRA and Replay-LoRA, N=1 | yes | 12/12 | 168 required remote files checked, 0 mismatches |
| Frozen-Stats FT and L2-SP, N=1 | yes | 12/12 | 144 required remote files checked, 0 mismatches |

Total final checkpoint coverage is **55/55**.

Verified Hugging Face revisions:

- seen expert: `96b6cfc66acf1c40c1d243961f88bbf13eb9efa9`;
- naive family: `bc21bfacc2007b011d986924da8d0e5d6b21956c`;
- LoRA family: `cb06a308d0ef3af345d1bb6bdf034084bc7ecc54`;
- stability family: `f6e4256329f7b6cf232d138d61529b183c7e03bd`.

The frozen seen weight SHA-256 is
`2cd510a594a87580f7368b782ca9b37332c0e5002d807093c759e95fbfb57c88`.
Both new families have `integrity_ok=true`; their remote indices contain 12
cells each and exactly match all 24 locally verified weight hashes.

## Portable evidence bundle

Release: [`v0.4.1-vm-shutdown-evidence`](https://github.com/alexsuw/smolvla-libero-fewshot/releases/tag/v0.4.1-vm-shutdown-evidence).

Asset: `smolvla-vm-shutdown-evidence-20260823.tar.gz`.

- archive size: 48,024,999 bytes;
- SHA-256: `946273045eb1268a0174d2e40293a9f29788ac57d9c6f80feae449ac40ead15f`;
- source set: 4,648 files and 113,915,519 uncompressed bytes;
- archive entries including directories: 6,424;
- included recorded rollout videos: 330;
- forbidden checkpoint/state entries after packaging: 0;
- credential-pattern matches before packaging: 0.

The bundle covers `/mnt/vla/{runs,eval,validation,bootstrap,doctor,logs,ops}`.
It contains run manifests, resolved configs, trainable scopes, metrics, event
logs, checkpoint completion/checksum metadata, evaluation summaries, raw
structured rollout records and traces, validation evidence, and rollout
videos.

Excluded by construction:

- model/adapter/optimizer/RNG files: `*.pt`, `*.pth`, `*.safetensors`;
- `train_state.json`;
- raw `*.ppm` frames;
- datasets and caches.

Verify a downloaded copy with:

```bash
sha256sum -c smolvla-vm-shutdown-evidence-20260823.tar.gz.sha256
```

## Reproducibility and publication checks

The repository contains experiment implementations, configs, fixed splits,
result tables, report source, plots, compiled PDF, HF model cards, and the
fail-closed extended publisher. The publisher verifies base/dataset/model
revisions, first-demo IDs, seeds, trainable scope, normalization digests,
rollout protocols and counts, initial-state fingerprints, completion markers,
and file checksums. It excludes optimizer/RNG state, datasets, raw rollouts,
traces, videos, and credentials from HF.

At audit time the GitHub CI for the checkpoint publication commit was green.
No object-storage credentials were configured on the VM; GitHub Releases and
Hugging Face are therefore the verified independent remote copies.

## Final checklist

- [x] No active training or evaluation process.
- [x] Source, configs, splits, tables, report source/PDF, and publisher on GitHub.
- [x] All 55 final checkpoint cells verified remotely on HF.
- [x] Portable non-weight evidence packaged, scanned, and checksummed.
- [x] Evidence archive excludes checkpoint/state/data files.
- [ ] GitHub release assets downloaded again and checksum-verified.

Only the last item must be completed before destroying the VM or `/mnt/vla`.
Stopping compute while retaining the volume is already safe.
