# Day 2 — Full Documentation

**Date**: May 13, 2026 (Wednesday)
**Hardware**: NVIDIA RTX A5000 (Ampere GA102, sm_86)
**Initial Toolchain**: CUDA 12.8.93, nvcc release 12.8
**New-pod Toolchain (post-migration)**: CUDA 12.4.1 devel, PyTorch 2.4.0, Ubuntu 22.04
**Hours estimated**: ~4.5 hours total (kernel work + infrastructure recovery)

---

## Day 2 outcome at a glance

| What | Result |
|---|---|
| First real CUDA kernel | **93.5% of A5000 peak DRAM bandwidth** |
| Block-size sweep | Plateau 64-1024 within 3% of itself, block=32 is only meaningfully bad |
| SASS analysis | 14-instruction kernel body, `LDG.E.CONSTANT` payoff for `__restrict__` |
| Files committed | 7 files (6 source/data + 1 writeup) |
| Pod-side push pipeline | Validated end-to-end with persistent credential helper |
| Windows-side push pipeline | Validated, GCM OAuth flow cached |
| Infrastructure resilience | Stress-tested via 9 accidental orphan pods, recovered cleanly |

---

## PART 1 — Kernel work (the planned content)

### `vector_add.cu` — first real CUDA kernel

**Operation**: `C[i] = A[i] + B[i]` for `N = 2^24 = 16,777,216` floats.

**Why this kernel is bandwidth-bound by design**: One FLOP per output element while moving 12 bytes (read 4 of A, read 4 of B, write 4 of C). Arithmetic intensity = 1/12 = **0.083 FLOP/byte**. The A5000's roofline crossover from bandwidth-bound to compute-bound sits at ~36 FLOP/byte (27.8 TFLOPS / 768.1 GB/s). The kernel is two orders of magnitude below the crossover — optimization analysis is about memory traffic, not arithmetic.

### Headline measurements (steady-state)

| Metric | Value |
|---|---|
| Kernel time (min of 10, with warmup) | **0.280 ms** |
| Effective bandwidth | **719 GB/s** |
| Peak DRAM bandwidth (A5000 spec) | 768.1 GB/s |
| **Fraction of peak** | **93.5%** |
| Theoretical occupancy at block=256 | 100% |
| Registers per thread | 12 (16 effective with sm_86 granularity) |
| Spills / stack frame / barriers | 0 / 0 / 0 |
| Correctness | PASS, 0 errors against CPU reference |
| Arithmetic intensity | 0.083 FLOP/byte (strongly bandwidth-bound) |

### Block-size sweep results (the full table)

Test methodology: 5 warmup runs, min-of-10 timing per config, single binary with shared device state.

| Block | Grid | Warps/Blk | Blks/SM | TheoryOcc% | BestTime (ms) | BW (GB/s) | %Peak |
|---|---|---|---|---|---|---|---|
| 32 | 524288 | 1 | 16 | 33.3 | 0.426 | 472.7 | **61.5** |
| 64 | 262144 | 2 | 16 | 66.7 | 0.277 | 726.6 | **94.6** |
| 128 | 131072 | 4 | 12 | 100.0 | 0.280 | 719.7 | 93.7 |
| 256 | 65536 | 8 | 6 | 100.0 | 0.280 | 718.2 | 93.5 |
| 512 | 32768 | 16 | 3 | 100.0 | 0.279 | 721.0 | 93.9 |
| 1024 | 16384 | 32 | 1 | 66.7 | 0.285 | 707.5 | **92.1** |

### Three kernel-level insights empirically validated

1. **Methodology dominates apparent performance.** Naive single-run timing reports 76.8% of peak; same kernel with warmup + min-of-N measures 93.5%. The 17 percentage-point gap is entirely measurement methodology, not the kernel. Rule going forward: warmup before measuring, min-of-N for headline numbers.

2. **Volkov "Better Performance at Lower Occupancy" effect is real on Ampere.** Block=64 (67% occupancy) ties block=256 (100% occupancy) at ~94% peak. For memory-bound kernels on Ampere, the latency-hiding curve has a knee around 32 warps/SM, not a slope to 48. Occupancy is necessary up to a knee, not a linear slope.

3. **`const float* __restrict__` is a free optimization** with concrete SASS payoff: the compiler emits `LDG.E.CONSTANT` (read-only L1 cache path) instead of plain `LDG.E`. Verified directly in cuobjdump output. Default-on for read-only inputs.

### SASS observations (14-instruction kernel body)

```
S2R R6, SR_CTAID.X                                   ; blockIdx.x
S2R R3, SR_TID.X                                     ; threadIdx.x
IMAD R6, R6, c[0x0][0x0], R3                         ; i = blockIdx.x * blockDim.x + threadIdx.x
ISETP.GE.AND P0, PT, R6, c[0x0][0x178], PT           ; P0 = (i >= N)
@P0 EXIT                                             ; predicated exit (no branch)
MOV R7, 0x4
IMAD.WIDE R4, R6, R7, c[0x0][0x168]                  ; &A[i]
IMAD.WIDE R2, R6.reuse, R7.reuse, c[0x0][0x160]      ; &B[i] (with reuse hints)
LDG.E.CONSTANT R4, [R4.64]                           ; load A[i] via read-only cache
LDG.E.CONSTANT R3, [R2.64]                           ; load B[i] via read-only cache
IMAD.WIDE R6, R6, R7, c[0x0][0x170]                  ; &C[i] (scheduled in load shadow)
FADD R9, R4, R3                                      ; the entire arithmetic of the kernel
STG.E [R6.64], R9                                    ; store C[i]
EXIT
```

Three additional observations from the SASS:

- **The compiler does instruction-level latency hiding statically.** The `IMAD.WIDE` for `&C[i]` is scheduled BETWEEN the two A/B loads and the FADD, because it has no data dependency on the load results. Executes while loads are outstanding.
- **`.reuse` operand hints** on the second `IMAD.WIDE` told the operand reuse cache to hold R6 and R7 across consecutive address computations, saving register-file ports.
- **The boundary check `if (i < N)` compiled to predication, not branching.** `ISETP` sets a 1-bit predicate, `@P0 EXIT` is masked execution. For the last block where some threads have `i >= N`, the hardware masks those lanes rather than diverging the warp.

### `ncu` profiling situation

`ncu --set full` failed with `ERR_NVGPUCTRPERM`. Root cause: NVIDIA driver host-level flag `NVreg_RestrictProfilingToAdminUsers=1`, set by RunPod for multi-tenant security on Secure Cloud. No in-container fix exists (host driver state can't be changed from inside container).

**Alternative measurements collected to replace ncu**: resource_usage from compile, SASS inspection via cuobjdump, block-size sweep, methodology comparison. These cover what ncu would have given for Day 2's kernel.

**Decision**: defer pod-replacement side quest until before Week 3 (tiled matmul, where profiling matters more). Strided/transpose kernel for Day 3 can still measure the bandwidth-drop signature without ncu.

---

## PART 2 — Git workflow validation

### Pod-side push pipeline (Phase 1)

Six files committed and pushed from the pod:

```
month_01/week_01/vector_add.cu                    (123 lines, first real CUDA kernel)
month_01/week_01/vector_add_output.txt            (bare-run output, 0.341ms / 589.9 GB/s)
month_01/week_01/vector_add_sweep.cu              (142 lines, sweep with warmup + min-of-10)
month_01/week_01/vector_add_sweep_output.txt      (sweep results, best 0.280ms / 719 GB/s)
month_01/week_01/resource_usage.txt               (nvcc --resource-usage output)
month_01/week_01/sass_dump.txt                    (cuobjdump --dump-sass output)
```

**Commit hash**: `16fa62f31e51443ad38105c0b0b11cdac2da8e4c` ("Day 2: vector_add kernel + block-size sweep + SASS analysis")

Push completed silently — credential helper at `/workspace/.git-credentials` (via the symlinked gitconfig) auto-authenticated against GitHub. This validates the pod-side persistence design end-to-end for the first time.

### Windows-side push pipeline (Phase 2)

One file (the writeup) committed and pushed from Windows Git Bash:

```
month_01/week_01/vector_add_writeup.md            (131 lines, full analysis + lessons learned)
```

**Commit hash**: `7305438` ("Day 2: writeup with full analysis and lessons learned")

The first Windows-side push triggered **Git Credential Manager OAuth flow**: `info: please complete authentication in your browser...` → GitHub authorize page → credential cached in Windows Credential Manager under `git:https://github.com`. Future pushes from this Windows machine will be silent.

### Phase 3 sync to pod (post-migration)

After the new pod was up, `git pull` brought the writeup commit (`7305438`) down. Fast-forward, no merge, no credential prompt. End-to-end sync now validated across **GitHub origin, Windows local, and pod local** — all three locations at the same commit.

---

## PART 3 — Infrastructure chaos and recovery (the "image debug thing")

This was the unplanned part of Day 2. After Phase 2 completed cleanly around 03:40 EDT, the pod was stopped to halt billing while the Windows-side writeup work was done. When the pod was started back up for Phase 3, RunPod's GPU availability problem cascaded into nine accidental pods. Full timeline:

### Trigger event

The original pod `q7t5e54yaggy80` (`envious_blush_cicada-migration`) on US-IL-1 Secure Cloud was stopped. When started again, RunPod responded with the **"Your Pod's GPUs are no longer available"** dialog — the host that previously held the A5000 had reassigned it to another tenant during the stop window. Three options offered:

1. **Automatically migrate your Pod data** (Recommended) — RunPod spawns a new pod with the same GPU type on a different host, network volume re-attaches
2. **Start Pod using CPUs** — same pod ID, no GPU (useless for kernel work)
3. **Do nothing** — close dialog, retry later

### What went wrong

The "Auto-migrate" option was clicked multiple times across the troubleshooting session. The dialog kept re-appearing — likely from refresh attempts and the user trying to "fix" the situation by clicking again. **Each click queued a separate migration job and spawned a separate destination pod.** RunPod did not deduplicate the migration jobs.

Result: 4 migration target pods all running simultaneously at $0.27/hr each, **burning $1.08/hr in compute for empty pods doing nothing**, while the original pod stayed locked in a migration state.

### Pod IDs accumulated through the chaos

| Pod ID | Pod name | State at end of chaos |
|---|---|---|
| `q7t5e54yaggy80` | envious_blush_cicada-migration | Original pod, stopped, $0.00/hr (the source) |
| `v1x1qefxq7wqrf` | envious_blush_cicada | Pre-existing stub from before Day 2, $0.00/hr |
| `q1mh24s09mym1g` | envious_blush_cicada-migration-migration | Orphan migration target #1 |
| `tj4bxg9c169a2u` | envious_blush_cicada-migration-migration | Orphan migration target #2 |
| `60q4vk7f40jtth` | envious_blush_cicada-migration-migration | Orphan migration target #3 |
| `hn85xbarrnn8nh` | envious_blush_cicada-migration-migration | Orphan migration target #4 |
| `xzxf8ew3jjh4z2` | envious_blush_cicada-migration-migration | Orphan migration target #5 |
| `9e0pvz1x6znojm` | envious_blush_cicada-migration-migration | Orphan migration target #6 |
| `g09r8v02tekabu` | envious_blush_cicada-migration-migration | Orphan migration target #7 (the last active one) |

**Total: 9 pods** in the RunPod console for what should have been a single working environment.

### Recovery path

The breakthrough was navigating to the **Storage page** rather than trying to identify which pod had the data. Key realizations:

1. **The network volume `cuda-workspace` is the source of truth for data.** Pods are just containers that mount it. The volume contains gitconfig, credentials, and the repo clone; pod-local disk doesn't matter.
2. **The volume has S3 API access** — a safety net not used today but worth knowing for future emergencies. The `cuda-workspace` volume (ID `6k7lxzjlrj`) can be browsed/downloaded via S3-compatible tools without ever attaching it to a pod.
3. **The "Configure Pod with volume" button** on the volume detail page is the clean exit from any pod-side mess. Click it, deploy a fresh pod, mount the existing volume — the data is preserved.

### The new working pod

Deployed fresh from "Configure Pod with volume":

- **Pod name**: `sudhakar_cuda-a5000`
- **Pod ID**: `eqddifo4txt5i7`
- **Hardware**: RTX A5000, 24 GB VRAM, 25 GB RAM, 12 vCPU
- **Region**: US-IL-1 Secure Cloud (locked by the volume location)
- **Template**: Runpod Pytorch 2.4.0 (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
- **External IP**: `203.57.40.106`, SSH port 10186
- **Cost**: $0.278/hr total ($0.27 GPU + $0.003 pod disk + $0.005 volume)

Volume contents verified intact after the new pod booted:

```
total 3908
drwxrwxrwx 3 root root 2000295 May 13 07:20 .
drwxr-xr-x 1 root root     109 May 13 08:39 ..
-rw-rw-rw- 1 root root     123 May 13 07:20 .git-credentials
-rw-rw-rw- 1 root root     176 May 13 03:25 .gitconfig
drwxrwxrwx 6 root root 2000295 May 12 07:40 cuda-gpu-kernel-learning
```

All three critical artifacts (`.gitconfig`, `.git-credentials`, the repo directory) present and intact. Symlink restored, git identity verified, Phase 3 pull completed without credential prompt. Day 2 work is now synchronized across all three locations.

---

## PART 4 — Cumulative lessons learned (kernel + infrastructure)

### Kernel-level lessons

1. **Methodology dominates apparent performance.** Always warmup, always min-of-N.
2. **Occupancy is necessary up to a knee, not a slope.** For memory-bound kernels on Ampere, the knee is near 32 warps/SM.
3. **`const` + `__restrict__` is a free optimization** with concrete SASS-level payoff (`LDG.E.CONSTANT`). Default-on for read-only inputs.
4. **The compiler does instruction-level latency hiding statically**, reordering address compute to fill load shadow.
5. **A first kernel hitting ~94% of peak DRAM is normal for a clean coalesced access pattern.** The remaining ~6% gap is fundamental memory subsystem overhead.
6. **Predication, not branching**, for simple boundary checks like `if (i < N)`. Hardware masks lanes; warp doesn't diverge.

### Infrastructure / workflow lessons

1. **RunPod stops don't reserve the GPU.** Any stop is a probability of GPU loss when you try to restart. The longer the gap, the worse the odds.
2. **Migration on RunPod is per-click, not per-intent.** Each click of "Auto-migrate" spawns a separate destination pod. The system doesn't deduplicate. Click ONCE then walk away.
3. **Pod ID changes through migration.** The original pod becomes a stub; you work with the new pod ID going forward. Update your context doc immediately when this happens.
4. **The network volume is the source of truth, not any pod.** Volume cost (~$0.005/hr) is continuous and small. Pod cost is only when running.
5. **S3 API access to the volume exists** as an emergency backup channel — you can read files off the volume without attaching any pod.
6. **"Configure Pod with volume"** is the clean recovery mechanism from any pod-side chaos. Use it instead of trying to resurrect broken pods.
7. **Container disk is ephemeral.** `~/.gitconfig` symlink to `/workspace/.gitconfig` must be re-created on every container restart. This is mechanical, not a bug.
8. **Network volume + cloud type are coupled.** A Secure Cloud volume can only attach to Secure Cloud pods. Switching to Community Cloud (for better availability or to unblock `ncu`) requires creating a fresh volume.

---

## PART 5 — Updated pitfalls catalog (cumulative across Day 1 and Day 2)

1. **Pod web terminal has paste size limits (~45 lines).** Heredocs longer than this truncate mid-paste. Workaround: chunk with `cat >> file <<'EOF'` ... `EOF`, paste each chunk separately.

2. **Pod web terminal corrupts Unicode characters during paste.** Right-arrow, approximately-equal, multiplication-x, em-dashes become control-char escape sequences. Solution: ASCII-only on the pod; Unicode-containing files go on Windows.

3. **Pod web terminal sessions time out after idle.** Reconnect via Connect → Start Web Terminal; container state preserved.

4. **`~/.gitconfig` is wiped on every container restart.** Persistent source: `/workspace/.gitconfig`. Restart procedure: `ln -sf /workspace/.gitconfig ~/.gitconfig`. NOT YET AUTOMATED — consider `/workspace/setup_session.sh` as a Day 3+ improvement.

5. **`ncu` profiling is BLOCKED on RunPod Secure Cloud** due to host-level `NVreg_RestrictProfilingToAdminUsers=1`. `ERR_NVGPUCTRPERM`. No in-container fix. Options: move to Community Cloud, or accept alt-measurements long-term.

6. **Naive single-run kernel timer under-reports performance by 15-20%.** Always: warmup + min-of-N for headline numbers.

7. **Migration creates new pod ID; original becomes stub.** Update context doc and pod references when migration happens. Don't accidentally start the wrong one — identify by ID, not name.

8. **RunPod auto-migration dialog spawns a NEW pod per click.** Multiple clicks on "Automatically migrate your Pod data" = multiple orphan pods burning compute simultaneously. **RULE: see migration dialog → close it with X if pod was just stopped; only click Auto-migrate ONCE and walk away for 5 minutes.** Each migration creates a new pod ID. The network volume follows automatically. Stopped pods on Secure Cloud are NOT guaranteed to come back on the same hardware.

9. **Network volume is the source of truth, not any specific pod.** Use Storage page (Volume detail) to identify volume attachment. Use "Configure Pod with volume" to create a fresh pod with the existing data, bypassing any orphan/locked pods.

10. **Volume billing is continuous** at $0.005/hr (~$3.50/month for 50 GB) regardless of pod state. Pods at $0.00/hr in the Pods list does NOT mean zero total cost.

---

## PART 6 — Outstanding cleanup tasks (for Day 3 morning)

**Orphan pod termination** — these are the 8 pods that should be terminated permanently. All are at $0.00/hr (stopped), so no urgency, but they clutter the pod list:

```
q1mh24s09mym1g    envious_blush_cicada-migration-migration   terminate
tj4bxg9c169a2u    envious_blush_cicada-migration-migration   terminate
60q4vk7f40jtth    envious_blush_cicada-migration-migration   terminate
hn85xbarrnn8nh    envious_blush_cicada-migration-migration   terminate
xzxf8ew3jjh4z2    envious_blush_cicada-migration-migration   terminate
9e0pvz1x6znojm    envious_blush_cicada-migration-migration   terminate
g09r8v02tekabu    envious_blush_cicada-migration-migration   terminate
q7t5e54yaggy80    envious_blush_cicada-migration             terminate (volume already detached)
v1x1qefxq7wqrf    envious_blush_cicada                       investigate or terminate (pre-existing stub)
```

**Procedure** for each: in RunPod console → Pods → click ⋮ on the row → Terminate → confirm. The pod `sudhakar_cuda-a5000` (ID `eqddifo4txt5i7`) is the keeper; do not terminate this one.

**Why defer to Day 3 morning**: at the end of an exhausting session, judgment about "which pod is the keeper" is degraded. Fresh-mind cleanup tomorrow is safer. The financial cost of deferring is zero.

---

## PART 7 — Day 3 kernel work plan

**Primary kernel**: Strided / transpose access pattern. Compare against today's coalesced baseline of 93.5% peak.

**Expected result**: Bandwidth drop from ~94% to <30% of peak as stride increases. The exact drop depends on stride and warp-level access pattern.

**Why this kernel matters**: It's the cleanest demonstration of memory coalescing as a first-order performance factor. The vector_add baseline showed what "good" looks like at peak; the strided kernel shows what "bad" looks like and why. Together they bracket the bandwidth-bound regime.

**Pre-reading recommendation**: Spend 30 minutes on the first three kernel iterations of Simon Boehm's matmul article (https://siboehm.com/articles/22/CUDA-MMM) before writing Day 3 code. His "naive → coalesced → shared memory" progression gives the mental model for what optimization moves look like at the kernel level.

**Without ncu, the strided kernel can still measure**:
- Bandwidth achieved at each stride
- Comparison against the coalesced baseline
- Resource usage and SASS for any compiler differences

**With ncu (if/when unblocked)**, additional metrics worth instrumenting:
- L1 hit rate degradation as stride grows
- L2 throughput as fraction of L2 capacity
- Warp stall reason breakdown
- Sector traffic per request (the mechanism of bandwidth loss)

The mechanistic-explanation part of the writeup is what ncu unlocks. Without ncu, the writeup will be empirical (bandwidth drops, predicted by coalescing theory). That's still valuable — just acknowledge the gap in the writeup.

---

## Repository state at end of Day 2

GitHub: https://github.com/margondla/cuda-gpu-kernel-learning
Latest commit on origin/main: `7305438` ("Day 2: writeup with full analysis and lessons learned")

```
cuda-gpu-kernel-learning/
├── .gitignore
├── README.md
├── month_01/
│   └── week_01/
│       ├── device_query.cu                  (Day 1)
│       ├── device_query_output.txt          (Day 1)
│       ├── resource_usage.txt               (Day 2)
│       ├── sass_dump.txt                    (Day 2)
│       ├── vector_add.cu                    (Day 2)
│       ├── vector_add_output.txt            (Day 2)
│       ├── vector_add_sweep.cu              (Day 2)
│       ├── vector_add_sweep_output.txt      (Day 2)
│       └── vector_add_writeup.md            (Day 2)
└── notes/
    └── rtx_a5000_constants.md               (Day 1)
```

All three locations synchronized:
- GitHub origin/main → `7305438`
- Windows local main → `7305438`
- Pod local main (sudhakar_cuda-a5000) → `7305438`

---

## Final closeout actions completed

- [x] Phase 1 (pod-side commit + push): 6 files, commit `16fa62f`
- [x] Phase 2 (Windows-side commit + push): writeup, commit `7305438`
- [x] Phase 3 (pod-side pull after migration): fast-forward to `7305438`
- [x] New pod deployed: `sudhakar_cuda-a5000` (`eqddifo4txt5i7`)
- [x] Volume integrity verified: gitconfig, credentials, repo all intact
- [x] Git identity working through symlink on new pod
- [x] Day 2 writeup committed and accessible from all three locations
- [ ] Stop the new pod for the night (final action before bed)
- [ ] Update context doc with new pod ID for Day 3 chat
- [ ] Orphan pod cleanup (deferred to Day 3 morning)
