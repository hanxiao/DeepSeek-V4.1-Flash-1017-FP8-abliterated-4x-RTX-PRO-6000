# The k staircase: an exact rule for the best draft length, and what it says for us

Source: arXiv 2610.34440, "Speculative Decoding for Mixture-of-Experts Models: Expert Bytes, Kernel Bandwidth
and the Best Draft Length" (Y Lan, Oct 2026; the main results are proved in Lean 4). Read 2026-11-08 for the
step axis. The arithmetic against our numbers is mine.

## The model

A step that drafts k tokens and verifies k+1 has

```
V(k) = tau(k+1) + h(k)
tau(n) = t0 + D(n)/beta_D + E_p(n)/beta_p(n) + phi_p(n)
```

with `D(n)` the bytes read once per pass (dense weights, KV), `E_p(n)` the expert bytes, `beta_p` the kernel's
*effective* bandwidth and `phi_p` the fixed work (launches). Two kernel families:

- **per-slot**: reads one expert per slot, `E_slot(n) = m n W` bytes.
- **deduplicating**: groups tokens by expert, `E_dedup(n) = sum_l w_l U_l(n)` where
  `U_l(n) = m * sum_{j=1..n} (1 - rho_l(j))` and `rho_l(j)` is the consecutive-token expert reuse. Dedup is
  faster exactly where `u_bar(n) = E_dedup/E_slot < r(n) = beta_dedup/beta_slot`.

Their measurements: next-token reuse `rho(2)` = 0.363 / 0.480 / 0.492 for OLMoE / Qwen3-30B-A3B / gpt-oss
(uniform routing would give 0.125 / 0.0625 / 0.125), rising to 0.762 / 0.780 / 0.856 at n=9. The per-slot
kernel has the **higher** effective bandwidth (r = 0.32-0.75) because repeated reads hit cache, so dedup wins
only from n = 5-8 tokens per pass, or never.

## The staircase

The best draft length is a monotone staircase in the acceptance rate a. With `c_j = V(j) - V(j-1)` the
marginal cost of the j-th draft and `G_a(k) = 1 + a + ... + a^k`:

```
k*(a) >= j   <=>   a^j V(j-1) > G_a(j-1) c_j          (exact, for V convex and non-decreasing)
```

The j-th step point is `a_j = inf{a : h_j(a) > c_j / V(j-1)}` with `h_j(a) = a^j / G_a(j-1)`, strictly
increasing, so every step point is exact and the staircase can jump by more than one.

This is the exact form of the row-economics rule we had been using in linearised form (`dA > A dT / T`).

## What it says for us

Our inputs: base step 9.1 ms, marginal draft c_j = 0.61 ms, FLy accepted length 4.24 at k=5, which implies a
geometric a = 0.86. k* and throughput:

| implied a | k* at c_j=0.61 | tok/s | k* at c_j=0.45, base 8.0 | tok/s |
|---|---:|---:|---:|---:|
| 0.80 | 6 | 407 | 6 | 468 |
| 0.86 (today + FLy) | **8** | 485 | **9** | 568 |
| 0.90 | 10 | 565 | 11 | 671 |
| 0.92 | 11 | 619 | 12 | 742 |

Three consequences:

1. **FLy moved the optimum right, to k=8.** It does not pay today only because the checkpoint's block is
   trained at 5 (`dspark_block_size: 5`), so positions 6+ accept far below the geometric rate - measured 3.55
   accepted at k=7 against a geometric 4.5. The retrain should therefore train **block 8**, not block 7, and
   the k-sweep must be re-run afterwards with measured (not geometric) accepted lengths.
2. **The row-cost work is multiplicative with k**: at a=0.86, c_j 0.61 -> 0.45 and base 9.1 -> 8.0 takes 485
   to 568 tok/s, more than the sum of the individual shavings.
3. **Dedup vs per-slot is a live question for our kernel, not a settled one.** We dedup (22.5 distinct of 36
   slots = 37.5% reuse, consistent with their measured 0.44-0.49 next-token reuse) at n=6, which is exactly
   their crossover range. Moet measured the deduplicating fused kernel winning (73.2 vs 75.7 us/layer), so it
   is settled empirically for now - but their method (fit each kernel's effective bandwidth on one layer's
   expert operation, then predict whole-model times from routing traces) is reusable if the kernel changes.

## Why host RAM / NVMe cannot help the bytes

4.2 GB per rank per step over 9.1 ms is **462 GB/s per rank**. PCIe Gen5 x16 is ~60 GB/s per direction and the
local NVMe RAID is the same order, so a step's expert bytes would take 4-7x the step time to cross PCIe, no
matter how well the transfer overlaps. The recent offload/prefetch work (Specter arXiv 2511.14102,
Cross-Precision Expert Caching arXiv 2611.03173, DeepInfer-Mac arXiv 2610.36564, the AWS Trainium dataflow
arXiv 2609.21137) all targets boxes where the experts do not fit in VRAM; ours fit, so the 720 GB of host RAM
and 5.9 TB of NVMe stay idle. The only byte reduction without requantisation is none; 2-bit experts halve the
3.0 ms MoE to 1.5 ms.
