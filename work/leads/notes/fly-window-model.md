# FLy at k=5: the window size is the lever, not the threshold

Researched 2026-11-08. Source: `vllm/v1/spec_decode/fly.py` at df3bd094 (= the base of the served image;
the module is present and wired in the image, verified by introspection on the production container).
Simulation is mine; the acceptance curve is the drafter teammate's measurement.

## What the kernel actually does

`apply_fly_random_acceptance_kernel` (probabilistic drafting; there is a greedy twin). For each draft
position `pos` in a request's `num_draft_tokens`:

```
can_defer = (not natively_accepted) and valid_draft and draft_prob > 0 and target_prob > 0
            and pos + FLY_WINDOW_SIZE < num_draft_tokens
            and entropy[pos] >= fly_entropy_threshold
for offset in 1..FLY_WINDOW_SIZE:  can_defer &= future_draft_id == future_target_id (native accept)
if can_defer: uniform_prob[pos] = 0.0        # forces the native p/q test to pass
```

`entropy` is the top-`fly_entropy_top_k` (default 3) entropy of the **target** distribution, computed with
the true probabilities of the top-3 tokens (renormalised against the full-vocab logsumexp), so the default
threshold 0.3 is low: it fires at almost any position that is not extremely peaked.

Two consequences that matter more than the threshold:

1. **`pos + FLY_WINDOW_SIZE < num_draft_tokens` means the last W positions can never defer.** At k=5 with the
   default `fly_window_size = min(6, k-1) = 4`, only position 0 is eligible - the least useful setting the
   knob can take.
2. A deferral requires the next W draft tokens to be **natively accepted**, so FLy repairs a position only
   when the chain after it is right. That is exactly the shape of our rejection pattern (position 1 at 0.81,
   later positions 0.67-0.73), which is why the window size dominates.

## Simulation

Independent-Bernoulli model of the native acceptances with the measured conditional rates
`c = [0.8124, 0.7585, 0.7295, 0.6723, 0.7251]` (native AL 3.40, reproduces the measurement), `g` =
P(entropy >= threshold | position is a mismatch). 400k draws, seed 7.

| window W | g=0.5 | g=0.8 | g=0.9 | g=1.0 |
|---|---:|---:|---:|---:|
| 1 | +23.3% | +38.2% | +43.6% | +49.0% |
| 2 | +15.1% | +24.3% | +27.3% | +30.3% |
| 3 | +8.9% | +14.4% | +16.0% | +17.8% |
| 4 (default) | +3.8% | +5.9% | +6.6% | +7.3% |

Script (run with `python3`, no dependencies):

```python
import random
random.seed(7)
c=[0.8124,0.7585,0.7295,0.6723,0.7251]; k=5
def sim(W,g,n=400000):
    tot=0
    for _ in range(n):
        A=[random.random()<ci for ci in c]; B=A[:]
        for p in range(k):
            if p+W<k and (not A[p]) and all(A[p+1:p+W+1]) and random.random()<g:
                B[p]=True
        al=1
        for b in B:
            if b: al+=1
            else: break
        tot+=al
    return tot/n
```

Caveats: positions are treated as independent (they are not - a rejection at position 0 changes the
distribution of what follows), and `g` is correlated with rejection (a rejected position is more likely to be
high entropy), so the true operating column is nearer g=1. Treat the table as an ordering, not a forecast.

## What to run

`fly_window_size` must be < `num_speculative_tokens`, so for k=5 the range is 1-4. Arms, one boot each,
40 min:

1. W=2, `fly_entropy_threshold` 0.3 - the expected-value pick.
2. W=1, threshold 0.3 - the maximum-acceptance arm.
3. W=1, threshold 0.5 - the quality trade.
4. W=4, threshold 0.3 - the control that tests the window model.

Keep `fly_entropy_top_k` at 3. Quality gates for every arm: GSM8K-100 with thinking (baseline 96% on r21 k5
FP8), the 9/9 smoke, and a greedy-agreement or KL check against the served baseline. FLy is lossy by
construction (vLLM warns "may degrade model outputs"), and the smaller the window the looser the rule: at W=1
a mismatch is accepted whenever the *next* token was accepted and entropy is above the threshold.
