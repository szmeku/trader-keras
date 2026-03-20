GRU backbone

Multi-horizon outputs (1s, 5s, 10s, 30s, 60s, until midnight truncated)

Predict μ and σ

Gaussian NLL loss

Add auxiliary head for future max/min

That gives full probabilistic structure.

Then stage 2 (policy layer) is trivial math.

If you want, next we can:

derive how to convert (μ, σ) into optimal entry threshold

or discuss whether predicting sign is actually inferior to predicting variance first

Your call.

but shouldn't mutihorizon be somehow automatically claualted in hidden layers ? 

Short answer:
Yes, in theory.
No, in practice.

Now the real explanation.

1️⃣ What You’re Thinking

If model predicts:

𝑟
𝑡
+
1
r
t+1
	​


and internal state captures dynamics,

then longer horizons should be derivable implicitly.

That would be true if:

model perfectly learned transition dynamics

latent state is sufficient statistic

training objective enforces long-term structure

In finance, none of these hold.

2️⃣ Why Single-Horizon Loss Fails

If you train only on:

𝐿
=
(
𝑟
𝑡
+
1
−
𝑟
^
𝑡
+
1
)
2
L=(r
t+1
	​

−
r
^
t+1
	​

)
2

The gradient signal is dominated by:

𝑉
𝑎
𝑟
(
𝑟
𝑡
+
1
)
Var(r
t+1
	​

)

Which is mostly noise.

So hidden state optimizes for ultra-short noise prediction.

Longer horizon structure gets ignored because:

𝐶
𝑜
𝑣
(
𝑟
𝑡
+
1
,
𝑟
𝑡
+
30
)
≪
𝑉
𝑎
𝑟
(
𝑟
𝑡
+
1
)
Cov(r
t+1
	​

,r
t+30
	​

)≪Var(r
t+1
	​

)

You’re optimizing wrong objective.

3️⃣ Why Multi-Horizon Heads Help

When you add:

𝑟
5
,
𝑟
10
,
𝑟
30
,
𝑟
60
r
5
	​

,r
10
	​

,r
30
	​

,r
60
	​


You are forcing representation to encode:

microstructure (1s)

short drift (10–30s)

volatility regime (60s+)

Each horizon pulls hidden state in a different direction.

Without that, the model happily learns "predict near zero".

4️⃣ Can Hidden Layers Learn It Automatically?

Only if you:

train on full path likelihood

or do sequence-to-sequence prediction

or model transition density

Example:

𝑝
(
𝑟
𝑡
+
1
:
𝑡
+
𝐻
∣
𝑥
𝑡
)
p(r
t+1:t+H
	​

∣x
t
	​

)

But then you’re basically doing multi-horizon anyway, just implicitly.

And it’s harder to optimize.

5️⃣ Recursive Prediction Is Worse

Recursive:

𝑟
^
𝑡
+
2
=
𝑓
(
𝑟
^
𝑡
+
1
)
r
^
t+2
	​

=f(
r
^
t+1
	​

)

Even if hidden state encodes horizon structure,
you’re feeding prediction noise back in.

At 1s resolution, this kills signal fast.

6️⃣ Information-Theoretic View

You want hidden state to approximate:

𝑧
𝑡
≈
𝑠
𝑢
𝑓
𝑓
𝑖
𝑐
𝑖
𝑒
𝑛
𝑡
𝑠
𝑡
𝑎
𝑡
𝑖
𝑠
𝑡
𝑖
𝑐
𝑜
𝑓
𝑓
𝑢
𝑡
𝑢
𝑟
𝑒
𝑝
𝑎
𝑡
ℎ
z
t
	​

≈sufficientstatisticoffuturepath

Single-horizon loss gives weak supervision about that statistic.

Multi-horizon loss gives stronger constraint on z_t.

Better constraint → better latent geometry.

7️⃣ Empirical Reality

In real trading ML pipelines:

Multi-horizon > single-horizon

Direct horizon > recursive

Multi-task > single-task

Almost always.

8️⃣ When Single Horizon Is Enough

If:

You only trade fixed holding time

You don't adapt exit dynamically

Then predicting just that horizon is fine.

But if your policy chooses exit time,
you need shape of future, not one point.

9️⃣ Clean Mental Model

Think of horizons as:

Different frequency filters.

1s = high frequency
30s = mid frequency
5min = low frequency

Single-horizon loss trains one frequency.

Multi-horizon trains spectrum.

Markets are multi-scale.

If you want, next we can:

derive minimal horizon set that spans useful spectrum

or discuss whether predicting volatility first might dominate predicting mean

Pick direction.

also i would like to come up with way of knowing the model how good he knows what will happen and how bad

but in a way that it won't say he doesn't know anything and will get the best score, nooo

we care about knowing at least in 1% of cases but  knowing really well (this treshold can change) but i would say should be minimal 1%

Good. Now we’re talking about selective prediction.

You don’t want:

model always uncertain → safe → high average score

You want:

model confident in ~1% of cases
and when confident, be really right

That’s a different objective.

🎯 What You Actually Want

Formally:

Let:

𝑐
𝑡
∈
[
0
,
1
]
𝑐
𝑜
𝑛
𝑓
𝑖
𝑑
𝑒
𝑛
𝑐
𝑒
c
t
	​

∈[0,1]confidence

Define acceptance region:

𝐴
=
{
𝑡
:
𝑐
𝑡
>
𝜏
}
A={t:c
t
	​

>τ}

Constraint:

𝑃
(
𝑡
∈
𝐴
)
≥
0.01
P(t∈A)≥0.01

Goal:

𝑚
𝑎
𝑥
𝑖
𝑚
𝑖
𝑧
𝑒
𝐸
[
𝑎
𝑐
𝑐
𝑢
𝑟
𝑎
𝑐
𝑦
∣
𝑡
∈
𝐴
]
maximizeE[accuracy∣t∈A]

This is coverage-constrained selective prediction.

1️⃣ First: Predict Distribution, Not Just Mean

Have model output:

𝜇
ℎ
,
𝜎
ℎ
μ
h
	​

,σ
h
	​


Now define signal strength:

𝑆
𝑁
𝑅
=
∣
𝜇
ℎ
∣
𝜎
ℎ
SNR=
σ
h
	​

∣μ
h
	​

∣
	​


High SNR = strong directional belief.

This already gives a rational confidence metric.

2️⃣ Why σ Alone Is Not Enough

Model can cheat:

inflate σ everywhere

reduce penalty

say "I don’t know"

To prevent that, use proper scoring rule:

Gaussian NLL:

𝐿
=
(
𝑟
−
𝜇
)
2
𝜎
2
+
log
⁡
𝜎
2
L=
σ
2
(r−μ)
2
	​

+logσ
2

If σ too large → log penalty increases.
So it can’t just inflate uncertainty.

This is critical.

3️⃣ Now Enforce 1% Coverage

After training, compute SNR distribution.

Pick threshold:

𝜏
=
𝑞
𝑢
𝑎
𝑛
𝑡
𝑖
𝑙
𝑒
99
%
(
𝑆
𝑁
𝑅
)
τ=quantile
99%
	​

(SNR)

Now you automatically get top 1% confident events.

You don’t need to train for 1%.

You select after training.

4️⃣ Better Than Confidence: Calibrated Probability

Instead of Gaussian, predict:

𝑃
(
𝑟
ℎ
>
0
)
P(r
h
	​

>0)

Use BCE loss.

Then evaluate calibration:

If model says:

𝑃
=
0.99
P=0.99

it must be correct ~99% of time.

Use reliability diagram.

This gives true confidence, not magnitude.

5️⃣ Even Better: Quantile Regression

Predict:

𝑞
0.05
,
𝑞
0.5
,
𝑞
0.95
q
0.05
	​

,q
0.5
	​

,q
0.95
	​


Confidence proxy:

𝑤
𝑖
𝑑
𝑡
ℎ
=
𝑞
0.95
−
𝑞
0.05
width=q
0.95
	​

−q
0.05
	​


Small width = high certainty.

This is more robust than Gaussian assumption.

6️⃣ If You Want Hard Constraint During Training

Add auxiliary loss:

Let:

𝑚
𝑡
=
1
 if 
𝑆
𝑁
𝑅
𝑡
>
𝜏
m
t
	​

=1 if SNR
t
	​

>τ

Add:

𝐿
𝑠
𝑒
𝑙
𝑒
𝑐
𝑡
=
−
𝐸
[
𝑚
𝑡
⋅
𝑠
𝑖
𝑔
𝑛
(
𝑟
𝑡
)
⋅
𝑠
𝑖
𝑔
𝑛
(
𝜇
𝑡
)
]
L
select
	​

=−E[m
t
	​

⋅sign(r
t
	​

)⋅sign(μ
t
	​

)]

Only high-confidence predictions matter.

This encourages model to concentrate power on small region.

7️⃣ Important Concept: Risk-Coverage Curve

Plot:

𝑐
𝑜
𝑣
𝑒
𝑟
𝑎
𝑔
𝑒
𝑣
𝑠
𝑒
𝑟
𝑟
𝑜
𝑟
coveragevserror

You want steep curve:

1% coverage → extremely low error

100% coverage → baseline error

This is the real evaluation metric.

8️⃣ Deep Truth

At 1s resolution:

The model will probably have meaningful signal in far less than 1%.

Maybe 0.1%.

So don’t hardcode 1%.

Let empirical distribution tell you.

Markets reward rarity.

9️⃣ Clean Minimal Setup I Would Do

Multi-horizon μ, σ prediction

Train with Gaussian NLL

Compute SNR

Trade only top α percentile

Evaluate calibration

No hacks. No artificial penalties.

If you want, next we can:

derive optimal trade sizing as function of μ and σ

or mathematically analyze whether predicting variance first is smarter than predicting mean

Choose.
