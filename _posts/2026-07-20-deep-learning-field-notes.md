---
title: "Deep Learning Field Notes: Losses, Layers, Training, and Deployment"
date: 2026-07-20 10:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [deep-learning, backpropagation, optimizers, focal-loss, cross-entropy, cnn, pooling, batch-normalization, dropout, model-compression, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/deep-learning-field-notes/
description: "A connected guide to deep-learning losses, backpropagation, optimizers, convolutional layers, regularization, and deployment optimization."
---

Deep learning is often taught as a catalog of layers and formulas. I find it more useful to ask what problem each idea solves.

- Cross-entropy tells a classifier how wrong its probabilities are.
- Focal loss stops easy examples from dominating an imbalanced training set.
- Pooling trades exact position for compact, locally robust features.
- Batch normalization and dropout change how a network trains and generalizes.
- Contrastive loss learns a geometry, not just a class label.

This post turns my original short notes into one connected guide. It is not a complete deep-learning textbook. It is a map of important concepts and the reasoning that connects them.

## Topic map

- [Cross-entropy and KL divergence](#cross-entropy-kl)
- [Focal loss](#focal-loss)
- [Multilayer perceptrons and activations](#mlp-activations)
- [Backpropagation](#backpropagation)
- [RMSProp and Adam](#optimizers)
- [Pooling in convolutional networks](#pooling)
- [Transposed convolution](#transposed-convolution)
- [Batch normalization](#batch-normalization)
- [Dropout](#dropout)
- [Inference optimization](#inference-optimization)

---

## 1. Cross-entropy and KL divergence {#cross-entropy-kl}

Suppose the true distribution over classes is $P$ and the model predicts $Q$. Their cross-entropy is

$$
H(P,Q)=-\sum_x P(x)\log Q(x).
$$

It measures the average surprise we experience when outcomes come from $P$, but we encode or predict them using $Q$.

Start with Shannon entropy, which measures the uncertainty inside one distribution:

$$
H(P)=-\sum_xP(x)\log P(x).
$$

If logarithms use base $2$, entropy is measured in bits. For $M$ equally likely outcomes,

$$
H(P)
=-M\left(\frac1M\log_2\frac1M\right)
=\log_2M.
$$

A fair coin therefore has $1$ bit of entropy, while a deterministic outcome has $0$. The coding interpretation is that $H(P)$ is the best achievable average description length for outcomes from $P$, whereas $H(P,Q)$ is the average length paid when the code is designed for the wrong distribution $Q$.

The KL divergence from $P$ to $Q$ is

$$
D_{\mathrm{KL}}(P\|Q)
=\sum_x P(x)\log\frac{P(x)}{Q(x)}.
$$

Expanding the logarithm gives

$$
\begin{aligned}
D_{\mathrm{KL}}(P\|Q)
&=\sum_x P(x)\log P(x)-\sum_xP(x)\log Q(x)\\
&=-H(P)+H(P,Q).
\end{aligned}
$$

Therefore,

$$
H(P,Q)=H(P)+D_{\mathrm{KL}}(P\|Q).
$$

During supervised training, the target distribution $P$ is fixed. Its entropy $H(P)$ does not depend on the model parameters. Minimizing cross-entropy is consequently equivalent to minimizing

$$
D_{\mathrm{KL}}(P\|Q).
$$

That is the conceptual link: **cross-entropy training pushes the predicted distribution toward the data distribution.**

### What do “true” and “predicted” distributions mean?

This language can sound more abstract than the actual training setup. Consider an image classifier with three classes: cat, dog, and rabbit.

For one training image labeled “cat,” the target is usually represented as the one-hot vector

$$
P=(1,0,0).
$$

The model's softmax output might be

$$
Q=(0.7,0.2,0.1).
$$

Here $P$ is the target distribution for this labeled example, while $Q$ is the model's predicted distribution. Cross-entropy is

$$
\begin{aligned}
H(P,Q)
&=-\left(1\log0.7+0\log0.2+0\log0.1\right)\\
&=-\log0.7\\
&\approx0.357.
\end{aligned}
$$

Only the probability assigned to the correct class contributes. This is why categorical cross-entropy is often written as the negative log-probability of the correct class.

There is a useful nuance in the phrase “true distribution.” The one-hot vector is the observed target for one sample; it is not necessarily the full, unknown distribution that generated all possible data. Label smoothing makes this distinction visible by replacing a perfectly one-hot target with a slightly softer training target.

### Why does cross-entropy equal KL divergence for a one-hot target?

The entropy of the one-hot target is

$$
H(P)=-\left(1\log1+0\log0+0\log0\right)=0,
$$

using the standard convention $0\log0=0$. There is no uncertainty in which class the label selects. Therefore,

$$
D_{\mathrm{KL}}(P\|Q)
=H(P,Q)-H(P)
=H(P,Q).
$$

For the cat example, both quantities are $-\log0.7\approx0.357$. More generally, if the targets are soft rather than one-hot, $H(P)$ need not be zero. Cross-entropy and KL divergence then differ by $H(P)$, but because $P$ is fixed with respect to the model parameters, minimizing one still minimizes the other.

### Binary cross-entropy

For a binary target $y\in\{0,1\}$ and predicted probability $p=P(y=1)$,

$$
\operatorname{BCE}(y,p)
=-\left[y\log p+(1-y)\log(1-p)\right].
$$

It is convenient to define $p_t$ as the probability assigned to the correct class:

$$
p_t=
\begin{cases}
p,&y=1,\\
1-p,&y=0.
\end{cases}
$$

Then binary cross-entropy becomes simply

$$
\operatorname{BCE}(p_t)=-\log p_t.
$$

As $p_t$ approaches $1$, the loss approaches $0$. A confidently wrong prediction has a small $p_t$ and receives a large penalty.

---

## 2. Focal loss: make hard examples matter {#focal-loss}

Cross-entropy works well when the examples contributing to training are reasonably balanced. In dense object detection and other highly imbalanced problems, however, a model can encounter an enormous number of easy negatives. Their individual losses may be small, but together they can dominate the gradient.

Focal loss adds a modulating factor:

$$
\operatorname{FL}(p_t)=-(1-p_t)^\gamma\log p_t,
$$

or equivalently,

$$
\operatorname{FL}(p_t)=(1-p_t)^\gamma\operatorname{BCE}(p_t),
$$

where $\gamma\ge 0$ is the focusing parameter.

- If $\gamma=0$, focal loss is exactly cross-entropy.
- If an example is easy and $p_t$ is close to $1$, $(1-p_t)^\gamma$ is tiny.
- If an example is hard and $p_t$ is small, the multiplier remains comparatively large.

The loss does not make a hard example's absolute penalty larger than ordinary cross-entropy. It **down-weights easy examples much more aggressively**, so hard examples matter more relative to them.

### A numerical example

Let $\gamma=2$ and compare a fairly confident correct prediction with a hard prediction.

| Example | $p_t$ | BCE $=-\ln(p_t)$ | Focal weight $(1-p_t)^2$ | Focal loss |
|---|---:|---:|---:|---:|
| Easy | $0.7$ | $0.357$ | $0.09$ | $0.032$ |
| Hard | $0.3$ | $1.204$ | $0.49$ | $0.590$ |

Under cross-entropy, the ratio of hard-example loss to easy-example loss is

$$
R_{\mathrm{BCE}}=\frac{1.204}{0.357}\approx3.37.
$$

Under focal loss, it becomes

$$
R_{\mathrm{FL}}
=\frac{0.49\times1.204}{0.09\times0.357}
\approx18.36
\approx5.44R_{\mathrm{BCE}}.
$$

So the hard example is about $5.44$ times more prominent *relative to the easy example* than it was under BCE. This is exactly the behavior needed when a sea of easy cases would otherwise drown out the useful learning signal.

An optional class-balancing factor $\alpha_t$ is also common:

$$
\operatorname{FL}(p_t)=-\alpha_t(1-p_t)^\gamma\log p_t.
$$

Here $\alpha_t$ addresses class imbalance, while $(1-p_t)^\gamma$ addresses difficulty imbalance. They solve related but distinct problems.

---

## 3. Multilayer perceptrons and activation functions {#mlp-activations}

A multilayer perceptron (MLP) repeatedly applies an affine transformation followed by a nonlinear activation:

$$
h^{(\ell)}=\phi\!\left(W^{(\ell)}h^{(\ell-1)}+b^{(\ell)}\right).
$$

The nonlinearity is essential. If every layer were linear, then

$$
W_3(W_2(W_1x))=(W_3W_2W_1)x,
$$

and the entire deep network would collapse into a single linear transformation. Depth becomes expressive because nonlinearities prevent this collapse.

### Why use ReLU in hidden layers instead of sigmoid?

The sigmoid function

$$
\sigma(z)=\frac{1}{1+e^{-z}}
$$

is smooth and maps values to $(0,1)$, making it a natural output activation for binary probability prediction. Its derivative is

$$
\sigma'(z)=\sigma(z)(1-\sigma(z)).
$$

For large positive or negative $z$, this derivative approaches zero. A deep stack of sigmoid units can therefore multiply gradients by many small values during backpropagation, producing vanishing gradients. Sigmoid activations are also not zero-centered.

The rectified linear unit is

$$
\operatorname{ReLU}(z)=\max(0,z).
$$

On its positive side, its derivative is $1$. That simple gradient path usually makes hidden layers easier to optimize, and its exact zeros create sparse activations.

ReLU is not perfect: a unit that stays on the negative side can receive zero gradient and “die.” Leaky ReLU, GELU, and SiLU are useful alternatives. The practical rule is not “sigmoid is bad”; it is:

> Use an activation whose behavior matches the layer's job. Sigmoid is useful for a binary output probability, while ReLU-like functions are usually easier to train in hidden layers.

---

## 4. Backpropagation: follow the chain rule backward {#backpropagation}

A forward pass computes a prediction and a scalar loss. Backpropagation computes how sensitive that loss is to every trainable parameter. It is an efficient application of the chain rule, not a separate learning rule.

For one layer,

$$
z^{(\ell)}=W^{(\ell)}a^{(\ell-1)}+b^{(\ell)},
\qquad
a^{(\ell)}=\phi(z^{(\ell)}).
$$

Let $L$ be the final loss and define the layer's error signal as

$$
\delta^{(\ell)}=\frac{\partial L}{\partial z^{(\ell)}}.
$$

The output error is obtained by differentiating the loss through the output activation. Hidden errors then move backward:

$$
\delta^{(\ell)}
=\left(W^{(\ell+1)T}\delta^{(\ell+1)}\right)
\odot\phi'(z^{(\ell)}).
$$

Once $\delta^{(\ell)}$ is known, the parameter gradients are

$$
\frac{\partial L}{\partial W^{(\ell)}}
=\delta^{(\ell)}a^{(\ell-1)T},
\qquad
\frac{\partial L}{\partial b^{(\ell)}}=\delta^{(\ell)}.
$$

The important computational idea is reuse. A naive derivative calculation would repeatedly recompute the same intermediate terms for different parameters. Backpropagation stores the forward values, computes each local derivative once, and reuses the downstream gradient while traversing the graph in reverse. This is reverse-mode automatic differentiation.

Backpropagation only computes gradients. An optimizer decides how to use them. This distinction is easy to blur: backprop says *which direction changes the loss*; SGD, RMSProp, or Adam says *how far and with what accumulated state to move*.

---

## 5. RMSProp and Adam: turn gradients into updates {#optimizers}

Plain stochastic gradient descent applies

$$
\theta_t=\theta_{t-1}-\eta g_t,
\qquad
g_t=\nabla_\theta L_t.
$$

A single learning rate $\eta$ can be awkward when different parameters see gradients of very different scales.

### RMSProp

#### Motivation

SGD multiplies every coordinate of the gradient by the same learning rate. On an anisotropic loss surface, one direction may be steep while another is flat. The steep direction produces large gradients and can make SGD overshoot or oscillate across a narrow valley. Reducing the global learning rate controls that oscillation, but then movement along the flat direction becomes painfully slow.

RMSProp addresses this mismatch by giving each parameter an adaptive step size. It tracks the recent scale of that parameter's gradients, then divides the current gradient by that scale:

- a coordinate with consistently large gradients receives a smaller effective step;
- a coordinate with consistently small gradients receives a relatively larger step.

This normalization lets optimization move cautiously in steep directions and more quickly in flat directions instead of forcing one global learning rate to handle both.

#### Moving average and update

RMSProp keeps an exponential moving average of squared gradients:

$$
v_t=\beta v_{t-1}+(1-\beta)g_t^2,
$$

then scales each coordinate of the update:

$$
\theta_t
=\theta_{t-1}
-\eta\frac{g_t}{\sqrt{v_t}+\varepsilon}.
$$

Here $\beta\in[0,1)$ controls how much history is retained. A value near $1$ produces a smoother but slower-changing estimate; a smaller value reacts more quickly to recent gradients. Squaring prevents positive and negative gradients from cancelling and makes $\sqrt{v_t}$ an estimate of their recent root-mean-square magnitude. The small $\varepsilon$ prevents division by zero and improves numerical stability.

Unlike AdaGrad's ever-growing sum of squared gradients, the exponential moving average can forget ancient gradients. Its effective learning rates therefore do not have to shrink forever, which is useful when the relevant gradient scale changes during training.

### Adam

Adam combines an exponential average of gradients with the RMSProp-style average of squared gradients. Its $\beta_2$ plays the same role as RMSProp's $\beta$:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

The two states solve different problems:

- $m_t$ is a momentum-like estimate of the gradient's direction. It damps rapid sign changes and carries consistent directions forward.
- $v_t$ estimates coordinate-wise gradient scale. Dividing by $\sqrt{v_t}$ makes the step smaller where gradients have recently been large.

Thus $\beta_1$ controls directional memory and $\beta_2$ controls scale memory. Common defaults, $\beta_1=0.9$ and $\beta_2=0.999$, make the second-moment estimate much smoother than the first, but they are defaults rather than laws. A rapidly changing objective may benefit from shorter memory.

Because both averages begin at zero, early values are biased toward zero. Adam corrects them:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

To see the bias, suppose the expected gradient is approximately constant at $g$. Starting from $m_0=0$ gives

$$
\mathbb E[m_t]=(1-\beta_1^t)g.
$$

The missing factor is exactly $1-\beta_1^t$, so dividing by it recovers an unbiased estimate under this simplified stationary picture. The same geometric-series argument applies to $v_t$.

and updates

$$
\theta_t
=\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

The $\varepsilon$ term prevents division by zero, but it can also affect effective step sizes when $v_t$ is tiny. Adam is a strong default, especially for sparse or noisy gradients, but it is not automatically the best-generalizing optimizer. SGD with momentum can still be preferable.

Ordinary L2 regularization adds $\lambda\theta$ to the gradient; Adam then rescales that term coordinate by coordinate. AdamW instead applies weight decay directly,

$$
\theta_t
\leftarrow
(1-\eta\lambda)\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon},
$$

which separates the intended parameter shrinkage from Adam's adaptive gradient normalization.

---

## 6. Pooling in convolutional networks {#pooling}

Early convolutional layers detect local patterns such as edges and corners. Deeper layers combine them into textures, parts, shapes, and objects. But a feature map also records *where* each response occurs. A small translation, crop, or deformation can move a strong activation to a neighboring position.

Pooling summarizes a local neighborhood and downsamples the spatial dimensions. The common forms are:

- **Max pooling:** keep the strongest response in each window.
- **Average pooling:** keep the mean response.
- **Global average pooling:** average every spatial position in each channel, often before a classifier.

For a $2\times2$ max-pooling window,

$$
\begin{bmatrix}
1&5\\
2&3
\end{bmatrix}
\longmapsto 5.
$$

A typical block is

$$
\text{input}\rightarrow\text{convolution}\rightarrow\text{nonlinearity}\rightarrow\text{pooling}.
$$

Pooling provides three useful effects:

1. It reduces computation and memory in later layers.
2. It expands the effective receptive field more quickly.
3. It makes the representation less sensitive to small local shifts.

The third statement needs care. Pooling does not create complete translation invariance. It creates limited local robustness while discarding precise spatial information. That trade-off is harmful in tasks such as segmentation or keypoint detection, where exact location matters. Strided convolutions are therefore often used instead, letting the network learn its own downsampling operation.

---

## 7. Transposed convolution: learned upsampling {#transposed-convolution}

A normal convolution often maps a large spatial grid to a smaller one. A transposed convolution performs the corresponding matrix operation in the opposite direction.

If a convolution is represented after flattening as

$$
y=Kx,
$$

then its transposed convolution computes

$$
z=K^Ty.
$$

This explains the name. It is the transpose of the linear operator used by the convolution; it is not generally the inverse of that convolution and does not recover information that downsampling destroyed.

### The spatial construction

A useful way to see the operation is to let every input value scale the entire kernel, place that scaled kernel at the input value's spatial offset, and add the overlapping contributions.

![A two-by-two input expanded by a two-by-two transposed-convolution kernel into a three-by-three output.](/assets/images/deep-learning-field-notes/transposed-convolution-overlap.png)

The Python implementation from the notes makes that scatter-and-add construction explicit:

```python
import torch


# This minimal example assumes stride s = 1 and padding p = 0.
def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i : i + h, j : j + w] += X[i, j] * K

    return Y
```

### Why the output-size equation looks reversed

For a standard convolution, ignoring the floor operation and using dilation $d=1$, the one-dimensional size equation is

$$
o=\frac{i+2p-k}{s}+1.
$$

Solving the same equation for its input size gives

$$
i=(o-1)s+k-2p.
$$

The transposed operation exchanges the roles of the ordinary convolution's input and output. After relabeling them, its output size is therefore

$$
o_{\mathrm{transposed}}
=(i_{\mathrm{transposed}}-1)s+k-2p.
$$

So the second handwritten image in the notes was not describing an unrelated rule: it rearranged the normal-convolution equation and then exchanged input and output. Including dilation and output padding gives the general form

$$
H_{\text{out}}
=(H_{\text{in}}-1)s-2p+d(k-1)+o_p+1,
$$

where $s$ is stride, $p$ is padding, $d$ is dilation, $k$ is kernel size, and $o_p$ is output padding.

Transposed convolutions are useful in decoders, segmentation networks, and generators because they learn how to increase spatial resolution.

Uneven kernel overlap can cause checkerboard artifacts. A common alternative is deterministic upsampling—nearest-neighbor or bilinear interpolation—followed by a standard convolution.

---

## 8. Batch normalization: stabilize the optimization path {#batch-normalization}

For one feature channel in a mini-batch, batch normalization computes

$$
\mu_B=\frac1m\sum_{i=1}^m x_i,
\qquad
\sigma_B^2=\frac1m\sum_{i=1}^m(x_i-\mu_B)^2,
$$

then normalizes and restores learnable scale and shift:

$$
\hat{x}_i=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\varepsilon}},
\qquad
y_i=\gamma\hat{x}_i+\beta.
$$

The parameters $\gamma$ and $\beta$ mean the layer is not forced to keep zero mean and unit variance. It can learn the scale and offset that the next layer needs.

### Which axes are normalized?

For a convolutional tensor with shape $(N,C,H,W)$, `BatchNorm2d` computes one mean and variance for each channel $c$ using all values across the batch and spatial axes $(N,H,W)$. It does **not** combine different channels into one statistic.

That distinction matters because channels usually represent different learned features—perhaps edges, textures, or colors. Mixing them would force unrelated feature maps to share a scale and would entangle their semantics. Per-channel normalization preserves that separation, while the learned $\gamma_c$ and $\beta_c$ let every channel recover the scale and offset it needs.

For a dense layer shaped $(N,D)$, batch normalization instead computes one pair of statistics for every feature dimension across $N$. The principle is the same: normalize repeated observations of the same feature, not different features together.

In practice, batch normalization often:

- permits larger learning rates;
- makes optimization less sensitive to initialization;
- keeps activation and gradient scales better behaved;
- introduces a small regularizing effect through noisy batch statistics.

Training and inference behave differently. Training uses the current mini-batch statistics. Inference uses running estimates accumulated during training. Forgetting to switch the model to evaluation mode can therefore change predictions.

The roles of the stored quantities are different:

- $\gamma$ and $\beta$ are trainable parameters updated by backpropagation.
- `running_mean` and `running_var` are buffers updated from batch statistics; they are saved with the model but are not learned through gradients.
- the current batch mean and variance are temporary values used for that training pass.

The distinction is visible in PyTorch-style code. During training, statistics come from the current batch and the running buffers are updated:

```python
import torch


# x has shape (N, C, H, W); c selects one channel.
mean_c = x[:, c, :, :].mean()
var_c = x[:, c, :, :].var(unbiased=False)

x_hat[:, c, :, :] = (
    x[:, c, :, :] - mean_c
) / torch.sqrt(var_c + eps)
y[:, c, :, :] = gamma[c] * x_hat[:, c, :, :] + beta[c]

# Here decay is the weight retained from the previous running estimate.
running_mean[c] = decay * running_mean[c] + (1 - decay) * mean_c
running_var[c] = decay * running_var[c] + (1 - decay) * var_c
```

During inference, the frozen running statistics replace the current batch statistics:

```python
x_hat[:, c, :, :] = (
    x[:, c, :, :] - running_mean[c]
) / torch.sqrt(running_var[c] + eps)
y[:, c, :, :] = gamma[c] * x_hat[:, c, :, :] + beta[c]
```

Frameworks differ in how they name the running-statistics update coefficient, so it is worth checking whether their `momentum` means the weight on the old estimate or on the new batch statistic.

Small or non-independent batches can make the estimates noisy. Layer normalization uses statistics within each example, while group normalization divides channels into groups and is independent of batch size. They are useful alternatives when reliable batch statistics are unavailable.

---

## 9. How dropout really works {#dropout}

The familiar description—“randomly turn off neurons to reduce overfitting”—is correct, but incomplete. Dropout must also prevent a systematic scale mismatch between training and inference.

### Step 1: train a randomly thinned network

During each training pass, dropout samples an independent binary mask for the activations. If $p$ is the probability of dropping an activation, then the keep probability is $q=1-p$:

$$
m_i\sim\operatorname{Bernoulli}(q).
$$

The masked activation would be $m_i h_i$. Different masks create different thinned subnetworks, but all of them share the same parameters. This discourages neurons from co-adapting—for example, a feature cannot assume that one particular companion feature will always be present.

There is a numerical problem, however. Suppose 100 incoming activations and their weights are all $1$. Without dropout, their sum is $100$. With $p=0.4$, only about 60 inputs survive during training, so the sum is roughly $60$. If inference uses the complete network, its input jumps back to $100$.

![Without scaling, dropout makes the expected training-time input smaller than the inference-time input.](/assets/images/deep-learning-field-notes/dropout-training-vs-inference.png)

*With a 40% drop rate, the training-time sum is roughly 60 while the full inference network produces 100.*

### Step 2: preserve the expected activation

Modern libraries normally use **inverted dropout**: each retained activation is divided by the keep probability during training,

$$
\tilde h_i
=\frac{m_i}{q}h_i
=\frac{m_i}{1-p}h_i.
$$

Because $\mathbb E[m_i]=q$, this preserves the activation in expectation:

$$
\mathbb E[\tilde h_i]
=\frac{\mathbb E[m_i]}{q}h_i
=h_i.
$$

In the example, the surviving sum is rescaled as

$$
60\times\frac{1}{1-0.4}=100.
$$

![Inverted dropout rescales the surviving training-time input from 60 to 100.](/assets/images/deep-learning-field-notes/dropout-inverted-scaling-example.png)

*Scaling the retained activations by $1/(1-p)$ aligns the expected training-time scale with inference.*

The actual number of retained units varies from mask to mask; 60 is the expected count in this simplified example. The equality is therefore about expected activation, not a promise that every training pass produces exactly the inference-time value.

### What happens at inference?

At inference time, dropout is disabled. Because the adjustment already happened during training, the complete network is used with no further scaling. This is why frameworks such as PyTorch change behavior between training and evaluation modes.

An older, equivalent convention leaves retained activations unscaled during training and multiplies them by $q$ at inference. Inverted dropout is generally preferred because inference stays simple and deterministic.

Conceptually, the full inference network approximates averaging many related subnetworks. It is an approximation rather than an exact enumeration, but it explains why dropout combines noise injection, reduced co-adaptation, and ensemble-like regularization.

Dropout is a regularizer, not a universal requirement. Heavy dropout can cause underfitting, and modern convolutional architectures often rely more on data augmentation, weight decay, normalization, and stochastic depth. It is still common and effective in MLP heads and transformer blocks.

---

## 10. Inference optimization: make the trained model practical {#inference-optimization}

Training minimizes a learning objective; deployment must also satisfy latency, memory, throughput, energy, and cost constraints. The main compression tools trade some redundancy or precision for efficiency.

### Pruning

Pruning removes weights, channels, heads, or entire blocks judged to be unimportant. Unstructured weight sparsity can reduce file size but may not improve wall-clock latency without sparse hardware and kernels. Structured pruning is easier for ordinary accelerators to exploit because it removes complete computational units.

Common importance signals include weight magnitude, activation statistics, gradient sensitivity, and the change in loss after removing a unit. Magnitude pruning is simple: remove weights with the smallest absolute values. More expensive sensitivity methods try to preserve weights whose removal would damage the objective most.

Pruning can happen at several points:

1. **During training:** gradually increase sparsity so the model adapts while learning.
2. **After training:** prune a converged model, then fine-tune to recover accuracy.
3. **Before or at initialization:** search for a sparse subnetwork that can be trained directly.

A practical iterative workflow is train, prune a modest fraction, fine-tune, evaluate, and repeat. One-shot aggressive pruning is faster but usually causes a larger quality drop. The target should be a measured latency or memory budget, not sparsity for its own sake.

The lottery-ticket hypothesis suggests that a large randomly initialized network may contain smaller subnetworks capable of training to comparable quality. It is an insight about overparameterization, not a guarantee that every pruning rule discovers such a subnetwork.

### Quantization

Quantization represents weights and sometimes activations with fewer bits—for example FP32 to FP16, BF16, INT8, or lower. This reduces memory bandwidth and can accelerate matrix operations on compatible hardware. Post-training quantization is cheap; quantization-aware training usually preserves quality better at aggressive precision.

Uniform affine quantization maps a real value $x$ to an integer $q$ using a scale $s$ and zero point $z$:

$$
q=\operatorname{clip}\!\left(\operatorname{round}(x/s)+z,
q_{\min},q_{\max}\right),
$$

with approximate reconstruction $\hat x=s(q-z)$. The range chosen for $s$ matters. A range that is too wide wastes integer levels; a range that is too narrow clips important outliers.

- **Dynamic quantization** quantizes weights ahead of time and activations at runtime. It is easy to apply and often useful for linear or recurrent layers.
- **Static post-training quantization (PTQ)** uses a representative calibration set to estimate activation ranges before deployment.
- **Quantization-aware training (QAT)** inserts fake quantization during training so weights adapt to rounding and clipping. It costs more but usually protects accuracy at INT8 or below.

Per-channel weight scales often preserve quality better than one scale for an entire tensor. Whether any scheme is faster depends on the target accelerator, kernel library, supported operators, and cost of inserting conversions around unsupported layers.

### Knowledge distillation and low-rank factorization

Distillation trains a smaller student to match a larger teacher's softened outputs or internal representations. With temperature $T$, the teacher and student class distributions are

$$
p_T=\operatorname{softmax}(z_{\mathrm{teacher}}/T),
\qquad
q_T=\operatorname{softmax}(z_{\mathrm{student}}/T).
$$

A common objective combines ordinary supervised loss with a soft-target loss:

$$
L
=\alpha L_{\mathrm{hard}}
+(1-\alpha)T^2D_{\mathrm{KL}}(p_T\|q_T).
$$

A higher temperature exposes relative preferences among incorrect classes—the teacher's “dark knowledge.” The factor $T^2$ compensates for the gradient scaling introduced by temperature. Distillation can also align intermediate features, attention maps, or pairwise relations, but the student architecture still needs enough capacity to represent what the teacher knows.

Low-rank factorization replaces a large matrix $W\in\mathbb R^{m\times n}$ with $UV^T$, where $U\in\mathbb R^{m\times r}$ and $V\in\mathbb R^{n\times r}$. Parameter count falls from $mn$ to $r(m+n)$ when $r\ll\min(m,n)$. The rank can be chosen from singular-value energy and then fine-tuned. Theoretical savings will not become latency savings if the two smaller operations introduce unfavorable memory traffic or kernel overhead.

### Validate on the deployment stack

Compression only counts as an optimization when measured on the target stack. Parameter count, theoretical FLOPs, model size, and observed latency are related but not interchangeable. Benchmark at realistic batch sizes and sequence or image shapes, including preprocessing, data transfer, warm-up, and memory use.

Also re-evaluate task quality after compression. Aggregate accuracy can conceal damage to rare classes, calibration, robustness, or important subgroups. A sensible comparison records at least model size, peak memory, median and tail latency, throughput, energy when relevant, and the task metrics that matter operationally.

---

## The common thread

These techniques live at different parts of a network, but they answer a small set of recurring questions:

| Question | Concepts that answer it |
|---|---|
| What should the model learn? | Cross-entropy and focal loss |
| How are gradients and updates computed? | Backpropagation, RMSProp, Adam |
| How should it represent the input? | MLPs, ReLU, and CNNs |
| How should spatial resolution change? | Pooling, strided convolution, transposed convolution |
| How can training remain stable? | ReLU-like activations, batch normalization |
| How can memorization be reduced? | Dropout and other regularizers |
| How can deployment become cheaper? | Pruning, quantization, distillation, low-rank methods |

The broader lesson is that an architecture is not just a stack of fashionable layers. Each component changes the learning problem: the geometry of the representation, the balance of the gradients, the information preserved by downsampling, or the path optimization takes through parameter space.

When a model fails, asking which of those jobs is failing is usually more productive than adding another layer.

---

## Further reading

- [Understanding the Backpropagation Algorithm](https://medium.com/data-science/understanding-backpropagation-algorithm-7bb3aa2f95fd)
- [Demystifying the Adam Optimizer in Machine Learning](https://medium.com/@weidagang/demystifying-the-adam-optimizer-in-machine-learning-4401d162cb9e)
- [A Lesser-Known Detail of Dropout](https://blog.dailydoseofds.com/p/a-lesser-known-detail-of-dropout)
- [Boosting AI Model Inference: Pruning, Distillation, Low-Rank Methods, and Quantization](https://medium.com/@minh.hoque/boosting-ai-model-inference-three-proven-methods-to-speed-up-your-models-3f2b439f8c8)
- [Model Pruning, Distillation, and Quantization](https://deepgram.com/learn/model-pruning-distillation-and-quantization-part-1#cool-but-when-do-i-prune-my-model)
