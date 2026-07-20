---
title: "Deep Learning Field Notes: Losses, Layers, Training, and Deployment"
date: 2026-07-20 10:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [deep-learning, backpropagation, optimizers, focal-loss, cross-entropy, cnn, pooling, batch-normalization, dropout, metric-learning, model-compression, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/deep-learning-field-notes/
description: "A connected guide to deep-learning losses, backpropagation, optimizers, convolutional layers, regularization, metric learning, and deployment optimization."
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
- [Contrastive and triplet loss](#contrastive-loss)
- [Inference optimization](#inference-optimization)

---

## 1. Cross-entropy and KL divergence {#cross-entropy-kl}

Suppose the true distribution over classes is $P$ and the model predicts $Q$. Their cross-entropy is

$$
H(P,Q)=-\sum_x P(x)\log Q(x).
$$

It measures the average surprise we experience when outcomes come from $P$, but we encode or predict them using $Q$.

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

These were the central follow-up doubts in the [shared ChatGPT conversation](https://chatgpt.com/share/670e47d4-8a48-8000-8da3-11b94aba98b9) linked from my original notes.

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

The derivation in my notes was prompted by this [worked backpropagation article](https://medium.com/data-science/understanding-backpropagation-algorithm-7bb3aa2f95fd).

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

RMSProp keeps an exponential moving average of squared gradients:

$$
v_t=\rho v_{t-1}+(1-\rho)g_t^2,
$$

then scales each coordinate of the update:

$$
\theta_t
=\theta_{t-1}
-\eta\frac{g_t}{\sqrt{v_t}+\varepsilon}.
$$

A coordinate with persistently large gradients gets a smaller effective step; a coordinate with small gradients gets a relatively larger one. Unlike AdaGrad's ever-growing sum, the moving average can forget ancient gradients.

### Adam

Adam combines an exponential average of gradients with the RMSProp-style average of squared gradients:

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

Because both averages begin at zero, early values are biased toward zero. Adam corrects them:

$$
\hat m_t=\frac{m_t}{1-\beta_1^t},
\qquad
\hat v_t=\frac{v_t}{1-\beta_2^t},
$$

and updates

$$
\theta_t
=\theta_{t-1}
-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\varepsilon}.
$$

The first moment smooths the direction; the second adapts the scale. Adam is a strong default, especially for sparse or noisy gradients, but it is not automatically the best-generalizing optimizer. SGD with momentum can still be preferable, and weight decay should usually be implemented in its decoupled AdamW form. The linked [Adam explanation](https://medium.com/@weidagang/demystifying-the-adam-optimizer-in-machine-learning-4401d162cb9e) motivated this part of the notes.

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

For the pictured example,

$$
X=K=
\begin{bmatrix}
0&1\\
2&3
\end{bmatrix}.
$$

The four input values place $0K$, $1K$, $2K$, and $3K$ at four neighboring offsets. Adding their overlaps produces

$$
\begin{bmatrix}
0&0&1\\
0&4&6\\
4&12&9
\end{bmatrix}.
$$

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

In practice, batch normalization often:

- permits larger learning rates;
- makes optimization less sensitive to initialization;
- keeps activation and gradient scales better behaved;
- introduces a small regularizing effect through noisy batch statistics.

Training and inference behave differently. Training uses the current mini-batch statistics. Inference uses running estimates accumulated during training. Forgetting to switch the model to evaluation mode can therefore change predictions.

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

Small or non-independent batches can make the estimates noisy. Layer normalization and group normalization are alternatives when reliable batch statistics are unavailable.

---

## 9. How dropout really works {#dropout}

The familiar description—“randomly turn off neurons to reduce overfitting”—is correct, but incomplete. Dropout must also prevent a systematic scale mismatch between training and inference. This second step is the key point illustrated in [A Lesser-Known Detail of Dropout](https://blog.dailydoseofds.com/p/a-lesser-known-detail-of-dropout).

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

## 10. Contrastive and triplet loss: learn a geometry {#contrastive-loss}

Ordinary classification asks, “Which class is this?” Metric learning asks, “Which examples should be close together?”

Let a network $f$ map an input to an embedding. For a pair $(x_1,x_2)$, define

$$
D=\left\|f(x_1)-f(x_2)\right\|_2.
$$

Let $y=1$ for a similar pair and $y=0$ for a dissimilar pair. One common contrastive loss is

$$
L
=\frac12yD^2
+\frac12(1-y)\max(0,m-D)^2,
$$

where $m$ is a margin.

- For a similar pair, $y=1$, so minimizing $L$ pulls the embeddings together.
- For a dissimilar pair, $y=0$, so the loss pushes them apart only while $D<m$.
- Once a dissimilar pair is at least $m$ apart, it contributes no loss.

The margin avoids wasting capacity by pushing every negative pair infinitely far away. The result is an embedding space in which distance carries semantic meaning. This is useful for face verification, image retrieval, clustering, few-shot learning, and self-supervised representation learning.

Label conventions differ across implementations: some use $y=0$ for similar pairs. The formula and label definition must always be read together.

### Triplet loss

Pairwise contrastive loss says whether two examples should be close. Triplet loss expresses a relative constraint using an anchor $a$, a positive $p$, and a negative $n$:

$$
L_{\text{triplet}}
=\max\!\left(0,
d(a,p)-d(a,n)+m\right).
$$

The loss asks the negative to be at least margin $m$ farther from the anchor than the positive. If that ordering already holds, the triplet contributes no gradient.

Triplet selection is the difficult part. Most easy triplets already have zero loss, while the very hardest examples may be mislabeled or destabilizing. Semi-hard negative mining and good in-batch sampling make the objective useful in practice. The triplet-loss discussion came from the [shared generative-model and representation-learning conversation](https://chatgpt.com/share/6a5216de-cca0-83e8-9b2a-933e3e843c36).

---

## 11. Inference optimization: make the trained model practical {#inference-optimization}

Training minimizes a learning objective; deployment must also satisfy latency, memory, throughput, energy, and cost constraints. The main compression tools trade some redundancy or precision for efficiency.

### Pruning

Pruning removes weights, channels, heads, or entire blocks judged to be unimportant. Unstructured weight sparsity can reduce file size but may not improve wall-clock latency without sparse hardware and kernels. Structured pruning is easier for ordinary accelerators to exploit because it removes complete computational units.

The lottery-ticket hypothesis suggests that a large randomly initialized network may contain smaller subnetworks capable of training to comparable quality. It is an insight about overparameterization, not a guarantee that every pruning rule discovers such a subnetwork.

### Quantization

Quantization represents weights and sometimes activations with fewer bits—for example FP32 to FP16, BF16, INT8, or lower. This reduces memory bandwidth and can accelerate matrix operations on compatible hardware. Post-training quantization is cheap; quantization-aware training usually preserves quality better at aggressive precision.

### Knowledge distillation and low-rank factorization

Distillation trains a smaller student to match a larger teacher's softened outputs or internal representations. Low-rank factorization replaces a large matrix $W$ with smaller factors $UV^T$, reducing parameters and multiplies when the required rank is small.

Compression only counts as an optimization when measured on the target stack. Parameter count, theoretical FLOPs, model size, and observed latency are related but not interchangeable. These notes were prompted by the linked overview of [pruning, distillation, low-rank methods, and quantization](https://medium.com/@minh.hoque/boosting-ai-model-inference-three-proven-methods-to-speed-up-your-models-3f2b439f8c8).

---

## The common thread

These techniques live at different parts of a network, but they answer a small set of recurring questions:

| Question | Concepts that answer it |
|---|---|
| What should the model learn? | Cross-entropy, focal, contrastive, and triplet loss |
| How are gradients and updates computed? | Backpropagation, RMSProp, Adam |
| How should it represent the input? | MLPs, ReLU, and CNNs |
| How should spatial resolution change? | Pooling, strided convolution, transposed convolution |
| How can training remain stable? | ReLU-like activations, batch normalization |
| How can memorization be reduced? | Dropout and other regularizers |
| How can deployment become cheaper? | Pruning, quantization, distillation, low-rank methods |

The broader lesson is that an architecture is not just a stack of fashionable layers. Each component changes the learning problem: the geometry of the representation, the balance of the gradients, the information preserved by downsampling, or the path optimization takes through parameter space.

When a model fails, asking which of those jobs is failing is usually more productive than adding another layer.
