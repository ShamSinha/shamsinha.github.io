---
title: "Shannon Information and Entropy — A Doubt-Driven Guide"
date: 2026-07-18 12:00:00 +0530
categories: [Mathematics, Information Theory]
tags: [shannon-information, entropy, self-information, compression, bits, probability, doubts, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/shannon-information-entropy-doubt-log/
description: "A doubt-by-doubt path from surprisal to bits, entropy, compression, and the reason logarithms appear in information theory."
---

Most explanations of information theory begin with

$$
I(x)=-\log_2 P(x)
$$

and move on as though the formula were self-evident. It was not self-evident to me.

Why should probability measure information? Why a logarithm? Why does choosing between four outcomes require two bits? Who is asking the yes-or-no questions? Is this quantity already entropy? And what does any of it have to do with compression?

This post keeps those doubts instead of editing them out. They are the path by which the subject started to make sense.

> A small name correction before we begin: I wrote “Claude Shanon” in my original question. The name is **Claude Shannon**.

---

## 1. What is the information content of an outcome?

> **Starting point:** “The Shannon information content of an outcome”

The **self-information**, **information content**, or **surprisal** of an outcome $x$ is

$$
I(x)=-\log_2 P(x).
$$

Here, $P(x)$ is the probability of observing $x$. The base of the logarithm determines the unit. Base $2$ gives bits; the natural logarithm gives nats.

The first intuition is:

- an almost certain outcome tells us very little;
- a rare outcome tells us a lot;
- an impossible outcome would have unbounded surprisal.

Some values make this concrete:

| Probability | Self-information |
|---:|---:|
| $1$ | $0$ bits |
| $1/2$ | $1$ bit |
| $1/4$ | $2$ bits |
| $1/8$ | $3$ bits |

For a fair coin,

$$
I(\text{Heads})=-\log_2(1/2)=1\text{ bit}.
$$

For an event with probability $1/8$,

$$
I(x)=-\log_2(1/8)=3\text{ bits}.
$$

The formula is easy to use. The real difficulty is understanding why this should be the formula at all.

---

## 2. How could Shannon have arrived at this?

> **My doubt:** “I still don't know how Claude Shanon think of this in first place”

A useful historical caution: Shannon did not invent these ideas in isolation. Ralph Hartley had already connected the amount of information with the logarithm of the number of possible symbols in 1928. Shannon's 1948 work built a much more general theory for probabilistic sources and noisy communication channels.

The formula can be reconstructed from a few requirements that an information measure ought to satisfy.

### Requirement 1: rarer events should be more informative

Learning that the sun rose today changes almost nothing in our knowledge. Learning that a fair coin produced twenty heads in a row changes much more.

Therefore, if $p$ decreases, $I(p)$ should increase.

### Requirement 2: independent information should add

Suppose two independent events have probabilities $p_1$ and $p_2$. Their joint probability is

$$
p_1p_2.
$$

Learning both outcomes should give the information from the first plus the information from the second:

$$
I(p_1p_2)=I(p_1)+I(p_2).
$$

This requirement is the key. Probabilities of independent events multiply, but we want their information to add.

### Requirement 3: the measure should behave smoothly

A tiny change in probability should not cause a wild jump in information.

Under these natural conditions, the logarithm is essentially forced on us because

$$
\log(ab)=\log a+\log b.
$$

Thus the measure must have the form

$$
I(p)=-K\log p
$$

for some positive constant $K$. Choosing base $2$ and $K=1$ measures information in bits:

$$
I(x)=-\log_2P(x).
$$

The minus sign is not mysterious. Since $0<p\le1$, $\log p\le0$. Negating it makes information nonnegative and makes rarer events produce larger values.

So the formula is not an arbitrary clever guess. It is what remains after asking information to behave in a few reasonable ways.

---

## 3. Why do 2, 4, and 8 outcomes correspond to 1, 2, and 3 bits?

> **My doubt:** “log is ok, but this — 2 outcomes → 1 bit, 4 outcomes → 2 bits, 8 outcomes → 3 bits”

This clicked only after I stopped thinking of a bit merely as a storage slot.

A bit represents the answer to one binary question: yes or no, $0$ or $1$. One such answer distinguishes between two possibilities.

With four equally likely possibilities, two binary answers are enough:

| Outcome | Binary code |
|---|---|
| A | `00` |
| B | `01` |
| C | `10` |
| D | `11` |

With eight possibilities, three binary digits are enough, from `000` through `111`.

In general, $k$ bits can distinguish among

$$
2^k
$$

possibilities. Therefore, distinguishing among $N$ equally likely outcomes requires

$$
k=\log_2N
$$

bits.

For equally likely outcomes,

$$
P(x)=\frac1N.
$$

Substituting this into self-information gives

$$
I(x)
=-\log_2\left(\frac1N\right)
=\log_2N.
$$

The probability formula and the binary-decision picture are therefore two views of the same quantity.

### What if $N$ is not a power of two?

If there are six equally likely outcomes, then

$$
I(x)=\log_2 6\approx2.585\text{ bits}.
$$

We cannot use $2.585$ physical binary digits for one isolated symbol. The fractional value is an average coding limit: by encoding long sequences together, a good code can approach $2.585$ bits per symbol.

That distinction between one codeword and a long-run average becomes important when entropy enters the story.

---

## 4. To whom are these questions being asked?

> **My doubt:** “to whom you are asking this question”

This was not a side issue. The yes-or-no explanation quietly assumes a communication setup:

- a **sender**, or nature, knows which outcome occurred;
- a **receiver** does not know it;
- enough binary information must pass from sender to receiver for the receiver to identify the outcome.

Suppose the sender secretly chooses one of $\{A,B,C,D\}$. The receiver could ask:

1. Is it in $\{A,B\}$?
2. Depending on the answer, is it A or C?

After two answers, the receiver knows the outcome.

No literal person needs to ask these questions. It is a decision-tree thought experiment. In a communication system, the answers become the zeros and ones sent through the channel.

The phrase “number of questions” is therefore shorthand for:

> How many binary distinctions must the receiver learn to remove the uncertainty?

---

## 5. Is this also called entropy?

> **My doubt:** “THIS is also called entropy”

Not exactly. This is the distinction I kept mixing up.

Self-information concerns **one observed outcome**:

$$
I(x)=-\log_2P(x).
$$

Entropy concerns the **average self-information** of the random variable before its outcome is known:

$$
H(X)=\mathbb E[I(X)]
=\sum_xP(x)I(x)
=-\sum_xP(x)\log_2P(x).
$$

In short:

- $I(x)$: how surprising was this particular outcome?
- $H(X)$: how much surprise should I expect on average?

### Why the two look identical for a fair coin

For a fair coin,

$$
P(H)=P(T)=\frac12.
$$

Each outcome has self-information

$$
I(H)=I(T)=1\text{ bit}.
$$

The entropy is

$$
H(X)=\frac12(1)+\frac12(1)=1\text{ bit}.
$$

Both numbers are $1$, which makes it easy to think they are the same concept.

### A biased coin separates them

Let

$$
P(H)=0.99,
\qquad
P(T)=0.01.
$$

Then

$$
I(H)=-\log_2(0.99)\approx0.014\text{ bits},
$$

while

$$
I(T)=-\log_2(0.01)\approx6.644\text{ bits}.
$$

The rare tail is highly surprising. But it occurs only one percent of the time. The average is

$$
\begin{aligned}
H(X)
&=0.99I(H)+0.01I(T)\\
&\approx0.99(0.014)+0.01(6.644)\\
&\approx0.081\text{ bits}.
\end{aligned}
$$

Entropy is neither $0.014$ nor $6.644$ bits. It is the long-run average surprise per flip.

An analogy:

- self-information is the height of one student;
- entropy is the average height of the class.

---

## 6. Why define entropy, and where does it help?

> **My doubt:** “why this entropy is defined, where it helps”

Engineers rarely design a communication system for one isolated outcome. They deal with a source producing thousands or millions of symbols. The useful engineering question is therefore:

> On average, how much new information does this source produce per symbol?

Entropy answers that question.

### Compression

Compare these two sources:

- a fair coin, whose next result is completely uncertain;
- a coin that lands heads $99\%$ of the time, whose next result is highly predictable.

Both sources have two possible symbols, but they should not require the same average number of bits to describe a long sequence. The predictable source contains more redundancy and can be compressed more.

Shannon's source coding result connects entropy to the fundamental limit of lossless compression. Roughly, for a long sequence generated by a source, no lossless code can beat its entropy in average bits per symbol, while suitable codes can approach that limit.

This idea sits beneath Huffman coding, arithmetic coding, and many practical compression systems.

### Measuring uncertainty

Entropy measures uncertainty before an outcome is observed.

For a fair die,

$$
H(X)=\log_2 6\approx2.585\text{ bits}.
$$

A heavily loaded die has lower entropy because its result is easier to predict.

### Machine learning

Cross-entropy turns the same idea into a loss. If the correct class is “cat” and a model assigns it probability $0.99$, the outcome has low surprisal:

$$
-\log(0.99).
$$

If the model assigns the correct class probability $0.01$, the loss is large:

$$
-\log(0.01).
$$

A confidently wrong prediction is penalized because the observed truth was extremely surprising under the model's claimed distribution.

### Communication through noise

Entropy is also one of the building blocks for mutual information and channel capacity. Those quantities let us ask how much information a noisy channel preserves and how fast it can transmit data reliably.

---

## 7. Why mention “you don't need 8 bits per character”?

> **My doubt:** “why you mention this”

The statement was meant to connect entropy with compression, but it needs qualification.

A storage encoding may allocate a fixed number of bits to every character. For example, an ASCII character occupies 8 bits. Sixteen `A` characters would therefore occupy

$$
16\times8=128\text{ bits}
$$

in that representation.

But storage size is not the same as information content. A message such as

```text
AAAAAAAAAAAAAAAA
```

is extremely predictable. If the receiver and sender share an encoding rule, it can be represented more compactly—for example, as the symbol `A` plus a count of sixteen.

The precise lesson is not that every repeated string magically occupies zero bits. The decoder still needs enough information to reconstruct the symbol, its count, and any coding conventions that are not already shared.

The lesson is:

> A fixed-width representation may use more bits than the source produces in new information. Compression exploits that gap, and entropy describes the asymptotic lower limit under a probabilistic source model.

---

## 8. The complete mental model

The chain now looks like this:

1. Before observing $x$, the receiver is uncertain.
2. Observing a rare $x$ removes more uncertainty than observing a predictable one.
3. The quantity $I(x)=-\log_2P(x)$ measures the information in that particular observation.
4. The logarithm makes information from independent events additive.
5. A bit represents one binary distinction.
6. Entropy $H(X)$ averages self-information over all possible outcomes.
7. That average determines the fundamental rate at which long outputs of a source can be losslessly encoded.

The shortest version is:

$$
\boxed{I(x)=-\log_2P(x)}
\qquad\text{and}\qquad
\boxed{H(X)=\mathbb E[I(X)]}.
$$

Self-information is **surprise after one outcome occurs**. Entropy is **expected surprise before it occurs**.

---

## Appendix: the Notion rendering doubt

> **My doubt:** “no it is not redenering I tried copy the content, it just creating markdown code block”

This was a publishing problem rather than an information-theory problem, but it was part of the learning conversation.

Markdown fenced code and mathematical display blocks are different things. In Jekyll with MathJax or KaTeX support, equations enclosed by `$$` can render as display mathematics. Notion's Markdown import does not reliably convert those expressions into native equation blocks; it may import them as plain text or code instead.

Inside Notion, the dependable approach is to create an equation block with `/math` and paste only the LaTeX expression, without the surrounding `$$` markers.

For this blog, the post front matter includes `math: true`, so the display equations are rendered by the site's math support.

---

## Primary sources

- Claude E. Shannon, [“A Mathematical Theory of Communication”](https://doi.org/10.1002/j.1538-7305.1948.tb01338.x), *Bell System Technical Journal*, 1948.
- R. V. L. Hartley, [“Transmission of Information”](https://doi.org/10.1002/j.1538-7305.1928.tb01236.x), *Bell System Technical Journal*, 1928.
