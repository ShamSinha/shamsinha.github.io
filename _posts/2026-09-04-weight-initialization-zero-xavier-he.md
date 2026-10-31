---
title: "Weight Initialization: Why Zero Fails and Xavier/He Stabilize Gradients"
date: 2026-09-04 10:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [weight-initialization, xavier-initialization, he-initialization, neural-networks, backpropagation, activation-functions]
math: true
toc: true
comments: true
published: true
permalink: /posts/weight-initialization-zero-xavier-he/
description: "A question-driven guide to zero and random initialization, vanishing and exploding gradients, activation-dependent scaling, and the derivation of Xavier and He initialization."
visual_guide: weight-initialization-zero-xavier-he
---

{% include visual-guide.html %}

## Full notes, proofs, and code {#full-notes}

<details class="visual-deep-dive">
<summary>Open the complete explanation</summary>

{% capture full_notes %}{% include notes/weight-initialization-zero-xavier-he.md %}{% endcapture %}
{{ full_notes | markdownify | replace: '<h2 ' , '<h2 data-toc-skip ' | replace: '<h3 ' , '<h3 data-toc-skip ' | replace: '<h4 ' , '<h4 data-toc-skip ' }}

</details>
