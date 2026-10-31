---
title: "How DINOv2 and DINOv3 Learn Without Labels"
date: 2026-09-04 11:00:00 +0530
categories: [Machine Learning, Self-Supervised Learning]
tags: [dino, dinov2, dinov3, self-supervised-learning, vision-transformer, knowledge-distillation, ibot, koleo, gram-anchoring, representation-learning]
math: true
toc: true
comments: true
published: true
permalink: /posts/how-dinov2-dinov3-learn-without-labels/
description: "A loss-by-loss explanation of DINOv2 and DINOv3 training: teacher-student self-distillation, iBOT masking, KoLeo regularization, collapse prevention, and why DINOv3 adds Gram anchoring for dense features."
visual_guide: how-dinov2-dinov3-learn-without-labels
---

{% include visual-guide.html %}

## Full notes, proofs, and code {#full-notes}

<details class="visual-deep-dive">
<summary>Open the complete explanation</summary>

{% capture full_notes %}{% include notes/how-dinov2-dinov3-learn-without-labels.md %}{% endcapture %}
{{ full_notes | markdownify | replace: '<h2 ' , '<h2 data-toc-skip ' | replace: '<h3 ' , '<h3 data-toc-skip ' | replace: '<h4 ' , '<h4 data-toc-skip ' }}

</details>
