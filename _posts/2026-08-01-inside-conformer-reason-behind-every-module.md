---
title: "Inside Conformer: Why Its Modules Fit Together"
date: 2026-08-01 11:00:00 +0530
categories: [Machine Learning, Deep Learning]
tags: [conformer, glu, pointwise-convolution, depthwise-convolution, self-attention, speech-recognition]
math: true
toc: true
comments: true
published: true
permalink: /posts/inside-conformer-reason-behind-every-module/
description: "A tensor-by-tensor explanation of the Conformer block, including its Macaron feed-forward layers, attention, pointwise convolution, GLU, depthwise convolution, and Swish activation."
visual_guide: inside-conformer-reason-behind-every-module
---

{% include visual-guide.html %}

## Full notes, proofs, and code {#full-notes}

<details class="visual-deep-dive">
<summary>Open the complete explanation</summary>

{% capture full_notes %}{% include notes/inside-conformer-reason-behind-every-module.md %}{% endcapture %}
{{ full_notes | markdownify | replace: '<h2 ' , '<h2 data-toc-skip ' | replace: '<h3 ' , '<h3 data-toc-skip ' | replace: '<h4 ' , '<h4 data-toc-skip ' }}

</details>
