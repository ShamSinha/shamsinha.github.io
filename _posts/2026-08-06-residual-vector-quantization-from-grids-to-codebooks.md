---
title: "Residual Vector Quantization: From Grid Intuition to Codebooks"
date: 2026-08-06 10:30:00 +0530
categories: [Machine Learning, Representation Learning]
tags: [vector-quantization, residual-vector-quantization, codebooks, embeddings, neural-audio-codecs, compression]
math: true
toc: true
comments: true
published: true
permalink: /posts/residual-vector-quantization-from-grids-to-codebooks/
description: "An intuitive explanation of residual vector quantization: how codebooks replace grids, how residuals refine an approximation, and why RVQ gives compact discrete representations."
visual_guide: residual-vector-quantization-from-grids-to-codebooks
---

{% include visual-guide.html %}

## Full notes, proofs, and code {#full-notes}

<details class="visual-deep-dive">
<summary>Open the complete explanation</summary>

{% capture full_notes %}{% include notes/residual-vector-quantization-from-grids-to-codebooks.md %}{% endcapture %}
{{ full_notes | markdownify | replace: '<h2 ' , '<h2 data-toc-skip ' | replace: '<h3 ' , '<h3 data-toc-skip ' | replace: '<h4 ' , '<h4 data-toc-skip ' }}

</details>
