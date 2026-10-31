---
title: "Generative AI from First Principles: Variational Inference, VAEs, and Diffusion"
date: 2026-07-20 12:00:00 +0530
categories: [Machine Learning, Generative AI]
tags: [generative-ai, variational-inference, vae, elbo, reparameterization-trick, diffusion-models, ddpm, latent-variables, bayesian-inference, study-notes]
math: true
toc: true
comments: true
published: true
permalink: /posts/generative-ai-variational-inference-vaes-diffusion/
description: "A first-principles path from Bayesian posterior approximation and the ELBO to variational autoencoders and denoising diffusion probabilistic models."
visual_guide: generative-ai-variational-inference-vaes-diffusion
---

{% include visual-guide.html %}

## Full notes, proofs, and code {#full-notes}

<details class="visual-deep-dive">
<summary>Open the complete explanation</summary>

{% capture full_notes %}{% include notes/generative-ai-variational-inference-vaes-diffusion.md %}{% endcapture %}
{{ full_notes | markdownify | replace: '<h2 ' , '<h2 data-toc-skip ' | replace: '<h3 ' , '<h3 data-toc-skip ' | replace: '<h4 ' , '<h4 data-toc-skip ' }}

</details>
