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
---

How can a vision model learn objects, shapes, textures, and spatial correspondences from images that have no class labels?

DINO's answer is not to train without targets. It is to **construct the targets from the images and from the model itself**.

The same image is transformed into several views. A student network must make its representation of one view agree with targets produced by a slowly changing teacher on another view. Some patches are hidden from the student, which must predict how the teacher represents the visible version. Additional regularizers keep the representation diverse and prevent collapse.

DINOv2 combines three main objectives:

$$
\mathcal{L}_{\mathrm{DINOv2}}
=
\mathcal{L}_{\mathrm{DINO}}
+
\mathcal{L}_{\mathrm{iBOT}}
+
0.1\mathcal{L}_{\mathrm{KoLeo}}.
$$

DINOv3 scales this recipe to much more data and a 7-billion-parameter teacher. It then introduces **Gram anchoring** because a surprising problem appears during long training: image-level recognition keeps improving while the spatial quality of patch features gets worse.

This post follows the training signal step by step, explains why every loss exists, and examines whether Gram anchoring actually helps.

> Here, DINO means **self-DIstillation with NO labels**. It should not be confused with the DETR-family object detector that is also named DINO.

## What “without labels” really means

In ordinary supervised classification, an image comes with a human target:

```text
image of a dog -> label: "dog"
```

The loss compares the model's prediction with that label. Self-supervised learning removes the human-written target, but it still needs a learning signal. DINO creates one from the structure already present in an image:

- two crops of the same image should describe the same underlying scene;
- a small crop should be compatible with the larger image from which it came;
- a hidden patch should be predictable from its visible context;
- different images should not all map to the same representation;
- patches belonging to related regions should retain meaningful relationships.

These are not category annotations. They are constraints derived automatically from augmentations, masking, and the model's own earlier predictions.

The pretraining data can still be curated. DINOv2 constructs LVD-142M from a pool of 1.2 billion deduplicated web images, while DINOv3 builds a 1.689-billion-image curated component from a pool of roughly 17 billion images and mixes it with retrieval-curated and public vision data. The important point is that the pretraining losses do not consume class, box, mask, or depth labels.

## The objects being trained: class and patch tokens

A Vision Transformer divides an image into patches and converts them into tokens. After the transformer blocks, it produces:

$$
\left[c, x_1,x_2,\ldots,x_P\right],
$$

where:

- $c$ is the **class token**, a global summary of the image;
- $x_i$ is the output feature for patch $i$;
- $P$ is the number of image patches.

The class token is useful for global tasks such as image classification and retrieval. Patch tokens are essential for dense tasks such as segmentation, depth estimation, tracking, and pixel-level correspondence.

DINOv2 and DINOv3 deliberately supervise both levels:

| Training signal | Token level | What it encourages |
| --- | --- | --- |
| DINO | Class token | View-invariant global semantics |
| iBOT | Patch tokens | Contextual, locally useful features |
| KoLeo | Class token geometry | Diversity across images |
| DINOv3 Gram anchoring | Pairwise patch relations | Stable spatial consistency during long training |

The backbone features are passed through small projection heads before the losses are computed. Their outputs are scores over learned **prototypes**. A prototype is not a named class such as dog or car. It is a learned direction that gives the networks a shared vocabulary for expressing visual structure.

After pretraining, these loss-specific heads can be discarded. The backbone representations are the useful product.

## One training step at a glance

For each source image, the pipeline constructs two large **global crops** and several smaller **local crops**.

```text
                         one unlabeled image
                                  │
                    augment, crop, and sometimes mask
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
      student network                         teacher network
   sees global + local crops                 sees global crops only
   some global patches masked               global patches are visible
              │                                       │
      receives gradients                         stop-gradient
              │                                       │
              └──────────── match predictions ────────┘
                                  │
                   update student with AdamW
                                  │
                  update teacher as EMA(student)
```

The teacher is initialized from the student but is not updated by backpropagation. After a student update, its parameters move as an exponential average:

$$
\theta_t
\leftarrow
m\theta_t+(1-m)\theta_s,
$$

where $\theta_s$ and $\theta_t$ are the student and teacher parameters, and $m$ is close to $1$.

The teacher is therefore a smoothed history of the student. It changes slowly enough to provide a more stable target, but it improves as the student improves.

This setup is called **self-distillation** because the model learns from a teacher produced by its own training trajectory.

## Loss 1: DINO learns a global representation

Take two augmented views $u$ and $v$ of the same image. They may differ in crop, scale, color, blur, or other augmentation. The student processes $u$, while the teacher processes the global view $v$.

Their class tokens pass through separate DINO projection heads. The student distribution is

$$
p_s^{(u)}
=
\operatorname{softmax}\!\left(
\frac{g_s(c_s^{(u)})}{\tau_s}
\right),
$$

where $g_s$ is the student head and $\tau_s$ is its temperature. The teacher similarly produces a target distribution $p_t^{(v)}$, but its assignments are sharpened and balanced before being used.

For one pair of views, the DINO loss is cross-entropy:

$$
H\!\left(p_t^{(v)},p_s^{(u)}\right)
=
-\sum_k p_{t,k}^{(v)}\log p_{s,k}^{(u)}.
$$

The complete image-level objective compares every teacher global crop with the compatible student crops, excluding a global crop's identical view:

$$
\mathcal{L}_{\mathrm{DINO}}
=
\frac{1}{N}
\sum_{v\in\mathcal{G}}
\sum_{\substack{u\in\mathcal{G}\cup\mathcal{L}\\u\ne v}}
H\!\left(p_t^{(v)},p_s^{(u)}\right).
$$

Here $\mathcal{G}$ is the set of global crops, $\mathcal{L}$ is the set of local crops, and $N$ normalizes the number of pairs.

### What is this loss teaching?

Suppose one global crop contains a dog in a park and a local crop contains only the dog's head. The teacher's global representation provides the target. The student must make the local view compatible with that target even though background, scale, and color may differ.

Across many images, the easiest stable solution is to represent visual content that survives augmentation rather than memorizing exact pixels. The network begins organizing images by shared appearance, parts, shapes, and eventually semantic structure.

The DINO loss therefore contributes:

- invariance to augmentations and crop scale;
- a global summary in the class token;
- semantic grouping without named classes;
- local-to-global consistency through the multi-crop setup.

It is a cross-entropy loss, but the target is a teacher distribution over learned prototypes—not a human one-hot label.

## Why does the teacher-student arrangement work?

A natural objection is: if the teacher is just an average of the student, where does new knowledge come from?

It comes from combining several asymmetries:

1. **Different views:** the student must explain one crop using a target from another crop.
2. **Different time scales:** the student changes quickly; the EMA teacher changes slowly.
3. **Stop-gradient:** the teacher target does not move to reduce the current loss.
4. **Different temperatures:** a sharper teacher supplies a decisive target to a smoother student prediction.
5. **Balanced teacher assignments:** the batch cannot obtain a low loss by using only one prototype.
6. **Masking:** for iBOT, the student has less visual information than the teacher.

The teacher is not an external oracle. It is a stable target generator. The student learns from the teacher's view-consistent patterns, and the teacher incorporates that learning gradually through the EMA update.

## Why does everything not collapse to one constant vector?

If student and teacher produced the same constant vector for every image, their outputs would agree perfectly but the representation would be useless. This is the **collapse problem**.

Original DINO avoids collapse through teacher centering and sharpening. DINOv2 and DINOv3 replace the teacher's centering operation with the Sinkhorn–Knopp assignment used in SwAV.

For a batch-to-prototype score matrix, Sinkhorn–Knopp alternates two normalizations:

- normalize across samples so prototypes receive balanced total mass;
- normalize across prototypes so each sample has a valid assignment distribution.

The released implementations perform three iterations. This balancing discourages the teacher from sending every input to the same prototype. Temperature sharpening prevents the opposite unhelpful solution in which every sample has the same uniform distribution over all prototypes.

Sinkhorn–Knopp is not another semantic loss. It is a target-normalization and collapse-prevention mechanism around the DINO and iBOT objectives.

## Loss 2: iBOT learns patch-level representations

The DINO class-token loss can learn an excellent global representation while saying too little about individual patches. A segmentation model needs to know much more than “this image contains a dog.” It needs spatially distinct features for dog, grass, tree, and boundary patches.

iBOT adds a masked-image objective in representation space.

For a global crop:

1. the teacher sees every patch;
2. a random subset of patches is replaced by mask tokens for the student;
3. the teacher produces a prototype distribution for each visible patch;
4. at each masked position, the student predicts the teacher's distribution for the corresponding unmasked patch.

If $\mathcal{M}$ is the set of masked positions, the patch loss is

$$
\mathcal{L}_{\mathrm{iBOT}}
=
\frac{1}{|\mathcal{M}|}
\sum_{i\in\mathcal{M}}
H\!\left(q_{t,i},q_{s,i}\right)
=
-\frac{1}{|\mathcal{M}|}
\sum_{i\in\mathcal{M}}
\sum_k q_{t,i,k}\log q_{s,i,k}.
$$

The important word is **latent** reconstruction. The student is not asked to reproduce RGB values. Pixel reconstruction can spend capacity on color and texture details that are irrelevant to semantics. iBOT instead asks the student to recover how the teacher represents the hidden region.

### What is this loss teaching?

To predict a masked dog's leg, the student can use the visible body, pose, surrounding patches, and patterns learned from other images. This encourages:

- contextual reasoning between patches;
- meaningful local descriptors;
- object-part and region structure;
- stronger transfer to segmentation and depth estimation.

The DINOv2 ablation makes the role concrete. Removing its masked-image term reduced the reported ADE20K linear-segmentation score from $47.1$ to $44.2$ mIoU, while its effect on Oxford image retrieval was small and slightly reversed. The loss is especially valuable where the **patch tokens themselves** matter.

## Loss 3: KoLeo spreads global features

Even when a network has not collapsed to one exact vector, its embeddings can crowd into a small part of the representation space. That hurts nearest-neighbor retrieval and wastes representational capacity.

KoLeo is an entropy-inspired spreading regularizer. Let $z_i$ be an $\ell_2$-normalized class-token feature and define its nearest-neighbor distance within a batch as

$$
d_i=\min_{j\ne i}\|z_i-z_j\|_2.
$$

The loss is

$$
\mathcal{L}_{\mathrm{KoLeo}}
=
-\frac{1}{B}\sum_{i=1}^{B}\log(d_i+\varepsilon).
$$

Minimizing the negative logarithm discourages very small nearest-neighbor distances. The features spread more uniformly over the unit hypersphere.

### What is this loss teaching?

KoLeo does not decide which images are semantically similar. Its narrower jobs are to:

- prevent feature concentration;
- encourage use of more representational directions;
- improve discriminability for nearest-neighbor search and retrieval.

In DINOv2's ablation, KoLeo increased Oxford-M retrieval mAP from $55.6$ to $63.9$, with no loss on the reported ADE20K score and small gains on the classification metrics.

### What if two genuinely similar images occur in the same batch?

Then KoLeo can push them apart. It has no labels and cannot know that two different dog photographs belong together. This is a real trade-off, similar to the false-negative problem in contrastive learning.

Several details keep it from becoming the dominant semantic signal:

- it is a regularizer with weight $0.1$, not the main objective;
- DINO pulls compatible views and learned semantics into agreement;
- iBOT structures the local features;
- batches and nearest neighbors change throughout training;
- the model can separate individual instances while preserving broader semantic organization in other directions.

The correct interpretation is not “KoLeo creates semantic clusters.” DINO and iBOT learn the content; KoLeo prevents the resulting global embedding distribution from becoming pathologically crowded.

DINOv3 uses a distributed KoLeo variant and applies it in small groups of 16 class tokens, possibly assembled across GPUs.

## Why the three DINOv2 losses complement one another

The objectives are not redundant:

| Objective | Failure without it | Representation benefit |
| --- | --- | --- |
| DINO | Weak cross-view global semantics | Image-level invariance and semantic organization |
| iBOT | Global token can improve while patches remain weak | Locally meaningful, contextual patch tokens |
| KoLeo | Global embeddings can crowd together | Better coverage of feature space and retrieval |
| Sinkhorn–Knopp | Prototype use can become unbalanced or collapse | Diverse, balanced teacher targets |

An image representation must be invariant enough to recognize the same content across views, detailed enough to preserve spatial information, and diverse enough to distinguish different inputs. The combined objective addresses all three.

## The DINOv2 training recipe

The largest DINOv2 model is a ViT-g/14 with about 1.1 billion parameters. Its main from-scratch training recipe is:

- two $224\times224$ global crops and eight $98\times98$ local crops per image;
- the student processes all crops;
- the EMA teacher processes only global crops;
- on half of the global-crop samples, between $10\%$ and $50\%$ of patches are masked for the student;
- separate projection heads compute the DINO and iBOT prototype distributions;
- Sinkhorn–Knopp balances teacher assignments;
- AdamW optimizes the student for 625,000 iterations;
- the teacher momentum follows a schedule from $0.994$ toward $1$;
- a short $518\times518$ high-resolution phase improves pixel-level transfer.

The ViT-g/14 is trained from scratch. Smaller DINOv2 models are distilled from the frozen ViT-g teacher. In that later distillation setting, “teacher” has a different meaning: it is no longer an EMA copy of the same-size student but a fixed, already trained larger network.

## What changes in DINOv3?

DINOv3 does not discard the DINOv2 objective. Its initial phase still uses

$$
\boxed{
\mathcal{L}_{\mathrm{Pre}}
=
\mathcal{L}_{\mathrm{DINO}}
+
\mathcal{L}_{\mathrm{iBOT}}
+
0.1\mathcal{L}_{\mathrm{DKoLeo}}
}
$$

but it scales and stabilizes the system:

| Component | DINOv2 flagship | DINOv3 flagship |
| --- | --- | --- |
| Backbone | ViT-g/14 | ViT-7B/16 |
| Parameters | About 1.1B | About 6.7B |
| Patch size | 14 | 16 |
| Positional representation | Learned embeddings | RoPE with coordinate jittering |
| Global/local crops | $224/98$ | $256/112$ |
| Training length | 625k iterations | 1M initial iterations |
| Main teacher update | Scheduled EMA | Constant $0.999$ EMA after warm-up setup |

DINOv3 uses two global and eight local crops. The student sees them all, while the teacher sees only the global crops. For iBOT, a proportion between $0.1$ and $0.5$ of global-crop patches is randomly masked with probability $0.5$. The flagship pretraining uses a total batch size of 4,096 images across 256 GPUs.

It also separates layer normalization for global- and local-crop backbone outputs. The paper reports that this small architectural detail stabilizes late k-NN classification and improves dense evaluation.

The scale is important, but it exposes a problem that the original three-loss recipe does not fully solve.

## The long-training problem: global improves, dense degrades

“Dense prediction” means making a spatial prediction over an image, as in:

- a semantic class per pixel;
- a depth value per pixel;
- an object or point correspondence across frames;
- a feature for every patch location.

During DINOv3 pretraining, Meta evaluated snapshots of the **frozen backbone**. ImageNet classification from the class token improved throughout training. Pascal VOC segmentation from patch tokens peaked at about 200,000 iterations and then declined; the 7B model eventually fell below its earlier dense performance.

This does not mean that a segmentation network was being trained alongside DINOv3 and its supervised loss grew worse. There is no segmentation loss in self-supervised pretraining. It means that when a simple downstream predictor was later placed on the current frozen patch features, those features had become less useful for segmentation.

The researchers inspected patch-to-patch cosine-similarity maps:

```text
early training:  related nearby patches form smooth, localized regions
late training:   unrelated patches increasingly receive high similarity
```

They also observed growing similarity between the global class token and patch tokens. The patches increasingly absorbed global information and lost locality. Register tokens kept patch norms stable, so this was different from the high-norm artifact addressed by registers.

This reveals an important distinction:

> A model can improve at recognizing the whole image while getting worse at preserving where visual information is located.

The global DINO objective and local iBOT objective had become imbalanced over the very long schedule. iBOT helped dense features, but it did not guarantee that the entire relational geometry between patches would remain coherent indefinitely.

## A Gram matrix is a map of patch relationships

Suppose an image produces $P$ patch features of dimension $d$. Normalize every feature to unit length and place them into a matrix

$$
X=
\begin{bmatrix}
x_1^T\\
x_2^T\\
\vdots\\
x_P^T
\end{bmatrix}
\in\mathbb{R}^{P\times d},
\qquad
\|x_i\|_2=1.
$$

Its Gram matrix is

$$
G=XX^T\in\mathbb{R}^{P\times P}.
$$

Entry $(i,j)$ is

$$
G_{ij}=x_i^Tx_j=\cos(x_i,x_j).
$$

So the Gram matrix is a table of every patch's similarity to every other patch:

$$
G=
\begin{bmatrix}
1 & \cos(x_1,x_2) & \cdots \\
\cos(x_2,x_1) & 1 & \cdots \\
\vdots & \vdots & \ddots
\end{bmatrix}.
$$

If several patches cover the same object, their relationships should form a coherent pattern. When unrelated background patches become spuriously similar, the Gram matrix exposes that deterioration.

## Loss 4 in DINOv3: Gram anchoring

DINOv3 saves an earlier model snapshot whose patch features still have good dense properties. This frozen historical model is called the **Gram teacher**.

For the same global crop, let

$$
X_S\in\mathbb{R}^{P\times d}
$$

contain the current student's normalized patch features, and let

$$
X_G\in\mathbb{R}^{P\times d}
$$

contain the Gram teacher's normalized patch features. The new objective is

$$
\boxed{
\mathcal{L}_{\mathrm{Gram}}
=
\left\|
X_SX_S^T-X_GX_G^T
\right\|_F^2
}
$$

where $\|\cdot\|_F$ is the Frobenius norm. The released implementation computes the mean-squared error between the two similarity matrices, which differs only by normalization of the summed entries.

The refinement objective becomes

$$
\mathcal{L}_{\mathrm{Ref}}
=
w_D\mathcal{L}_{\mathrm{DINO}}
+
\mathcal{L}_{\mathrm{iBOT}}
+
w_{DK}\mathcal{L}_{\mathrm{DKoLeo}}
+
w_G\mathcal{L}_{\mathrm{Gram}}.
$$

The paper uses a Gram-loss weight of $2$. For efficiency, the flagship run begins this refinement only after the initial one million iterations, although the authors note that the objective can be introduced earlier. The Gram teacher is updated from the main EMA teacher every 10,000 steps, at most three times, after refinement begins.

### Why match relationships instead of patch vectors directly?

Direct feature matching would use something like

$$
\|X_S-X_G\|_F^2.
$$

That would force every current patch vector toward the exact coordinates used by an old checkpoint. It could freeze too much of the earlier representation and obstruct further global learning.

Gram matching is less restrictive. If all student features undergo the same orthogonal transformation $Q$, then

$$
(X_SQ)(X_SQ)^T
=
X_SQQ^TX_S^T
=
X_SX_S^T.
$$

The individual vectors may rotate or reorganize while their pairwise geometry remains intact. Gram anchoring says:

> Keep learning, but do not destroy which patches are similar to which other patches.

This is exactly the information needed to protect spatial coherence without forcing the late model to copy every early feature coordinate.

## Why use an early checkpoint as the Gram teacher?

The EMA teacher and Gram teacher solve different problems:

| Teacher | How it is formed | What it supplies |
| --- | --- | --- |
| EMA teacher | Moving average of the current student | Current DINO and iBOT targets |
| Gram teacher | Earlier checkpoint with strong dense features | Historical patch-relation anchor |
| Distillation teacher | Frozen, fully trained large model | Targets for a smaller deployment model |

The current EMA teacher follows the same training trajectory that is losing locality. It cannot act as a historical reference unless an earlier version is retained.

The DINOv3 ablation compared Gram teachers from 100k, 200k, and 1M iterations. The 100k and 200k teachers performed similarly, while the 1M teacher was worse because its patch consistency had already degraded. The quality of the anchor matters.

## High-resolution Gram targets

DINOv3 improves the Gram target further:

1. run the Gram teacher at twice the student's crop resolution;
2. obtain its finer patch-feature grid;
3. bicubically downsample the feature grid to the student's size;
4. compute the teacher Gram matrix from the smoothed, downsampled features.

The higher-resolution pass captures finer structure. Downsampling smooths isolated patch noise while retaining coherent spatial regions. The student learns this improved relational structure without paying the higher-resolution cost during every ordinary backbone path.

DINOv3 later performs a separate 10,000-iteration high-resolution adaptation phase with mixed global resolutions of $512$ and $768$. Gram anchoring remains important in that phase; the paper reports significant dense-performance degradation when it is removed.

## Does Gram anchoring actually help?

In the DINOv3 paper's Gram-teacher ablation, the answer is yes for the tested dense tasks:

| Method | Gram teacher | Teacher resolution | ImageNet linear $\uparrow$ | ADE20K mIoU $\uparrow$ | NYUv2 RMSE $\downarrow$ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Long-training baseline | — | — | **88.2** | 50.3 | 0.307 |
| Gram | 200k | $1\times$ | 88.0 | 53.6 | 0.285 |
| Gram | 200k | $2\times$ | 88.0 | **55.7** | **0.281** |
| Gram | 100k | $2\times$ | 87.9 | **55.7** | 0.284 |
| Gram | 1M | $2\times$ | 88.1 | 54.9 | 0.290 |

Relative to the baseline, the 200k high-resolution anchor gains $5.4$ mIoU on ADE20K and reduces depth RMSE from $0.307$ to $0.281$, while ImageNet linear accuracy changes from $88.2$ to $88.0$.

The paper also reports that dense improvements begin within the first 10,000 refinement iterations. The iBOT loss falls faster after Gram anchoring is enabled, whereas the DINO global loss changes much less. This supports the authors' interpretation that Gram and iBOT both act strongly on local structure while the global DINO objective remains comparatively independent.

So Gram anchoring is not merely producing prettier cosine maps. In the reported evaluation it recovers substantial segmentation and depth quality with little change to global classification.

### What Gram anchoring does not guarantee

The result should still be interpreted carefully:

- it preserves relationships supplied by the teacher, including any mistakes in them;
- it needs a checkpoint with good local structure;
- the $P\times P$ Gram matrix adds quadratic work in the number of patches;
- it does not assign semantic names to regions;
- it does not prove that every dense task or domain improves equally;
- it was needed for the long-trained 7B trajectory, but the paper did not observe the same patch-consistency issue while distilling smaller models from a fixed 7B teacher.

It is a targeted regularizer for an observed training pathology, not a universal replacement for DINO or iBOT.

## The complete DINOv3 training story

The flagship model's training can be understood as a sequence of jobs.

### Phase 1: large-scale self-supervised pretraining

For one million iterations:

```text
DINO   -> learn global view-invariant semantics
iBOT   -> learn contextual patch representations
KoLeo  -> keep global embeddings spread out
EMA    -> provide slowly evolving targets
SK     -> keep prototype assignments balanced
```

The student receives gradients; the EMA teacher does not.

### Phase 2: Gram refinement

After long training has produced strong global features but degraded patch geometry:

```text
continue DINO + iBOT + KoLeo
                  +
match student patch Gram matrix
to a strong early-checkpoint Gram matrix
```

This restores local consistency while allowing the representation itself to continue moving.

### Phase 3: high-resolution adaptation

Train for 10,000 additional iterations using mixed crop resolutions so the model can exploit fine detail and remain useful over a range of inference sizes. Gram anchoring protects dense features during this transition.

### Phase 4: distill a family of smaller models

Use the finished 7B model as a fixed teacher for ViT-S through ViT-H+ and ConvNeXt variants. The training retains the DINO/iBOT-style targets, but this is now conventional large-to-small knowledge distillation rather than same-model EMA self-distillation. The paper does not apply Gram anchoring in this stage because it does not observe the local-consistency failure there.

## Pseudocode for the core idea

The real implementation is distributed and heavily optimized, but the conceptual loop is compact:

```python
for images in unlabeled_loader:
    global_views, local_views = make_multicrop_views(images)
    masked_globals, mask = mask_random_patches(global_views)

    # No gradient flows through either teacher path.
    with no_grad():
        teacher_global = ema_teacher(global_views)
        teacher_gram = gram_teacher(high_resolution(global_views))

    student_global = student(masked_globals)
    student_local = student(local_views)

    loss_dino = cross_view_cls_loss(
        student_global,
        student_local,
        teacher_global,
    )
    loss_ibot = masked_patch_loss(
        student_global,
        teacher_global,
        mask,
    )
    loss_koleo = feature_spreading_loss(student_global.cls_token)

    loss = loss_dino + loss_ibot + 0.1 * loss_koleo

    if gram_refinement_is_active:
        loss += 2.0 * gram_relation_loss(
            student_global.patch_tokens,
            teacher_gram.patch_tokens,
        )

    update_student(loss)
    update_ema_teacher_from_student()
```

The Gram-teacher branch is absent during ordinary DINOv2 training and the initial DINOv3 phase. It appears only during DINOv3's refinement and relevant high-resolution adaptation.

## The main idea to remember

DINO does not discover semantics from a single mysterious loss. It combines complementary constraints:

1. **DINO:** make different views of one image agree globally.
2. **iBOT:** predict the teacher's local representation where the student is masked.
3. **KoLeo:** stop global embeddings from crowding into a small region.
4. **Sinkhorn–Knopp and teacher sharpening:** prevent trivial prototype assignments.
5. **EMA teacher:** turn the student's history into a stable target.
6. **DINOv3 Gram anchoring:** preserve the pairwise geometry of good patch features when long training begins to erase locality.

No human says which image contains a dog or which pixels belong to it. The supervision comes from agreement across views, prediction across missing information, diversity across samples, and—in DINOv3—consistency of spatial relationships across training time.

The most interesting lesson from DINOv3 is broader than its new loss:

> Better global representations do not automatically imply better local representations. If a foundation model must serve both, the training objective has to protect both.

## References

- Maxime Oquab et al., [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193), TMLR 2024, and the [official DINOv2 repository](https://github.com/facebookresearch/dinov2).
- Oriane Siméoni et al., [DINOv3](https://arxiv.org/abs/2508.10104), 2025, and the [official DINOv3 repository](https://github.com/facebookresearch/dinov3).
- Mathilde Caron et al., [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294), ICCV 2021.
- Jinghao Zhou et al., [iBOT: Image BERT Pre-Training with Online Tokenizer](https://arxiv.org/abs/2111.07832), ICLR 2022.
- Mathilde Caron et al., [Unsupervised Learning of Visual Features by Contrasting Cluster Assignments](https://arxiv.org/abs/2006.09882), NeurIPS 2020.
- Alexandre Sablayrolles et al., [Spreading Vectors for Similarity Search](https://arxiv.org/abs/1806.03198), ICLR 2019.
