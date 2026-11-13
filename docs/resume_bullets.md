# Résumé-ready descriptions / 简历项目表述

> Select one version based on the target role; do not paste all versions onto a one-page résumé. All metrics below are synthetic **Hard100**, not RefCOCO. Project created for algorithm research and empirical learning; do not claim publication, production deployment, or real-image benchmark achievements.

## 项目名称

**视觉语言指代表达定位与生成式 Grounding 研究（Visual Grounding Lab）**

技术栈：Python、PyTorch、Transformers、CLIP ViT-B/16、Qwen3-VL-2B-Instruct、PEFT、ms-swift、QLoRA（4-bit）、Transformer Decoder、OpenCV/PIL、IoU/GIoU

### VLM / 多模态算法工程师版（推荐）

- **图文定位模型研究：** 围绕 Referring Expression Grounding 构建从 CLIP Spatial Grounding、Region/Anchor-free 候选框，到自回归坐标生成与 Qwen3-VL 指令微调的 **V1–V8 可复现实验体系**；设计多目标、同色/同形干扰、遮挡及目标尺寸变化的合成测试集，统一使用 Acc@IoU≥0.5 与 Mean IoU 评估。
- **视觉语言空间对齐与定位优化：** 基于冻结 CLIP ViT-B/16 提取 14×14 Patch Tokens，设计文本条件空间打分与 Coarse-to-Fine ROI 残差精修；在 100 条合成 Hard 测试集上将显式定位方案 **Acc@0.5 从 80% 提升至 86%**，小目标子集从 **67.44% 提升至 83.72%**。
- **生成式 Grounding 与 QLoRA：** 实现 Coordinate Tokenization、Teacher Forcing、Causal Decoder 与基于 IoU 的生成式诊断；进一步采用 **Qwen3-VL-2B-Instruct + 4-bit QLoRA**（rank 8，冻结视觉编码器及对齐模块），使用 800 条合成图文指令进行任务适配，将 Hard100 **Acc@0.5 从 48% 提升至 94%（+46 pp）**，Mean IoU **0.4736→0.7547**；完成逐样本错误迁移分析。

### CV / 视觉算法工程师版

- **细粒度定位与空间建模：** 基于 CLIP ViT-B/16 构建文本条件的 Patch-level 空间响应图和单目标 Bounding Box 预测；探索 Region Proposal、Anchor-free 及 Coarse-to-Fine ROI Refinement，对尺度、遮挡及干扰目标进行分组评估。
- **检测头与指标优化：** 在合成 Hard100 测试集上，空间定位模型从 **80%→86% Acc@0.5**，小目标从 **29/43→36/43**；进一步采用冻结空间特征的残差回归头，达到 **87% Acc@0.5**、**0.6323 Mean IoU**，并复盘失败案例与精修负迁移。

### 纯算法 / 研究版

- **受控实验与消融：** 搭建 Referring Expression Grounding 的多架构实验体系，比较连续坐标回归与离散 Token 自回归生成；实现 0–1000 / 0–100 坐标量化、Teacher Forcing、EOS/Coordinate Accuracy 分离、约束解码，以及教师强制与实际自回归 IoU 对照。
- **实验归因与误差分析：** 发现小规模随机初始化 Decoder 虽有训练 Cross-Entropy 收敛，但 Hard100 Acc@0.5 为 **0%～1%**；引入预训练空间特征仍不能直接解决坐标生成问题；通过 Qwen3-VL QLoRA 适配实验验证预训练 VLM 对任务学习的重要性，同时指出预训练规模、结构归纳偏置等混杂因素，避免过度因果推断。

## English résumé version (VLM)

**Visual Grounding Lab — Referring Expression Localization and Generative Grounding Research**

- Developed a reproducible **V1–V8 visual grounding research pipeline** spanning CLIP-based patch-level spatial alignment, region/anchor-free localization, autoregressive coordinate generation, and Qwen3-VL instruction tuning; evaluated robustness to same-color/shape distractors, scale variation, and occlusion on a controlled synthetic test set.
- Designed a text-conditioned spatial prediction head with CLIP ViT-B/16 patch features and coarse-to-fine ROI residual refinement, improving **Acc@IoU≥0.5 from 80% to 86%** on the 100-example synthetic Hard split; small-object accuracy increased from **67.44% to 83.72%**.
- Implemented coordinate-token teacher forcing, causal decoding, and per-coordinate/EOS diagnostics; adapted **Qwen3-VL-2B-Instruct via 4-bit QLoRA** on 800 synthetic multimodal instructions, increasing Hard100 **Acc@0.5 from 48% to 94% (+46 pp)** and Mean IoU **from 0.4736 to 0.7547**, with paired error-transition analysis.

## Interview talking points / 面试表达边界

1. **Why is V7 low while Qwen's adapted model performs well?** V7 trains a small autoregressive decoder from scratch on 800 examples; Qwen3-VL starts with extensive pretrained visual-language and grounding capabilities. Different architecture, pretrained training and optimizer details prevent clean causal attribution.
2. **What is actually trainable?** V6 trains its spatial/refinement heads with frozen CLIP. V7 trains projections/coordinate decoder with frozen CLIP. Qwen experiment uses 4-bit QLoRA adapters with ViT and aligner frozen; the exact injected layer names should be confirmed in ms-swift logs before claiming specific attention matrices.
3. **What does 94% mean?** 94/100 **synthetic Hard100** examples have predicted xyxy-box IoU≥0.5, not COCO mAP or a RefCOCO score. The output JSON and box validity were 100% both pre-/post-LoRA.
4. **Why not conclude generation is inferior?** V7 decoders were randomly initialized and trained on small data, and V7.2 regression is initialized to preserve V6's predicted box. The experimental controls are not symmetric.
5. **Does Qwen search around a proposal?** This project does **not** show that. Qwen's common grounding interface generates coordinate representations via multimodal language modeling, whereas the V6/V7.2 residual pipeline explicitly refines a pre-existing box.
6. **What next?** Evaluate on genuine referring-expression data, check image visibility vs amodal-box labels, run multiple random seeds, and test prompts/cross-domain robustness.

