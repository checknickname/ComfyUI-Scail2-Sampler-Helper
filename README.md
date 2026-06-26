# ComfyUI-Scail2-Sampler-Helper

SCAIL-2 视频生成的采样辅助节点，包括分块循环采样、关键帧采样、Prompt 增强等。

所有节点均为实验性 (`is_experimental=True`)，V3 API。

---

## 节点列表

### 1. SCAIL2LoopSampler
分块视频生成节点。每次调用生成一个 chunk，通过输入/输出 `video_frame_offset` 和 `trim_pixel_frames` 在工作流中串联多个 chunk。

#### 输入
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model | MODEL | - | SCAIL-2 模型 |
| positive | CONDITIONING | - | 正向条件 |
| negative | CONDITIONING | - | 负向条件 |
| vae | VAE | - | VAE 模型 |
| width / height | INT | 512 / 896 | 生成分辨率 |
| length | INT | 81 | 该 chunk 的像素帧数 (step=4) |
| batch_size | INT | 1 | 批大小 |
| pose_video | IMAGE | 可选 | 姿态驱动视频，会被降采样到半分辨率 |
| pose_video_mask | IMAGE | 可选 | SAM3 彩色身份遮罩视频 |
| replacement_mode | BOOLEAN | False | False=动画模式, True=替换模式 |
| pose_strength | FLOAT | 1.0 | 姿态潜变量强度 |
| pose_start / pose_end | FLOAT | 0.0 / 1.0 | 姿态条件生效的采样步百分比区间 |
| reference_image | IMAGE | 可选 | 参考图像 |
| reference_image_mask | IMAGE | 可选 | 参考图像彩色遮罩 |
| clip_vision_output | CLIP_VISION_OUTPUT | 可选 | CLIP 视觉特征 |
| video_frame_offset | INT | 0 | 累积帧偏移，从上个 chunk 串联 |
| previous_frame_count | INT | 5 | 用于锚定的 previous_frames 尾部帧数 (step=4) |
| previous_frames | IMAGE | 可选 | 上个 chunk 解码后的输出 |
| previous_frame_max_noise | FLOAT | 0.425 | 锚定帧的 noise_mask 上限 (0=冻结, 1=自由) |
| last_prev_dynamic_noise | BOOLEAN | True | 对锚定末尾帧做动态 noise shift，消除硬阶跃色块 |
| seed | INT | 0 | 随机种子 |
| steps | INT | 4 | 采样步数 |
| cfg | FLOAT | 1.0 | CFG 引导强度 |
| sampler_name / scheduler | COMBO | euler / normal | 采样器与调度器 |
| denoise | FLOAT | 1.0 | 去噪强度 |

#### 输出
| 输出 | 类型 | 说明 |
|------|------|------|
| latent | LATENT | 包含生成帧和 noise_mask 的潜变量输出 |
| video_frame_offset | INT | 调整后的帧偏移 + length，传给下一个 chunk |
| trim_pixel_frames | INT | 与上一个 chunk 重叠的像素帧数，用于裁切拼接 |

#### 串联方式
1. 第一个 chunk 的 `video_frame_offset=0`，`previous_frames` 留空。
2. 后续 chunk 将上一个的 `latent` 解码后的图像作为 `previous_frames`，将 `video_frame_offset` 输出连接到下一个的 `video_frame_offset` 输入。
3. 最终拼接时每个 chunk 输出裁掉前 `trim_pixel_frames` 帧。

#### last_prev_dynamic_noise 说明
开启后，锚定段的末尾帧不再固定冻结（`noise_mask = 0`），而是随采样步数动态变化：
- 早期步（sigma 大）：接近冻结，与锚定段无缝衔接。
- 后期步（sigma 小）：逐渐放开到 `previous_frame_max_noise`，与生成段自然融合。

从而消除锚定段与生成段边界的硬阶跃色块。

---

### 2. SCAIL2KeyFrameSampler
关键帧视频采样器。单次采样整段视频，pose 帧非均匀选取——chunk 首尾使用全密度 pose（边界段），中间使用大步长 pose（稀疏段），以减少计算量同时保持边界质量。

#### 额外参数
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| length | INT | 243 | 总像素帧数 |
| chunk_length | INT | 81 | 每个 chunk 的帧数，用于边界/中间段划分 |
| boundary_start | INT | 4 | chunk 头部的全密度 pose 帧数 (第一个 chunk 跳过) |
| boundary_end | INT | 4 | chunk 尾部的全密度 pose 帧数 (最后一个 chunk 跳过) |
| pose_stride | INT | 4 | 中间段的 pose 帧步长 (1=全密度) |

#### 输出
- `latent`：生成潜变量
- `boundary_indices`：JSON 字符串，每个 chunk 边界帧的索引，供 SCAIL2KeyFrameSelector 提取首尾帧

---

### 3. SCAIL2KeyFrameSelector
从 SCAIL2KeyFrameSampler 的输出中提取指定 chunk 的首尾边界帧。

#### 输入
| 参数 | 类型 | 说明 |
|------|------|------|
| images | IMAGE | SCAIL2KeyFrameSampler 的解码图像批 |
| boundary_indices | STRING | SCAIL2KeyFrameSampler 的 boundary_indices 输出 |
| chunk_index | INT | 要提取边界的 chunk 索引 |
| start_count | INT | 从起始边界取的最大帧数 |
| end_count | INT | 从结束边界取的最大帧数 |

#### 输出
- `start_images`：chunk 首部边界帧
- `end_images`：chunk 尾部边界帧

---

### 4. SCAIL2PromptEnhancer
基于 VLM (视觉语言模型) 的 Prompt 增强器。从源视频采样帧，自动生成视频描述，再结合参考图像生成增强后的替换/动画 Prompt。

#### 输入
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| reference_image | IMAGE | 可选 | 替换角色的参考图像 |
| video_frames | IMAGE | 可选 | 用于 VLM 描述的源视频采样帧 |
| max_frames | INT | 6 | 为 VLM 采样的最大帧数 |
| sample_method | COMBO | uniform | uniform/first_middle_last/middle |
| instruction | STRING | "" | 替换/动画指令，留空则按模式使用默认 |
| caption_instruction | STRING | "" | VLM 提取源视频内容的指令 |
| source_caption | STRING | "" | 预写的源视频描述，留空由 VLM 生成 |
| examples | STRING | "" | few-shot 示例，留空用内置默认 |
| prompt_mode | COMBO | replacement | replacement=角色替换, animation=动作模仿 |
| llm_model | COMBO | - | models/LLM/ 下的 .gguf 模型 |
| mmproj | COMBO | - | mmproj/CLIP 模型文件 |
| chat_handler | COMBO | Qwen3-VL | 模型架构 / chat handler |
| disable_thinking | BOOLEAN | True | 禁用思考/推理模式 |
| n_ctx | INT | 65536 | 上下文长度上限 |
| temperature | FLOAT | 0.4 | 采样温度 |
| seed | INT | 42 | 随机种子 |

#### 输出
- `enhanced_prompt`：增强后的生成 Prompt（英文段落，约 90-140 词）

#### 使用前提
- 安装 `llama-cpp-python`。
- 在 `models/LLM/` 目录放置 `.gguf` 模型和可选的 mmproj 文件。

---

### 5. SCAIL2MultiRefImages
多参考图像输入节点。通过上传组件管理多个参考图像及其对应的全局 pose 下标。

#### 输入
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| images_data | STRING | "[]" | 上传组件管理的内部 JSON |
| width / height | INT | 512/896 | 目标分辨率 |
| resize_method | COMBO | bicubic | 缩放方法 |
| crop | COMBO | center | center/disabled |

#### 输出
- `images`：处理后的参考图像批
- `indices`：每个参考图像对应的全局 pose 下标（JSON 整数数组，与 `SCAIL2LoopSampler.indices` 配合使用）

---

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <repo_url> ComfyUI-Scail2-Sampler-Helper
```

无需额外 Python 依赖。如使用 `SCAIL2PromptEnhancer` 需额外安装 `llama-cpp-python`。

## 依赖的 ComfyUI 内部机制

- `node_helpers.conditioning_set_values` — 注入模型专用 conditioning 键
- `node_helpers.conditioning_set_values_with_timestep_range` — 按采样步区间调度 conditioning
- `comfy.context_windows` — IndexListContextHandler / ContextFuseMethods / ContextSchedules
- `comfy.sample.sample` / `comfy.sample.prepare_noise` — 采样入口
- `comfy.samplers.SAMPLER_NAMES` / `KSampler.SCHEDULERS` — 采样器/调度器选项
- `model.set_model_denoise_mask_function` — 动态 noise_mask 注入 (last_prev_dynamic_noise)

## 技术说明

- VAE 时间压缩系数为 4：1 个 latent 帧 = 4 个像素帧（Causal VAE）。
- SAM3 遮罩使用 28 通道格式：7 种基础颜色 × 4 帧时间堆叠 → 28 通道。
- Pose 视频在 VAE 编码前被降采样到半分辨率 (`width//2, height//2`)。
- 蒸馏 LoRA 推荐 `steps=4~6, cfg=1.0`。
