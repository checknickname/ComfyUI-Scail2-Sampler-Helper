from typing_extensions import override
import math
import json
import os
import torch
import node_helpers
import comfy.utils
import comfy.model_management
import comfy.samplers
import comfy.context_windows
import folder_paths
from comfy_api.latest import ComfyExtension, io
import logging
logger = logging.getLogger("SCAIL2ContextWindowSampler")

def _extract_mask_to_28ch(rgb_video: torch.Tensor) -> torch.Tensor:
    T, H, W, _ = rgb_video.shape
    _ON_THRESH = 225.0 / 255.0
    mask = rgb_video.movedim(-1, 1).float()
    R = (mask[:, 0:1] > _ON_THRESH).float()
    G = (mask[:, 1:2] > _ON_THRESH).float()
    B = (mask[:, 2:3] > _ON_THRESH).float()
    nR, nG, nB = 1 - R, 1 - G, 1 - B
    binary_7ch = torch.cat([
        R * G * B, R * nG * nB, nR * G * nB,
        nR * nG * B, R * G * nB, R * nG * B, nR * G * B,
    ], dim=1)
    H_lat, W_lat = H, W
    for _ in range(3):
        H_lat = (H_lat + 1) // 2
        W_lat = (W_lat + 1) // 2
    binary_7ch = torch.nn.functional.interpolate(binary_7ch, size=(H_lat, W_lat), mode='area')
    T_latent = (T - 1) // 4 + 1
    padded = torch.cat([binary_7ch[:1].repeat(4, 1, 1, 1), binary_7ch[1:]], dim=0)
    expected = T_latent * 4
    if padded.shape[0] != expected:
        if padded.shape[0] > expected:
            padded = padded[:expected]
        else:
            padded = torch.cat([padded, padded[-1:].repeat(expected - padded.shape[0], 1, 1, 1)], dim=0)
    out = padded.view(T_latent, 28, H_lat, W_lat)
    return out.unsqueeze(0)


class SCAIL2LoopSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2LoopSampler",
            category="sampling",
            inputs=[
                io.Model.Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Vae.Input("vae"),
                io.Int.Input("width", default=512, min=32, max=8192, step=32),
                io.Int.Input("height", default=896, min=32, max=8192, step=32),
                io.Int.Input("length", default=81, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("pose_video", optional=True,
                    tooltip="\u7528\u4e8e\u59ff\u52bf\u6761\u4ef6\u7684\u89c6\u9891\uff0c\u4f1a\u88ab\u964d\u91c7\u6837\u5230\u534a\u5206\u8fa8\u7387\u3002"),
                io.Image.Input("pose_video_mask", optional=True,
                    tooltip="\u6309\u8eab\u4efd\u7740\u8272\u7684 SAM3 \u906e\u7f69\u89c6\u9891\u3002"),
                io.Boolean.Input("replacement_mode", default=False, optional=True),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Image.Input("reference_image", optional=True),
                io.Image.Input("reference_image_mask", optional=True,
                    tooltip="\u4e0e reference_image \u540c\u5206\u8fa8\u7387\u7684\u5f69\u8272\u53c2\u8003\u906e\u7f69\u3002"),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Int.Input("video_frame_offset", default=0, min=0, max=1048576, step=1,
                    tooltip="\u7d2f\u8ba1\u5e27\u504f\u79fb\u3002\u4ece\u4e0a\u4e00\u4e2a chunk \u8fde\u63a5\u800c\u6765\u3002"),
                io.Int.Input("previous_frame_count", default=5, min=1, max=4096, step=4,
                    tooltip="\u7528\u4e8e\u951a\u5b9a\u7684 previous_frames \u5c3e\u90e8\u5e27\u6570\u3002"),
                io.Image.Input("previous_frames", optional=True,
                    tooltip="\u4e0a\u4e00\u4e2a chunk \u5b8c\u6574\u89e3\u7801\u540e\u7684\u8f93\u51fa\u3002"),
                io.Float.Input("previous_frame_max_noise", default=0.425, min=0.0, max=1.0, step=0.001,
                    tooltip="previous_frame \u5e27\u7684 noise_mask \u4e0a\u9650\uff080=\u51bb\u7ed3\uff0c1=\u81ea\u7531\uff09\u3002"),
                io.Boolean.Input("last_prev_dynamic_noise", default=True, optional=True,
                    tooltip="\u5bf9 previous_frame \u672b\u5c3e\u5e27\u52a8\u6001 noise_mask\uff0c\u6d88\u9664\u786c\u9636\u8dc3\u5f15\u8d77\u7684\u8272\u5757\u3002"),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.fixed),
                io.Int.Input("steps", default=4, min=1, max=10000),
                io.Float.Input("cfg", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal"),
                io.Float.Input("denoise", default=1.00, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="video_frame_offset"),
                io.Int.Output(display_name="trim_pixel_frames"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, positive, negative, vae,
                width=512, height=896, length=81, batch_size=1,
                pose_video=None, pose_video_mask=None,
                replacement_mode=False, pose_strength=1.0, pose_start=0.0, pose_end=1.0,
                reference_image=None, reference_image_mask=None,
                clip_vision_output=None,
                video_frame_offset=0, previous_frame_count=5,
                previous_frames=None,
                previous_frame_max_noise=0.0, last_prev_dynamic_noise=False,
                seed=0, steps=20, cfg=8.0,
                sampler_name="euler", scheduler="normal", denoise=1.0) -> io.NodeOutput:
        import comfy.sample

        if video_frame_offset is None:
            video_frame_offset = 0

        T_lat = ((length - 1) // 4) + 1
        H_lat = height // 8
        W_lat = width // 8

        ref_mask_flag = not replacement_mode
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_flag": ref_mask_flag})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_flag": ref_mask_flag})

        prev_trimmed = None
        trim_pixel_frames = 0
        prev_latent_frames = 0
        encoded_prev_latent = None

        if previous_frames is not None and previous_frames.shape[0] > 0:
            prev_trimmed = previous_frames[-previous_frame_count:]
            video_frame_offset -= prev_trimmed.shape[0]
            video_frame_offset = max(0, video_frame_offset)
            trim_pixel_frames = previous_frame_count

        total_pose_pixel = (T_lat - 1) * 4 + 1

        if prev_trimmed is not None:
            pf = comfy.utils.common_upscale(prev_trimmed.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            encoded_prev_latent = vae.encode(pf[:, :, :, :3])
            prev_latent_frames = min(encoded_prev_latent.shape[2], T_lat)

        latent = torch.zeros([batch_size, 16, T_lat, H_lat, W_lat],
                             device=comfy.model_management.intermediate_device())
        noise_mask = None

        if encoded_prev_latent is not None:
            latent[:, :, :prev_latent_frames] = encoded_prev_latent[:, :, :prev_latent_frames].to(device=latent.device, dtype=latent.dtype)
            noise_mask = torch.ones((1, 1, latent.shape[2], latent.shape[-2], latent.shape[-1]), device=latent.device, dtype=latent.dtype)
            noise_mask[:, :, :prev_latent_frames] = previous_frame_max_noise
            
        if reference_image is not None:
            ref_imgs = comfy.utils.common_upscale(reference_image.movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            n_ref = ref_imgs.shape[0]
            if replacement_mode and reference_image_mask is not None:
                rm = comfy.utils.common_upscale(reference_image_mask.movedim(-1, 1), width, height, "nearest-exact", "center").movedim(1, -1)
                rm = rm[[min(i, rm.shape[0] - 1) for i in range(n_ref)]]
                is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_imgs.dtype)
                ref_imgs = ref_imgs * is_char
            ref_latents = [vae.encode(ref_imgs[i:i + 1, :, :, :3]) for i in range(n_ref)]
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": ref_latents}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": ref_latents}, append=True)

        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        if pose_video is not None and pose_video.shape[0] <= video_frame_offset:
            pose_video = None
        elif pose_video is not None:
            pose_video = pose_video[video_frame_offset:]
        if pose_video_mask is not None and pose_video_mask.shape[0] <= video_frame_offset:
            pose_video_mask = None
        elif pose_video_mask is not None:
            pose_video_mask = pose_video_mask[video_frame_offset:]

        if reference_image_mask is not None and reference_image is not None:
            ref_mask_hw = comfy.utils.common_upscale(reference_image_mask.movedim(-1, 1), width, height, "nearest-exact", "center").movedim(1, -1)
            n_masks = ref_mask_hw.shape[0]
            n_ref = reference_image.shape[0]
            add_masks = [_extract_mask_to_28ch(ref_mask_hw[min(i, n_masks - 1)][None]) for i in range(1, n_ref)]
            ref_mask_1f = _extract_mask_to_28ch(ref_mask_hw[:1])
            mask_t = latent.shape[2]
            zeros_count = mask_t - ref_mask_1f.shape[1] + 1
            if zeros_count > 0:
                zeros = torch.zeros((1, zeros_count, 28, ref_mask_1f.shape[-2], ref_mask_1f.shape[-1]),
                                    device=ref_mask_1f.device, dtype=ref_mask_1f.dtype)
                ref_mask_28ch = torch.cat(add_masks + [ref_mask_1f, zeros], dim=1)
            else:
                ref_mask_28ch = torch.cat(add_masks + [ref_mask_1f], dim=1)
            positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_28ch})

        if pose_video is not None and pose_video.shape[0] < total_pose_pixel:
            pad = pose_video[-1:].repeat(total_pose_pixel - pose_video.shape[0], 1, 1, 1)
            pose_video = torch.cat([pose_video, pad], dim=0)
        if pose_video_mask is not None and pose_video_mask.shape[0] < total_pose_pixel:
            pad = pose_video_mask[-1:].repeat(total_pose_pixel - pose_video_mask.shape[0], 1, 1, 1)
            pose_video_mask = torch.cat([pose_video_mask, pad], dim=0)

        ts = [v.shape[0] for v in (pose_video, pose_video_mask) if v is not None]
        if ts:
            T_kept = ((min(min(ts), total_pose_pixel) - 1) // 4) * 4 + 1
            if pose_video is not None:
                pose_video = pose_video[:T_kept]
            if pose_video_mask is not None:
                pose_video_mask = pose_video_mask[:T_kept]

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(pose_video[:total_pose_pixel].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            pose_video_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
            positive = node_helpers.conditioning_set_values_with_timestep_range(positive, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)
            negative = node_helpers.conditioning_set_values_with_timestep_range(negative, {"pose_video_latent": pose_video_latent}, pose_start, pose_end)

        if pose_video_mask is not None:
            mask_video_hw = comfy.utils.common_upscale(pose_video_mask[:total_pose_pixel].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            driving_mask_28ch = _extract_mask_to_28ch(mask_video_hw)
            positive = node_helpers.conditioning_set_values(positive, {"driving_mask_28ch": driving_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"driving_mask_28ch": driving_mask_28ch})

        pbar = comfy.utils.ProgressBar(steps)

        def progress_callback(step, denoised, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        def px_to_lat(p):
            return 0 if p <= 0 else (p - 1) // 4 + 1

        B, C, T, Hl, Wl = latent.shape
        seed_mask = 0xffffffffffffffff
        noise = torch.empty(latent.shape, dtype=torch.float32, layout=latent.layout, device="cpu")
        for j in range(T):
            gp = video_frame_offset if j == 0 else video_frame_offset + 1 + 4 * (j - 1)
            glat = px_to_lat(gp)
            gen = torch.manual_seed((seed + glat) & seed_mask)
            noise[:, :, j] = torch.randn([B, C, Hl, Wl], generator=gen, device="cpu")
        noise = noise.to(dtype=latent.dtype)
        sample_seed = (seed + px_to_lat(video_frame_offset)) & seed_mask

        dyn_model = model
        if last_prev_dynamic_noise and encoded_prev_latent is not None and prev_latent_frames > 0:

            def _noise_fn(sigma, denoise_mask, extra_options={}):
                sigmas = extra_options.get("sigmas")
                if sigmas is not None and len(sigmas) > 0:
                    progress = previous_frame_max_noise * (1.0 - sigma / sigmas[0])
                else:
                    progress = 1.0
                boundary_idx = prev_latent_frames - 1
                if boundary_idx >= 0:
                    denoise_mask[:, :, boundary_idx, :, :] = progress
                return denoise_mask

            dyn_model = model.clone()
            dyn_model.set_model_denoise_mask_function(_noise_fn)

        samples = comfy.sample.sample(
            model=dyn_model,
            noise=noise,
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent,
            denoise=denoise,
            noise_mask=noise_mask,
            seed=sample_seed,
            callback=progress_callback,
        )

        out = {"samples": samples}
        if noise_mask is not None:
            out["noise_mask"] = noise_mask

        return io.NodeOutput(out, video_frame_offset + length, trim_pixel_frames)


class SCAIL2KeyFrameSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2KeyFrameSampler",
            category="sampling",
            inputs=[
                io.Model.Input("model"),
                io.Vae.Input("vae"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("negative"),
                io.Int.Input("width", default=512, min=32, max=8192, step=32),
                io.Int.Input("height", default=896, min=32, max=8192, step=32),
                io.Int.Input("length", default=243, min=1, max=4096, step=4),
                io.Int.Input("batch_size", default=1, min=1, max=4096),
                io.Image.Input("pose_video", optional=True,
                    tooltip="\u7528\u4e8e\u59ff\u52bf\u6761\u4ef6\u7684\u89c6\u9891\uff0c\u4f1a\u88ab\u964d\u91c7\u6837\u5230\u534a\u5206\u8fa8\u7387\u3002"),
                io.Image.Input("pose_video_mask", optional=True,
                    tooltip="\u6309\u8eab\u4efd\u7740\u8272\u7684 SAM3 \u906e\u7f69\u89c6\u9891\u3002"),
                io.Int.Input("chunk_length", default=81, min=4, max=4096, step=4,
                    tooltip="\u6bcf\u4e2a chunk \u7684\u5e27\u6570\uff0c\u7528\u4e8e\u8fb9\u754c/\u4e2d\u95f4\u6bb5\u5212\u5206\u3002"),
                io.Int.Input("boundary_start", default=4, min=4, max=4096, step=4,
                    tooltip="chunk \u5934\u90e8\u7684\u5168\u5bc6\u5ea6 pose \u5e27\uff08\u7b2c\u4e00\u4e2a chunk \u8df3\u8fc7\uff09\u3002"),
                io.Int.Input("boundary_end", default=4, min=4, max=4096, step=4,
                    tooltip="chunk \u5c3e\u90e8\u7684\u5168\u5bc6\u5ea6 pose \u5e27\uff08\u6700\u540e\u4e00\u4e2a chunk \u8df3\u8fc7\uff09\u3002"),
                io.Int.Input("pose_stride", default=4, min=1, max=256, step=1,
                    tooltip="\u4e2d\u95f4\u6bb5\u7684 pose \u5e27\u6b65\u957f\u30021=\u5168\u5bc6\u5ea6\u3002"),
                io.Boolean.Input("replacement_mode", default=False, optional=True),
                io.Float.Input("pose_strength", default=1.0, min=0.0, max=10.0, step=0.01),
                io.Float.Input("pose_start", default=0.0, min=0.0, max=1.0, step=0.01),
                io.Float.Input("pose_end", default=1.0, min=0.0, max=1.0, step=0.01),
                io.Image.Input("reference_image", optional=True),
                io.Image.Input("reference_image_mask", optional=True,
                    tooltip="\u4e0e reference_image \u540c\u5206\u8fa8\u7387\u7684\u5f69\u8272\u53c2\u8003\u906e\u7f69\u3002"),
                io.ClipVisionOutput.Input("clip_vision_output", optional=True),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.randomize),
                io.Int.Input("steps", default=20, min=1, max=10000),
                io.Float.Input("cfg", default=8.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("sampler_name", options=comfy.samplers.SAMPLER_NAMES, default="euler"),
                io.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="normal"),
                io.Float.Input("denoise", default=1.0, min=0.0, max=1.0, step=0.01),
            ],
            outputs=[
                io.Latent.Output(display_name="latent"),
                io.String.Output("boundary_indices"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, vae, positive, negative,
                width=512, height=896, length=243, batch_size=1,
                pose_video=None, pose_video_mask=None,
                chunk_length=81, boundary_start=5, boundary_end=4, pose_stride=4,
                replacement_mode=False, pose_strength=1.0, pose_start=0.0, pose_end=1.0,
                reference_image=None, reference_image_mask=None,
                clip_vision_output=None,
                seed=0, steps=20, cfg=8.0,
                sampler_name="euler", scheduler="normal", denoise=1.0) -> io.NodeOutput:
        import comfy.sample

        device = comfy.model_management.get_torch_device()

        # ---- ref_mask_flag ----
        ref_mask_flag = not replacement_mode
        positive = node_helpers.conditioning_set_values(positive, {"ref_mask_flag": ref_mask_flag})
        negative = node_helpers.conditioning_set_values(negative, {"ref_mask_flag": ref_mask_flag})

        # ---- reference image ----
        if reference_image is not None:
            ref_upscaled = comfy.utils.common_upscale(reference_image[:1].movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            if replacement_mode and reference_image_mask is not None:
                rm = comfy.utils.common_upscale(reference_image_mask[:1].movedim(-1, 1), width, height, "nearest-exact", "center").movedim(1, -1)
                is_char = (rm[..., :3].max(dim=-1, keepdim=True).values > 0.1).to(ref_upscaled.dtype)
                ref_upscaled = ref_upscaled * is_char
            ref_latent = vae.encode(ref_upscaled[:, :, :, :3])
            positive = node_helpers.conditioning_set_values(positive, {"reference_latents": [ref_latent]}, append=True)
            negative = node_helpers.conditioning_set_values(negative, {"reference_latents": [ref_latent]}, append=True)

        # ---- clip vision ----
        if clip_vision_output is not None:
            positive = node_helpers.conditioning_set_values(positive, {"clip_vision_output": clip_vision_output})
            negative = node_helpers.conditioning_set_values(negative, {"clip_vision_output": clip_vision_output})

        # ---- pose frame selection ----
        boundary_indices = []
        if pose_video is not None:
            total = pose_video.shape[0]
            num_chunks = (total + chunk_length - 1) // chunk_length
            kept_indices = []

            for ci in range(num_chunks):
                c_s = ci * chunk_length
                c_e = min(c_s + chunk_length, total)
                chunk_boundaries = []

                if ci > 0 and boundary_start > 0:
                    e = min(c_s + boundary_start, c_e)
                    b_s = len(kept_indices)
                    kept_indices.extend(range(c_s, e))
                    chunk_boundaries.append([b_s, len(kept_indices)])

                m_s = c_s + (boundary_start if ci > 0 else 0)
                m_e = c_e - (boundary_end if ci < num_chunks - 1 else 0)
                if m_e > m_s:
                    kept_indices.extend(range(m_s, m_e, pose_stride))

                if ci < num_chunks - 1 and boundary_end > 0:
                    s = max(c_e - boundary_end, c_s)
                    b_s = len(kept_indices)
                    kept_indices.extend(range(s, c_e))
                    chunk_boundaries.append([b_s, len(kept_indices)])

                boundary_indices.append(chunk_boundaries)

            pose_video = pose_video[kept_indices]
            if pose_video_mask is not None:
                pose_video_mask = pose_video_mask[kept_indices]
            length = len(kept_indices)

        T_lat = ((length - 1) // 4) + 1
        samples = torch.zeros([batch_size, 16, T_lat, height // 8, width // 8],
                              device=comfy.model_management.intermediate_device())
        noise_mask = None

        if pose_video is not None:
            pose_video = comfy.utils.common_upscale(pose_video[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            pose_latent = vae.encode(pose_video[:, :, :, :3]) * pose_strength
            positive = node_helpers.conditioning_set_values_with_timestep_range(positive, {"pose_video_latent": pose_latent}, pose_start, pose_end)
            negative = node_helpers.conditioning_set_values_with_timestep_range(negative, {"pose_video_latent": pose_latent}, pose_start, pose_end)

        if pose_video_mask is not None:
            mask_video_hw = comfy.utils.common_upscale(pose_video_mask[:length].movedim(-1, 1), width // 2, height // 2, "area", "center").movedim(1, -1)
            driving_mask_28ch = _extract_mask_to_28ch(mask_video_hw)
            positive = node_helpers.conditioning_set_values(positive, {"driving_mask_28ch": driving_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"driving_mask_28ch": driving_mask_28ch})

        # ---- reference image mask ----
        if reference_image_mask is not None:
            ref_mask_hw = comfy.utils.common_upscale(reference_image_mask[:1].movedim(-1, 1), width, height, "bicubic", "center").movedim(1, -1)
            ref_mask_1f = _extract_mask_to_28ch(ref_mask_hw)
            zeros = torch.zeros((1, samples.shape[2], 28, ref_mask_1f.shape[-2], ref_mask_1f.shape[-1]), device=ref_mask_1f.device, dtype=ref_mask_1f.dtype)
            ref_mask_28ch = torch.cat([ref_mask_1f, zeros], dim=1)
            positive = node_helpers.conditioning_set_values(positive, {"ref_mask_28ch": ref_mask_28ch})
            negative = node_helpers.conditioning_set_values(negative, {"ref_mask_28ch": ref_mask_28ch})

        latent_image = samples

        pbar = comfy.utils.ProgressBar(steps)

        def _callback(step, denoised, x, total_steps):
            pbar.update_absolute(step + 1, total_steps)

        samples = comfy.sample.sample(
            model=model,
            noise=comfy.sample.prepare_noise(latent_image, seed, None),
            steps=steps,
            cfg=cfg,
            sampler_name=sampler_name,
            scheduler=scheduler,
            positive=positive,
            negative=negative,
            latent_image=latent_image,
            denoise=denoise,
            noise_mask=noise_mask,
            seed=seed,
            callback=_callback,
        )

        out = {"samples": samples}
        if noise_mask is not None:
            out["noise_mask"] = noise_mask
        return io.NodeOutput(out, json.dumps(boundary_indices))


class SCAIL2KeyFrameSelector(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2KeyFrameSelector",
            category="sampling",
            inputs=[
                io.Image.Input("images", tooltip="\u6765\u81ea SCAIL2KeyFrameSampler \u7684\u89e3\u7801\u56fe\u50cf\u6279\u6b21\u3002"),
                io.String.Input("boundary_indices",
                    tooltip="\u6765\u81ea SCAIL2KeyFrameSampler \u7684 JSON boundary_indices \u8f93\u51fa\u3002"),
                io.Int.Input("chunk_index", default=0, min=0, max=1048576, step=1,
                    tooltip="\u8981\u63d0\u53d6\u8fb9\u754c\u7684 chunk \u7d22\u5f15\u3002"),
                io.Int.Input("start_count", default=4, min=4, max=4096, step=4,
                    tooltip="\u4ece\u8d77\u59cb\u8fb9\u754c\u53d6\u7684\u6700\u5927\u5e27\u6570\u3002999=\u5168\u90e8\u53ef\u7528\u3002"),
                io.Int.Input("end_count", default=4, min=4, max=4096, step=4,
                    tooltip="\u4ece\u7ed3\u675f\u8fb9\u754c\u53d6\u7684\u6700\u5927\u5e27\u6570\u3002999=\u5168\u90e8\u53ef\u7528\u3002"),
            ],
            outputs=[
                io.Image.Output("start_images"),
                io.Image.Output("end_images"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, images, boundary_indices, chunk_index=0,
                start_count=999, end_count=999) -> io.NodeOutput:
        boundaries = json.loads(boundary_indices)
        if chunk_index >= len(boundaries):
            return io.NodeOutput(None, None)

        start_images = None
        end_images = None

        if chunk_index > 0:
            prev_chunk = boundaries[chunk_index - 1]
            if prev_chunk:
                ps, pe = prev_chunk[-1]
                start_images = images[ps:ps + min(start_count, pe - ps)]

        if chunk_index + 1 < len(boundaries):
            next_chunk = boundaries[chunk_index + 1]
            if next_chunk:
                ns, ne = next_chunk[0]
                end_images = images[ns:ns + min(end_count, ne - ns)]

        return io.NodeOutput(start_images, end_images)


def _get_llm_files(kind):
    try:
        if "LLM" not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path("LLM", os.path.join(folder_paths.models_dir, "LLM"))
        all_files = folder_paths.get_filename_list("LLM")
        if kind == "mmproj":
            return [""] + [f for f in all_files if "mmproj" in f.lower()]
        models = [f for f in all_files if "mmproj" not in f.lower() and f.endswith(".gguf")]
        return models if models else [""]
    except Exception:
        return [""] if kind == "model" else [""]


VIDEO_CAPTION_PROMPT = """You are captioning sampled frames from a source video for a character replacement video generation task.

Describe the source video in one detailed English paragraph. Focus on:
- the scene, location, lighting, camera framing, and background;
- the action, motion, timing, and camera movement across the sampled frames;
- the clothing, pose, body motion, and nearby objects touched or interacted with by the person/character being replaced.

If the user specifies who should be replaced, identify that source subject clearly in the caption. Pay special attention to the source subject's clothing and any objects they hold, touch, operate, sit on, stand near, or otherwise interact with, because those details help locate the replacement region.
Do not mention the replacement target image. Do not invent an identity for the replacement target.
Output only the source-video caption."""

REPLACEMENT_TEMPLATE = """You are a prompt enhancer for SCAIL-2 character replacement.

Your task is to write one detailed English description of the final replaced video. This is not an editing instruction. The output must describe the video AFTER replacement has already happened: the replacement character from the reference image is performing the source subject's motion in the source scene.

Replacement instruction from user:
{instruction}

Source video caption:
{caption}

Few-shot examples of the desired prompt style:
{examples}

Rules:
1. Output a positive video-generation prompt describing the replaced video itself. Do not output wording like "replace X with Y", "swap", "edit", or "the task is". Do not output thinking, reasoning, or analysis steps.
2. Remove the original source subject's identity and appearance. Keep only the original subject's motion, pose, timing, spatial position, and interaction with the scene.
3. The final prompt should describe the replacement character's visible clothing and appearance in enough detail, using the reference image only for identity, wardrobe and appearance details; IGNORE the reference person's pose and actions - all motion, pose and gestures come from the source video.
4. The final prompt should also describe important objects the character interacts with or stays close to in the source video, such as tools, instruments, furniture, vehicles, doors, tables, handheld items, or work surfaces.
5. Keep the original video environment, lighting, camera angle, shot scale, background objects, and motion trajectory.
6. If the source caption mentions the original subject's clothing only to locate body regions or interactions, translate those grounding details into the replacement character's final appearance instead of preserving the original identity.
7. Use natural video wording with concrete verbs. Avoid mentioning masks, segmentation, editing software, or the prompt generation process.
8. Output only the final enhanced prompt, in one English paragraph, around 90-140 words."""

ANIMATION_TEMPLATE = """You are a prompt enhancer for SCAIL-2 motion imitation.

Your task is to write one detailed English description of the final video where the person in the reference image performs the motion described in the source video. This is not an editing instruction. Describe the final video as if it already exists.

Reference instruction:
{instruction}

Source video description:
{caption}

Few-shot examples of the desired prompt style:
{examples}

Rules:
1. Output a positive video-generation prompt describing the final video. Do not output thinking, reasoning, or analysis steps. Describe the reference person's appearance, clothing, identity and setting based on the reference image, but IGNORE the reference person's pose and actions - all motion, pose and gestures come only from the source video.
2. Describe the exact motion, action, timing, and physical movements from the source video that the reference person should perform.
3. Place the reference person in their own setting from the reference image (or a clean, simple background if the reference shows none). Do NOT reuse the source video's scene or location; take only the motion from the source video.
4. Use natural video wording with concrete verbs. Do not mention masks, segmentation, editing software, or the prompt generation process.
5. Output only the final enhanced prompt, in one English paragraph, around 90-140 words."""


REPLACEMENT_EXAMPLES = """Example 1:
A young woman with long black hair, wearing a fitted red leather jacket and dark jeans, walks briskly along a rain-slicked city street at night. Neon signs cast pink and blue reflections across the wet pavement as she weaves between pedestrians, one hand gripping the strap of a small crossbody bag. The handheld camera tracks her from a low three-quarter angle, keeping the glowing storefronts and passing headlights in the background. Her boots splash through shallow puddles in time with her confident stride.

Example 2:
A bearded man in a worn olive workshop apron over a grey henley shirt leans over a wooden workbench, carefully sanding the curved edge of a guitar body. Warm tungsten light from an overhead lamp falls across sawdust drifting in the air, while shelves of hand tools blur in the soft-focus background. The fixed medium shot holds steady as his forearms flex with each pass of the sanding block, fingers steadying the instrument against the bench vise."""


ANIMATION_EXAMPLES = """Example 1:
A woman with shoulder-length blonde hair, wearing a flowing white summer dress, performs a slow contemporary dance routine on a clean, softly lit studio background. She extends one arm overhead, pivots on the ball of her foot, and sweeps into a controlled spin, the hem of her dress trailing the motion. The static camera frames her full body in calm, even light, emphasizing the fluid timing and precise footwork of the choreography.

Example 2:
A man in a navy tracksuit jogs in place and then breaks into a sequence of energetic jumping jacks against a plain grey backdrop. His movements are crisp and rhythmic, arms snapping up and down in sync with his feet, brightly and evenly lit in a steady medium shot that keeps his whole body in frame throughout the exercise."""


DEFAULT_INSTRUCTION_REPLACEMENT = "Replace the main subject in the source video with the reference character, keeping the exact same actions, motion, pose, timing and scene interactions."
DEFAULT_INSTRUCTION_ANIMATION = "Make the reference character perform the exact actions, motion, pose, timing and scene interactions shown in the source video."
CAPTION_INSTRUCTION = "Objectively describe the source video: the main subject's actions, motion, pose and timing, the scene, location, lighting and camera framing, and the objects the subject interacts with. Mention the subject's clothing only to ground body regions and interactions. Do not invent or describe any replacement target."


_HANDLERS = {
    "Qwen3-VL":            ("Qwen3VLChatHandler",           "force_reasoning"),
    "Qwen3.5":             ("Qwen35ChatHandler",            "enable_thinking"),
    "MiniCPM-v4.5":        ("MiniCPMv45ChatHandler",        "enable_thinking"),
    "GLM-4.6V":            ("GLM46VChatHandler",            "enable_thinking"),
    "Gemma4":              ("Gemma4ChatHandler",            "enable_thinking"),
    "Step3-VL":            ("Step3VLChatHandler",           "enable_thinking"),
    "Qwen2.5-VL":          ("Qwen25VLChatHandler",          None),
    "LLaVA-1.6":           ("Llava16ChatHandler",           None),
    "LLaVA-1.5":           ("Llava15ChatHandler",           None),
    "MiniCPM-v2.6":        ("MiniCPMv26ChatHandler",        None),
    "Gemma3":              ("Gemma3ChatHandler",            None),
    "Moondream2":          ("MoondreamChatHandler",         None),
    "nanoLLaVA":           ("NanoLlavaChatHandler",         None),
    "llama3-Vision-Alpha": ("Llama3VisionAlphaChatHandler", None),
    "LFM2-VL":             ("LFM2VLChatHandler",            None),
    "LFM2.5-VL":           ("LFM25VLChatHandler",           None),
    "Granite-Docling":     ("GraniteDoclingChatHandler",    None),
    "PaddleOCR-VL-1.5":    ("PaddleOCRChatHandler",         None),
    "DeepSeek-OCR":        ("MTMDChatHandler",              None),
    "None":                (None,                           None),
}


_llm = None
_chat_handler = None
_has_vision = False


class SCAIL2PromptEnhancer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2PromptEnhancer",
            category="sampling",
            inputs=[
                io.Image.Input("reference_image", optional=True,
                    tooltip="\u66ff\u6362\u89d2\u8272\u7684\u53c2\u8003\u56fe\u50cf\u3002"),
                io.Image.Input("video_frames", optional=True,
                    tooltip="\u7528\u4e8e VLM \u63cf\u8ff0\u7684\u6e90\u89c6\u9891\u91c7\u6837\u5e27\u3002"),
                io.Int.Input("max_frames", default=6, min=1, max=32, step=1,
                    tooltip="\u4ece video_frames \u4e3a VLM \u91c7\u6837\u7684\u6700\u5927\u5e27\u6570\u3002"),
                io.Combo.Input("sample_method", options=["uniform", "first_middle_last", "middle"],
                    default="uniform",
                    tooltip="uniform=\u5747\u5300\u95f4\u9694\uff0cfirst_middle_last=3 \u4e2a\u5173\u952e\u5e27\uff0cmiddle=\u4ec5\u4e2d\u95f4\u3002"),
                io.String.Input("instruction", multiline=True, default="",
                    tooltip="\u66ff\u6362/\u52a8\u753b\u6307\u4ee4\u3002\u7559\u7a7a\u5219\u6309 prompt_mode \u4f7f\u7528\u5bf9\u5e94\u7684\u901a\u7528\u6307\u4ee4\u3002\u4f8b\u5982 'replace the man with the person'\u3002"),
                io.String.Input("caption_instruction", default="", multiline=True, optional=True,
                    tooltip="VLM \u63d0\u53d6\u6e90\u89c6\u9891\u5185\u5bb9\u7684\u6307\u4ee4\uff08\u72ec\u7acb\u4e8e\u751f\u6210 instruction\uff09\u3002\u7559\u7a7a\u5219\u7528\u5185\u7f6e\u9ed8\u8ba4\u3002"),
                io.String.Input("source_caption", default="", multiline=True,
                    tooltip="\u9884\u5148\u5199\u597d\u7684\u6e90\u89c6\u9891\u63cf\u8ff0\u3002\u7559\u7a7a\u5219\u7531 VLM \u81ea\u52a8\u751f\u6210\u3002"),
                io.String.Input("examples", default="", multiline=True, optional=True,
                    tooltip="few-shot \u793a\u4f8b\uff0c\u5f15\u5bfc\u8f93\u51fa\u98ce\u683c\u3002\u7559\u7a7a\u5219\u7528\u5f53\u524d\u6a21\u5f0f\u7684\u5185\u7f6e\u9ed8\u8ba4\u793a\u4f8b\u3002"),
                io.Combo.Input("prompt_mode", options=["replacement", "animation"],
                    default="replacement",
                    tooltip="replacement=\u89d2\u8272\u66ff\u6362\uff0canimation=\u52a8\u4f5c\u6a21\u4eff\u3002"),
                io.Combo.Input("llm_model", options=_get_llm_files("model"),
                    tooltip="models/LLM/ \u4e0b\u7684 llama.cpp .gguf \u6a21\u578b\u6587\u4ef6\u3002"),
                io.Combo.Input("mmproj", options=_get_llm_files("mmproj"),
                    tooltip="mmproj / CLIP \u6a21\u578b\u6587\u4ef6\u3002\u53ef\u9009\u3002"),
                io.Combo.Input("chat_handler", options=list(_HANDLERS.keys()), default="Qwen3-VL",
                    tooltip="\u9009\u62e9 chat handler / \u6a21\u578b\u67b6\u6784\u3002"),
                io.Boolean.Input("disable_thinking", default=True, optional=True,
                    tooltip="\u7981\u7528\u601d\u8003/\u63a8\u7406\u6a21\u5f0f\uff08\u901a\u8fc7 handler \u6784\u9020\u53c2\u6570\u4ece\u6e90\u5934\u4e0d\u751f\u6210\u601d\u8003\uff09\u3002\u4ec5\u5bf9\u652f\u6301\u601d\u8003\u7684\u6a21\u578b\u751f\u6548\uff1aQwen3-VL / Qwen3.5 / MiniCPM-v4.5 / GLM-4.6V / Step3-VL / Gemma4\u3002\u5173\u95ed\u5b83\u5373\u5f00\u542f\u601d\u8003\u3002"),
                io.Int.Input("n_ctx", default=65536, min=1024, max=327680, step=128,
                    tooltip="\u4e0a\u4e0b\u6587\u957f\u5ea6\u4e0a\u9650\u3002"),
                io.Float.Input("temperature", default=0.4, min=0.0, max=2.0, step=0.05),
                io.Int.Input("seed", default=42, min=0, max=0xffffffffffffffff, control_after_generate=io.ControlAfterGenerate.randomize),
            ],
            outputs=[
                io.String.Output("enhanced_prompt"),
            ],
            is_experimental=True,
        )

    @classmethod
    def _load_model(cls, llm_model, mmproj, chat_handler_str, n_ctx, disable_thinking):
        global _llm, _chat_handler, _has_vision

        model_path = os.path.join(folder_paths.models_dir, "LLM", llm_model)
        mmproj_path = os.path.join(folder_paths.models_dir, "LLM", mmproj) if mmproj else None

        cls_name, think_param = _HANDLERS.get(chat_handler_str, (None, None))
        handler_cls = None
        if cls_name:
            try:
                import llama_cpp.llama_chat_format as lcf
                handler_cls = getattr(lcf, cls_name, None)
            except ImportError:
                handler_cls = None

        kwargs = {"verbose": False}
        if handler_cls is not None:
            if mmproj_path:
                kwargs["clip_model_path"] = mmproj_path
            if think_param:
                kwargs[think_param] = not disable_thinking
            _chat_handler = handler_cls(**kwargs)
            _has_vision = mmproj_path is not None
        else:
            _chat_handler = None
            _has_vision = False

        from llama_cpp import Llama
        _llm = Llama(
            model_path=model_path,
            chat_handler=_chat_handler,
            n_ctx=n_ctx,
            verbose=False)

    @classmethod
    def _chat(cls, system_prompt, user_text, image_tensors=None, max_size=512, temperature=0.4, seed=42):
        import numpy as np
        import base64
        import io
        from PIL import Image

        _llm.n_tokens = 0
        try:
            _llm._ctx.memory_clear(True)
        except Exception:
            pass

        messages = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt})

        content = []
        if image_tensors is not None:
            if not _has_vision:
                user_text = f"[{len(image_tensors)} image(s) provided but model has no mmproj.]\n" + user_text
            else:
                for img_t in image_tensors:
                    img = img_t.cpu().numpy()
                    img = (np.clip(img * 255.0, 0, 255)).astype(np.uint8)
                    if max_size > 0:
                        pil_img = Image.fromarray(img)
                        w, h = pil_img.size
                        if max(w, h) > max_size:
                            scale = max_size / max(w, h)
                            pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=92)
                    else:
                        buf = io.BytesIO()
                        Image.fromarray(img).save(buf, format="JPEG", quality=92)
                    b64 = base64.b64encode(buf.getvalue()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
        content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content})

        output = _llm.create_chat_completion(
            messages=messages, temperature=temperature, seed=seed)
        text = output["choices"][0]["message"]["content"].strip()
        end_tag = "<" + "/think>"
        if end_tag in text:
            text = text.split(end_tag)[-1].strip()

        return text

    @classmethod
    def execute(cls, reference_image, instruction,
                video_frames=None, source_caption="", examples="", caption_instruction="",
                prompt_mode="replacement",
                max_frames=6, sample_method="uniform",
                llm_model="", mmproj="", chat_handler="Qwen3-VL", n_ctx=65536,
                disable_thinking=True,
                temperature=0.4, seed=42) -> io.NodeOutput:
        global _has_vision
        instruction = instruction.strip() or (DEFAULT_INSTRUCTION_REPLACEMENT if prompt_mode == "replacement" else DEFAULT_INSTRUCTION_ANIMATION)
        if not llm_model:
            return io.NodeOutput("[Error] Specify llm_model.")

        try:
            cls._load_model(llm_model, mmproj, chat_handler, n_ctx, disable_thinking)
        except Exception as e:
            return io.NodeOutput(f"[Error] Failed to load model: {e}")

        frames = None
        if not source_caption.strip():
            if video_frames is not None:
                total = video_frames.shape[0]
                n = max(1, min(max_frames, total))
                if sample_method == "first_middle_last":
                    if total <= 3:
                        indices = list(range(total))
                    else:
                        indices = [0, total // 2, total - 1]
                elif sample_method == "middle":
                    indices = [0, total // 2] if total > 1 else [0]
                else:
                    indices = [0] if n <= 1 else \
                        [0] + [round(i * (total - 1) / (n - 1)) for i in range(1, n)]
                frames = [video_frames[idx] for idx in sorted(set(indices))]
                cap_inst = caption_instruction.strip() or CAPTION_INSTRUCTION
                cap_prompt = f"{VIDEO_CAPTION_PROMPT}\n\nCaption focus: {cap_inst}\nThe following are {len(frames)} sampled source video frames in chronological order."
                source_caption = cls._chat("", cap_prompt, frames, temperature=temperature, seed=seed)
            else:
                source_caption = instruction

        if prompt_mode == "replacement":
            ex = examples.strip() or REPLACEMENT_EXAMPLES
            system = REPLACEMENT_TEMPLATE.format(instruction=instruction, caption=source_caption, examples=ex)
        else:
            ex = examples.strip() or ANIMATION_EXAMPLES
            system = ANIMATION_TEMPLATE.format(instruction=instruction, caption=source_caption, examples=ex)

        ref_imgs = [reference_image[0]] if reference_image is not None else None
        if ref_imgs is not None and _has_vision:
            if prompt_mode == "replacement":
                user_text = ("The image is the reference character, used ONLY for appearance, identity and clothing - not its pose or actions. "
                             "Use the source video description for the scene, motion, pose, timing and interactions; "
                             "do not copy the source subject's appearance. "
                             "Generate the enhanced prompt per the system instructions.")
            else:
                user_text = ("The image is the reference character, used ONLY for appearance, identity and setting - not its pose or actions. "
                             "Take ALL motion, pose, timing, gestures and actions from the source video description; "
                             "keep the reference person in their own setting from the image, NOT the source video's scene; "
                             "do not copy the source subject's appearance. "
                             "Generate the enhanced prompt per the system instructions.")
            enhanced = cls._chat(system, user_text, ref_imgs, temperature=temperature, seed=seed)
        else:
            enhanced = cls._chat(system, "Generate the enhanced prompt per the system instructions.", None,
                                 temperature=temperature, seed=seed)

        global _llm, _chat_handler
        _has_vision = False
        if _llm is not None:
            try:
                if _chat_handler and hasattr(_chat_handler, '_exit_stack'):
                    _chat_handler._exit_stack.close()
            except Exception:
                pass
            try:
                _llm.close()
            except Exception:
                pass
            _llm = None
            _chat_handler = None

        return io.NodeOutput(enhanced)


class SCAIL2MultiRefImages(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="SCAIL2MultiRefImages",
            category="sampling",
            inputs=[
                io.String.Input("images_data", default="[]",
                    tooltip="\u7531\u4e0a\u4f20\u7ec4\u4ef6\u7ba1\u7406\u7684\u5185\u90e8 JSON\u3002"),
                io.Int.Input("width", default=512, min=32, max=8192, step=32),
                io.Int.Input("height", default=896, min=32, max=8192, step=32),
                io.Combo.Input("resize_method",
                    options=["bicubic", "bilinear", "area", "nearest-exact", "lanczos"],
                    default="bicubic"),
                io.Combo.Input("crop", options=["center", "disabled"], default="center"),
            ],
            outputs=[
                io.Image.Output("images"),
                io.String.Output("indices"),
            ],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, images_data="[]", width=512, height=896,
                resize_method="bicubic", crop="center") -> io.NodeOutput:
        import numpy as np
        from PIL import Image, ImageOps

        try:
            entries = json.loads(images_data) if images_data else []
        except (ValueError, TypeError):
            entries = []

        device = comfy.model_management.intermediate_device()
        dtype = comfy.model_management.intermediate_dtype()

        imgs = []
        indices = []
        for e in entries:
            name = e.get("name")
            if not name:
                continue
            subfolder = e.get("subfolder", "")
            ftype = e.get("type", "input")
            ref = (f"{subfolder}/{name}" if subfolder else name) + f" [{ftype}]"
            path = folder_paths.get_annotated_filepath(ref)

            img = node_helpers.pillow(Image.open, path)
            img = node_helpers.pillow(ImageOps.exif_transpose, img)
            img = img.convert("RGB")
            t = torch.from_numpy(np.array(img).astype(np.float32) / 255.0)[None,]
            t = comfy.utils.common_upscale(t.movedim(-1, 1), width, height, resize_method, crop).movedim(1, -1)
            imgs.append(t)
            indices.append(int(e.get("index", 0)))

        if imgs:
            out_img = torch.cat(imgs, dim=0)
        else:
            out_img = torch.zeros((1, height, width, 3))

        return io.NodeOutput(out_img.to(device=device, dtype=dtype), json.dumps(indices))


class SCAILExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [SCAIL2LoopSampler, SCAIL2KeyFrameSampler, SCAIL2KeyFrameSelector, SCAIL2PromptEnhancer, SCAIL2MultiRefImages]
