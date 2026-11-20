# NeSeg

NeSeg trains a vision-language model to segment the object a viewer describes in a video. The policy, **Qwen2.5-VL-7B-Instruct**, chooses a frame and places positive clicks on the target and negative clicks on nearby distractors. **SAM 2 produces the pixel masks and propagates them through the sampled clip.** Qwen inspects the returned mask, adds corrective clicks if needed, and decides when to finish.

![Comparison of segmentation prompts](assets/comparison.png)

Two similar objects crossing paths illustrate the motivation. A positive click can identify the intended object in one frame, yet the predicted mask may also include its neighbor. A negative click gives the policy a way to exclude that neighbor. Inspecting the resulting mask gives it a basis for the next decision. Segmentation quality depends directly on SAM 2's mask decoder and temporal propagation.

## How training works

1. **Build supervision.** The pipeline pairs referring expressions from ReVOS, RVOS and Ref-SAV with target masks. Ref-SAV supplies language for objects in Meta's SA-V videos. The sampler keeps frames from across the clip and favors intervals where the target is visible. It chooses a representative annotated frame, samples positive clicks inside its mask, and mines negative clicks just outside the boundary. Nearby patches with DINOv2 features similar to target patches make useful hard negatives. The original project data build produced **30k+ examples**. Each new export reports its own counts.
2. **Cold-start SFT.** Qwen learns to choose a displayed timestamp and emit labeled point coordinates. The feedback-enabled export runs the frozen SAM 2 predictor to construct conversations with real mask observations and a final stop action. Only the next assistant action contributes to the training loss.
3. **GRPO.** EasyR1, built on veRL, samples several independent interactions for the same expression. Each interaction starts with a fresh SAM 2 state. Clicks accumulate across turns. After each call, the policy sees the predicted foreground bounding box and a 12 × 12 occupancy map at the clicked time. It can finish early or make another call, up to three by default. The final propagated masks are compared with the annotations. Ground truth and DINOv2 features are available to the reward function only during rollout scoring.

### Reward

The default score is a weighted sum. Point terms are averaged within each call and then across calls. Coordinates range from 0 to 1000 on each axis.

| Component | Weight | Measurement |
| --- | ---: | --- |
| Action format | 0.05 | A valid sequence of JSON actions, ending after at least one SAM 2 call |
| Final segmentation | 0.50 | Total intersection divided by total union over all sampled frames, including false foreground on target-absent frames |
| Positive hit | 0.10 | Fraction of positive clicks inside the target mask on the selected frame |
| Positive distance | 0.10 | For each positive click, distance to its nearest sampled foreground reference. Credit decays as `exp(-distance / scale)`, with scale equal to 10% of the frame diagonal (at least one pixel) |
| Negative hit | 0.10 | +1 outside the target and −1 inside it |
| Negative DINOv2 | 0.10 | For a correct negative click, the highest cosine similarity to sampled foreground references, clipped to [0, 1]. A click inside the target earns zero feature credit |
| Tool efficiency | 0.05 | Final segmentation IoU divided by the number of executed calls |

A malformed action or a finish before the first tool call earns zero for the episode. Reaching the call limit finalizes the last prediction. Clicks on a frame with no visible target receive zero point credit. A valid finish uses the latest mask. The training loss excludes SAM 2's feedback tokens.

## Prepare the environment

Use a Linux CUDA environment compatible with the bundled EasyR1 version. Its upstream image is `hiyouga/verl:ngc-th2.8.0-cu12.9-vllm0.11.0`. Run the commands below from the NeSeg repository root. The vendored EasyR1 source is based on the project's [pinned upstream revision](https://github.com/hiyouga/EasyR1/tree/3f527bd266f964f33c6cd6ec83ac85a6a9e5bf37) and retains its Apache 2.0 license.

```bash
python -m pip install -e ./site-packages/easyr1 -r requirements-neseg.txt
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'

export DATA_ROOT=/absolute/path/to/Sa2VA-Training
export SAM2_CHECKPOINT=/absolute/path/to/sam2.1_hiera_tiny.pt
```

The default config matches the [official SAM 2.1 Tiny checkpoint](https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt). Set `SAM2_CONFIG` when using a different checkpoint. SAM 2 defaults to CPU so its state can coexist with the policy on a training node. Set `SAM2_DEVICE=cuda` when each rollout GPU has room for both models.

## Build the training data

| Source | Files under `$DATA_ROOT/video_datas/` |
| --- | --- |
| ReVOS | `revos/meta_expressions_train_.json`, `revos/mask_dict.json`, frame directories named by the video IDs |
| RVOS | `rvos/meta_expressions/train/meta_expressions.json`, `rvos/mask_dict.pkl`, `rvos/train/JPEGImages/<video>/` |
| Ref-SAV | `Ref-SAV.json` and `sam_v_full/` containing the referenced SA-V video files and masklet JSON files |

Ref-SAV uses every fourth source frame to align 24 fps video with 6 fps mask annotations. The loader caches the selected JPEGs in `video_datas/.neseg_frames/`. ReVOS and RVOS default to a 5 fps frame sequence. Set `--revos_fps` or `--rvos_fps` to the frame rate of your extraction if it differs. Each policy prompt shows the sampled frames in order with their timestamps. The loaders and annotation layouts follow [Sa2VA](https://github.com/bytedance/Sa2VA/tree/main/projects/sa2va) and the [SA-V format](https://github.com/facebookresearch/sam2/tree/main/sav_dataset).

```bash
python -m preprocess_data.preprocess \
  --data_root "$DATA_ROOT" --datasets revos rvos ref_sav \
  --max_frames 8 --sft_tool_calls 2 --output_json runs/records.json

python -m neseg.prepare runs/records.json \
  --data-root "$DATA_ROOT" --output runs/prepared --with-feedback
```

The first command writes the annotated records and a DINOv2 feature cache shared by expressions with the same frames. Existing NeSeg record files with real masks can enter at the second command. The export assigns entire videos to train or validation using a seeded split (5% validation by default). SFT uses training videos only.

| Output | Contents |
| --- | --- |
| `sft.jsonl` | Timestamped frames, instruction, conversation history and next assistant action |
| `train.jsonl`, `val.jsonl` | Policy question and frames, plus a separate scoring label with per-frame masks, reference points and feature path |

For a point-only cold start, omit `--with-feedback`. For a DINOv2 ablation, preprocess with `--skip_features` and pass `worker.reward.reward_function_kwargs.negative_dino_weight=0` to the GRPO launcher.

## Train the policy

```bash
torchrun --standalone --nproc_per_node=4 -m neseg.sft \
  --data runs/prepared/sft.jsonl --data-root "$DATA_ROOT" --output runs/sft

GPUS=4 bash scripts/train_grpo.sh
```

SFT defaults to LoRA on Qwen's language attention layers. Pass `--full-finetune` for full-parameter SFT. The exported `runs/sft/policy` contains a Hugging Face policy with the adapter merged and the processor alongside it. GRPO starts from that directory. Override `MODEL_PATH` or `PREPARED` for other locations. The default batch uses eight training questions, four trajectories per question and one GPU per rollout replica. Both training and validation exports must contain data.

The launcher accepts EasyR1 overrides. For example, `bash scripts/train_grpo.sh worker.rollout.neseg_max_calls=2 worker.reward.reward_function_kwargs.positive_distance_weight=0.15` changes the interaction budget and the distance coefficient. Component scores are logged separately from the total.

## Segment a clip

Pass sampled frames in chronological order, their original timestamps in seconds, and a referring expression. This uses the same action parser, feedback and SAM 2 propagation as training. It writes one binary PNG mask per input frame in input order.

```bash
python -m neseg.infer --policy runs/sft/policy \
  --frames clip/00000.jpg clip/00005.jpg clip/00010.jpg --times 0 0.2 0.4 \
  --expression "the dog running behind the cyclist" --output runs/masks
```

To run this with a GRPO checkpoint, first export the actor using the [EasyR1 model merger](site-packages/easyr1/scripts/model_merger.py) and pass the resulting Hugging Face directory to `--policy`.
