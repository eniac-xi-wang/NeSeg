"""Cold-start instruction tuning for Qwen2.5-VL."""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Trainer, TrainingArguments


class Examples(torch.utils.data.Dataset):
    def __init__(self, file):
        with open(file) as stream:
            self.rows = [json.loads(line) for line in stream if line.strip()]
        if not self.rows:
            raise ValueError(f'Empty SFT split: {file}')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def encode_context(processor, root, row):
    images = []
    content = []
    for i, text in enumerate(row['problem'].split('<image>')):
        if i:
            with Image.open(root / row['images'][i - 1]) as image:
                images.append(image.convert('RGB'))
            content.append({'type': 'image'})
        if text:
            content.append({'type': 'text', 'text': text})
    messages = [dict(role='user', content=content), *row['history']]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(images=images, text=[prompt], add_special_tokens=False, return_tensors='pt')


class ActionCollator:
    def __init__(self, processor, root):
        self.processor, self.root = processor, Path(root)

    def encode(self, row):
        inp = encode_context(self.processor, self.root, row)
        prefix = inp.input_ids[0]
        tok = self.processor.tokenizer
        target = torch.tensor(tok.encode(row['target'], add_special_tokens=False) + [tok.convert_tokens_to_ids('<|im_end|>')])
        return (torch.cat((prefix, target)), torch.cat((torch.full_like(prefix, -100), target)),
                inp.pixel_values, inp.image_grid_thw)

    def __call__(self, rows):
        ids, labels, pixels, grids = zip(*(self.encode(row) for row in rows))
        batch = pad_sequence(ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id)
        lengths = torch.tensor([len(x) for x in ids])
        return dict(input_ids=batch, labels=pad_sequence(labels, batch_first=True, padding_value=-100),
                    attention_mask=(torch.arange(batch.shape[1]) < lengths[:, None]).long(),
                    pixel_values=torch.cat(pixels), image_grid_thw=torch.cat(grids))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='Qwen/Qwen2.5-VL-7B-Instruct')
    parser.add_argument('--data', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=float, default=1)
    parser.add_argument('--full-finetune', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    proc = AutoProcessor.from_pretrained(args.model, min_pixels=56*56, max_pixels=448*448)
    cuda = torch.cuda.is_available()
    bf16 = cuda and torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float32
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype)
    model.config.use_cache = False
    if not args.full_finetune:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, task_type='CAUSAL_LM',
                                                 target_modules=r'.*language_model.*\.(q_proj|k_proj|v_proj|o_proj)'))
    trainer = Trainer(model=model, train_dataset=Examples(args.data), data_collator=ActionCollator(proc, args.data_root),
                      args=TrainingArguments(output_dir=args.output, per_device_train_batch_size=1,
                                             gradient_accumulation_steps=8, learning_rate=2e-5, bf16=bf16, use_cpu=not cuda,
                                             num_train_epochs=args.epochs, remove_unused_columns=False,
                                             gradient_checkpointing=True, gradient_checkpointing_kwargs={'use_reentrant': False},
                                             save_strategy='epoch', report_to='none', ddp_find_unused_parameters=False))
    trainer.train(resume_from_checkpoint=True if args.resume else None)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        policy = trainer.accelerator.unwrap_model(trainer.model)
        if not args.full_finetune:
            policy = policy.merge_and_unload()
        policy.config.use_cache = True
        policy.save_pretrained(Path(args.output, 'policy'))
        proc.save_pretrained(Path(args.output, 'policy'))


if __name__ == '__main__':
    main()
