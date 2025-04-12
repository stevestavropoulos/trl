# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
# regular:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing

# peft:
python examples/scripts/sft.py \
    --model_name_or_path="facebook/opt-350m" \
    --dataset_text_field="text" \
    --report_to="wandb" \
    --learning_rate=1.41e-5 \
    --per_device_train_batch_size=64 \
    --gradient_accumulation_steps=16 \
    --output_dir="sft_openassistant-guanaco" \
    --logging_steps=1 \
    --num_train_epochs=3 \
    --max_steps=-1 \
    --push_to_hub \
    --gradient_checkpointing \
    --use_peft \
    --lora_r=64 \
    --lora_alpha=16
"""

# Login to Hugging Face
import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    print("[INFO] Logging in to Hugging Face...")
    login(token=hf_token)
else:
    raise ValueError("[ERROR] Hugging Face token not found! Ensure it's passed to SageMaker.")


from trl.commands.cli_utils import SFTScriptArguments, TrlParser


from datasets import load_dataset

from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    SFTConfig,
    SFTTrainer,
    get_peft_config,
    get_quantization_config,
    get_kbit_device_map,
)

if __name__ == "__main__":
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model init kwargs & Tokenizer
    ################
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        trust_remote_code=model_config.trust_remote_code,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=model_config.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else "auto",
        quantization_config=quantization_config,
    )
    training_args.model_init_kwargs = model_kwargs
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, use_fast=True
    )
    # We patched SFTTrainer to always resize embeddings to handle our additions
    tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    # pad_token should NOT be equal to eos_token, otherwise the model will not
    # know when it's turn is stopped
    #tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    #dataset = load_dataset(args.dataset_name)
    dataset = load_dataset("json", data_files={"train": os.path.join(os.environ["SM_CHANNEL_TRAIN"], "*.jsonl")})


    # Confirm splits
    print("Dataset splits:", dataset.keys())

    # Inspect first sample of the training split
    print("Dataset columns in train split:", dataset["train"].column_names)
    print("First sample from dataset train split:", dataset["train"][0])

    from trl import apply_chat_template
    print(apply_chat_template(dataset["train"][0], tokenizer))
    print(tokenizer.eos_token)        # Should print <|eot_id|>
    print(tokenizer.eos_token_id)     # Should print id of <|eot_id|>
    print(tokenizer.convert_tokens_to_ids("<|eot_id|>"))  # Should print same id

    from trl import DataCollatorForCompletionOnlyLM
    collator = DataCollatorForCompletionOnlyLM(
            tokenizer=tokenizer,
            instruction_template="<|start_header_id|>user<|end_header_id|>\n\n",
            response_template="<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    ################
    # Training
    ################
    trainer = SFTTrainer(
        model=model_config.model_name_or_path,
        args=training_args,
        train_dataset=dataset[args.dataset_train_split],
        eval_dataset=None,
        tokenizer=tokenizer,
        peft_config=get_peft_config(model_config),
        data_collator=collator,
        #eval_dataset=dataset[args.dataset_test_split],
    )

    trainer.train()
    print("Model dtype: ", next(trainer.model.parameters()).dtype)
    trainer.save_model(training_args.output_dir)
