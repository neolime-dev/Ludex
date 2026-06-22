"""Script de Fine-Tuning do Llama 3 para o Ludex usando Unsloth (LoRA 4-bit)."""

import os
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments

# Configurações do Treinamento
MAX_SEQ_LENGTH = int(os.getenv("LUDEX_TRAIN_MAX_SEQ_LENGTH", "1024"))
MODEL_NAME = os.getenv("LUDEX_TRAIN_MODEL", "unsloth/llama-3-8b-Instruct-bnb-4bit")
DATASET_PATH = os.getenv("LUDEX_TRAIN_DATASET", "data/processed/fine_tuning/ludex_instruction_dataset.jsonl")
OUTPUT_DIR = os.getenv("LUDEX_TRAIN_OUTPUT_DIR", "ludex-llama3-finetuned")
LORA_R = int(os.getenv("LUDEX_TRAIN_LORA_R", "8"))
LORA_ALPHA = int(os.getenv("LUDEX_TRAIN_LORA_ALPHA", str(LORA_R * 2)))
BATCH_SIZE = int(os.getenv("LUDEX_TRAIN_BATCH_SIZE", "1"))
GRAD_ACCUMULATION = int(os.getenv("LUDEX_TRAIN_GRAD_ACCUMULATION", "8"))
MAX_STEPS = int(os.getenv("LUDEX_TRAIN_MAX_STEPS", "120"))
LEARNING_RATE = float(os.getenv("LUDEX_TRAIN_LEARNING_RATE", "2e-4"))
GGUF_OUTPUT_DIR = os.getenv("LUDEX_TRAIN_GGUF_OUTPUT_DIR", "ludex-llama3-finetuned_gguf")


def print_gpu_memory_hint():
    if not torch.cuda.is_available():
        print("⚠️ CUDA indisponível. O treino com Unsloth precisa de GPU NVIDIA.")
        return

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    print(f"GPU VRAM livre antes do load: {free_gb:.2f} GiB de {total_gb:.2f} GiB.")
    if free_gb < 9.5:
        print(
            "⚠️ Pouca VRAM livre para Llama 3 8B 4-bit. "
            "Pare Ollama, Streamlit ou outros processos usando GPU antes de treinar."
        )


print("Configuração do treino:")
print(f"- Modelo base: {MODEL_NAME}")
print(f"- max_seq_length: {MAX_SEQ_LENGTH}")
print(f"- LoRA rank/alpha: {LORA_R}/{LORA_ALPHA}")
print(f"- batch/grad_accumulation: {BATCH_SIZE}/{GRAD_ACCUMULATION}")
print(f"- max_steps: {MAX_STEPS}")
print(f"- dataset: {DATASET_PATH}")
print(f"- saída LoRA: {OUTPUT_DIR}")
print(f"- saída GGUF: {GGUF_OUTPUT_DIR}")
print_gpu_memory_hint()
CUDA_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

# 1. Carregando o Modelo Base (4-bit quantizado)
print("Carregando o modelo...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = MODEL_NAME,
    max_seq_length = MAX_SEQ_LENGTH,
    dtype = None, # Auto-detecta (float16/bfloat16)
    load_in_4bit = True, # Essencial para caber na RTX 4070
)

# 2. Configurando o Adaptador LoRA (O que realmente será treinado)
model = FastLanguageModel.get_peft_model(
    model,
    r = LORA_R,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = LORA_ALPHA,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 3. Preparando o Dataset
print("Preparando o dataset...")
alpaca_prompt = """Abaixo está uma instrução que descreve uma tarefa, combinada com uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente a solicitação.

### Instrução:
{}

### Entrada:
{}

### Resposta:
{}"""

EOS_TOKEN = tokenizer.eos_token # Finaliza a conversa

def format_dataset(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

dataset = load_dataset("json", data_files=DATASET_PATH, split="train")
dataset = dataset.map(format_dataset, batched = True,)

# 4. Configurando o Treinador (Trainer)
print("Iniciando configuração do SFTTrainer...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = MAX_SEQ_LENGTH,
    dataset_num_proc = 2,
    packing = False, # Pode acelerar trenos pequenos
    args = TrainingArguments(
        per_device_train_batch_size = BATCH_SIZE,
        gradient_accumulation_steps = GRAD_ACCUMULATION,
        warmup_steps = 5,
        max_steps = MAX_STEPS,
        learning_rate = LEARNING_RATE,
        fp16 = not CUDA_BF16,
        bf16 = CUDA_BF16,
        logging_steps = 10,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 5. Executando o Treinamento
print("Rodando o Fine-Tuning! Isso pode demorar algumas horas dependendo dos steps...")
trainer_stats = trainer.train()

# 6. Salvando o Modelo para o Ollama
print("Treino concluído. Salvando o modelo adaptado no formato GGUF para o Ollama...")

# Salva em formato 8-bit ou q4_k_m (padrão Ollama)
model.save_pretrained_gguf(GGUF_OUTPUT_DIR, tokenizer, quantization_method = "q4_k_m")

print(f"Modelo salvo em {GGUF_OUTPUT_DIR}! Você agora pode importá-lo no Ollama usando um Modelfile.")
