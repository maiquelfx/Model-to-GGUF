# 0. Instalar dependencias:

!pip install transformers
!pip install torch

#1. Basta carregar modelo GPT-2 small em português:

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "/content/gpt2-small-portuguese"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

#2. Bora pesos do modelo em um arquivo usando o formato GGUF:

import torch

output_file = "gpt2-small-portuguese.gguf"
torch.save(model.state_dict(), output_file)
