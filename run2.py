import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Carregar o modelo e o tokenizador
model_name = "pierreguillou/gpt2-small-portuguese"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Salvar o modelo em um diretório local no formato PyTorch
model.save_pretrained("./gpt2-portuguese-converted")
tokenizer.save_pretrained("./gpt2-portuguese-converted")

print("Modelo e tokenizador salvos com sucesso.")

#############################################################

from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_path = "/content/gpt2-small-portuguese"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

#############################################################

import torch

torch.save(model.state_dict(), "gpt2-small-portuguese.pt")

!pip install gguf

#############################################################


import gguf

model_weights = torch.load("gpt2-small-portuguese.pt")

#############################################################


import torch

output_file = "gpt2-small-portuguese.gguf"
torch.save(model.state_dict(), output_file)
