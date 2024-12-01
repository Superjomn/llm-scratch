import os
import random
from pathlib import Path

import torch

from llama_from_scratch.config import LlamaConfig
from llama_from_scratch.model import LlamaForCausalLM


def get_model_path(model_name:str) -> Path:
    return Path(os.environ.get("LLM_MODELS_ROOT")) / mode_name

def test_model():
    torch_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = LlamaForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = model.to(torch_device)
    config : LlamaConfig = model.config
    print(model)
    '''
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(32000, 2048)
    (layers): ModuleList(
      (0-21): 22 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=256, bias=False)
          (v_proj): Linear(in_features=2048, out_features=256, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (up_proj): Linear(in_features=2048, out_features=5632, bias=False)
          (down_proj): Linear(in_features=5632, out_features=2048, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=32000, bias=False)
)
    '''

    batch_size = 2
    seq_length = 128
    input_ids = ids_tensor([batch_size, seq_length], config.vocab_size, torch_device=model.device)
    print(f"input_ids: {input_ids}")
    print(f"torch_device: {torch_device}")

    input_mask = torch.tril(torch.ones_like(input_ids).to(torch_device))

    result = model(input_ids=input_ids, attention_mask=input_mask)
    print(f"result: {result})")


global_rng = random.Random()
def ids_tensor(shape, vocab_size, rng=None, name=None, torch_device=None):
    #  Creates a random int32 tensor of the shape within the vocab size
    if rng is None:
        rng = global_rng

    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(rng.randint(0, vocab_size - 1))

    return torch.tensor(data=values, dtype=torch.long, device=torch_device).view(shape).contiguous()






if __name__ == '__main__':
    test_model()
