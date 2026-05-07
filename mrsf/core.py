import os
import numpy as np
from llama_cpp import Llama

GGUF_PATH       = "./models/mistral-7b-v0.1.Q4_K_M.gguf"
MODEL_VERSION   = "mistral-7b-q4km-v1"
N_CTX           = 4096
N_GPU_LAYERS    = 0
LOGIT_PRECISION = 6

print("Loading Mistral 7B...")
lm = Llama(
    model_path   = GGUF_PATH,
    n_ctx        = N_CTX,
    n_gpu_layers = N_GPU_LAYERS,
    logits_all   = True,
    verbose      = False,
    n_threads    = os.cpu_count()
)
print("Mistral loaded.\n")

def tokenize(text: str) -> list:
    return lm.tokenize(text.encode("utf-8"), add_bos=True)

def detokenize(ids: list) -> str:
    return lm.detokenize(ids).decode("utf-8", errors="replace")

def quantized_argmax(raw_logits: np.ndarray) -> int:
    quantized = np.round(raw_logits.astype(np.float64), decimals=LOGIT_PRECISION)
    return int(np.argmax(quantized))

def next_token_greedy(context_ids: list) -> int:
    lm.reset()
    lm.eval(context_ids)
    raw = np.array(lm.scores[len(context_ids) - 1])
    return quantized_argmax(raw)