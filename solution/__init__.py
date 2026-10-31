"""Your solution."""
from .swiglu_q8 import patch_qwen2_mlp
from .rmsnorm import patch_qwen2_rmsnorm
from .rope import patch_qwen2_rope
from .lora_fast import patch_lora_dropout
from .fast_rewind import patch_fast_rewind


def apply():
    patch_qwen2_mlp()
    patch_qwen2_rmsnorm()
    patch_qwen2_rope()
    patch_lora_dropout()
    patch_fast_rewind()
