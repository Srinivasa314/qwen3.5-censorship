"""qwc — mechanistic-interpretability tools for Qwen3.5-9B censorship circuitry.

Tap-numbering convention (matches Hugging Face `output_hidden_states`):
    tap 0  = embedding output (input to layer 0)
    tap k  = output of layer k-1 (== input to layer k), for k=1..L
    tap L  = output of the final transformer layer (== input to lm_head)

So a direction extracted at tap 14 is the residual on entry to layer 14,
equivalently the output of layer 13. Steering "at L13" hooks layer 13's
forward output, which is the same location as tap 14.
"""
