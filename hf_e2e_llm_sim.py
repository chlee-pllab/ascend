import torch
import torch_npu
from transformers import AutoModelForCausalLM, AutoTokenizer
import fused_matmul_lib

model_id = "Qwen/Qwen2.5-0.5B-Instruct"
print(f"Loading model from Hugging Face: {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map="cpu"
)

def custom_mlp_forward(self, hidden_states):
    weight_gate = self.gate_proj.weight
    fused_output = fused_matmul_lib.matmul_activation_forward(hidden_states, weight_gate)
    up_output = self.up_proj(hidden_states)
    return self.down_proj(fused_output * up_output)

print("Injecting custom Ascend C fused matmul op...")
for layer in model.model.layers:
    layer.mlp.forward = custom_mlp_forward.__get__(layer.mlp, layer.mlp.__class__)

print("Op injection succeeded!")

prompt = "Please explain what matrix multiplication fusion is."
inputs = tokenizer(prompt, return_tensors="pt").to("cpu")

print("=== Starting end-to-end Hugging Face model simulated inference ===")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False
    )

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n=== End-to-end model generation result ===")
print(response)
