#!/usr/bin/env python3
"""Download and cache Florence-2-large model."""
from transformers import AutoModelForCausalLM, AutoProcessor

print("Downloading Florence-2-large processor...")
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
print("Processor OK")

print("Downloading Florence-2-large model weights...")
model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
n_params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model downloaded successfully! Params: {n_params:.1f}M")
