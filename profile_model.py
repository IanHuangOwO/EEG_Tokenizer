import torch
import torch.nn as nn
import time
import json
import os
import logging
import warnings
import pandas as pd
from collections import defaultdict
from model.factory import build_model_from_config

# Suppress warnings
logging.getLogger("fvcore").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    from fvcore.nn import FlopCountAnalysis, flop_count_table
    HAS_FVCORE = True
except ImportError:
    HAS_FVCORE = False

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class ProfilerHooks:
    def __init__(self):
        self.timings = defaultdict(list)
        self.starts = {}
        self.device = None

    def _sync(self):
        if self.device and self.device.type == 'cuda':
            torch.cuda.synchronize()

    def register(self, model, device):
        self.device = device
        hooks = []
        for name, module in model.named_children():
            # Skip empty containers if any
            if list(module.parameters()) or list(module.buffers()):
                h1 = module.register_forward_pre_hook(self._make_pre_hook(name))
                h2 = module.register_forward_hook(self._make_hook(name))
                hooks.extend([h1, h2])
        return hooks

    def _make_pre_hook(self, name):
        def hook(module, input):
            self._sync()
            self.starts[name] = time.perf_counter()
        return hook

    def _make_hook(self, name):
        def hook(module, input, output):
            self._sync()
            end = time.perf_counter()
            if name in self.starts:
                duration = (end - self.starts[name]) * 1000 # ms
                self.timings[name].append(duration)
        return hook

    def get_summary(self):
        stats = []
        for name, times in self.timings.items():
            if not times: continue
            avg_ms = sum(times) / len(times)
            stats.append((name, avg_ms))
        return stats

def profile_model():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Profiling on device: {device}")
    
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    model = build_model_from_config(config).to(device)
    model.eval()
    
    # 2. Dummy Input
    B, N, T = 32, 64, 200
    x = torch.randn(B, N, T).to(device)
    coords = torch.randn(B, N, 3).to(device)
    
    print(f"\nModel: {config['training_params']['model_type']}")
    print(f"Input: Batch={B}, Channels={N}, Time={T}")
    print("-" * 60)
    
    # 3. Component Discovery & Param Count
    print("Detected Components:")
    children = list(model.named_children())
    param_map = {}
    total_params = 0
    
    for name, module in children:
        p = count_parameters(module)
        param_map[name] = p
        total_params += p
        print(f"  - {name:<20} : {p/1e6:>6.2f} M params")
    print("-" * 60)

    # 4. Profiling (Timing via Hooks)
    profiler = ProfilerHooks()
    hooks = profiler.register(model, device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model(x, coords)
            
    # Clear warmup timings
    profiler.timings.clear()
    
    # Run Timing Loop
    n_iters = 20
    start_total = time.perf_counter()
    if device.type == 'cuda': torch.cuda.synchronize()
    
    with torch.no_grad():
        for _ in range(n_iters):
            model(x, coords)
            
    if device.type == 'cuda': torch.cuda.synchronize()
    total_avg_ms = ((time.perf_counter() - start_total) / n_iters) * 1000
    
    # Remove hooks
    for h in hooks: h.remove()
    
    # 5. FLOPs Analysis
    flop_stats = {}
    if HAS_FVCORE:
        flop_analysis = FlopCountAnalysis(model, (x, coords))
        # This gives breakdown by module
        flop_stats = flop_analysis.by_module() 
        total_flops = flop_analysis.total()
    else:
        total_flops = 0

    # 6. Report
    print(f"\nPerformance Summary (Avg of {n_iters} runs):")
    print(f"{'Component':<22} | {'Params (M)':<10} | {'Time (ms)':<10} | {'% Time':<7} | {'FLOPs (G)':<10} | {'% FLOPs':<7}")
    print("-" * 90)
    
    time_stats = dict(profiler.get_summary())
    
    # Sort by structure order (order of children)
    for name, _ in children:
        t_ms = time_stats.get(name, 0.0)
        t_pct = (t_ms / total_avg_ms) * 100
        
        # Params
        p_m = param_map.get(name, 0) / 1e6
        
        # Match FLOPs name
        f_count = flop_stats.get(name, 0.0)
        f_g = f_count / 1e9
        f_pct = (f_count / total_flops * 100) if total_flops > 0 else 0
        
        print(f"{name:<22} | {p_m:<10.2f} | {t_ms:<10.2f} | {t_pct:<6.1f}% | {f_g:<10.3f} | {f_pct:<6.1f}%")
        
    print("-" * 90)
    print(f"{'Total Model':<22} | {total_params/1e6:<10.2f} | {total_avg_ms:<10.2f} | {'100.0':<7}% | {total_flops/1e9:<10.3f} | {'100.0':<7}%")
    print("-" * 90)

if __name__ == "__main__":
    profile_model()