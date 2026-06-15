import torch
import torch.nn as nn
import time
import json
import logging
import warnings
from collections import defaultdict
from model.factory import build_model_from_config

# Suppress warnings
logging.getLogger("fvcore").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

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
            if isinstance(module, nn.ModuleList):
                for sub_module in module:
                    h1 = sub_module.register_forward_pre_hook(self._make_pre_hook(name))
                    h2 = sub_module.register_forward_hook(self._make_hook(name))
                    hooks.extend([h1, h2])
            else:
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

    def get_summary(self, n_iters, model):
        stats = []
        for name, module in model.named_children():
            times = self.timings.get(name, [])
            if not times: continue
            if isinstance(module, nn.ModuleList):
                total_ms = sum(times) / n_iters
            else:
                total_ms = sum(times) / len(times)
            stats.append((name, total_ms))
        return stats

def profile_model():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Profile in train mode (eigh skipped)')
    args = parser.parse_args()

    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_str = 'TRAIN (eigh skipped)' if args.train else 'EVAL (eigh active)'
    print(f"Profiling on device: {device}  |  Mode: {mode_str}")
    
    with open('config/config.json', 'r') as f:
        config = json.load(f)
    
    model_type = config['training_params'].get('model_type', 'MeFSQ')
    preprocess = config['preprocess_params']
    
    # 2. Dummy Input - Dynamically extracted from config
    B = 16
    C = 64
    L = preprocess.get('patch_length', 25) 
    N = 800 // L  # 4 seconds @ 200Hz = 800 samples
    model = build_model_from_config(config).to(device)
    model.train() if args.train else model.eval()
    
    x = torch.randn(B, C, N, L).to(device)
    coords = torch.randn(B, C, 3).to(device)
    time_idx = torch.zeros(B, N, dtype=torch.long).to(device)
    
    with torch.no_grad():
        model(x, coords, time_idx)

    print(f"\nModel: {model_type}")
    print(f"Input: Batch={B}, Channels={C}, Patches={N}, Samples={L}")
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
    
    with torch.no_grad():
        for _ in range(5):
            model(x, coords, time_idx)

    profiler.timings.clear()

    n_iters = 20
    start_total = time.perf_counter()
    if device.type == 'cuda': torch.cuda.synchronize()

    loss_times = []

    bool_masked_pos = torch.zeros(B, C, N, dtype=torch.bool).to(device)

    with torch.no_grad():
        for _ in range(n_iters):
            recon, _, v_q = model(x, coords, time_idx, bool_masked_pos=bool_masked_pos)
            if device.type == 'cuda': torch.cuda.synchronize()

            t1 = time.perf_counter()
            model.get_loss(x, recon, bool_masked_pos, v_q)
            if device.type == 'cuda': torch.cuda.synchronize()
            t2 = time.perf_counter()
            loss_times.append((t2 - t1) * 1000)
            
    if device.type == 'cuda': torch.cuda.synchronize()
    total_avg_ms = ((time.perf_counter() - start_total) / n_iters) * 1000
    
    # 5. Report
    print(f"\nPerformance Summary (Avg of {n_iters} runs):")
    print(f"{'Component':<22} | {'Params (M)':<10} | {'Time (ms)':<10} | {'% Total'}")
    print("-" * 70)
    
    time_stats = dict(profiler.get_summary(n_iters, model))
    
    # children
    for name, _ in children:
        t_ms = time_stats.get(name, 0.0)
        print(f"{name:<22} | {param_map.get(name, 0)/1e6:<10.2f} | {t_ms:<10.2f} | {(t_ms/total_avg_ms)*100:>6.1f}%")
        
    # Extra Methods
    avg_loss_ms = sum(loss_times) / n_iters
    print(f"{'Method: get_loss':<22} | {'-':<10} | {avg_loss_ms:<10.2f} | {(avg_loss_ms/total_avg_ms)*100:>6.1f}%")
    
    print("-" * 70)
    print(f"{'Total (Fwd + Loss + Rec)':<22} | {total_params/1e6:<10.2f} | {total_avg_ms:<10.2f} | 100.0%")
    print("-" * 70)

if __name__ == "__main__":
    profile_model()