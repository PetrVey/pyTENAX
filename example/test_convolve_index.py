import numpy as np

# Fake precipitation event — long enough for 24h duration (window=144 at 10min res)
np.random.seed(42)
event = np.random.uniform(0.5, 3.0, 300).astype(np.float64)

# ---- FLAT EVENT TEST ----
# Nearly flat: all values 1.0 with tiny noise at machine-epsilon level
flat_event = np.ones(300, dtype=np.float64)
flat_event += np.random.uniform(-1e-13, 1e-13, 300)  # near-ties

window_size_flat = 144
half_left_flat = window_size_flat // 2
half_right_flat = (window_size_flat - 1) // 2
n_flat = len(flat_event)

conv_np_flat = np.convolve(flat_event, np.ones(window_size_flat), 'same')
idx_np_flat  = int(np.argmax(conv_np_flat))

conv_lr_flat = np.zeros(n_flat)
for j in range(n_flat):
    s = 0.0
    start_k = max(0, j - half_left_flat)
    end_k   = min(n_flat, j + half_right_flat + 1)
    for k in range(start_k, end_k):
        s += flat_event[k]
    conv_lr_flat[j] = s
idx_lr_flat = int(np.argmax(conv_lr_flat))

conv_rl_flat = np.zeros(n_flat)
for j in range(n_flat):
    s = 0.0
    start_k = max(0, j - half_left_flat)
    end_k   = min(n_flat, j + half_right_flat + 1)
    for k in range(end_k - 1, start_k - 1, -1):
        s += flat_event[k]
    conv_rl_flat[j] = s
idx_rl_flat = int(np.argmax(conv_rl_flat))

print("=== FLAT EVENT TEST ===")
print(f"{'Method':<25} {'argmax index':>12} {'value':>25}")
print("-" * 65)
print(f"{'numpy convolve':<25} {idx_np_flat:>12} {conv_np_flat[idx_np_flat]:>25.15f}")
print(f"{'manual left-to-right':<25} {idx_lr_flat:>12} {conv_lr_flat[idx_lr_flat]:>25.15f}")
print(f"{'manual right-to-left':<25} {idx_rl_flat:>12} {conv_rl_flat[idx_rl_flat]:>25.15f}")

top3 = np.argsort(conv_np_flat)[-3:][::-1]
print("\nTop 3 positions (flat event):")
for i in top3:
    print(f"  idx={i:>4}  np={conv_np_flat[i]:.15f}  lr={conv_lr_flat[i]:.15f}  rl={conv_rl_flat[i]:.15f}")

print()
print("--- MATLAB flat event ---")
print(f"flat_event = [{', '.join(f'{v:.15f}' for v in flat_event)}];")
print(f"conv_flat = conv(flat_event, ones({window_size_flat}, 1), 'same');")
print(f"[val_f, idx_f] = max(conv_flat);")
print(f"fprintf('MATLAB flat argmax (1-based): %d, value: %.15f\\n', idx_f, val_f);")
print()
print("=== ORIGINAL EVENT TEST ===")


window_size = 144  # 24h / 10min
half_left = window_size // 2    # 72
half_right = (window_size - 1) // 2  # 71

n = len(event)

# --- numpy convolve ---
conv_np = np.convolve(event, np.ones(window_size), 'same')
idx_np  = int(np.argmax(conv_np))
val_np  = conv_np[idx_np]

# --- manual loop left-to-right ---
conv_lr = np.zeros(n)
for j in range(n):
    start_k = max(0, j - half_left)
    end_k   = min(n, j + half_right + 1)
    s = 0.0
    for k in range(start_k, end_k):
        s += event[k]
    conv_lr[j] = s
idx_lr = int(np.argmax(conv_lr))
val_lr = conv_lr[idx_lr]

# --- manual loop right-to-left ---
conv_rl = np.zeros(n)
for j in range(n):
    start_k = max(0, j - half_left)
    end_k   = min(n, j + half_right + 1)
    s = 0.0
    for k in range(end_k - 1, start_k - 1, -1):
        s += event[k]
    conv_rl[j] = s
idx_rl = int(np.argmax(conv_rl))
val_rl = conv_rl[idx_rl]

print(f"event length : {n}")
print(f"window_size  : {window_size}")
print()
print(f"{'Method':<25} {'argmax index':>12} {'value':>20}")
print("-" * 60)
print(f"{'numpy convolve':<25} {idx_np:>12} {val_np:>20.10f}")
print(f"{'manual left-to-right':<25} {idx_lr:>12} {val_lr:>20.10f}")
print(f"{'manual right-to-left':<25} {idx_rl:>12} {val_rl:>20.10f}")
print()

# Show top 5 positions and their values for numpy vs manual
top5_np = np.argsort(conv_np)[-5:][::-1]
print("Top 5 positions (numpy):")
for i in top5_np:
    print(f"  idx={i:>4}  np={conv_np[i]:.15f}  lr={conv_lr[i]:.15f}  rl={conv_rl[i]:.15f}")

print()
print("--- paste this into MATLAB ---")
print(f"event = [{', '.join(f'{v:.15f}' for v in event)}];")
print(f"window_size = {window_size};")
print(f"conv_result = conv(event, ones(window_size, 1), 'same');")
print(f"[val, idx] = max(conv_result);")
print(f"fprintf('MATLAB argmax index (1-based): %d, value: %.15f\n', idx, val);")
