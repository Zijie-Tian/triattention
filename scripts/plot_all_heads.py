import os
import glob
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import gc

def plot_layer_q_k(file_path, q_dir, k_dir):
    layer_name = os.path.basename(file_path).split('.')[0]
    print(f"Loading {layer_name}...")
    
    data = torch.load(file_path, map_location='cpu', weights_only=True)
    
    N = 5000  # Number of tokens to plot
    
    q_pre = data['pre_rope_q'][:N, :, 0:2].float().numpy()   # [N, 32, 2]
    q_post = data['post_rope_q'][:N, :, 0:2].float().numpy() # [N, 32, 2]
    
    k_pre = data['pre_rope_k'][:N, :, 0:2].float().numpy()   # [N, 4, 2]
    k_post = data['post_rope_k'][:N, :, 0:2].float().numpy() # [N, 4, 2]
    
    del data
    gc.collect()

    num_q_heads = q_pre.shape[1]
    num_k_heads = k_pre.shape[1]

    scatter_kwargs = {'s': 1, 'alpha': 0.1, 'linewidths': 0}
    pre_color = '#1f77b4'  # Blue for Pre-RoPE
    post_color = '#d62728' # Red for Post-RoPE

    # ---------------------------------------------------------
    # Plot Q Heads
    # ---------------------------------------------------------
    # 32 heads -> 4 rows x 8 cols
    rows_q, cols_q = 4, 8
    fig_q, axes_q = plt.subplots(rows_q, cols_q, figsize=(24, 12))
    fig_q.suptitle(f'{layer_name.capitalize()} - Q Heads (Pre-RoPE vs Post-RoPE)', fontsize=20, y=0.98)
    
    max_val_q = max(np.abs(q_pre).max(), np.abs(q_post).max())
    limit_q = max_val_q * 1.1

    for i in range(num_q_heads):
        row = i // cols_q
        col = i % cols_q
        ax = axes_q[row, col]
        
        ax.scatter(q_post[:, i, 0], q_post[:, i, 1], c=post_color, label='Post-RoPE', **scatter_kwargs)
        ax.scatter(q_pre[:, i, 0], q_pre[:, i, 1], c=pre_color, label='Pre-RoPE', **scatter_kwargs)
        
        ax.set_xlim(-limit_q, limit_q)
        ax.set_ylim(-limit_q, limit_q)
        ax.set_aspect('equal')
        ax.set_title(f'Q Head {i}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Only add legend to the first subplot to save space
        if i == 0:
            leg = ax.legend(loc='upper right', frameon=True, fontsize=8, handletextpad=0.1)
            for lh in leg.legend_handles: 
                lh.set_alpha(1)
                lh.set_sizes([20])

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    fig_q.savefig(os.path.join(q_dir, f"{layer_name}_q.png"), bbox_inches='tight')
    plt.close(fig_q)

    # ---------------------------------------------------------
    # Plot K Heads
    # ---------------------------------------------------------
    # 4 heads -> 1 row x 4 cols
    fig_k, axes_k = plt.subplots(1, num_k_heads, figsize=(16, 4))
    if num_k_heads == 1:
        axes_k = [axes_k]
        
    fig_k.suptitle(f'{layer_name.capitalize()} - K Heads (Pre-RoPE vs Post-RoPE)', fontsize=16, y=1.05)
    
    max_val_k = max(np.abs(k_pre).max(), np.abs(k_post).max())
    limit_k = max_val_k * 1.1

    for i in range(num_k_heads):
        ax = axes_k[i]
        
        ax.scatter(k_post[:, i, 0], k_post[:, i, 1], c=post_color, label='Post-RoPE', **scatter_kwargs)
        ax.scatter(k_pre[:, i, 0], k_pre[:, i, 1], c=pre_color, label='Pre-RoPE', **scatter_kwargs)
        
        ax.set_xlim(-limit_k, limit_k)
        ax.set_ylim(-limit_k, limit_k)
        ax.set_aspect('equal')
        ax.set_title(f'K Head {i}', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if i == 0:
            leg = ax.legend(loc='upper right', frameon=True, fontsize=10)
            for lh in leg.legend_handles: 
                lh.set_alpha(1)
                lh.set_sizes([20])

    plt.tight_layout()
    fig_k.savefig(os.path.join(k_dir, f"{layer_name}_k.png"), bbox_inches='tight')
    plt.close(fig_k)
    
    del q_pre, q_post, k_pre, k_post
    gc.collect()

def main():
    sns.set_theme(style="darkgrid")
    plt.rcParams['figure.dpi'] = 150  # Lower DPI for grids to avoid huge files
    plt.rcParams['savefig.dpi'] = 150

    input_dir = 'data/kvcache-rope'
    q_dir = 'docs/assets/rope_distributions/q_heads'
    k_dir = 'docs/assets/rope_distributions/k_heads'
    
    os.makedirs(q_dir, exist_ok=True)
    os.makedirs(k_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(input_dir, 'layer_*.pt')))
    if not files:
        print(f"No layer files found in {input_dir}")
        return

    print(f"Found {len(files)} layer files. Starting batch plotting for all heads...")
    for f in files:
        plot_layer_q_k(f, q_dir, k_dir)
        
    print(f"All done! Plotted Q heads to {q_dir}/ and K heads to {k_dir}/")

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings('ignore')
    main()
