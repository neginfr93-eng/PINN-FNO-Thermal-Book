import matplotlib.pyplot as plt
import numpy as np
from config import GRID, MAX_STEPS

def plot_results(epochs_recorded, train_loss_history, val_loss_history, true_traj, fno_traj, abs_error, T0_test):
    plt.rcParams.update({'font.size': 11})

    # Create a figure with a 2-row, 1-column grid layout
    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.1])

    # ==========================================
    # ROW 1: Training & Validation Convergence 
    # ==========================================
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(epochs_recorded, train_loss_history, color='#2c3e50', lw=2, label='Train Loss (Batch)')
    ax0.plot(epochs_recorded, val_loss_history, color='#e74c3c', lw=2, ls='--', label='Validation Loss (Unseen)')
    ax0.set_yscale('log')
    ax0.set_title("Training & Validation Convergence", fontweight='bold', fontsize=14, pad=10)
    ax0.set_xlabel("Actual Epochs")
    ax0.set_ylabel("Loss (MSE)")
    ax0.grid(True, which="both", ls="-", alpha=0.2)
    ax0.legend(loc='upper right')

    # Time and Grid extensions for 2D plotting
    time_steps = np.arange(MAX_STEPS)
    T, X = np.meshgrid(time_steps, GRID)

    # ==========================================
    # ROW 2: Space-Time Rollout Trajectories
    # ==========================================
    # Absolute Error Only
    ax3 = fig.add_subplot(gs[1, 0])
    im3 = ax3.imshow(abs_error.T, aspect='auto', origin='lower', cmap='inferno',
                     extent=[0, MAX_STEPS-1, 0, 1])
    ax3.set_title("Absolute Prediction Error", fontweight='bold')
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Spatial Grid (x)")
    fig.colorbar(im3, ax=ax3, label="Error Magnitude")

    plt.tight_layout()
    plt.show()
