
import config
import numpy as np

if config.PLOT_Switch == True:
    import matplotlib.pyplot as plt


def plot_and_save_lr_schedule(schedule, total_steps, save_path):
    """繪製學習率變化曲線並儲存。"""
    steps = np.arange(total_steps)
    lrs = [schedule(step) for step in steps]
    plt.figure(figsize=(10, 5))
    plt.plot(steps, lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Learning rate schedule plot saved to {save_path}")

def plot_and_save_loss_curve(history, save_path):
    """繪製損失曲線並儲存。"""
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history) + 1), history)
    plt.title('Distillation Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    print(f"📉 Loss curve plot saved to {save_path}")
