import signal
import sys

def install_interrupt_handlers(trainer):
    """安裝 Ctrl+C 攔截器，確保訓練中斷時能儲存模型"""
    def handler(sig, frame):
        print("\n[System] Keyboard Interrupt (Ctrl+C) detected!")
        print("[System] Stopping training gracefully...")
        trainer.stop_requested = True
        
    signal.signal(signal.SIGINT, handler)
    print("[System] Interrupt handlers installed.")