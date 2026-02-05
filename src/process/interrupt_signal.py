
import config
import signal

'''
偵測中斷訊號以儲存模型
'''

def install_interrupt_handlers():
    """安裝 SIGINT/SIGTERM 處理器；收到訊號後將 config.STOP_REQUESTED 設為 True。"""
    def _handler(sig, frame):
        if not config.STOP_REQUESTED:
            config.STOP_REQUESTED = True
            name = {signal.SIGINT: "SIGINT (Ctrl+C)", signal.SIGTERM: "SIGTERM"}.get(sig, f"signal {sig}")
            print(f"\n\n[⚠️ Interrupt] Caught {name}. Will stop after current step and export current weights...\n")
    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)