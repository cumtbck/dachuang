import socket
import json
import time
import os
import csv

LISTEN_IP = '0.0.0.0'
UDP_PORT = 5005
min_latency_fix = 9999.0  # 动态校准值

def main():
    global min_latency_fix
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((LISTEN_IP, UDP_PORT))

    print(f"[*] Performance Monitor Active on port {UDP_PORT}")

    while True:
        data, addr = sock.recvfrom(4096)
        recv_ts = time.time()

        try:
            msg = json.loads(data.decode())
            send_ts = msg.get("send_ts", 0)

            # --- 核心校准逻辑 ---
            raw_latency = (recv_ts - send_ts) * 1000

            # 自动捕获物理极限最小值（过滤掉异常值）
            if 0.1 < raw_latency < min_latency_fix:
                min_latency_fix = raw_latency

            # 修正延迟 = 原始值 - 观测最小值 + 物理线路固有延迟(假设0.2ms)
            calibrated_latency = raw_latency - min_latency_fix + 0.2
            if calibrated_latency < 0: calibrated_latency = 0.1 # 极端抖动保护

            # --- 显示仪表盘 ---
            os.system('clear')
            print(f"=== Zynq-Jetson Performance Dashboard ===")
            print(f"IP Source      : {addr[0]}")
            print(f"Net Latency    : {calibrated_latency:.2f} ms")
            print(f"-----------------------------------------")
            print(f"DPU Execute    : {msg['dpu_ms']:.2f} ms")
            print(f"Pre-process    : {msg['pre_ms']:.2f} ms")
            print(f"Post-process   : {msg['post_ms']:.2f} ms")
            print(f"Total Inference: {msg['dpu_ms']+msg['pre_ms']+msg['post_ms']:.2f} ms")
            print(f"Objects Found  : {msg['count']}")
            print(f"-----------------------------------------")
            print(f"Current Sync Fix: {min_latency_fix:.4f} ms")

        except Exception as e:
            pass

if __name__ == "__main__":
    main()