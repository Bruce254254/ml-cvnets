# main_benchmark.py
"""
Benchmark model inference speed (throughput and latency).
Usage:
python main_benchmark.py --ckpt checkpoints/best...pth --img-size 224 --batch-size 64 --iters 200
"""
import time
import torch
import argparse
from main_train import EnhancedMobileViT

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EnhancedMobileViT(num_classes=2, img_size=args.img_size, pretrained=False, cbam=args.cbam, fusion=args.fusion)
    model = model.to(device)
    if args.ckpt:
        state = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(state.get("model_state", state))
    model.eval()

    # create synthetic input
    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(args.iters):
            t0 = time.time()
            _ = model(x)
            t1 = time.time()
            times.append(t1 - t0)

    avg_time = sum(times) / len(times)
    per_image = avg_time / args.batch_size
    throughput = args.batch_size / avg_time
    print(f"Batch size: {args.batch_size} | Avg batch time: {avg_time:.6f}s | Throughput: {throughput:.2f} img/s | Per-image latency: {per_image*1000:.3f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cbam", action="store_true")
    parser.add_argument("--fusion", action="store_true")
    args = parser.parse_args()
    main(args)
