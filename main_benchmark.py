# main_conversion.py
"""
Model export utilities.
- Exports to TorchScript (script or trace)
- Notes how to export to CoreML using coremltools (optional)
Usage:
python main_conversion.py --ckpt checkpoints/best...pth --out-model models/mobilevit_malaria_ts.pt
"""
import argparse
import torch
from pathlib import Path
from main_train import EnhancedMobileViT

def export_torchscript(ckpt_path, out_path, img_size=224):
    device = torch.device("cpu")
    model = EnhancedMobileViT(num_classes=2, img_size=img_size, pretrained=False, cbam=False, fusion=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state.get("model_state", state))
    model.eval()
    example = torch.randn(1, 3, img_size, img_size)
    # Trace the model
    traced = torch.jit.trace(model, example)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    traced.save(out_path)
    print(f"Saved TorchScript traced model to: {out_path}")

def main(args):
    export_torchscript(args.ckpt, args.out)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()
    main(args)

# NOTE: CoreML conversion (optional)
# If you want CoreML, see coremltools and use:
#   import coremltools as ct
#   traced = torch.jit.load("model_ts.pt")
#   mlmodel = ct.convert(traced, inputs=[ct.ImageType(name="input_1", shape=(1,3,224,224), scale=1/255.0)])
# Save mlmodel: mlmodel.save("MobileViTMalaria.mlmodel")
#
# Make sure to test inference numerically after conversion.
