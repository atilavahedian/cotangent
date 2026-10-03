"""Load a research snapshot and generate bytes with exact inference.

This is a qualitative smoke test, not a language-model capability benchmark.
"""
from pathlib import Path
import argparse
import json
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from cotangent.backward import Policy
from cotangent.model import ModelConfig, Transformer
from cotangent.runtime import model_hash


def load(checkpoint, device="cpu"):
    saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
    model=Transformer(ModelConfig(**saved["model_config"]),Policy(mode="native"))
    model.load_state_dict(saved["model"],strict=True)
    model.eval().to(device)
    return model,saved


@torch.no_grad()
def generate(model,prompt,new_bytes,seed,device):
    torch.manual_seed(seed)
    values=list(prompt.encode("utf-8"))
    if not values:
        raise ValueError("Prompt must not be empty")
    for _ in range(new_bytes):
        x=torch.tensor([values[-model.cfg.sequence:]],device=device,dtype=torch.long)
        logits,_=model(x)
        probabilities=torch.softmax(logits[0,-1].float()/.8,dim=-1)
        assert torch.isfinite(probabilities).all()
        values.append(int(torch.multinomial(probabilities,1).item()))
    return bytes(values).decode("utf-8",errors="replace")


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("checkpoint",type=Path)
    parser.add_argument("--prompt",default="The history of science ")
    parser.add_argument("--new-bytes",type=int,default=100)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--device",choices=["cpu","mps"],default="cpu")
    args=parser.parse_args()
    torch.set_num_threads(4)
    model,saved=load(args.checkpoint,args.device)
    print(json.dumps({"trained_method":saved["mode"],"trained_seed":saved["seed"],
                      "model_sha256":model_hash(model),"inference":"exact",
                      "text":generate(model,args.prompt,args.new_bytes,args.seed,args.device)},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
