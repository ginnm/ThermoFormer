from argparse import ArgumentParser
from model.modeling_thermoformer import ThermoFormer
from model.tokenization_thermoformer import ThermoFormerTokenizer
import torch
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import pandas as pd


def parse_args():
    psr = ArgumentParser()
    psr.add_argument(
        "--model_name",
        type=str,
        default="GinnM/ThermoFormer",
    )
    psr.add_argument("--file", type=str, required=True)
    psr.add_argument("--batch_size", type=int, default=1)
    psr.add_argument("--output", type=str, default=None)
    args = psr.parse_args()
    return args


def load_model(args):
    tokenizer = ThermoFormerTokenizer()
    model = ThermoFormer.from_pretrained(args.model_name)
    model.eval()
    model.cuda()
    return tokenizer, model


def load_data(args):
    file = Path(args.file)
    df = pd.read_csv(file)
    sequences = df["sequence"].values.tolist()
    df = pd.DataFrame(
        {"S": sequences, "L": [len(i) for i in sequences]}
    )
    df = df.sort_values("L", ascending=False)
    return df


def infer(model, tokenizer, df, batch_size=2, truncate_length=2048):
    results = []
    sequences = df["S"].values.tolist()
    over_length = [i for i in sequences if len(i) > truncate_length]
    if over_length:
        print(f"Warning: {len(over_length)} sequences are longer than {truncate_length}")
    sequences = [i[:truncate_length] for i in sequences]
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch = sequences[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = inputs["input_ids"].cuda()
        attention_mask = inputs["attention_mask"].cuda()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
        results += outputs.predicted_values.cpu().numpy().tolist()
    df["predicted_values"] = results
    return df


@torch.no_grad()
def main():
    args = parse_args()
    df = load_data(args)
    tokenizer, model = load_model(args)
    df = infer(model, tokenizer, df, batch_size=args.batch_size)
    df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
