import os

import pandas as pd

base = "/storage/counting-manifolds"
dataset = "fineweb"
fname = "lm_metrics.csv"


def func(models):
    dfs = []

    for display_name, name in models.items():
        path = f"{base}/{name}/{dataset}/{fname}"
        df = pd.read_csv(path)
        df = df.T
        df.columns = [display_name]
        dfs.append(df)

    combined = pd.concat(dfs, axis=1).T

    for column in combined.columns:
        if "acc" in column:
            combined[column] *= 100

    return combined


family_name = "qwen3s"

if family_name == "pythias":
    models = {
        "Pythia-14m": "pythia-14m-deduped",
        "Pythia-70m": "pythia-70m-deduped",
        "Pythia-160m": "pythia-160m-deduped",
        "Pythia-410m": "pythia-410m-deduped",
        "Pythia-1B": "pythia-1b-deduped",
        "Pythia-1.4B": "pythia-1.4b-deduped",
        "Pythia-1.8B": "pythia-2.8b-deduped",
    }
elif family_name == "gpt2s":
    models = {
        "GPT2-Small": "gpt2",
        "GPT-Medium": "gpt2-medium",
        "GPT2-Large": "gpt2-large",
        "GPT2-XL": "gpt2-xl",
    }
elif family_name == "gemma3s":
    models = {
        "Gemma3-270m": "gemma-3-270m",
        "Gemma3-1B": "gemma-3-1b-pt",
        "Gemma3-4B": "gemma-3-4b-pt",
        "Gemma3-12B": "gemma-3-12b-pt",
    }
elif family_name == "gemma2s":
    models = {
        "Gemma2-2B": "gemma-2-2b",
        "Gemma2-9B": "gemma-2-9b",
    }
elif family_name == "qwen3s":
    models = {
        "Qwen3-0.6B": "Qwen3-0.6B",
        "Qwen3-1.7B": "Qwen3-1.7B",
        "Qwen3-4B": "Qwen3-4B",
        "Qwen3-8B": "Qwen3-8B",
        "Qwen3-14B": "Qwen3-14B",
    }
else:
    raise NotImplementedError(family_name)


combined = func(models)
os.makedirs(
    os.path.join(base, family_name, dataset),
    exist_ok=True,
)
combined.to_csv(os.path.join(base, family_name, dataset, "lm_metrics.csv"))
