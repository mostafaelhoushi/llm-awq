import torch
from datasets import Dataset, load_dataset

prompt1 = """- Fiction: \"Once upon a time, a girl named Alice was living alone on an island. One day, she met a wizard ...\"
- News: \"The United Nations held its General Assembly meeting this year amid multiple world crises and wars. In his speech, the General Secretary called for ...\"
- Code: `public static void main(String[] args) \nSystem.out.println(``Hello world!'');\n`
- Math: (5.2 + 2.7) / 0.6 - 1.9 * 2.2 =
- Facts: \"The capital of Egypt is Cairo. It is the largest city in the region and is home to...\"
"""


def get_calib_dataset(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif data == "prompt1":
        dataset = Dataset.from_list([{"text": prompt1}])
    else:
        raise NotImplementedError
    dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        line = data["text"]
        line = line.strip()
        line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    if n_split > 0:
        return [
            cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
        ]
    else:
        return [cat_samples]
