# ThermoFormer: Temperature-Aware Protein Representations 🏆🔥

![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=flat-square)
![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-orange.svg?style=flat-square)
![Transformers](https://img.shields.io/badge/HF%20Transformers-compatible-green.svg?style=flat-square)
![Biopython](https://img.shields.io/badge/Biopython-yes-brightgreen.svg?style=flat-square)

ThermoFormer is a model built on top of the [Hugging Face Transformers](https://github.com/huggingface/transformers) and PyTorch frameworks. It is designed to learn **temperature-aware representations** from millions of annotated protein sequences. 

---

## Prerequisites

- **Environment**:  
  - Python 3.8+
  - [PyTorch 2.4+](https://pytorch.org)
  - [Transformers (Hugging Face)](https://github.com/huggingface/transformers)
  - [Biopython](https://biopython.org/)

---

## 🏗️ Building the OGT-labeled Dataset

### Step 1: Download Growth Temperatures for 21,498 Microorganisms
1. Download the dataset [temperature_data.tsv](https://zenodo.org/records/1175609).  
2. Place `temperature_data.tsv` into the `ogt_data/` directory.

### Step 2: Download UniRef100
1. Download [Uniref100.fasta](https://www.uniprot.org/help/downloads) from UniProt.  
2. Place `Uniref100.fasta` into the `ogt_data/` directory.

### Step 3: Check the Directory
```bash
ls ogt_data
uniref100.fasta  temperature_data.tsv
```

### Step 4: Generate the OGT-Annotated Dataset
```bash
python build_ogt_dataset.py \
    --fasta_file ogt_data/uniref100.fasta \
    --ogt_file ogt_data/temperature_data.tsv \
    --output_file ogt_data/annotated.csv
```

---

## 🚀 Running Inference
Once you have the annotated dataset, you can run inference:

```bash
python inference.py --file ogt_data/annotated.csv --output ogt_data/infer.csv
```

---

## 🧪 Using ThermoFormer in Python

```python
from model.modeling_thermoformer import ThermoFormer
from model.tokenization_thermoformer import ThermoFormerTokenizer

tokenizer = ThermoFormerTokenizer()
model = ThermoFormer.from_pretrained("GinnM/ThermoFormer")

# Example usage:
sequence = "MSSKLLL..."
inputs = tokenizer(sequence, return_tensors="pt")
outputs = model(**inputs)
```

---

## 📜 Citation

If you use ThermoFormer in your research, please cite:

```bibtex
@inproceedings{
li2024learning,
title={Learning temperature-aware representations from millions of annotated protein sequences},
author={Mingchen Li and Liang Zhang and Zilan Wang and Bozitao Zhong and Pan Tan and Jiabei Cheng and Bingxin Zhou and Liang Hong and Huiqun Yu},
booktitle={Neurips 2024 Workshop Foundation Models for Science: Progress, Opportunities, and Challenges},
year={2024},
url={https://openreview.net/forum?id=sOU2rNqo90}
}
```

---

**Happy hacking!** ✨