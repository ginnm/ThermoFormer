# ProPrime

```python
from transformers import AutoModel, AutoTokenizer
model_path = "AI4Protein/ThermoFormer"

model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

seqs = [
    "ADHJSJHSLGJSGLSGGAD",
    "MSJKFHLSKGJSG",
    "SDKHF"
]

inputs = tokenizer(seqs, padding=True, return_tensors="pt")
outputs = model(**inputs)
print(outputs.predicted_values)
```