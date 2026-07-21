import torch

from transformers import AutoModel, AutoTokenizer

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
DEVICE_TYPE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE_TYPE).eval()


def embed(input):
    tokens = tokenizer(input, return_tensors="pt", truncation=True).to(DEVICE_TYPE)

    with torch.no_grad():
        output = model(**tokens)

    vectors = output.last_hidden_state.squeeze(0)

    return vectors.cpu().numpy()
