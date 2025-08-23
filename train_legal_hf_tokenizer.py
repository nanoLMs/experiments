import os
import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.processors import TemplateProcessing

# Paths
DATASET_JSON = "data/Contextualized_Bangladesh_Legal_Acts.json"
ACTS_DIR = "data/"
TOKENIZER_OUT_DIR = "hf_legal_tokenizer/"
VOCAB_SIZE = 32000  # You can adjust this

# 1. Load dataset JSON
with open(DATASET_JSON, "r", encoding="utf-8") as f:
    dataset = json.load(f)

# 2. Collect all act file paths
act_files = [os.path.join(ACTS_DIR, act["source_file"]) for act in dataset["acts"] if "source_file" in act]

# 3. Read all texts
texts = []
for file in act_files:
    if os.path.exists(file):
        with open(file, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Try to extract main text content
                if isinstance(data, dict):
                    for k in ["text", "content", "body", "act_text"]:
                        if k in data:
                            texts.append(data[k])
                elif isinstance(data, str):
                    texts.append(data)
            except Exception:
                # Fallback: treat as plain text
                f.seek(0)
                texts.append(f.read())

# 4. Write all texts to a temporary file (for training)
tmp_corpus = "legal_corpus.txt"
with open(tmp_corpus, "w", encoding="utf-8") as f:
    for t in texts:
        f.write(t.strip() + "\n")

# 5. Train a new tokenizer (BPE)
tokenizer.decoder = decoders.BPEDecoder()

tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[HEADER]", "[REASONING]"])
tokenizer.train([tmp_corpus], trainer)

# Set post-processor for special tokens
tokenizer.post_processor = TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[("[CLS]", tokenizer.token_to_id("[CLS]")), ("[SEP]", tokenizer.token_to_id("[SEP]"))],
)

tokenizer.decoder = decoders.BPEDecoder()

# 6. Save the tokenizer in Hugging Face format
os.makedirs(TOKENIZER_OUT_DIR, exist_ok=True)
tokenizer.save(os.path.join(TOKENIZER_OUT_DIR, "tokenizer.json"))

# Save config for transformers
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=os.path.join(TOKENIZER_OUT_DIR, "tokenizer.json"),
                                       unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]",
                                       additional_special_tokens=["[HEADER]", "[REASONING]"])
hf_tokenizer.save_pretrained(TOKENIZER_OUT_DIR)

print(f"Tokenizer trained and saved to {TOKENIZER_OUT_DIR}")
