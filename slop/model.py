"""Shared model and tokenization settings for training and prediction."""

BASE_MODEL = "bert-base-cased"
MAX_LENGTH = 512
STRIDE = 64  # tokens of overlap between windows, so sentences at a
             # boundary are seen whole by at least one window

ID2LABEL = {0: "human", 1: "ai"}
LABEL2ID = {"human": 0, "ai": 1}


def window_encode(tokenizer, texts, labels=None):
    """Tokenize texts into overlapping windows of at most MAX_LENGTH tokens.

    Long texts become several windows. Returns a dict of lists; `owner` maps
    each window back to the index of the text it came from.
    """
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=MAX_LENGTH,
        stride=STRIDE,
        return_overflowing_tokens=True,
    )
    owner = enc.pop("overflow_to_sample_mapping")
    out = {k: enc[k] for k in ("input_ids", "token_type_ids", "attention_mask")}
    out["owner"] = list(owner)
    if labels is not None:
        out["labels"] = [labels[i] for i in owner]
    return out
