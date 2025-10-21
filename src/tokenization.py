
def get_tokenizer(model_name: str, length=64):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def apply_tokenizer(batch):
        return tokenizer(batch['text'], padding=True, truncation=True, max_length = length)

    return tokenizer, apply_tokenizer 
