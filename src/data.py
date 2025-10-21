
def get_banking_data(share, tokenizer=None, seed=42):

    from datasets import load_dataset
    dataset = load_dataset("banking77")
    if 'validation' not in dataset.keys():
        dataset_split = dataset['train'].train_test_split(seed=seed, test_size=share, stratify_by_column = 'label')
        dataset = {'train': dataset_split['train'], 'valid': dataset_split['test'], 'test': dataset['test']}
    if tokenizer:
        dataset = {split: content.map(tokenizer, batched=True, remove_columns = 'text') for split, content in dataset.items()}

    return dataset
