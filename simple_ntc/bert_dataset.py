import torch
from torch.utils.data import Dataset


# The following codes may not be the same as the instruction for Hugging Face
# because they are designed to concatenate paddings as many as the difference between each sample and the longest one for every mini batch.
# It would be much better to divide the whole data into mini batches in terms of the number of tokens, which is efficient memorywise.
# That is, the mini-batch size is subject to change for every mini batches. -> the best approach but not easy to implement
# However, in this case, we divide the entire data into mini batches based on the predetermined mini-batch size.
class TextClassificationCollator():

    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples): # samples: list type
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        # Refer to the call magic method in the following URL for the following lines.
        # https://huggingface.co/docs/transformers/main_classes/tokenizer
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        ) # the resulting tensor size -> (batch_size, max_length, 1)

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'], # Paddings must not be involved in attention operations.
            'labels': torch.tensor(labels, dtype=torch.long),
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):

    # Override three methods below.
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item): # Call (iterate) this magic method as many as the mini-batch size.
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        } # Each output will be appended in a list for a mini batch.
          # And the list is the input for the call magic method of TextClassificationCollator class.