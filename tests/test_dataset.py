from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

dset = load_from_disk('data/')

print(dset)
print(dset['train'])

example = dset['train'][0]

print(example)

dset['train'].set_format(type="torch", columns=['input_ids', 'attention_mask'])

train_dl = DataLoader(dset['train'], batch_size=2, shuffle=True)

for batch in train_dl:
    print(batch)
    break