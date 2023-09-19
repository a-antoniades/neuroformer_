from tqdm.notebook import tqdm
from torch.utils.data.dataloader import default_collate

def my_collate_fn(batch):
    try:
        return default_collate(batch)
    except RuntimeError as e:
        print(f"There was an error with collating the batch: {str(e)}")
        for idx, item in enumerate(batch):
            recursive_print(item)
        raise e  # Re-raise the exception to stop the training


loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0, collate_fn=my_collate_fn)
pbar = tqdm(loader, total=len(loader), colour='purple')
for x, y in pbar:
    continue


from tqdm import tqdm
import cProfile

def profile_dataloader(dataloader, max_iters=1):
    """
    dataloader: instance of your DataLoader
    max_iters: number of batches to fetch from the DataLoader
    """
    pr = cProfile.Profile()
    pr.enable()

    pbar = tqdm(enumerate(dataloader), total=max_iters)
    for i, _ in pbar:
        if i >= max_iters:
            break

    pr.disable()
    pr.dump_stats("dataloader_2.profile")  # saving stats to a file

dataloader = DataLoader(train_dataset, batch_size=32*6, shuffle=True, num_workers=0)
profile_dataloader(dataloader)

import pstats

p = pstats.Stats('dataloader.profile')
p.sort_stats('cumulative').print_stats(10)  # print top 10 functions that took the most time