import torch
import time
import sys
from retro_pytorch.retro_pytorch import RETRO
from retro_pytorch.training import TrainingWrapper
import gc
from tqdm import tqdm
from datetime import datetime
#%%

### Important notes!!!!
### !!!! I had to change autofaiss files C:\Users\Timur.Galimzyanov\anaconda3\lib\site-packages\autofaiss\indices\index_utils.py
### line 23     with NamedTemporaryFile() as tmp_file: -->> with NamedTemporaryFile(delete=False) as tmp_file:

#%%


'''
Creates embeddings and finding knns for each chuncks in dataset
'''

# instantiate RETRO, fit it into the TrainingWrapper with correct settings

retro = RETRO(
    max_seq_len = 512,                      # max sequence length
    enc_dim = 896,                           # encoder model dimension 896
    enc_depth = 3,                           # encoder depth
    dec_dim = 768,                           # decoder model dimensions
    dec_depth = 12,                          # decoder depth
    dec_cross_attn_layers = (1, 3, 6, 9),    # decoder cross attention layers (with causal chunk cross attention)
    heads = 8,                               # attention heads
    dim_head = 64,                           # dimension per head
    dec_attn_dropout = 0.25,                 # decoder attention dropout
    dec_ff_dropout = 0.25                    # decoder feedforward dropout
).cuda()


folder = '../../data/texts_folder_val/'

wrapper_val = TrainingWrapper(
    retro = retro,                                 # path to retro instance
    knn = 2,                                       # knn (2 in paper was sufficient)
    chunk_size = 64,                               # chunk size (64 in paper)
    documents_path = folder,              # path to folder of text
    glob = '**/*.txt',                             # text glob
    processed_stats_json_path = folder + 'processed-stats.json',
    chunks_memmap_path = folder + 'train.chunks.dat',     # path to chunks
    seqs_memmap_path = folder + 'train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = folder + 'train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    #max_chunks = 1_000_000,                        # maximum cap to chunks
    #max_seqs = 100_000,                            # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = '10G',
    current_memory_available = '32G'
)
print('--------------VAL read --------------')
#%%
folder = '../../data/texts_folder_train/'

gc.collect()
torch.cuda.empty_cache()

wrapper_train = TrainingWrapper(
    retro = retro,                                 # path to retro instance
    knn = 2,                                       # knn (2 in paper was sufficient)
    chunk_size = 64,                               # chunk size (64 in paper)
    documents_path = folder,              # path to folder of text
    glob = '**/*.txt',                             # text glob
    processed_stats_json_path = folder + 'processed-stats.json',
    chunks_memmap_path = folder + 'train.chunks.dat',     # path to chunks
    seqs_memmap_path = folder + 'train.seq.dat',          # path to sequence data
    doc_ids_memmap_path = folder + 'train.doc_ids.dat',   # path to document ids per chunk (used for filtering neighbors belonging to same document)
    #max_chunks = 1_000_000,                        # maximum cap to chunks
    #max_seqs = 100_000,                            # maximum seqs
    knn_extra_neighbors = 100,                     # num extra neighbors to fetch
    max_index_memory_usage = '10G',
    current_memory_available = '32G'
)
print('--------------TRAIN read --------------')
gc.collect()
torch.cuda.empty_cache()

#%%
print('--------------TRAINING --------------')
### TRAIN
# get the dataloader and optimizer (AdamW with all the correct settings)
batch_size = 4
train_dl = iter(wrapper_train.get_dataloader(batch_size = batch_size, shuffle = False)) #, shuffle = True
val_dl = iter(wrapper_val.get_dataloader(batch_size = batch_size, shuffle = False)) #, shuffle = True
optim = wrapper_train.get_optimizer(lr = 3e-4, wd = 0.01)
 
#%%

losses_train = []
losses_val = []
i = 0
f_train = open("losses_train_no_retrieve.txt", "a")
f_val = open("losses_val_no_retrieve.txt", "a")
current_time = datetime.now()
f_train.write(f"\n ------- NEW TRAINING {str(current_time)}, batch size = {batch_size} ------- \n")
f_val.write(f"\n ------- NEW TRAINING {str(current_time)}, batch size = {batch_size} ------- \n")

tt = time.time()

freq_val = 3000
num_val = 280

for seq, retrieved in tqdm(train_dl, ncols=80):
    
    i += 1
    seq = seq.cuda()
    #retrieved = retrieved.cuda()
    #seq, retrieved = map(lambda t: t.cuda(), next(train_dl))    
    loss = retro(
        seq,
        retrieved = None,
        return_loss = True
    )
    
    # one gradient step
    
    loss.backward()
    optim.step()
    optim.zero_grad()
    losses_train.append(loss.item())
    f_train.write(str(loss.item()) + "\n")
    
    losses_val_cur = []

    if i%freq_val == 0:
        del loss
        retro.eval()
        f_train.flush()
        gc.collect()
        torch.cuda.empty_cache()
        print('------ Validation ------')
        j = 0
        for seq, retrieved in tqdm(val_dl, total=num_val, ncols=80):
            
            j += 1
            seq = seq.cuda()
            #retrieved = retrieved.cuda()
 
            loss = retro(
                seq,
                retrieved = None,
                return_loss = True
            )

            losses_val_cur.append(loss.item())
            
            if j>=num_val:
                break
        if len(losses_val_cur)!=0:
            loss_cur = sum(losses_val_cur)/(len(losses_val_cur))
            losses_val.append(loss_cur)
            f_val.write(str(loss_cur) + "\n")
            f_val.flush()
            del loss
        retro.train()
        gc.collect()
        torch.cuda.empty_cache()

time_used = time.time() - tt
print(f'Time used = {time_used:.2f} s')

#%%





















