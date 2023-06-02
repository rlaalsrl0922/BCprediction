import os
import gc
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn
import numpy as np
from BPM_MT import BPM_MT
import json
from time import time
import datetime

from transformers import BertModel, AutoTokenizer
import torchvision.transforms as transforms
from torch.utils.data import Subset
from ETRI_Dataset import ETRI_Corpus_Dataset
from SWBD_Dataset import SWBD_Dataset
from HuBert import HuBert
from Audio_LSTM import Audio_LSTM

from transformers import AutoTokenizer, AutoModelForPreTraining

class Trainer:
    def __init__(self, args) -> None:
        print(args)
        self.model = args.model
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.world_size = args.world_size
        self.is_MT = args.is_MT
        self.language = args.language
        self.audio = args.audio

        self.seed = args.seed
        self.distributed = False
        self.rank = args.rank
        self.ngpus_per_node = torch.cuda.device_count()
        self.world_size = args.world_size * self.ngpus_per_node
        self.distributed = self.world_size > 1
        self.lossfunction = args.lossfunction
        self.hierarchy = False

        self.batch_size = int(self.batch_size / self.world_size)

        print("is_MT: ", self.is_MT)

        if os.environ.get("MASTER_ADDR") is None:
            os.environ["MASTER_ADDR"] = "localhost"
        if os.environ.get("MASTER_PORT") is None:
            os.environ["MASTER_PORT"] = "8888"

    def run(self):
        if self.distributed:
            mp.spawn(self._run, nprocs=self.world_size, args=(self.world_size,))
        else:
            self._run(0, 1)

    def _run(self, rank, world_size):

        self.local_rank = rank
        self.rank = self.rank * self.ngpus_per_node + rank
        self.world_size = world_size

        self.init_distributed()
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed) # if use multi-GPU
            cudnn.deterministic = True
            cudnn.benchmark = False
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
                
        
        if self.language == 'koBert':
            tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
            bert = BertModel.from_pretrained("skt/kobert-base-v1", add_pooling_layer=False, output_hidden_states=True, output_attentions=False)
            sentiment_dict = json.load(open('/data/minjae/BC/SentiWord_info.json', encoding='utf-8-sig', mode='r'))
            self.num_class = 4
        elif self.language == 'Bert':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, output_hidden_states=True, output_attentions=False)
            sentiment_dict = {}
            self.num_class = 2
        elif self.language == 'ELECTRA':
            tokenizer = AutoTokenizer.from_pretrained("google/electra-base-discriminator")
            bert = AutoModelForPreTraining.from_pretrained("google/electra-base-discriminator")
            sentiment_dict = {}
            self.num_class = 2
        else:
            raise NotImplementedError
        
        if self.audio == 'LSTM':
            audio_model = Audio_LSTM()
        elif self.audio == 'HuBert':
            audio_model = HuBert()

        tf = transforms.ToTensor()

        if self.language == 'koBert':
            dataset = ETRI_Corpus_Dataset(path = '/local_datasets', tokenizer=tokenizer, transform=tf, length=1.5)
        else :
            dataset = SWBD_Dataset(path = '/local_datasets', tokenizer=tokenizer, length=1.5)
            
        self.train_dataset = Subset(dataset, range(0, int(len(dataset)*0.8)))
        self.val_dataset = Subset(dataset, range(int(len(dataset)*0.8), len(dataset)))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True, num_replicas=self.world_size, rank=self.rank)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False, num_replicas=self.world_size, rank=self.rank)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.val_sampler, num_workers=self.num_workers)

        if self.is_MT:
            self.model = BPM_MT(language_model=bert, audio_model=audio_model, sentiment_dict=sentiment_dict, num_class=self.num_class)
        else:
            self.model = BPM_MT(language_model=bert, audio_model=audio_model, sentiment_dict=None, num_class=self.num_class)
        
        self.model = self.model.to(self.local_rank)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # Get the model parameters divided into two groups : bert and others
        language_model_params = []
        audio_model_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'language_model' in name:
                language_model_params.append(param)
            elif 'audio_model' in name:
                audio_model_params.append(param)
            else:
                other_params.append(param)
                
        adam_l_optimizer = torch.optim.Adam(language_model_params, lr=0.000001, weight_decay = 0.0001)#, weight_decay=0.00001)
        adam_a_optimizer = torch.optim.Adam(audio_model_params, lr=0.000001, weight_decay = 0.0001)#, weight_decay=0.00001)
        sgd_optimizer = torch.optim.Adam(other_params, lr=0.00005, weight_decay=0.001)
        
        print(self.lossfunction)
        for epoch in range(20):
            start=time()
            loss = 0
            for b, batch in enumerate(self.train_dataloader):
                for key in batch:
                    batch[key] = batch[key].cuda()
                with torch.cuda.amp.autocast():
                    y = self.model(batch)
                # Get the logit from the model
                logit     = y["logit"]
                if self.is_MT:
                    sentiment = y["sentiment"]
                
                # Calculate the loss
                loss = self.model.pretext_forward(batch)

                loss.backward()# Update the model parameters
                adam_l_optimizer.step()
                adam_a_optimizer.step()
                sgd_optimizer.step()

                # Zero the gradients
                adam_l_optimizer.zero_grad()
                adam_a_optimizer.zero_grad()
                sgd_optimizer.zero_grad()

#                print("Epoch : {}, Loss : {}".format(epoch, loss.item()))
                loss    += loss.item() * len(batch["label"])
            
                gc.collect()
            sec = time() -start
            times = str(datetime.timedelta(seconds=sec))
            short = times.split(".")[0]
                
            loss     /= len(self.train_dataloader)    
            print("Epoch : {}, Loss : {}, Time taken : {} ".format(epoch, loss, short))
                
        # Training loop
        
        for epoch in range(40):
            start=time()
            for b, batch in enumerate(self.train_dataloader):
                # Move the batch to GPU if CUDA is available
                for key in batch:
                    batch[key] = batch[key].cuda()
                with torch.cuda.amp.autocast():
                    y = self.model(batch)
                # Get the logit from the model
                if self.language=='koBert':
                    logit     = y["logit"]
                    if self.hierarchy:
                        c2 = y['c2']
                        c3 = logit
                else:
                    logit     = y['logit']
                
                if self.is_MT:
                    sentiment = y["sentiment"]

                # Calculate the loss
                if self.lossfunction == 'count':
                    loss_BC = F.cross_entropy(logit, batch["label"], reduction='none')
                    unq, cnt = batch["label"].unique(return_counts=True)
                    unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                    loss_BC = (loss_BC * unq).mean()
                    
                elif self.lossfunction == 'weight':
                    if self.language == 'koBert':
                        class_weight = torch.tensor([1,1.22,9.5,13.43]).float().cuda()
                    else:
                        class_weight = torch.tensor([1,2.5]).float().cuda()
                    loss_BC = F.cross_entropy(logit, batch["label"],weight=class_weight)
                    
                elif self.lossfunction == 'focal':
                    loss = criterion(logit,batch["label"])
                    loss_BC = loss.requires_grad_(True)
                    
                elif self.lossfunction == 'focalcount':
                    loss = criterion(logit,batch["label"])
                    unq, cnt = batch["label"].unique(return_counts=True)
                    unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                    loss = (loss * unq).mean()
                    loss_BC = loss.requires_grad_(True)
                    
                elif self.lossfunction == 'cosine':
                    unq, cnt = batch["label"].unique(return_counts=True)
                    unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                    class_labels = np.arange(self.num_class)
                    weights = np.cos(class_labels / num_classes * np.pi)
                    weights = torch.tensor(weights / np.sum(weights))
                    loss_BC = F.cross_entropy(logit, batch['label'], weight=wegihts)
                    
                elif self.lossfunction == 'countweight':
                    loss_BC = F.cross_entropy(logit, batch["label"], reduction='none')
                    cnt = torch.zeros(self.num_class).float().cuda()
                    for i in batch['label']:
                        cnt[i] += 1
                    for i in range(self.num_class):
                        if cnt[i]==0:
                            cnt[i]=1
                    cnt *= torch.tensor([18.45,9.5,1.22,1]).float().cuda()
                    unq = torch.tensor([0,1,2,3]).cuda()
                    unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                    loss_BC = (loss_BC * unq).mean()
                    
                elif self.lossfunction == 'conditional':
                    c2_m0 = (logit.argmax(dim=1) == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                    c2_m1 = (logit.argmax(dim=1) == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                    c2_m2 = (logit.argmax(dim=1) == 2).nonzero(as_tuple=True)[0].cpu().numpy()
                    c2_m3 = (logit.argmax(dim=1) == 3).nonzero(as_tuple=True)[0].cpu().numpy()

                    # Get corresponding indices in labels
                    ans_m0 = (batch['label'] == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                    ans_m1 = (batch['label'] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                    ans_m2 = (batch['label'] == 2).nonzero(as_tuple=True)[0].cpu().numpy()
                    ans_m3 = (batch['label'] == 3).nonzero(as_tuple=True)[0].cpu().numpy()

                    l = logit
                    b = batch['label']

                    common0_0 = np.intersect1d(c2_m0, ans_m0)
                    mask0_0 = torch.from_numpy(common0_0)
                    common0_1 = np.intersect1d(c2_m0, ans_m1)
                    mask0_1 = torch.from_numpy(common0_1)
                    common0_2 = np.intersect1d(c2_m0, ans_m2)
                    mask0_2 = torch.from_numpy(common0_2)
                    common0_3 = np.intersect1d(c2_m0, ans_m3)
                    mask0_3 = torch.from_numpy(common0_3)
                    
                    common1_0 = np.intersect1d(c2_m1, ans_m0)
                    mask1_0 = torch.from_numpy(common1_0)
                    common1_1 = np.intersect1d(c2_m1, ans_m1)
                    mask1_1 = torch.from_numpy(common1_1)
                    common1_2 = np.intersect1d(c2_m1, ans_m2)
                    mask1_2 = torch.from_numpy(common1_2)
                    common1_3 = np.intersect1d(c2_m1, ans_m3)
                    mask1_3 = torch.from_numpy(common1_3)
                    
                    common2_0 = np.intersect1d(c2_m2, ans_m0)
                    mask2_0 = torch.from_numpy(common2_0)
                    common2_1 = np.intersect1d(c2_m2, ans_m1)
                    mask2_1 = torch.from_numpy(common2_1)
                    common2_2 = np.intersect1d(c2_m2, ans_m2)
                    mask2_2 = torch.from_numpy(common2_2)
                    common2_3 = np.intersect1d(c2_m2, ans_m3)
                    mask2_3 = torch.from_numpy(common2_3)
                    
                    common3_0 = np.intersect1d(c2_m3, ans_m0)
                    mask3_0 = torch.from_numpy(common3_0)
                    common3_1 = np.intersect1d(c2_m3, ans_m1)
                    mask3_1 = torch.from_numpy(common3_1)
                    common3_2 = np.intersect1d(c2_m3, ans_m2)
                    mask3_2 = torch.from_numpy(common3_2)
                    common3_3 = np.intersect1d(c2_m3, ans_m3)
                    mask3_3 = torch.from_numpy(common3_3)
                    

                    l1 = l[mask1_2]
                    l2 = b[mask1_2]
                    l3 = l[mask1_3]
                    l4 = b[mask1_3]
                    l5 = l[mask2_1]
                    l6 = b[mask2_1]
                    l7 = l[mask2_3]
                    l8 = b[mask2_3]
                    l9 = l[mask3_1]
                    l10 = b[mask3_1]
                    l11 = b[mask3_2]
                    l12 = b[mask3_2]
                    
                    b1 = l[mask0_1]
                    b2 = b[mask0_1]
                    b3 = l[mask0_2]
                    b4 = b[mask0_2]
                    b5 = l[mask0_3]
                    b6 = b[mask0_3]
                    b7 = l[mask3_0]
                    b8 = b[mask3_0]
                    b9 = l[mask2_0]
                    b10 = b[mask2_0]
                    b11 = l[mask1_0]
                    b12 = b[mask1_0]
                    
                    
                    n_w = len(b1)+len(b3)+len(b5)+1e-10
                    c_w = len(l1)+len(l3)+len(b11)+1e-10
                    u_w = len(l5)+len(l7)+len(b9)+1e-10
                    e_w = len(b7)+len(l9)+len(l11)+1e-10
                    
                    #weights = torch.tensor([n_w,c_w,u_w,e_w]).float().cuda() + torch.tensor([1,1.22,9.5,18.45]).float().cuda()
                    weights = torch.tensor([n_w,c_w,u_w,e_w]).float().cuda()
                    loss_BC = F.cross_entropy(logit,batch['label'],weight=weights,reduction='none')
                    unq, cnt = batch["label"].unique(return_counts=True)
                    unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                    loss_BC = (loss_BC * unq).mean()

                    # 가중치 계산
                    #unq, cnt = batch["label"].unique(return_counts=True)
                    #unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                    
                    #loss_BC = F.cross_entropy(logit,batch['label'],reduction='none')
                    #print((loss_BC * weights).mean())
                    
                elif self.lossfunction == 'exponential':
                    c2_m0 = (logit.argmax(dim=1) == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                    c2_m1 = (logit.argmax(dim=1) == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                    ans_m0 = (batch['label'] == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                    ans_m1 = (batch['label'] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                    length=torch.tensor([len(ans_m0),len(ans_m1)]).cpu()
                    length2=torch.tensor([len(c2_m0),len(c2_m1)]).cpu()
                    if self.language =='koBert':
                        c2_m2 = (logit.argmax(dim=1) == 2).nonzero(as_tuple=True)[0].cpu().numpy()
                        c2_m3 = (logit.argmax(dim=1) == 3).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_m2 = (batch['label'] == 2).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_m3 = (batch['label'] == 3).nonzero(as_tuple=True)[0].cpu().numpy()
                        length=torch.tensor([len(ans_m0),len(ans_m1),len(ans_m2),len(ans_m3)]).cpu()
                        length2=torch.tensor([len(c2_m0),len(c2_m1),len(c2_m2),len(c2_m3)]).cpu()
                    
                    labels = batch['label'].view(-1)
                    num_classes = logit.size(1)  # 클래스 개수 (4)
                    indices = [torch.nonzero(labels == i, as_tuple=True)[0] for i in range(num_classes)] # 각 클래스별 인덱스 리스트

                    graph_x = torch.zeros_like(logit[:,0]).float()
                    for i, idx in enumerate(indices):
                        graph_x[idx] = (torch.max(length2) / torch.cosh(length[i]))+2
                        #graph_x[idx] = (torch.max(length)*2 / torch.cosh(length[i]))+1
                    
                    loss_BC = (graph_x * F.cross_entropy(logit, labels, reduction='none')).mean()

                    
                else:
                    if self.hierarchy==False:
                        loss_BC = F.cross_entropy(logit,batch['label'])
                    else:
                        batch_2={}
                        batch_2['label'] = batch['BClabel'].unsqueeze(-1)
                        
                        c2_1_m = (c2 > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
                        c2_0_m = (c2 < 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_0_m = (batch_2['label'] == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_1_m = (batch_2['label'] == 1).nonzero(as_tuple=True)[0].cpu().numpy()

                        common1 = np.intersect1d(c2_1_m, ans_1_m)
                        t_mask = torch.from_numpy(common1)
                        common2 = np.intersect1d(c2_0_m, ans_1_m)
                        f_mask = torch.from_numpy(common2)
                        common3 = np.intersect1d(c2_1_m, ans_0_m)
                        a_mask = torch.from_numpy(common3)
                        common4 = np.intersect1d(c2_0_m, ans_0_m)
                        b_mask = torch.from_numpy(common4)

                        c3_t = c3[t_mask]
                        a3_t = batch['label'][t_mask]

                        c3_f = c3[f_mask]
                        a3_f = batch['label'][f_mask]

                        c3_a = c3[a_mask]
                        a3_a = batch['label'][a_mask]

                        c3_b = c3[b_mask]
                        a3_b = batch['label'][b_mask]
                            
                        
                        # t,b = Poisitive, foreground / f,a = Negative, background
                        
                        #loss_BC = BCE(c2,batch["BClabel"].float().reshape(-1,1))
                        #P1 = criterion2(c3_t,a3_t) + criterion2(c3_b,a3_b)
                        #P2 = criterion(c3_f,a3_f) + criterion(c3_a,a3_a)
                        
                        # wrong batch
                        '''
                        n_w = self.batch_size-n_w
                        c_w = self.batch_size-c_w
                        u_w = self.batch_size-u_w
                        e_w = self.batch_size-e_w
                        weights = torch.tensor([n_w,c_w,u_w,e_w]).float().cuda()
                        loss_BC = F.cross_entropy(logit,batch['label'],weight=weights)
                        '''
                        
                        # wrong
                        # weights = torch.tensor([n_w,c_w,u_w,e_w]).float().cuda() + torch.tensor([1,1.22,9.5,18.45]).float().cuda()
                        # loss_BC = F.cross_entropy(logit,batch['label'],weight=weights)
                        
                        
                        # wrong count
                        
                        weights = torch.tensor([n_w,c_w,u_w,e_w]).float().cuda()
                        loss_BC = F.cross_entropy(logit,batch['label'],weight=weights,reduction='none')
                        unq, cnt = batch["label"].unique(return_counts=True)
                        unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                        loss_BC = (loss_BC * unq).mean()
                        

                
                if self.is_MT:
                    loss_SP = F.binary_cross_entropy(torch.sigmoid(sentiment), batch["sentiment"])
                    loss = 0.9 * loss_BC + 0.1 * loss_SP
                else:
                    loss = loss_BC
            
                accuracy = (logit.argmax(dim=-1) == batch["label"]).float().mean()

                # Backpropagation
                loss.backward()

                # Update the model parameters
                adam_l_optimizer.step()
                adam_a_optimizer.step()
                sgd_optimizer.step()

                # Zero the gradients
                adam_l_optimizer.zero_grad()
                adam_a_optimizer.zero_grad()
                sgd_optimizer.zero_grad()
                
                l, c = logit.argmax(dim=-1).unique(return_counts=True)
                
                gc.collect()
            
            with torch.no_grad():
                # Validation loop
                accuracy   = 0
                loss       = 0
                accuracyc2 = 0
                loss_c2     = 0
                accuracyc3 = 0
                loss_c3     = 0
                # F1 initialize
                if self.hierarchy:
                    tp2 = torch.tensor([0 for _ in range(2)],device=self.local_rank)
                    fp2 = torch.tensor([0 for _ in range(2)],device=self.local_rank)
                    fn2 = torch.tensor([0 for _ in range(2)],device=self.local_rank)
                    tn2 = torch.tensor([0 for _ in range(2)],device=self.local_rank)
                    tp3 = torch.tensor([0 for _ in range(4)],device=self.local_rank)
                    fp3 = torch.tensor([0 for _ in range(4)],device=self.local_rank)
                    fn3 = torch.tensor([0 for _ in range(4)],device=self.local_rank)
                    tn3 = torch.tensor([0 for _ in range(4)],device=self.local_rank)
                else:
                    tp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                    fp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                    fn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                    tn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)

                for batch in self.val_dataloader:
                    # Move the batch to GPU if CUDA is available
                    for key in batch:
                        batch[key] = batch[key].cuda()
                    with torch.cuda.amp.autocast():
                        y = self.model(batch)

                    # Get the logit from the model
                    if self.language=='koBert':
                        logit     = y["logit"]
                    else:
                        logit     = y['logit']
                    
                    if self.is_MT:
                        sentiment = y["sentiment"]
                        
                    if self.hierarchy:
                        c2 = y['c2']
                    # Calculate the loss
                    if self.lossfunction == 'count':
                        loss_BC = F.cross_entropy(logit, batch["label"], reduction='none')
                        unq, cnt = batch["label"].unique(return_counts=True)
                        unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                        loss_BC = (loss_BC * unq).mean()
                        
                    elif self.lossfunction == 'weight':
                        if self.language == 'koBert':
                            class_weight = torch.tensor([1,1.22,9.5,18.45]).float().cuda()
                        else:
                            class_weight = torch.tensor([1,2.5]).float().cuda()
                        loss_BC = F.cross_entropy(logit, batch["label"],weight=class_weight)
                        
                    elif self.lossfunction == 'focal':
                        loss = criterion(logit,batch["label"])
                        loss_BC = loss.requires_grad_(True)
                        
                    elif self.lossfunction == 'focalcount':
                        loss = criterion(logit,batch["label"])
                        unq, cnt = batch["label"].unique(return_counts=True)
                        unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                        loss = (loss * unq).mean()
                        loss_BC = loss.requires_grad_(True)
                    
                    elif self.lossfunction == 'countweight':
                        loss_BC = F.cross_entropy(logit, batch["label"], reduction='none')
                        cnt = torch.zeros(self.num_class).float().cuda()
                        for i in batch['label']:
                            cnt[i] += 1
                        for i in range(self.num_class):
                            if cnt[i]==0:
                                cnt[i]=1
                        cnt *= torch.tensor([18.45,9.5,1.22,1]).float().cuda()
                        unq = torch.tensor([0,1,2,3]).cuda()
                        unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                        loss_BC = (loss_BC * unq).mean()
                    elif self.lossfunction == 'conditional':
                        c2_m0 = (logit.argmax(dim=1) == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                        c2_m1 = (logit.argmax(dim=1) == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                        c2_m2 = (logit.argmax(dim=1) == 2).nonzero(as_tuple=True)[0].cpu().numpy()
                        c2_m3 = (logit.argmax(dim=1) == 3).nonzero(as_tuple=True)[0].cpu().numpy()

                        # Get corresponding indices in labels
                        ans_m0 = (batch['label'] == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_m1 = (batch['label'] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_m2 = (batch['label'] == 2).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_m3 = (batch['label'] == 3).nonzero(as_tuple=True)[0].cpu().numpy()

                        l = logit
                        b = batch['label']

                        common0_0 = np.intersect1d(c2_m0, ans_m0)
                        mask0_0 = torch.from_numpy(common0_0)
                        common0_1 = np.intersect1d(c2_m0, ans_m1)
                        mask0_1 = torch.from_numpy(common0_1)
                        common0_2 = np.intersect1d(c2_m0, ans_m2)
                        mask0_2 = torch.from_numpy(common0_2)
                        common0_3 = np.intersect1d(c2_m0, ans_m3)
                        mask0_3 = torch.from_numpy(common0_3)
                        
                        common1_0 = np.intersect1d(c2_m1, ans_m0)
                        mask1_0 = torch.from_numpy(common1_0)
                        common1_1 = np.intersect1d(c2_m1, ans_m1)
                        mask1_1 = torch.from_numpy(common1_1)
                        common1_2 = np.intersect1d(c2_m1, ans_m2)
                        mask1_2 = torch.from_numpy(common1_2)
                        common1_3 = np.intersect1d(c2_m1, ans_m3)
                        mask1_3 = torch.from_numpy(common1_3)
                        
                        common2_0 = np.intersect1d(c2_m2, ans_m0)
                        mask2_0 = torch.from_numpy(common2_0)
                        common2_1 = np.intersect1d(c2_m2, ans_m1)
                        mask2_1 = torch.from_numpy(common2_1)
                        common2_2 = np.intersect1d(c2_m2, ans_m2)
                        mask2_2 = torch.from_numpy(common2_2)
                        common2_3 = np.intersect1d(c2_m2, ans_m3)
                        mask2_3 = torch.from_numpy(common2_3)
                        
                        common3_0 = np.intersect1d(c2_m3, ans_m0)
                        mask3_0 = torch.from_numpy(common3_0)
                        common3_1 = np.intersect1d(c2_m3, ans_m1)
                        mask3_1 = torch.from_numpy(common3_1)
                        common3_2 = np.intersect1d(c2_m3, ans_m2)
                        mask3_2 = torch.from_numpy(common3_2)
                        common3_3 = np.intersect1d(c2_m3, ans_m3)
                        mask3_3 = torch.from_numpy(common3_3)
                        

                        l1 = l[mask1_2]
                        l2 = b[mask1_2]
                        l3 = l[mask1_3]
                        l4 = b[mask1_3]
                        l5 = l[mask2_1]
                        l6 = b[mask2_1]
                        l7 = l[mask2_3]
                        l8 = b[mask2_3]
                        l9 = l[mask3_1]
                        l10 = b[mask3_1]
                        l11 = b[mask3_2]
                        l12 = b[mask3_2]
                        
                        b1 = l[mask0_1]
                        b2 = b[mask0_1]
                        b3 = l[mask0_2]
                        b4 = b[mask0_2]
                        b5 = l[mask0_3]
                        b6 = b[mask0_3]
                        b7 = l[mask3_0]
                        b8 = b[mask3_0]
                        b9 = l[mask2_0]
                        b10 = b[mask2_0]
                        b11 = l[mask1_0]
                        b12 = b[mask1_0]
                        
                        
                        n_w = len(b1)+len(b3)+len(b5)+1e-10
                        c_w = len(l1)+len(l3)+len(b11)+1e-10
                        u_w = len(l5)+len(l7)+len(b9)+1e-10
                        e_w = len(b7)+len(l9)+len(l11)+1e-10
                        
                        # wrong batch
                        '''
                        n_w = self.batch_size-n_w
                        c_w = self.batch_size-c_w
                        u_w = self.batch_size-u_w
                        e_w = self.batch_size-e_w
                        weights = torch.tensor([n_w,c_w,u_w,e_w]).float().cuda()
                        loss_BC = F.cross_entropy(logit,batch['label'],weight=weights)
                        '''
                        
                        # wrong
                        # weights = torch.tensor([n_w,c_w,u_w,e_w]).float().cuda() + torch.tensor([1,1.22,9.5,18.45]).float().cuda()
                        # loss_BC = F.cross_entropy(logit,batch['label'],weight=weights)
                        
                        
                        # wrong count
                        
                        weights = torch.tensor([n_w,c_w,u_w,e_w]).float().cuda()
                        loss_BC = F.cross_entropy(logit,batch['label'],weight=weights,reduction='none')
                        unq, cnt = batch["label"].unique(return_counts=True)
                        unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                        loss_BC = (loss_BC * unq).mean()
                        
                    elif self.lossfunction == 'exponential':
                        c2_m0 = (logit.argmax(dim=1) == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                        c2_m1 = (logit.argmax(dim=1) == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_m0 = (batch['label'] == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                        ans_m1 = (batch['label'] == 1).nonzero(as_tuple=True)[0].cpu().numpy()
                        length=torch.tensor([len(ans_m0),len(ans_m1)]).cpu()
                        length2=torch.tensor([len(c2_m0),len(c2_m1)]).cpu()
                        if self.language =='koBert':
                            c2_m2 = (logit.argmax(dim=1) == 2).nonzero(as_tuple=True)[0].cpu().numpy()
                            c2_m3 = (logit.argmax(dim=1) == 3).nonzero(as_tuple=True)[0].cpu().numpy()
                            ans_m2 = (batch['label'] == 2).nonzero(as_tuple=True)[0].cpu().numpy()
                            ans_m3 = (batch['label'] == 3).nonzero(as_tuple=True)[0].cpu().numpy()
                            length=torch.tensor([len(ans_m0),len(ans_m1),len(ans_m2),len(ans_m3)]).cpu()
                            length2=torch.tensor([len(c2_m0),len(c2_m1),len(c2_m2),len(c2_m3)]).cpu()
        
                        labels = batch['label'].view(-1)
                        num_classes = logit.size(1)  # 클래스 개수 (4)
                        indices = [torch.nonzero(labels == i, as_tuple=True)[0] for i in range(num_classes)] # 각 클래스별 인덱스 리스트

                        graph_x = torch.zeros_like(logit[:,0]).float()
                        for i, idx in enumerate(indices):
                            graph_x[idx] = (torch.max(length2) / torch.cosh(length[i])) + 2
                            #graph_x[idx] = (torch.max(length)*2 / torch.cosh(length[i]))+1
                        
                        loss_BC = (graph_x * F.cross_entropy(logit, labels,reduction='none')).mean()
                    else:
                        if self.hierarchy==False:
                            loss_BC = F.cross_entropy(logit,batch['label'])
                        else:
                            batch_2={}
                            batch_2['label'] = batch['BClabel'].unsqueeze(-1)
                            c3 = logit
                            c2_1_m = (c2 > 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
                            c2_0_m = (c2 < 0.5).nonzero(as_tuple=True)[0].cpu().numpy()
                            ans_0_m = (batch_2['label'] == 0).nonzero(as_tuple=True)[0].cpu().numpy()
                            ans_1_m = (batch_2['label'] == 1).nonzero(as_tuple=True)[0].cpu().numpy()

                            common1 = np.intersect1d(c2_1_m, ans_1_m)
                            t_mask = torch.from_numpy(common1)
                            common2 = np.intersect1d(c2_0_m, ans_1_m)
                            f_mask = torch.from_numpy(common2)
                            common3 = np.intersect1d(c2_1_m, ans_0_m)
                            a_mask = torch.from_numpy(common3)
                            common4 = np.intersect1d(c2_0_m, ans_0_m)
                            b_mask = torch.from_numpy(common4)

                            c3_t = c3[t_mask]
                            a3_t = batch['label'][t_mask]

                            c3_f = c3[f_mask]
                            a3_f = batch['label'][f_mask]

                            c3_a = c3[a_mask]
                            a3_a = batch['label'][a_mask]

                            c3_b = c3[b_mask]
                            a3_b = batch['label'][b_mask]
                                
                                # t,b = Poisitive, foreground / f,a = Negative, background
                            #loss_BC = BCE(c2,batch["BClabel"].float().reshape(-1,1))
                            #loss_BC = F.cross_entropy(logit,batch['label'])
                            #P1 = criterion2(c3_t,a3_t) + criterion2(c3_b,a3_b)
                            #P2 = criterion(c3_f,a3_f) + criterion(c3_a,a3_a)
                            unq, cnt = batch["label"].unique(return_counts=True)
                            unq = torch.tensor([1/(cnt[l==unq]+1) if l in unq else 1 for l in batch["label"]], device=batch["label"].device)
                            #loss = (loss * unq).mean()
                            
                            loss_BC = F.cross_entropy(c2,batch['BClabel'])
                            class_weight = torch.tensor([1,1.22,9.5,18.45]).float().cuda()
                            P1_1 = (F.cross_entropy(c3_f, a3_f) * unq).mean()
                            P1_2 = (F.cross_entropy(c3_a, a3_a) * unq).mean()
                            P2_1 = F.cross_entropy(c3_t, a3_t, weight=class_weight)
                            P2_2 = F.cross_entropy(c3_b, a3_b, weight=class_weight)

                            P1 = P1_1 + P1_2
                            P2 = P2_1 + P2_2
                            
                            if epoch < 20:
                                loss_BC = (loss_BC + P1 + P2)/3
                            else:
                                loss_BC = loss_BC * 0.25 + (P1+P2)*0.75
                        
                        if self.is_MT:
                            loss_SP = F.binary_cross_entropy(torch.sigmoid(sentiment), batch["sentiment"])
                            loss = 0.9 * loss_BC + 0.1 * loss_SP
                        else:
                            loss = loss_BC

                        # Calculate the accuracy
                        #accuracy += (torch.argmax(logit, dim=1) == batch["label"]).sum().item()
                    if self.hierarchy:
                        accuracyc3 += (torch.argmax(logit,dim=1)==batch['label']).sum().item()
                        accuracyc2 += (torch.argmax(c2,dim=1)==batch['BClabel']).sum().item()
                        
                    accuracy += (torch.argmax(logit,dim=1)==batch['label']).sum().item()
                    loss    += loss * len(batch["label"])

                    # Calculate the confusion matrix

                    if self.hierarchy==False:
                        for i in range(len(batch["label"])):
                            for l in range(self.num_class):
                                if batch["label"][i] == l:
                                    if logit.argmax(dim=-1)[i] == l:
                                        tp[l] += 1
                                    else:
                                        fn[l] += 1
                                else:
                                    if logit.argmax(dim=-1)[i] == l:
                                        fp[l] += 1
                                    else:
                                        tn[l] += 1
                                        

                        precision = tp / (tp + fp)
                        recall    = tp / (tp + fn)
                        score  = 2 * precision * recall / (precision + recall)
                        score = score.cpu().tolist()
                        
                        if self.distributed:
                            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                            dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
                            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
                            dist.all_reduce(fn, op=dist.ReduceOp.SUM)
                            dist.all_reduce(tn, op=dist.ReduceOp.SUM)

                    else:
                        for i in range(len(batch["BClabel"])):
                            for l in range(2):#self.num_class):
                                if batch["BClabel"][i] == l:
                                    if c2.argmax(dim=-1)[i] == l:
                                        tp2[l] += 1
                                    else:
                                        fn2[l] += 1
                                else:
                                    if c2.argmax(dim=-1)[i] == l:
                                        fp2[l] += 1
                                    else:
                                        tn2[l] += 1
                        
                        precision = tp2 / (tp2 + fp2)
                        recall    = tp2 / (tp2 + fn2)
                        f1_c2_score  = 2 * precision * recall / (precision + recall)
                        c2_score = f1_c2_score.cpu().tolist()
                        
                        
                        # BC 3categories             
                        for i in range(len(batch["label"])):
                            for l in range(4):#self.num_class):
                                if batch["label"][i] == l:
                                    if c3.argmax(dim=-1)[i] == l:
                                        tp3[l] += 1
                                    else:
                                        fn3[l] += 1
                                else:
                                    if c3.argmax(dim=-1)[i] == l:
                                        fp3[l] += 1
                                    else:
                                        tn3[l] += 1
                                        
                        precision = tp3 / (tp3 + fp3)
                        recall    = tp3 / (tp3 + fn3)
                        f1_c3_score  = 2 * precision * recall / (precision + recall)
                        c3_score = f1_c3_score.cpu().tolist()
                        
                        if self.distributed:
                            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                            dist.all_reduce(accuracyc2, op=dist.ReduceOp.SUM)
                            dist.all_reduce(loss2, op=dist.ReduceOp.SUM)
                            dist.all_reduce(accuracyc3, op=dist.ReduceOp.SUM)
                            dist.all_reduce(loss3, op=dist.ReduceOp.SUM)
                            dist.all_reduce(tp2, op=dist.ReduceOp.SUM)
                            dist.all_reduce(fp2, op=dist.ReduceOp.SUM)
                            dist.all_reduce(fn2, op=dist.ReduceOp.SUM)
                            dist.all_reduce(tn2, op=dist.ReduceOp.SUM)
                            dist.all_reduce(tp3, op=dist.ReduceOp.SUM)
                            dist.all_reduce(fp3, op=dist.ReduceOp.SUM)
                            dist.all_reduce(fn3, op=dist.ReduceOp.SUM)
                            dist.all_reduce(tn3, op=dist.ReduceOp.SUM)
                                        

                        precision = tp / (tp + fp)
                        recall    = tp / (tp + fn)
                        score  = 2 * precision * recall / (precision + recall)
                        score = score.cpu().tolist()
                        
                        if self.distributed:
                            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                            dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                            dist.all_reduce(tp, op=dist.ReduceOp.SUM)
                            dist.all_reduce(fp, op=dist.ReduceOp.SUM)
                            dist.all_reduce(fn, op=dist.ReduceOp.SUM)
                            dist.all_reduce(tn, op=dist.ReduceOp.SUM)
                            
                            
                accuracy    /= len(self.val_dataset)
                accuracyc2  /= len(self.val_dataset)
                loss_c2     /= len(self.val_dataset)
                accuracyc3  /= len(self.val_dataset)
                loss_c3     /= len(self.val_dataset)
                loss        /= len(self.val_dataset)
                
                # Time check
                sec = time() -start
                times = str(datetime.timedelta(seconds=sec))
                short = times.split(".")[0]
                for i in range(len(score)):
                    if score[i]==float('nan'):
                        score[i]=0



                print('-'*40)
                print()
                if self.hierarchy:
                    print(f"Epoch : {epoch}")
                    print(f"Accuracy_c2 : {accuracyc2}, Accuracy_c3 : {accuracyc3}")
                    print(f"Loss : {loss}, Loss_c2 : {loss_c2}, Loss_c3 : {loss_c3}")
                    print("C2_F1 score : {},  All : {}, Time taken : {} ".format(c2_score, c2_score[0]*0.5+c2_score[1]*0.5 , short))
                    #print("C3_F1 score : {},  All : {}, Time taken : {} ".format(c3_score, c3_score[1]*0.82+c3_score[2]*0.105+c3_score[3]*0.074 , short))
                    print("C3_F1 score : {},  All : {}, Time taken : {} ".format(c3_score, c3_score[0]*0.5+c3_score[1]*0.41+c3_score[2]*0.053+c3_score[3]*0.037 , short))
                    #portion : 49.5% 38.85% 7.44% 4.2%
                else:
                    for i in range(len(score)):
                        if score[i]==float('nan'):
                            score[i]=0
                    print("Epoch : {}, Accuracy : {}, Loss : {}".format(epoch, accuracy, loss))
                    if self.language == 'koBert':
                        print("F1 score : {},  All : {}, Time taken : {} ".format(score, score[0]*0.5+score[1]*0.41+score[2]*0.053+score[3]*0.037 , short))
                    else:
                        print("F1 score : {},  All : {}, Time taken : {} ".format(score, score[0]*0.715+score[1]*0.2854 , short))
                print()
                print('-'*40)
            gc.collect()

    def init_distributed(self):
        if self.distributed:
            if torch.cuda.is_available():
                self.gpu    = self.local_rank % self.ngpus_per_nodes
                self.device = torch.device(self.gpu)
                if self.distributed:
                    self.local_rank = self.gpu
                    self.rank = self.node_rank * self.ngpus_per_nodes + self.gpu
                    time.sleep(self.rank * 0.1) # prevent port collision
                    print(f'rank {self.rank} is running...')
                    dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                            world_size=self.world_size, rank=self.rank)
                    dist.barrier()
                    self.setup_for_distributed(self.is_main_process())
        else:
            self.device = torch.device('cpu')


    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def get_rank(self):
        if self.distributed:
            return dist.get_rank()
        return 0
    
    def get_world_size(self):
        if self.distributed:
            return dist.get_world_size()
        return 1
    