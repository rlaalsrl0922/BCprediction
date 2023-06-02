import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, HubertModel,  Wav2Vec2Processor

class HuBert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    #    self.processor =  Wav2Vec2Processor.from_pretrained("facebook/hubert-base-ls960")
        self.model = HubertModel.from_pretrained("facebook/hubert-base-ls960") # AutoModel

    def forward(self, x):
    #    input_values = self.processor(x.squeeze(1), return_tensors="pt", sample_rate=16000, padding=True).input_values.squeeze()
        batch_size = x.size(0)
        x = x.squeeze(1)
        x = x.reshape(batch_size, -1)
        
        outputs = self.model(x.to("cuda"))
        return outputs.last_hidden_state
    
    def get_feature_size(self):
        return 768