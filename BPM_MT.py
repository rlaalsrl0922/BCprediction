import json
import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super(CrossAttentionLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.dropout = nn.Dropout(0.5) ################### 0.5
        self.norm_1 = nn.LayerNorm(d_model)
        self.ffn_1 = nn.Linear(d_model, d_model * 4)
        self.ffn_2 = nn.Linear(d_model * 4, d_model)
        self.norm_2 = nn.LayerNorm(d_model)

    def forward(self, x, y):
        x = self.norm_1(x)
        y = self.norm_1(y)
        x2, _ = self.self_attn(x, y, y)
        x = x + x2
        
        x = self.norm_2(x)
        x2 = self.ffn_1(self.dropout(x))
        x2 = F.relu(x2)
        x2 = self.ffn_2(self.dropout(x2))
        x = x + x2
        return x

class BPM_MT(nn.Module):
    def __init__(self, language_model=None, audio_model=None, sentiment_dict = None, output_size=128, num_class=4, sentiment_output_size=64, dropout=0.5):#########0.5
        super(BPM_MT, self).__init__()

        self.register_module("language_model", language_model)

        # if bert and vocab are not provided, raise an error
        assert self.language_model is not None, "bert and vocab must be provided"

        self.sentiment_dict = sentiment_dict
        self.is_MT = self.sentiment_dict is not None       

        self.register_module("audio_model", audio_model)
        # define the LSTM layer, 4 of layers
        self.audio_feature_size = audio_model.get_feature_size()    # 768

        self.dropout = nn.Dropout(dropout)
        # FC layer that has 128 of nodes which fed concatenated feature of audio and text
        # self.fc_layer_1 = nn.Linear(77568, output_size)
        ## FLATTEN
    #    self.fc_layer_1 = nn.Linear(768*84, output_size*16)    # 77568
    #    self.fc_layer_2 = nn.Linear(output_size * 16, output_size * 4)
    #    self.fc_layer_3 = nn.Linear(output_size * 4, output_size)
        ## AVGPOOL
        self.fc_layer_1 = nn.Linear(768, 128)    # 77568
        ##self.fc_layer_2 = nn.Linear(256, 128)
        ## self.fc_layer_3 = nn.Linear(768, output_size)
        
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(output_size, num_class) # 128 -> 2

        # FC layer that has 64 of nodes which fed the text feature
        # FC layer that has 5 of nodes which fed the sentiment feature
##        if self.is_MT:
##            self.sentiment_fc_layer_1 = nn.Linear(768, sentiment_output_size)
##            self.sentiment_relu = nn.ReLU()
##            self.sentiment_classifier = nn.Linear(sentiment_output_size, 5)

##        self.audio_downproject = nn.Linear(self.audio_feature_size, 512)    # 768 -> 512
##        self.text_downproject = nn.Linear(768, 512)                         # 768 -> 512

        self.encoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=768, nhead=8, batch_first=True) for _ in range(8)]) ############ 6 / 2

#        self.audio_to_text_attention = nn.Sequential(*[CrossAttentionLayer(d_model=768, nhead=8) for _ in range(4)])
#        self.audio_mask = nn.Parameter(torch.zeros(1, 1, 768))  # mask 위치 0으로
        
#        self.text_to_audio_attention = nn.Sequential(*[CrossAttentionLayer(d_model=768, nhead=8) for _ in range(4)])
#        self.text_mask = nn.Parameter(torch.zeros(1, 1, 768))   # mask 위치 0으로

        self.full_mask = nn.Parameter(torch.zeros(1, 1, 768))
        
#        self.audio_decoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True) for _ in range(2)])
#        self.text_decoder = nn.Sequential(*[nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True) for _ in range(2)])

        self.audio_decoder = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=2) for _ in range(2)])
        self.text_decoder = nn.Sequential(*[CrossAttentionLayer(d_model=192, nhead=2) for _ in range(2)])
        
##        self.audio_upproject = nn.Linear(512, self.audio_feature_size)
##        self.text_upproject = nn.Linear(512, 768)

#######        self.audio_upproject = nn.Linear(192, 768)
#######        self.text_upproject = nn.Linear(192, 768)
        
        self.full_downproject = nn.Linear(768, 192)
        self.audio_downproject = nn.Linear(768, 192)   
        self.text_downproject = nn.Linear(768, 192)   
        
        self.audio_predictor = nn.Linear(192*74, 24000)
        self.text_predictor = nn.Linear(192, 30522)
        
    def pretext_forward(self, x):
        audio = x["audio"]
        text  = x["text"]
        audio = audio.squeeze(1)
        original_audio = audio.clone()
        original_text = text.clone()
        
        batch_size = audio.size(0)
        
        y = {}
        
        text = self.language_model(text).last_hidden_state  # B, 64, 768 
        audio = self.audio_model(audio)                     # B, 74, 768
        
        full = torch.cat((text, audio), dim = 1)            # B, 138, 768

        original_full = full.clone()
#######        original_audio = audio.clone()
#######        original_text = text.clone()

        masked_full = torch.rand_like(full.mean(-1)) < 0.15
        
    #    masked_audio = torch.rand_like(audio.mean(-1))<0.70
    #    masked_text = torch.rand_like(text.mean(-1))<0.40
        
    #    audio[masked_audio] = self.audio_mask
    #    text[masked_text] = self.text_mask
        
        
##        visible_full = (masked_full == False)

        full[masked_full] = self.full_mask
        
##        visible_tokens = full[visible_full]             # B, ??, 768
        
##        n_tokens = visible_tokens.size(1)                   
 
    #    for layer in self.encoder:
    #        visible_tokens = layer(visible_tokens)


        
        full = self.encoder(full)
        
##        full[visible_full] = visible_tokens
        
        encoded_text = full[:,:10,:]
        encoded_audio = full[:,10:,:]        
        
        encoded_audio = self.audio_downproject(encoded_audio)
        encoded_text = self.text_downproject(encoded_text)
        
        full = self.full_downproject(full)
        
        for layer in self.audio_decoder:
            encoded_audio = layer(encoded_audio, full)  
        
        for layer in self.text_decoder:
            encoded_text = layer(encoded_text, full)  

    #######    audio = self.audio_upproject(encoded_audio)
    #######    text = self.text_upproject(encoded_text)   
    
        audio = self.audio_predictor(encoded_audio.flatten(start_dim=1))
        text = self.text_predictor(encoded_text)

        self.pretext_loss = F.mse_loss(audio, original_audio) + F.cross_entropy(text.transpose(-1,-2), original_text)
        

    #######    audio = audio.reshape(batch_size,-1)
    #######    text = text.reshape(batch_size,-1)
    #######    original_audio = original_audio.reshape(batch_size,-1)
    #######    original_text = original_text.reshape(batch_size,-1)
    #######    target_a = torch.ones(batch_size).to("cuda:0")
    #######    target_b = torch.ones(batch_size).to("cuda:0")
        
    #######    loss = torch.nn.CosineEmbeddingLoss()
        
    #######    self.pretext_loss = loss(audio, original_audio, target_a) + loss(text, original_text, target_b)
        
        return self.pretext_loss

    def forward(self, x):
        audio = x["audio"]
        text  = x["text"]

        text = self.language_model(text).last_hidden_state
        audio = self.audio_model(audio)

        y = {}

        x = torch.cat((audio, text), dim=1)
        x = self.encoder(x)
        ## print(x.shape)
##        x = x.flatten(start_dim=1)
        
        x = torch.nn.functional.avg_pool2d(x, kernel_size=(84,1))
        x = x.squeeze(1)
        
        # print(x.shape)
        x = self.fc_layer_1(self.dropout(x))
        x = self.relu(x)
    ##    x = self.fc_layer_2(self.dropout(x))
    ##    x = self.relu(x)
    ##    x = self.fc_layer_3(self.dropout(x))
    ##    x = self.relu(x)
        y["logit"] = self.classifier(self.dropout(x))
        
        return y