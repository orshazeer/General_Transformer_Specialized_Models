from transformers import BertTokenizer
from datasets import load_dataset
from tokenizers import Tokenizer
import torch.nn.functional as tnnf
import torch.optim as optim
import pandas as pd
import linecache
import random
import csv
import torch
import torch.nn.functional as tnnf
import torch.optim as optim
import math
import os
import torch
import mmap
from itertools import islice
import itertools
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter
import random
import tkinter as tk
f_step_num = 0
class Tokenizer:
    def __init__(self):
   
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        custom_token = "<pad>"
        custom_token2 = "<start>"

        self.tokenizer.add_tokens([custom_token])
        self.tokenizer.add_tokens([custom_token2])
        
    def convert(self, a):
        return self.tokenizer.convert_ids_to_tokens(a)

    def tokenize_string(self, a):
        
        return self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(a))
    
    def decode(self, a):
       

        return self.tokenizer.decode(a)
    
    def vocab_size(self):
        return self.tokenizer.vocab_size
    
tokenizer = Tokenizer()

TOKEN_BYTES = 2 if tokenizer.vocab_size() <= 32768 else 4


 
class Preprocessing:
    def __init__(self):
        
        self.lines = 155455700
        # self.random_integers = random.sample(range(0, int(self.lines*0.8)), int(self.lines*0.8))
        
        
    def preprocessing(self):
  
        dataset = load_dataset("Skylion007/openwebtext")
        dataset1 = dataset['train']
        length = len(dataset1)

        a = open('General_LLM_tokenizer_data.txt','w')
        z = 0
        for i in dataset1:
            
            a.write(next(iter(i.values())))
        
            z += 1
        a.close()


    def data_preperation(self):
    
        
        tokens = 0
        lines = 0
        
        with open('tokenized_data.bin', mode='wb') as file:

            with open('General_LLM_tokenizer_data.txt', "r", encoding="utf-8") as input_file:
                bytes_read = 0
                for line in input_file:
                    bytes_read += len(line)
                    if line != '\n':
                        tokenized_line = tokenizer.tokenize_string(line)
                        for i in tokenized_line:
                            file.write(i.to_bytes(TOKEN_BYTES, 'big'))
                            tokens += 1
                    lines += 1
                    
               

    def fetch_training_batch_old_version(self, tokens_per_batch, step_num):
        #find actual number of lines
        
        
        file = open('tokenized_data.bin','rb')
        total_bytes = os.path.getsize('tokenized_data.bin')
        total_tokens = total_bytes // TOKEN_BYTES
        total_batches = total_tokens // tokens_per_batch
        batch_num = step_num % total_batches
        # print(f"{step_num=}{batch_num=}{total_batches=}")
        start_byte = batch_num * tokens_per_batch * TOKEN_BYTES
        file.seek(start_byte)
        bytes = file.read(tokens_per_batch * TOKEN_BYTES)
        tokens = []
        for i in range(tokens_per_batch):
            tokens.append(int.from_bytes(bytes[i * TOKEN_BYTES:(i+1) * TOKEN_BYTES],"big"))
        return torch.tensor(tokens)


    def fetch_training_batch(self, B, S):
        #find actual number of lines
                
        file = open('tokenized_data.bin','rb')
        total_bytes = os.path.getsize('tokenized_data.bin')
        total_tokens = total_bytes // TOKEN_BYTES
        batch = []
        for b in range(B):
            start_token = random.randint(0, total_tokens - S)
            # print(f"{step_num=}{batch_num=}{total_batches=}")
            start_byte = start_token * TOKEN_BYTES
            file.seek(start_byte)
            bytes = file.read(S * TOKEN_BYTES)
            tokens = []
            for i in range(S):
                tokens.append(int.from_bytes(bytes[i * TOKEN_BYTES:(i+1) * TOKEN_BYTES],"big"))
            batch.append(tokens)
        ret = torch.tensor(batch)
        # print(f"{B=} {S=} {ret.size()=}")
        return ret
    
   

def positional_encoding(tensor_BSE):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_model = 2048
    n = 10000 
    i = torch.arange(d_model//2).to(device)

    B, S, E = tensor_BSE.shape
    pos_S1 = torch.arange(S, dtype=torch.float32, device=device).unsqueeze(1)
    
    e = E // 2
    i_e = torch.arange(e, dtype=torch.float32, device=device) 
    exponent_e = (2 * i_e) / E
    # print(e)
    denom_1e = torch.pow(1e4, exponent_e).view(1, e)
    pe_sin_Se = torch.sin(pos_S1 / denom_1e)
    pe_cos_Se = torch.cos(pos_S1 / denom_1e)
    pe_SE = torch.cat([pe_sin_Se, pe_cos_Se], dim=1).to(device)
    
    return tensor_BSE + pe_SE

def norm(x_BSE):
    square = torch.square(x_BSE)
    sum_BS1 = torch.mean(square, 2, keepdim=True)
    square_root_BS1 = torch.sqrt(sum_BS1)
    return x_BSE/square_root_BS1

class FeedForward(torch.nn.Module):
    def __init__(self, d_model, ffn_size):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ffn = ffn_size
        self.w_1_EF = torch.nn.Parameter(torch.randn(d_model,self.ffn).to(self.device) * (d_model ** -0.5))
        self.b_1_1F = torch.nn.Parameter(torch.randn(1,self.ffn).to(self.device) * (d_model ** -0.5))
        self.w_2_FE = torch.nn.Parameter(torch.randn(self.ffn,d_model).to(self.device) * (d_model ** -0.5))
        self.b_2_1E = torch.nn.Parameter(torch.randn(1,d_model).to(self.device) * (d_model ** -0.5))
        self.a = torch.nn.ReLU6()
    def forward(self,tensor_BSE):
        tensor_BSE = tensor_BSE.to(self.device)
        out1 = tensor_BSE @ self.w_1_EF + self.b_1_1F
        out1 = self.a(out1)
        out1 = out1 @ self.w_2_FE + self.b_2_1E
        return out1


class Attention(torch.nn.Module):
    def __init__(self, d_model, Heads, Kquerys):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Kquerys = Kquerys
        self.w_EHK_k = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_v = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_q = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        self.w_EHK_o = torch.nn.Parameter(torch.randn(d_model,Heads,self.Kquerys).to(self.device) * (d_model ** -0.5))
        
    def MultiHeadAttention(self,x_BSE):
  
        query_BSHK = torch.einsum('BSE,EHK->BSHK',x_BSE,self.w_EHK_q)
        key_BMHK = torch.einsum('BSE,EHK->BSHK',x_BSE,self.w_EHK_k)
        value_BMHK = torch.einsum('BSE,EHK->BSHK',x_BSE,self.w_EHK_v)
        logits_BSHM = torch.einsum('BSHK,BMHK->BSHM',query_BSHK, key_BMHK) / math.sqrt(self.Kquerys)
        
        B, S, H, M = logits_BSHM.shape
        query_pos_1S11 = torch.arange(S, device=logits_BSHM.device).view(1, S, 1, 1)
        memory_pos_111M = torch.arange(M, device=logits_BSHM.device).view(1, 1, 1, M)
        visiBSE_1S1M = query_pos_1S11 >= memory_pos_111M
        mask_1S1M = torch.where(visiBSE_1S1M, 0, -torch.inf)
        logits_BSHM = logits_BSHM + mask_1S1M
        softmax_BSHM = torch.softmax(logits_BSHM, dim=3)
 
        output_BSHK = torch.einsum('BSHM,BMHK->BSHK',softmax_BSHM, value_BMHK)

        a = torch.einsum('BSHK,EHK->BSE',output_BSHK,self.w_EHK_o)
        return a
class Encoder(torch.nn.Module):
    def __init__(self, Heads, ffn1):
        super().__init__()
        d_model = 2048
        Kquerys = int(d_model/Heads)
        self.mha = Attention(d_model, Heads, Kquerys)
        self.ffn = FeedForward(d_model,ffn1)
    def forward(self, x_BSE):
        x_BSE = x_BSE + self.mha.MultiHeadAttention(norm(x_BSE))
        x_BSE = x_BSE + self.ffn.forward(norm(x_BSE))
        return x_BSE
        

class Transformer(torch.nn.Module):
    """
    B is batch size
    S is sentence length
    E is embedding size or d_model
    V is vocab size
    """
    def __init__(self, writing):
        self.length_of_sentence = 325
        super().__init__()
        self.Heads = 8
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.training_iterations = 100000000000
        self.Batch_size = 5
        self.d_model = 2048
        self.writing = writing
        self.Vocab_size = 30524
        self.embedding_matrix_VE = torch.nn.Parameter(torch.randn((self.Vocab_size, self.d_model)).to(self.device))
        self.step_num = 0
        self.encoder_layers_num = 16
        self.lr = 0.0000025
        self.ffn = self.d_model*4
        self.encoder_layers = torch.nn.ModuleList([Encoder(self.Heads, self.ffn) for i in range(self.encoder_layers_num)])
        self.final_linear = torch.nn.Linear(self.d_model, self.Vocab_size, device=self.device)
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.relu = torch.nn.ReLU()
        self.gelu = torch.nn.GELU()

        self.preprocessing = Preprocessing()
        self.tokenizer = Tokenizer()
        
    def state_dict(self):
        ret = super().state_dict()
        ret["step_num"] = self.step_num
        return ret
    
    def load_state_dict(self, state_dict, *args, **kwargs):
        self.step_num = state_dict["step_num"]
        del state_dict["step_num"]
        super().load_state_dict(state_dict, *args, **kwargs)

    def make_batch(self, step, istraining):
        global f_step_num

        
        if istraining:
            batch_all_1 = []
            batch_all_2 = []
            start_tokenized = self.tokenizer.tokenize_string("<start>")
            start_token, = start_tokenized
            B = self.Batch_size
            S = self.length_of_sentence
            
            # target_token_BS = self.preprocessing.fetch_training_batch(B * S, step).view(B, S)
            target_token_BS = self.preprocessing.fetch_training_batch(B, S)
            input_token_BS = torch.cat(
                (torch.full((B, 1), start_token), target_token_BS[:, :-1]), dim=1
            )
            return input_token_BS, target_token_BS           
            batch1 = start_tokenized + batch[:-1]
            batch_all_1.append(batch1)
            batch2 = batch
            batch_all_2.append(batch2)
            
            return torch.tensor(batch_all_1), torch.tensor(batch_all_2)
        else: 
            #write this at some point
            pass

    def forward(self, encoder_inputs_BS):
        # take the un-embedded encoder input and put it to the device
        encoder_inputs_BS = encoder_inputs_BS.to(self.device)

        
        # Embed the inputs
        # print(f"{encoder_inputs_BS=}  {encoder_inputs_BS.shape=} {encoder_inputs_BS.device=} {self.embedding_matrix_VE.shape=} {self.embedding_matrix_VE.device=}")
        embedded_inputs_BSE = tnnf.embedding(encoder_inputs_BS, self.embedding_matrix_VE)

 
        # Positionally encode the inputs
        positionally_encoded_inputs_BSE = positional_encoding(embedded_inputs_BSE)

 
        #forward through the encoder layers
        for i in self.encoder_layers:
            positionally_encoded_inputs_BSE = i.forward(positionally_encoded_inputs_BSE)
        # return the final output of the decoder layers
        return norm(positionally_encoded_inputs_BSE)

    def training_step(self,i,writer,training,input_sentence, lr):
        
            
        # make the batch of inputs for the model
        if training:
            inputs_BS, targets_BS = self.make_batch(i+1680000, True)
            inputs_BS = inputs_BS.to(self.device)
            targets_BS = targets_BS.to(self.device).float()
        
        else:
            inputs_BS = input_sentence.to(self.device)

        
        # put the batch through the forward to produce the outputs of the model
 
        logits_BSE = self.forward(inputs_BS).to(self.device)
        logits_BSV = self.final_linear(logits_BSE).to(self.device)
        logits_flat = logits_BSV.reshape(-1, logits_BSV.size(-1)).to(self.device)  # Reshape to (Batch size * Sequence Length, Vocab size)\
        if training:
            target_flat = targets_BS.to(torch.int64).reshape(-1).to(self.device)
        

        if training == False:
            temperature = 0.72
            probabilities = torch.nn.functional.softmax(logits_flat[-1] / temperature, dim=0)
            predicted = torch.multinomial(probabilities, 1).item()
            # a = random.randint(0,n_selections-1)
            # predicted = torch.topk(probabilities, n_selections, dim=1)[-1][-1][a]
            
            return predicted
        # B,S,V = logits_BSV.shape
        # put the logits into crossentropy
        # print("targets_shape",targets_B1.shape)
        # print(logits_BSV.shape)
        loss_function = torch.nn.CrossEntropyLoss()
        loss_B = loss_function(logits_flat, target_flat)
        # get gradients and do weight updates:
        if self.writing:   
            writer.add_scalar(str([f"{self.training_iterations=}",f"{self.Batch_size=}", f"{self.encoder_layers_num=}", f"{self.length_of_sentence=}"]), loss_B.item(), i)
            # writer.add_scalar("lr", lr, i)
        # print(i,loss_B.item())
        self.optimizer.zero_grad() 
        loss_B.backward()# Zero gradients
          # Compute gradients
        self.optimizer.step()
        return loss_B.item()

    def train_loop(self):
        i = self.step_num
        losses = []
        if self.writing:
            writer = SummaryWriter("/home/or/")
        
        save_interval = 2000
        saved_model_path = "Models/model.XLP_512000"
        
        write_model_path = "Models/model.XLP_"+str(i)
        if os.path.exists(saved_model_path):
            print("EXISTS")
            self.load_state_dict(torch.load(saved_model_path))
            state_dict = torch.load(saved_model_path)
            print(f"{state_dict.keys()=}")
        
        debt = 0
        while(self.step_num < self.training_iterations):

        # for i in range(self.training_iterations):
            average = 100

            # print(f"{self.step_num=}")
            i = self.step_num
            lr = self.lr
            # lr = self.lr / max(i, 1000) ** 0.5 * min(1.0,i/1200)
            # for g in self.optimizer.param_groups:
            #     g['lr'] = lr 

            if i % save_interval == 0 and i > 0:
                write_model_path = "Models/model.XLP_"+str(i)
                torch.save(self.state_dict(), write_model_path)
            if i % average == 0 and i >= 2:
                print(f"step={self.step_num} {lr=} avg_loss={sum(losses)/average}")
                if debt:
                    debt += sum([prev < next for prev, next in zip(losses[:-1], losses[1:])])
                    print(f"i vow to pay prajit $1 for every step where my loss increases. total money I owe prajit: ${debt}")
                losses = []

            
            if self.writing:
            
                a = self.training_step(i,writer,True,0, lr=lr)

            elif self.writing == False: 
                a = self.training_step(i,5,True,0, lr=lr)

            # print(f"step {i}",f"loss {a}")
            losses.append(a)
            self.step_num += 1
            
        

# Close the writer when done
        if self.writing:
            writer.close()
    def sample(self,a):
        saved_model_path = "Models/model.XLP_512000"
        if os.path.exists(saved_model_path):
            self.load_state_dict(torch.load(saved_model_path))
            torch.load(saved_model_path)
        sentence_length = 125
        tokens1 = self.tokenizer.tokenize_string("<start>") + self.tokenizer.tokenize_string(a)
        for i in range(sentence_length):     
            tokens2 = torch.tensor(tokens1).unsqueeze(0)
            b = self.training_step(0,0,False,tokens2,0)
            tokens1.append(b)
        return ' '.join(self.tokenizer.convert(tokens1))


    def GUI_sample(self):
        root = tk.Tk()
        root.title("Input Box Example")
        root.geometry("400x300")
        text_widget = tk.Text(root, width=100, height=10)
        text_widget.pack()
        

        def on_button_click():
            text_widget.delete("1.0", tk.END)
            entered_text = entry.get() 
            print(entered_text)
            #do stuff here
            text_widget.insert(tk.END, self.sample(entered_text).split("<start>")[1])



        entry = tk.Entry(root, width=125)
        entry.pack()


        button = tk.Button(root, text="Submit", command=on_button_click)
        button.pack()


        root.mainloop()

    def AutoCorrect(self):
        pass

b = Transformer(False)
b.train_loop()
# b.GUI_sample()

# a = Preprocessing()
# a.preprocessing()
# a.data_preperation()

# while True:
#     s = input()
#     b.sample(s)
# def tune():
#     model = Transformer()
#     model.encoder_layers = 5
# print(Tokenizer().vocab_size())
