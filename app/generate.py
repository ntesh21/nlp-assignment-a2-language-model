import math
import torch
import torch.nn as nn
import torchtext
import pickle


class LSTMLanguageModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_layers = num_layers
        self.hid_dim    = hid_dim
        self.emb_dim    = emb_dim
        
        self.embedding  = nn.Embedding(vocab_size, emb_dim)
        self.lstm       = nn.LSTM(emb_dim, hid_dim, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.dropout    = nn.Dropout(dropout_rate)
        self.fc         = nn.Linear(hid_dim, vocab_size)
        
        self.init_weights()
    
    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1/math.sqrt(self.hid_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_other)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        for i in range(self.num_layers):
            self.lstm.all_weights[i][0] = torch.FloatTensor(self.emb_dim,
                self.hid_dim).uniform_(-init_range_other, init_range_other) #We
            self.lstm.all_weights[i][1] = torch.FloatTensor(self.hid_dim,   
                self.hid_dim).uniform_(-init_range_other, init_range_other) #Wh
    
    def init_hidden(self, batch_size, device):
        hidden = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        cell   = torch.zeros(self.num_layers, batch_size, self.hid_dim).to(device)
        return hidden, cell
        
    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach() #not to be used for gradient computation
        cell   = cell.detach()
        return hidden, cell
        
    def forward(self, src, hidden):
        #src: [batch_size, seq len]
        embedding = self.dropout(self.embedding(src)) #harry potter is
        #embedding: [batch-size, seq len, emb dim]
        output, hidden = self.lstm(embedding, hidden)
        #ouput: [batch size, seq len, hid dim]
        #hidden: [num_layers * direction, seq len, hid_dim]
        output = self.dropout(output)
        prediction =self.fc(output)
        #prediction: [batch_size, seq_len, vocab_size]
        return prediction, hidden


class TextGeneration:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = torchtext.data.utils.get_tokenizer('basic_english')
        self.vocab = pickle.load(open('../models/vocab.pkl',"rb"))
        self.model = torch.load('../models/best-val-lstm_lm.pt')

    def generate(self, prompt, max_seq_len, temperature, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.model.eval()
        tokens = self.tokenizer(prompt)
        indices = [self.vocab[t] for t in tokens]
        batch_size = 1
        hidden = self.model.init_hidden(batch_size, self.device)
        with torch.no_grad():
            for i in range(max_seq_len):
                src = torch.LongTensor([indices]).to(self.device)
                prediction, hidden = self.model(src, hidden)
                
                #prediction: [batch size, seq len, vocab size]
                #prediction[:, -1]: [batch size, vocab size] #probability of last vocab
                
                probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
                prediction = torch.multinomial(probs, num_samples=1).item()    
                
                while prediction == self.vocab['<unk>']: #if it is unk, we sample again
                    prediction = torch.multinomial(probs, num_samples=1).item()

                if prediction == self.vocab['<eos>']:    #if it is eos, we stop
                    break

                indices.append(prediction) #autoregressive, thus output becomes input

        itos = self.vocab.get_itos()
        tokens = [itos[i] for i in indices]
        return ' '.join(tokens)+'\n'


