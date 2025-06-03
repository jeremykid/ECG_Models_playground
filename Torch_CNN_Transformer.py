"""
@Modified by: Tayyib Ul Hassan
@Original Author  : QuYue
@File    : cnn + transformer model.py
https://link.springer.com/article/10.1186/s12911-021-01546-2#availability-of-data-and-materials
@Software: VSCode
@Time    : 2024/09/07 15:42
"""
#%% Importing Libraries
import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import torch.optim as optim

#%% Enable anomaly detection
# torch.autograd.set_detect_anomaly(True)
 
#%% Transformer
class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention'
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        '''
        shape of query: (batch_size, num_heads, seq_len, d_k)
        shape of key: (batch_size, num_heads, seq_len, d_k)
        shape of value: (batch_size, num_heads, seq_len, d_v)
        '''
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
        self.attention = Attention()
        self.output_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        '''
        Note: Here query, key, value are of shape (batch_size, seq_len, d_model)
        '''
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x) 
    
class GELU(nn.Module): 
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU()
 
    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))
    
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2 
    
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))
    
class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()
 
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape is (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
    
class TransformerBlock(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden)
        self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=None))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)
    
class TRANSFORMER(nn.Module):
    def __init__(self, input_size, hidden =128,n_layers =6,attn_heads=8,dropout=0.1):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads
        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4
        self.position = PositionalEmbedding(d_model=128, max_len=input_size)
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])
    def forward(self, x):
        output = x
        pe = self.position(x)
        output +=pe
        for transformer in self.transformer_blocks:
            x = transformer.forward(output)
        return x
#%% Test tramsformer
# trans = TRANSFORMER()
# input = torch.FloatTensor(torch.randn(1,100,128))
# X = trans(input)
# print(X.shape) 

#%% Model
class TransformerWithCNNEmbeddings(nn.Module):
    def __init__(self, num_classes=17, input_size=5120):
        super(TransformerWithCNNEmbeddings, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(12, 32, (30, 1), 1, 0),
            nn.ReLU(inplace=False),  # Changed inplace to False
            nn.MaxPool2d(kernel_size=(3, 1))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (10, 1), 1, 0),
            nn.ReLU(inplace=False),  # Changed inplace to False
            nn.MaxPool2d((3, 1))
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, (10, 1), 1, 0),
            nn.ReLU(inplace=False),  # Changed inplace to False
            nn.MaxPool2d((3, 1))
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, (5, 1), 1, 0),  
            nn.ReLU(inplace=False),  # Changed inplace to False
            nn.MaxPool2d((3, 1))
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 128, (5, 1), 1, 0),  
            nn.ReLU(inplace=False),  # Changed inplace to False
            nn.MaxPool2d((3, 1))
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(128, 128, (3, 1), 1, 0),  
            nn.ReLU(inplace=False),  # Changed inplace to False
            nn.MaxPool2d((3, 1))
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, (2, 1), 1, 0),  
            nn.ReLU(inplace=False),  # Changed inplace to False
            nn.MaxPool2d((1, 1))
        )
        self.network = nn.Sequential(
            nn.Linear(128 * 5 + 10, 100),  
            nn.Dropout(0.4),
            nn.Linear(100, 60),  
            nn.Dropout(0.4),
            nn.Linear(60, num_classes),  
        )
        self.dropout = nn.Dropout(0.5)
        self.batchnorm = nn.BatchNorm1d(128)
        self.transform = TRANSFORMER(input_size=input_size)
        self.age_sex_fc = nn.Linear(2, 10)

    def forward(self, x):
        age_sex = x[1]
        age_sex = self.age_sex_fc(age_sex)

        x = x[0]
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        x = x.reshape(x.size(0), 128, -1)
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        x = self.transform(x)
        x = self.dropout(x)
        x = x.reshape(x.size(0), -1)
        x = torch.cat([x, age_sex], dim=1)

        x = self.network(x)
        # x = F.sigmoid(x)
        return x
    
#%% Helper Functions
def initialize_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    
def print_number_of_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')

def print_model_size_in_MB(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model size: {total_params * 4 / (1024 ** 2):.2f} MB')

#%% Main Function
if __name__ == '__main__':
    batch_size = 64
    sequence_length = 5120
    num_leads = 12
    ecg_signals = torch.randn(batch_size, num_leads, sequence_length)
    age_sex = torch.randn(batch_size, 2)
    model = TransformerWithCNNEmbeddings(num_classes=1, input_size=5120)
    model.apply(initialize_weights)

    # Generate fake labels (for example, random integers between 0 and 16, inclusive, for classification)
    fake_labels = torch.randint(0, 1, (batch_size,1)).to(torch.float32)
    lossfun = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for multi-label classification

    # Compute the loss
    output = model({0: ecg_signals.to(torch.float32), 1: age_sex.to(torch.float32)})
    print ('fake_labels:', fake_labels.shape)
    print ('output:', output.shape)
    loss = lossfun(output, fake_labels)
    # Backpropagate one step
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Assuming you use the Adam optimizer
    optimizer.zero_grad()  # Clear existing gradients
    loss.backward()  # Compute gradients
    optimizer.step()  # Update model parameters

    print('input shape:', ecg_signals.shape)
    print('output shape:', output.shape, "(assumming 1 disease classes)")
    print_number_of_parameters(model)
    print_model_size_in_MB(model)