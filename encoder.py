import torch as t
import torch.nn as nn
import numpy as np
import math
from torch import Tensor


class Encoder(nn.Module):

    def __init__(
        self,
        num_heads: int,
        sent: Tensor) -> None:
        
        super(Encoder, self).__init__()
        
        self.sent = sent
        self.num_heads = num_heads
        '''
        We are making output dim same as the input dim, 
        as we are taking 2 heads for multi-head attention, 
        as a result, 1024/2 = 512 for the output dim.
        The will become 1024 when it will be concatenated. 
        '''
        self.k_dim = 512
        self.v_dim = 512
        self.q_dim = 512
        
        self.W_q = nn.Linear(512, self.k_dim * self.num_heads)
        self.W_k = nn.Linear(512, self.k_dim * self.num_heads)
        self.W_v = nn.Linear(512, self.v_dim * self.num_heads)

        self.seq_len = sent.size()[0]
        assert self.seq_len == 4, "The sequence length should be 4."
        
        self.W_o = nn.Linear(512 * self.num_heads, 512)
        
        self.layer_norm = nn.LayerNorm(512)
        
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()

        # Ensure all parameters require gradients
        for param in self.parameters():
            param.requires_grad = True
        
        self.d_model = 512
    

    # we will make two heads for multi-head attention
    def multi_head_attention(self):
        '''
        We are making this function for just 1 sample. 
        The words of which will be computed to have similarity with each other.
        '''

        '''
        The query, key, and value are the three vectors that are used to computed with the embedding layer dim to assign a new dim.
        '''
        query = self.W_q(self.input_embedding).view(1, self.num_heads, self.seq_len, self.q_dim) # (1, 2, 4, 512)
        key = self.W_k(self.input_embedding).view(1, self.num_heads, self.seq_len, self.k_dim) # (1, 2, 4, 512)
        value = self.W_v(self.input_embedding).view(1, self.num_heads, self.seq_len, self.v_dim) # (1, 2, 4, 512)
        
        # we will take the dot product of query and key to get the similarity score.
        attention_score = t.softmax(t.matmul(query, key.transpose(2,3))/t.sqrt(t.tensor(self.k_dim)), dim=-1) # (1, 2, 4, 4)
        overall_attention = t.matmul(attention_score, value).view(1, 4, 512*self.num_heads)
        print(overall_attention.shape)        
        final_attention = self.W_o(overall_attention) # (1, 4, 512)
                
        return final_attention
        
    def position_embedding(self, sent: Tensor, d_model: int) -> Tensor:

        '''
        We will require the position embedding to be unique for each word in the sentence. 
        As it represents different position and having the same position will affect the learning of the model.
        Hence, we will need to create a bits and bytes type of representation for each word in the sentence.
        The LSB bit is alternating on every number, the second-lowest bit is rotating on every two numbers, and so on.
        However, using binary values would be inefficient in a world dominated by floating-point numbers. 
        Instead, we can represent them with their continuous float equivalents—sinusoidal functions. 
        These functions essentially act like alternating bits.

        Some open Questions:
        1. Is it somehow related to Fourier series and sin and cos defined on the circle.
        2. Why are we using 10000 as the base for the power of the sinusoidal function?
        3. Why are we using the power of 2*i/d_model?
        4. Is there any better positional encoding method? How much does this positional encoding affect the model?

        Arguments:
        sent: Tensor - The sentence converted into embeddings that will be passed into the encoder.
        d_model: int - The dimension of the model.
        '''

        pe = np.zeros((sent.size()[0], d_model))

        for pos, word in enumerate(range(sent.size()[0])):
            for i in range(0,d_model, 2):
                pe[pos][i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos][i+1] = math.cos(pos/(10000**(2*i/d_model)))

        # adding positional encoding to the sentence, that will be passed into the transformer (encoder/decoder).
        final_sent = sent.unsqueeze(1) + pe
        return t.tensor(final_sent, dtype = t.float32)

    def ffn(self, x:Tensor) -> Tensor:
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
    
        return x3

    def forward(self):
        self.input_embedding = self.position_embedding(self.sent, self.d_model)
        multi_head_attn = self.multi_head_attention()
        input_embedding = self.layer_norm(multi_head_attn + self.input_embedding)
        ffn_out = self.ffn(input_embedding)
        encoder_out = self.layer_norm(ffn_out + input_embedding)
        return encoder_out

