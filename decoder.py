from imports import *

class Decoder(nn.Module):

    def __init__(self, num_heads: int, out_sent: Tensor) -> None:
        super(Decoder, self).__init__()
        self.out_sent = out_sent
        self.num_heads = num_heads
        '''
        We are making output dim same as the input dim, 
        as we are taking 2 heads for multi-head attention, 
        as a result, 1024/2 = 512 for the output dim.
        The will become 1024 when it will be concatenated. 
        '''
        
        self.k_dim = 512; self.v_dim = 512; self.q_dim = 512
        self.W_q = nn.Linear(512, self.k_dim*self.num_heads, dtype = t.float32) 
        self.W_k = nn.Linear(512, self.k_dim*self.num_heads, dtype = t.float32)
        self.W_v = nn.Linear(512, self.v_dim*self.num_heads, dtype = t.float32)
        self.W_q_m = nn.Linear(512, self.k_dim*self.num_heads, dtype = t.float32)
        self.W_k_m = nn.Linear(512, self.k_dim*self.num_heads, dtype = t.float32)
        self.W_v_m = nn.Linear(512, self.v_dim*self.num_heads, dtype = t.float32)
 
        self.seq_len = out_sent.size()[0]
        assert self.seq_len == 4, "The sequence length should be 4."
        self.masking_tensor = t.triu(t.full((1, self.num_heads, self.seq_len, self.seq_len), float("-inf")), diagonal = 1)
        
        self.W_o = nn.Linear(512*self.num_heads,512, dtype = t.float32) 
        self.W_o_m = nn.Linear(512*self.num_heads,512, dtype = t.float32)
        
        self.layer_norm = nn.LayerNorm(512, dtype = t.float32)
        
        self.fc1 = nn.Linear(512, 1024, dtype = t.float32)
        self.fc2 = nn.Linear(1024, 512, dtype = t.float32)
        self.relu = nn.ReLU()

        self.d_model = 512
    
    def position_embedding(self, sent: Tensor, d_model: int) -> Tensor:
        '''
        Defined in depth in the encoder.py file. 
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

    def masked_multi_head_attention(self) -> Tensor:
        '''
        We are making this as the masked multi-head attention, as we are masking the future words in the sentence.
        For reference you can look at its diagram before implementation to get an intuition about it.  
        ''' 
        query = self.W_q_m(self.input_embedding).view(1, self.num_heads, self.seq_len, self.q_dim) # (1, 2, 4, 512)
        key = self.W_k_m(self.input_embedding).view(1, self.num_heads, self.seq_len, self.k_dim) # (1, 2, 4, 512)
        value = self.W_v_m(self.input_embedding).view(1, self.num_heads, self.seq_len, self.v_dim) # (1, 2, 4, 512)
        
        # we will take the dot product of query and key to get the similarity score.
        attention_score = t.softmax(t.matmul(query, key.transpose(2,3))/t.sqrt(t.tensor(self.k_dim)), dim=-1) # (1, 2, 4, 4)
        overall_attention = t.matmul(attention_score, value).view(1, 4, self.d_model*self.num_heads)
        
        final_attention = self.W_o(overall_attention) # (1, 4, 512)
                
        return final_attention

    def multi_head_attention(self, encoder_output: Tensor, dec_attn: Tensor) -> Tensor:
        '''
        We are making this function for just 1 sample. 
        The words of which will be computed to have similarity with each other.

        The query, key, and value are the three vectors that are used to computed with the embedding layer dim to assign a new dim.
        '''

        query = self.W_q(dec_attn).view(1, self.num_heads, self.seq_len, self.q_dim)
        key = self.W_k(encoder_output).view(1, self.num_heads, self.seq_len, self.k_dim)
        value = self.W_v(encoder_output).view(1, self.num_heads, self.seq_len, self.v_dim)

        attention_score = t.matmul(query, key.transpose(2,3))/t.sqrt(t.tensor(self.k_dim))
        # Adding the attention score with the masking tensor to mask the future words in the sentence.
        attention_score = t.softmax((attention_score + self.masking_tensor), dim = -1)
        
        overall_attention = t.matmul(attention_score, value).view(1, 4, self.d_model*self.num_heads)
        final_attention = self.W_o_m(overall_attention)
        
        return final_attention 
        

    def forward(self, encoder_output) -> Tensor:
        x = self.input_embedding = self.position_embedding(self.out_sent, 512)
        x_ = self.masked_multi_head_attention()
        x = self.layer_norm(x_ + x)
        x_ = self.multi_head_attention(encoder_output, x) # this is creating the issue.
        x = self.layer_norm(x + x_)
        x_ = self.ffn(x)
        x = self.layer_norm(x_ + x)
        return x