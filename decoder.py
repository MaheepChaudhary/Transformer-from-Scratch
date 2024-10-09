from imports import *

class decoder:

    def __init__(self, num_heads: int, out_sent: Tensor) -> None:
        super(decoder, self).__init__()
        self.out_sent = out_sent
        self.num_heads = num_heads
        '''
        We are making output dim same as the input dim, 
        as we are taking 2 heads for multi-head attention, 
        as a result, 1024/2 = 512 for the output dim.
        The will become 1024 when it will be concatenated. 
        '''
        self.k_dim = 512; self.v_dim = 512; self.q_dim = 512
        self.W_q = nn.Linear(512, self.k_dim*self.num_heads) 
        self.W_k = nn.Linear(512, self.k_dim*self.num_heads)
        self.W_v = nn.Linear(512, self.v_dim*self.num_heads)

        self.seq_len = out_sent.size()[0]
        assert self.seq_len == 4, "The sequence length should be 4."
        
        self.W_o = nn.Linear(512*self.num_heads,512) 
        
        self.layer_norm = nn.LayerNorm(512)
        
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu = nn.ReLU()

    
    def masked_multi_head_attention(self):
        pass

    def position_embedding(self, sent: Tensor, d_model: int) -> Tensor:
        '''
        Defined in depth in the encoder.py file. 
        '''
        pe = np.zeros((sent.size()[0], d_model))

        for pos, word in enumerate(sent.size()[0]):
            for i in range(0,d_model, 2):
                pe[pos][i] = math.sin(pos/(10000**(2*i/d_model)))
                pe[pos][i+1] = math.cos(pos/(10000**(2*i/d_model)))

        # adding positional encoding to the sentence, that will be passed into the transformer (encoder/decoder).
        final_sent = sent + pe
        return t.tensor(final_sent)

    def ffn(self, x:Tensor) -> Tensor:
        x1 = self.fc1(x)
        x2 = self.relu(x1)
        x3 = self.fc2(x2)
    
        return x3

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
        overall_attention = t.matmul(attention_score, value)

        overall_attention = t.cat(overall_attention).view(1, self.seq_len, self.k_dim*self.num_heads) # (1, 4, 512)
        
        final_attention = self.W_o(overall_attention) # (1, 4, 512)
                
        return final_attention

    def forward(self):
        pass