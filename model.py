from imports import *

'''
So in the encoder there are three main components:

1. The input embedding layer
2. The positional encoding layer
3. Multi-head attention layer
4. Add and norm
5. Feed forward layer
6. Add and norm
'''

class Transformer(nn.Module):

    def __init__(
        self,
        num_heads: int,
        sent: Tensor) -> None:
        
        super(Transformer, self).__init__()
        
        self.sent = sent
        self.num_heads = num_heads
        self.W_q_1 = nn.Linear(512, 1024)
        self.W_k_1 = nn.Linear(512, 1024)
        self.W_v_1 = nn.Linear(512, 1024)
        self.W_q_2 = nn.Linear(512, 1024)
        self.W_k_2 = nn.Linear(512, 1024)
        self.W_v_2 = nn.Linear(512, 1024)
        self.W_o = nn.Linear(1024,1024) 
        self.layer_norm = nn.LayerNorm(1024)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 2048)
        self.relu = nn.ReLU()

    
    def encoder(self):
        # we will make two heads for multi-head attention
        def self_attention(self):
            '''
            We are making this function for just 1 sample. 
            The words of which will be computed to have similarity with each other.
            '''

            '''
            The query, key, and value are the three vectors that are used to computed with the embedding layer dim to assign a new dim.
            '''
            #TODO: Split tensors into multiple heads
            query1 = t.matmul(self.input_embeddings, self.W_q_1) # (4, 512) * (512, 1024) = (4, 1024)
            key1 = t.matmul(self.input_embeddings, self.W_k_1)        # (4, 512) * (512, 1024) = (4, 1024) 
            value1 = t.matmul(self.embeddings, self.W_v_1)       # (4, 512) * (512, 1024) = (4, 1024)
            
            product1 = t.softmax(t.matmul(query1, key1.T)/t.sqrt(key1.size()[0]))  # (4, 1024) * (1024, 4) = (4, 4)
            attention1 = t.matmul(product1, value1)
            
            query2 = t.matmul(self.input_embeddings, self.W_q_2) # (4, 512) * (512, 1024) = (4, 1024)
            key2 = t.matmul(self.input_embeddings, self.W_k_2)        # (4, 512) * (512, 1024) = (4, 1024) 
            value2 = t.matmul(self.embeddings, self.W_v_2)
        
            product2 = t.softmax(t.matmul(query2, key2.T)/t.sqrt(key2.size()[0]))  # (4, 1024) * (1024, 4) = (4, 4)
            attention2 = t.matmul(product2, value2) # (4, 4) * (4, 1024) = (4, 1024)

            overall_attention = t.concat(attention1, attention2) # (4, 1024) + (4, 1024) = (4, 2048)
            
            return overall_attention
            
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

        def forward(self):
            self.input_embedding = self.position_embedding(self.sent, 4)
            multi_head_attn = self.self_attention()
            multi_head_attn_out = t.matmul(multi_head_attn, self.W_o.T) #(4,2048) * (2048, 4) = (4, 4)
            input_embedding = self.layernorm(multi_head_attn_out + self.input_embedding)
            ffn_out = self.ffn(input_embedding)
            encoder_out = self.layernorm(ffn_out + input_embedding)
            return encoder_out
