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
        self.input_emebdding = position_embedding(self.sent, 4)

        def Encoder(self):

        def __init__(self,
                    num_heads: int,
                    sent: Tensor) -> None:
            
            self.sent = sent
            self.num_heads = num_heads
            self.input_emebdding = position_embedding(self.sent, 4)    

        # we will make two heads for multi-head attention
        def self_attention(self):
            '''
            We are making this function for just 1 sample. 
            The words of which will be computed to have similarity with each other.
            '''
            for pos in range(self.sent.size()[0]):
                query = self.sent[pos]
                key = self.sent
                value = self.sent[pos]
                
                # computing the dot product of query and key
                similarity = t.dot(query, key)
                product_similarity = t.softmax(similarity, dim = 0)/t.sqrt(t.tensor(self.sent.size()[0]))
                values = t.matmul(product_similarity, value)
            
            
            

        def position_embedding(sent: Tensor, d_model: int) -> Tensor:

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
            return final_sent
