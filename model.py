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

class Encoder:
    
    def __init__(self,
                num_heads: int,
                sent: Tensor) -> None:
        
        self.sent = sent
        self.num_heads = num_heads
        
    def self_attention(self):
        pass
        

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

    final_sent = sent + pe
    return final_sent

        