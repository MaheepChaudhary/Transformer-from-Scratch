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
                num_heads,
                sent):
        
        self.sent = sent
        self.num_heads = num_heads
        
    def self_attention(self):
        pass
        

def position_embedding(sent, d_model):
    pe = np.zeros((sent.size()[0], d_model))
    for pos, word in enumerate(sent.size()[0]):
        for i in range(0,d_model, 2):
            pe[pos][i] = math.sin(pos/(10000**(2*i/d_model)))
            pe[pos][i+1] = math.cos(pos/(10000**(2*i/d_model)))

    final_sent = sent + pe
    return final_sent

        