import torch as t
import torch.nn as nn
import argparse

from imports import *
from encoder import encoder
from decoder import decoder

# Taking a simple sent that will be passed into the transformer. 
sent: t.Tensor = t.randint(0, 8, (4,))
out_sent: t.Tensor = t.randint(0, 8, (4,))

print(f"Input sent: {sent}")
print(f"Output sent: {out_sent}")

def output(decoder_output: t.Tensor) -> t.Tensor:
    linear_output = nn.Linear(512, 8)(decoder_output)  # 8 is the vocabulary size
    return linear_output

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads for multi-head attention.")
    
    args = parser.parse_args()

    ce_loss = nn.CrossEntropyLoss()
    
    # Creating the encoder object
    enc_output = encoder(args.num_heads, sent).forward()
    decoder_output = decoder(args.num_heads, out_sent, enc_output).forward()
    predicted_logits = output(decoder_output)
    print(predicted_logits)

    # To get the contribution of each word to the loss, we can use nn.CrossEntropyLoss with reduction='none'
    ce_loss_none = nn.CrossEntropyLoss(reduction='none')
    loss_contributions = ce_loss_none(predicted_logits.view(4, 8), out_sent)
    print(f"Contribution of each word to the loss: {loss_contributions}")