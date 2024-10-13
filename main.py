import torch as t
import torch.nn as nn
import argparse

from imports import *
from encoder import Encoder as encoder
from decoder import Decoder as decoder

# Taking a simple sent that will be passed into the transformer. 
sent: t.Tensor = t.randint(0, 8, (4,))
out_sent: t.Tensor = t.randint(0, 8, (4,))

print(f"Input sent: {sent}")
print(f"Output sent: {out_sent}")

class Transformer(nn.Module):
    def __init__(self, num_heads: int, sent: t.Tensor, out_sent: t.Tensor):
        super(Transformer, self).__init__()
        self.encoder = encoder(num_heads, sent)
        self.decoder = decoder(num_heads, out_sent)

    def output(self, decoder_output: t.Tensor) -> t.Tensor:
        linear_output = nn.Linear(512, 8)(decoder_output)  # 8 is the vocabulary size
        return linear_output

    def forward(self) -> t.Tensor:
        encoder_output = self.encoder.forward()
        decoder_output = self.decoder.forward(encoder_output) 
        return self.output(decoder_output)
# parser = argparse.ArgumentParser(
ce_loss = nn.CrossEntropyLoss()
num_heads = 2

predicted_logits = Transformer(num_heads, sent, out_sent).forward()
ce_loss_none = nn.CrossEntropyLoss(reduction='none')
loss_contributions = ce_loss_none(predicted_logits.view(4, 8), out_sent)
print(f"Contribution of each word to the loss: {loss_contributions}")