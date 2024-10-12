from imports import *
from encoder import encoder
from decoder import decoder

# Taking a simple sent that will be passed into the transformer. 
sent: Tensor = t.randint(0, 8, (4,))
out_sent: Tensor = t.randint(0, 8, (4,))

print(f"Input sent: {sent}")
print(f"Output sent: {out_sent}")

def output(decoder_output: Tensor, sent: Tensor, out_sent: Tensor):
    linear_output = t.nn.Linear(512, 1)(decoder_output)
    return linear_output

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, predicted: Tensor, output: Tensor) -> Tensor:
        loss_value = t.mean((predicted - output) ** 2)  # Example: Mean Squared Error
        return loss_value

if __name__ == "__main__":
    parser = parser.ArgumentParser()
    
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads for multi-head attention.")
    
    args = parser.parse_args()
    
    # Creating the encoder object
    enc_output = encoder(args.num_heads, sent).forward()
    decoder_output = decoder(args.num_heads, out_sent, enc_output).forward()
    predicted_logits = output(decoder_output, sent, out_sent).view(4,)
    print(predicted_logits)
    loss = Loss()
    print(loss(predicted=predicted_logits, output = out_sent))