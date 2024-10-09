from imports import *
from encoder import encoder
from decoder import decoder

# Taking a simple sent that will be passed into the transformer. 
sent: Tensor = t.randn((4))
out_sent: Tensor = t.randn((4))

def output(decoder_output: Tensor, sent: Tensor, out_sent: Tensor):
    linear_output = t.nn.Linear(512, 1)(decoder_output)
    output = t.nn.Softmax(dim = -1)(linear_output.squeeze()) 
    return output 

if __name__ == "__main__":
    parser = parser.ArgumentParser()
    
    parser.add_argument("--num_heads", type=int, default=2, help="Number of heads for multi-head attention.")
    
    args = parser.parse_args()
    
    # Creating the encoder object
    enc_output = encoder(args.num_heads, sent)
    decoder_output = decoder(args.num_heads, out_sent, enc_output)
    predicted_output_probs = output(decoder_output, sent, out_sent)
    print(predicted_output_probs)