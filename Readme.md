# 🤖🗒️ Transformer From Scratch:

I will be implementing a 1 layer Transformer architecure with with no dropouts, custom optimisation and layers.
The implementation is built by taking the [Jay Alammar's Blog](https://jalammar.github.io/illustrated-transformer) as basis.
This repo will be aimed to provide insights to me and to other how really Transformers work, even at a gradient level.
This will not only enable the users to build upon this repo but will also be able to do toy experiments; as we all are GPU poor 😛,
There have been recent discoveries regarding the behaviour of transformers, i.e. do they learn or generalize?
Furthermore, different experiments such as grokking for very simple experiments like prediting addition or other operation on numbers.
This could also be used by others better understand Transformer.

I would be excited to post open questions and their answers that i would encounter during this implementation (some of them posted below).
I hope it would be useful resource for the community.

I am constantly active on Twitter and can be found posting useful things i find during my learning journey: [Twitter](https://x.com/ChaudharyMaheep)

## 🧿 Positional Encoding:

- We will require the position embedding to be unique for each word in the sentence. As it represents different position and having the same position will affect the learning of the model. Hence, we will need to create a bits and bytes type of representation for each word in the sentence. The LSB bit is alternating on every number, the second-lowest bit is rotating on every two numbers, and so on. However, using binary values would be inefficient in a world dominated by floating-point numbers. Instead, we can represent them with their continuous float equivalents—sinusoidal functions. These functions essentially act like alternating bits.

![alt text](figures/positional_encoding.png)

<p align="center"><em>Figure 2 - The 128-dimensional positonal encoding for a sentence with the maximum lenght of 50. Each row represents the embedding vector</em></p>

<!-- please write about this image also -->

### 🙋🏻‍♂️ Open Questions:

1. Is there any better positional encoding method?
2. How much does this positional encoding affect the model and overall generalization?

## 🎀🙇🏻 Acknowledgements:

- [Transformer Architecture: The Positional Encoding by Amirhossein Kazemnejad's](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

Connect with me or to have an e-coffee 😉 on [Twitter](https://x.com/ChaudharyMaheep) 

