# CS 224N Default Final Project: Build GPT-2

This is the default final project for the Stanford CS 224N class. 

This project comprises two parts. In the first part, you will implement some important components of the GPT-2 model to better understand its architecture.
In the second part, you will use the token embeddings produced by your GPT-2 model on two downstream tasks: paraphrase detection and sonnet generation. You will implement extensions to improve your model's performance on these tasks.

In broad strokes, Part 1 of this project targets:

* modules/attention.py: Missing code blocks.
* modules/gpt2_layer.py: Missing code blocks.
* models/gpt2.py: Missing code blocks.
* classifier.py: Missing code blocks.
* optimizer.py: Missing code blocks.

To test Part 1, you will run:

* `optimizer_test.py`: To test your implementation of `optimizer.py`.
* `sanity_check.py`: To test your implementation of GPT models.
* `classifier.py` : To perform sentiment classification using your models.

In Part 2 of this project, you will use GPT2 (via cloze-style classification) detect if one sentence is a paraphrase of 
another as well as generate sonnets via autoregressive language modeling.  

To test Part 2, you will run:

* `paraphrase_detection.py`: To perform paraphrase detection. 
* `sonnet_generation.py`: To perform sonnet generation.

Important: Adjust training hyperparameters, particularly batch size, according to your GPU's specifications to optimize performance and prevent out-of-memory errors.

## Pre-testing instructions

While there are missing code blocks that you need to implement in both of these files, the main focus of this second part are the extensions: how you modify your GPT2 model to improve its ability to determine if one sentence is a paraphrase of another as well as its ability to generate sonnets. 

## Setup instructions

Follow `setup.sh` to properly setup a conda environment and install dependencies.

## Acknowledgement

This project is adapted from a prior year's CS 224N
project [Implement BERT](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1246/project/default-final-project-handout-minbert-spr2024-updated.pdf)
.

Parts of the code are from the [`transformers`](https://github.com/huggingface/transformers)
library ([Apache License 2.0](./LICENSE)).


# GPT2 Implementation 
Attention 

AdamW 

# Sentiment Classifier vs Paraphrase Detection 
Both are classifiers with the identical architecture. The architecture is that, the last_token from GPT2 output is passed into a task head (a linear layer for the specific downstream task, which is sentiment classification and paraphrase detection in our case). And the cross entropy loss of the output of the task head is calculated and backpropagated so that the task head learns and updates its weights. 


The difference, as emphasised in the lecture note, is that the former is a pure classification problem, whilst the latter is structured as a cloze-style next word generation problem. The only cause for this difference is that we structured the input to the gpt2 model for the Paraphrase Detection sentences as: `Is "{s1}" a paraphrase of "{s2}"? Answer "yes" or "no": ` whilst in the Sentiment Classifier the sentence is simply a sentence. So the `last_token` from the gpt2 output before passing to the Sentiment Classifier layer only has information about the sentence itself, whilst the `last_token` that get passed into the Paraphrase Detection layer already has read the question and is prime to give a yes/no answer to fill the cloze. 

As the last_token gets passed into the task heads, the Sentiment Classifier layer maps the last_token which contains sentence information to sentiment labels (by giving a value for each category, later using argmax to choose the most likely category for it), and cross entropy loss is calculated and minimised as the layer learns. As the last_token passes into the Paraphrase Detection layer, the layer maps the last_token (prime to answer yes/no) to yes/no labels (logit with dimension 2), and after comparing with actual labels, the layer learns what state of the last_token is more likely a yes and which one a no. 

For the Paraphrase Detection layer, the logit (i.e. output from the layer) has size [batch, 2] rather than [batch, vocab_size] because vocab_size is way too large, so restricting it to 2 possible outputs is more efficient. If using the vocab_size, then the Paraphrase Detection will be a pure cloze problem. Restricting to two outputs makes the Paraphrase Detection a hybrid of cloze and classification. And because of the way we framed the cloze `Answer "yes" or "no": `, the final output is likely yes or no rather than any other words in the vocabulary.  


# Sonnet Generation 


A note on embedding: 
When turning input_ids [B, T] to input_embeds [B, T, D], we use inputs_embeds = self.word_embedding(input_ids), where `self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)`. We are mapping each word token to a vector of dimension D. The word_embedding matrix has shape [V, D]. In this embedding layer defined by nn.Embedding, it's a lookup table for mapping tokens to embedding vectors not a linear transformation. The lookup table is simply to make computation faster. If we represent the input tokens as one-hot vectors (wherever the element is 1, that means this input token corresponds to that particular work in the entire vocabulary) with dimension = vocabulary size, then the embedding layer is mathematically equivalent to a linear transformation, which takes input token [B, T, V] to input embedding [B, T, D] with the embedding matrix [V, D]. 
When we want to turn the hidden state back to words, we use the same embedding matrix and perform a linear transformation. The hidden state [B, T, D] (it has all words in the sentence) is transformed to the logit [B, T, V]. The logit isn't one-hot vectors -- they contain probabilities for every word in the vocabulary. It's only when we pass it through a softmax and take argmax that we turn it to a one-hot vector which represents the most likely word. 


How do we use it to generate? Why don't we use the last_token as we did for the classification problems? 


One way Sonnet Generation is different to the classification is that we need to write an additional generate() function to generate new texts. With the classification problem, the last hidden state already contains information on which class the sentence belongs to, so we just stopped there. But for sonnet generation, we couldn't stop there, and we need to keep going to generate new texts. 