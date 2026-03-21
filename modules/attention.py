import torch

from einops import rearrange
from torch import nn


class CausalSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # Initialize the linear transformation layers for key, value, query.
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # This dropout is applied to normalized attention scores following the original
    # implementation of transformer. Although it is a bit unusual, we empirically
    # observe that it yields better performance.
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # The corresponding linear_layer of k, v, q are used to project the hidden_state (x).
    proj = linear_layer(x)
    # Next, we need to produce multiple heads for the proj. This is done by spliting the
    # hidden state to self.num_attention_heads, each of size self.attention_head_size.
    proj = rearrange(proj, 'b t (h d) -> b t h d', h=self.num_attention_heads)
    # By proper transpose, we have proj of size [bs, num_attention_heads, seq_len, attention_head_size].
    proj = rearrange(proj, 'b t h d -> b h t d')
    return proj

  def attention(self, key, query, value, attention_mask):

    ### YOUR CODE HERE

    # key: Key matrix [bs, num_attention_heads, seq_len, attention_head_size] = [B, H, T, D] (because in forward, we see key_layer in being passed into attentino. And key_layer is the output of self.transform.)
    # attention_mask: [bs, 1, 1, seq_len] = [B, 1, 1, T]
    
    B,H,T,D = key.shape
    
    # Score, S = Q * K^T
    S = torch.matmul(query, key.transpose(-1, -2))
    A = S / D ** 0.5
    # A has shape [B, H, T, T] = [2,12,8,8]


    # torch.triu returns the upper triangular part of a matrix, the other elements of the result tensor are set to 0. 
    #mask = torch.triu(torch.ones(T, T, device=query.device), diagonal=1) * -1e10
    

    causal_mask = torch.triu(
      torch.ones((T, T), device=A.device, dtype=torch.bool),diagonal=1,
    )[None, None, :, :]
    # device = A.device because pyTorch strictly requires A and the mask to be on the same device. By default the device is CPU. If A is on GPU, and we don't specify the device, the mask will be on CPU, and the program crashes.
    # dtype = torch.bool is the modern best practices for masking. It's compatible with masked_fill, which requires a boolean tensor so it knows which position to overwrite. Using bool (8 bits) compared to float (32 bits) saves significant memory.  
    # [None, None, :, :] is to make the shape of causal_mask to be [1, 1, T, T] so that it can be broadcasted when added to A which has shape [B, H, T, T].

    # causal_mask has shapre [1,1,T,T] = [1,1,8,8]
    # causal_mask's uppper triangle has value True, and the rest is False. The upper triangle will be filled with negative infinity. 


    A = A.masked_fill(causal_mask, float('-inf')) 
    # you could do A = A+ causal_mask * -1e10, but it's better to use masked_fill instead. Because masked_fill is incredibly fast, but multiplication requires multiple type convesion. using -inf is also safer. 

    # attention_mask is [B, 1, 1, T] = [2,1,1,8] with 0 for keep and -10000 for masked pads.
    # attention_mask is already in the form of 0 and -inf (remember when we passed att_mask to gpt2, it was 1 for keep and 0 for padding. This transformation is done in utils.get_extended_attention_mask function. )
    # Add directly to scores so it broadcasts over heads and query positions.
    A = A + attention_mask 
    
    # shape of A now is [B,H,T,T] = [2,12,8,8]


    A = torch.softmax(A, dim=-1)
    A = self.dropout(A)
    attn_value = torch.matmul(A, value)
    attn_value = rearrange(attn_value, 'b h t d -> b t (h d)')
    return attn_value


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # First, we have to generate the key, value, query for each token for multi-head attention
    # using self.transform (more details inside the function).
    # Size of *_layer is [bs, num_attention_heads, seq_len, attention_head_size].
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    
    # Calculate the multi-head attention.
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value
