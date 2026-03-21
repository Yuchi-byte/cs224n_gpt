import torch

from models.gpt2 import GPT2Model

from transformers import GPT2Model as OpenAIGPT2Model
from utils import model_size_to_params




def test_gpt2(model_size='gpt2'):
  # sent_ids is the sentence. The first sentence has 4 tokens, with the other 4 being zero to represent padding. The second sentence has 8 tokens. 
  sent_ids = torch.tensor([[101, 7592, 2088, 102, 0, 0, 0, 0],
                           [101, 7592, 15756, 2897, 2005, 17953, 2361, 102]])
  # att_mask is required because the gpt2 doesn't know the value 0 in sent_ids means padding. So att_mask makes the padding explicit, and later we set the 0 values in att_mask to be -inf. 
  # 1 means keep and 0 means ignore (padding). 
  att_mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])

  # Load both the OpenAI and your own model.
  openai_model = OpenAIGPT2Model.from_pretrained(model_size)
  gpt = GPT2Model.from_pretrained(model=model_size, **model_size_to_params(model_size))

  outputs = gpt(sent_ids, att_mask)
  openai_outputs = openai_model(input_ids=sent_ids, attention_mask=att_mask, output_hidden_states=True).hidden_states[-1]

  att_mask = att_mask.unsqueeze(-1)
  outputs['last_hidden_state'] = outputs['last_hidden_state'] * att_mask
  openai_outputs *= att_mask

  assert torch.allclose(outputs['last_hidden_state'], openai_outputs, atol=1e-1, rtol=1e-2)

  print("Your GPT2 implementation is correct!")

if __name__ == '__main__':
  test_gpt2('gpt2')
