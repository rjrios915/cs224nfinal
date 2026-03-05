'''
Sonnet generation starter code.

Running:
  `python sonnet_generation.py --use_gpu`

trains your SonnetGPT model and writes the required submission files.
'''

import argparse
import random
import torch

import numpy as np
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Tokenizer
from einops import rearrange
from evaluation import test_sonnet

from datasets import (
  SonnetsDataset,
)
from models.gpt2 import GPT2Model

from optimizer import AdamW
from shampoo import Shampoo

TQDM_DISABLE = False


# Fix the random seed.
def seed_everything(seed=11711):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


class SonnetGPT(nn.Module):
  """Your GPT-2 Model designed for paraphrase detection."""

  def __init__(self, args):
    super().__init__()
    self.gpt = GPT2Model.from_pretrained(model=args.model_size, d=args.d, l=args.l, num_heads=args.num_heads)
    print([name for name, _ in self.gpt.named_parameters() if "emb" in name or "tok" in name or "wte" in name])
    self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    self.tokenizer.pad_token = self.tokenizer.eos_token
    self.lm_head = nn.Linear(args.d, self.tokenizer.vocab_size, bias=False)
    self.lm_head.weight = self.gpt.word_embedding.weight

    # By default, fine-tune the full model. TODO: this is maybe not idea.
    for param in self.gpt.parameters():
      param.requires_grad = False
    for param in self.lm_head.parameters():
      param.requires_grad = True

  def forward(self, input_ids, attention_mask):
    """
    This is similar to the forward for ParaphraseGPT, but we now want to produce a logit for each token in our sequence;
    not just the last token! This will allow our model to learn the natural language distribution that composes sonnets,
    not just the distribution over next tokens for the last token!
    """
    # TODO: Implement the forward pass for the SonnetGPT model.
    outputs = self.gpt(input_ids, attention_mask)
    logits = outputs['last_hidden_state']
    return self.lm_head(logits)


  def get_device(self):
    for param in self.gpt.parameters():
      return param.device

  @torch.no_grad()
  def generate(self, encoding, temperature=0.7, top_p=0.9, max_length=128):
    """
    Generates an original sonnet using top-p sampling and softmax temperature.

    TODO: this is probably not ideal. You can look at hugging face's model.generate(...) function for inspiration.
    In particular, generating multiple sequences and choosing the best with beam search is one avenue. Top_k is another;
    there are many.
    """
    token_ids = encoding.to(self.get_device())
    attention_mask = torch.ones(token_ids.shape, dtype=torch.int64).to(self.get_device())


    for _ in range(max_length):
      # Forward pass to get logits
      logits_sequence = self.forward(token_ids, attention_mask)
      logits_last_token = logits_sequence[:, -1, :] / temperature  # Apply temperature scaling

      # Convert logits to probabilities
      probs = torch.nn.functional.softmax(logits_last_token, dim=-1)

      # Top-p (nucleus) sampling
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
      top_p_mask = cumulative_probs <= top_p
      top_p_mask[..., 1:] = top_p_mask[..., :-1].clone()  # Shift mask right for proper thresholding
      top_p_mask[..., 0] = True  # Always include the highest probability token
      filtered_probs = sorted_probs * top_p_mask  # Zero out unlikely tokens
      filtered_probs /= filtered_probs.sum(dim=-1, keepdim=True)  # Normalize probabilities

      # Sample from filtered distribution
      sampled_index = torch.multinomial(filtered_probs, 1)
      sampled_token = sorted_indices.gather(dim=-1, index=sampled_index)

      # Stop if end-of-sequence token is reached
      if sampled_token.item() == self.tokenizer.eos_token_id:
        break

      # Append sampled token
      token_ids = torch.cat([token_ids, sampled_token], dim=1)
      attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.int64).to(self.get_device())], dim=1
      )

    generated_output = self.tokenizer.decode(token_ids[0].cpu().numpy().tolist())[3:]
    return token_ids, generated_output


  # @torch.no_grad()
  # def generate(
  #     self,
  #     input_ids,
  #     max_new_tokens=128,
  #     min_new_tokens=64,
  #     do_sample=False,
  #     temperature=1.0,
  #     top_k=0,
  #     top_p=1.0,
  #     repetition_penalty=1.1,
  #     no_repeat_ngram_size=3,
  #     eos_token_id=None,
  #     pad_token_id=None,
  #   ):
  #     """
  #     HF-like generate for a custom GPT2Model.

  #     Returns:
  #       generated_ids: LongTensor [B, T + <=max_new_tokens]
  #       decoded_texts: List[str] length B
  #     """
  #     device = self.get_device()
  #     input_ids = input_ids.to(device)

  #     if eos_token_id is None:
  #         eos_token_id = self.tokenizer.eos_token_id
  #     if pad_token_id is None:
  #         pad_token_id = self.tokenizer.eos_token_id

  #     B = input_ids.size(0)
  #     generated = input_ids
  #     attention_mask = torch.ones_like(generated, dtype=torch.long, device=device)
  #     finished = torch.zeros(B, dtype=torch.bool, device=device)

  #     def apply_repetition_penalty(logits, gen_ids, penalty: float):
  #         if penalty == 1.0:
  #             return logits
  #         # HF-style: penalize tokens already generated
  #         for b in range(logits.size(0)):
  #             prev = set(gen_ids[b].tolist())
  #             for t in prev:
  #                 if logits[b, t] < 0:
  #                     logits[b, t] *= penalty
  #                 else:
  #                     logits[b, t] /= penalty
  #         return logits

  #     def ban_no_repeat_ngrams(logits, gen_ids, n: int):
  #         if n <= 0 or gen_ids.size(1) < n:
  #             return logits
  #         for b in range(logits.size(0)):
  #             if finished[b]:
  #                 continue
  #             tokens = gen_ids[b].tolist()
  #             prefix_to_next = {}
  #             for i in range(len(tokens) - n + 1):
  #                 prefix = tuple(tokens[i : i + n - 1])
  #                 nxt = tokens[i + n - 1]
  #                 prefix_to_next.setdefault(prefix, set()).add(nxt)
  #             current_prefix = tuple(tokens[-(n - 1) :])
  #             banned = prefix_to_next.get(current_prefix, set())
  #             if banned:
  #                 logits[b, list(banned)] = -float("inf")
  #         return logits

  #     def top_k_top_p_filtering(logits, top_k: int, top_p: float):
  #         if top_k and top_k > 0:
  #             topk_vals, _ = torch.topk(logits, top_k, dim=-1)
  #             kth = topk_vals[:, -1].unsqueeze(-1)
  #             logits = torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)

  #         if top_p is not None and top_p < 1.0:
  #             sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
  #             probs = F.softmax(sorted_logits, dim=-1)
  #             cumprobs = torch.cumsum(probs, dim=-1)

  #             sorted_mask = cumprobs > top_p
  #             sorted_mask[:, 0] = False  # keep at least 1 token

  #             mask = torch.zeros_like(logits, dtype=torch.bool)
  #             mask.scatter_(dim=-1, index=sorted_idx, src=sorted_mask)
  #             logits = logits.masked_fill(mask, -float("inf"))
  #         return logits

  #     for step in range(max_new_tokens):
  #         logits_seq = self.forward(generated, attention_mask)  # [B, T, V]
  #         logits = logits_seq[:, -1, :].clone()                 # [B, V]

  #         # force pad on finished sequences
  #         logits[finished, :] = -float("inf")
  #         logits[finished, pad_token_id] = 0.0

  #         # prevent immediate EOS (min_new_tokens)
  #         if step < min_new_tokens:
  #             logits[:, eos_token_id] = -float("inf")

  #         # repetition controls
  #         logits = apply_repetition_penalty(logits, generated, repetition_penalty)
  #         logits = ban_no_repeat_ngrams(logits, generated, no_repeat_ngram_size)

  #         # temperature
  #         if temperature and temperature != 1.0:
  #             logits = logits / temperature

  #         # pick next token
  #         if do_sample:
  #             logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
  #             probs = F.softmax(logits, dim=-1)
  #             next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]
  #         else:
  #             next_token = torch.argmax(logits, dim=-1, keepdim=True)  # [B, 1]

  #         # append
  #         generated = torch.cat([generated, next_token], dim=1)
  #         attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

  #         # mark finished
  #         finished = finished | (next_token.squeeze(-1) == eos_token_id)
  #         if torch.all(finished):
  #             break

  #     decoded_texts = [
  #         self.tokenizer.decode(g.tolist(), skip_special_tokens=True)
  #         for g in generated
  #     ]
  #     return generated, decoded_texts

def save_model(model, optimizer, args, filepath):
  save_info = {
    'model': model.state_dict(),
    'optim': optimizer.state_dict(),
    'args': args,
    'system_rng': random.getstate(),
    'numpy_rng': np.random.get_state(),
    'torch_rng': torch.random.get_rng_state(),
  }

  torch.save(save_info, filepath)
  print(f"save the model to {filepath}")

def train(args):
  """Train GPT-2 for paraphrase detection on the Quora dataset."""
  # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
  # Create the data and its corresponding datasets and dataloader.
  sonnet_dataset = SonnetsDataset(args.sonnet_path)
  sonnet_dataloader = DataLoader(sonnet_dataset, shuffle=True, batch_size=args.batch_size,
                                 collate_fn=sonnet_dataset.collate_fn)

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  args = add_arguments(args)
  model = SonnetGPT(args)
  model = model.to(device)

  lr = args.lr
  # optimizer = AdamW(model.parameters(), lr=lr)
  optimizer = Shampoo(model.parameters(), lr=lr)

  # Run for the specified number of epochs.
  for epoch in range(args.epochs):
    model.train()
    train_loss = 0
    num_batches = 0

    for batch in tqdm(sonnet_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
      # Get the input and move it to the gpu (I do not recommend training this model on CPU).
      b_ids, b_mask = batch['token_ids'], batch['attention_mask']
      b_ids = b_ids.to(device)
      b_mask = b_mask.to(device)

      # Compute the loss, gradients, and update the model's parameters.
      optimizer.zero_grad()
      logits = model(b_ids, b_mask)
      logits = rearrange(logits[:, :-1].contiguous(), 'b t d -> (b t) d')  # Ignore the last prediction in the sequence.
      labels = b_ids[:, 1:].contiguous().flatten()  # Ignore the first token to compose the labels.
      # print(logits, labels)
      loss = F.cross_entropy(logits, labels, reduction='mean')
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      num_batches += 1

    train_loss = train_loss / num_batches
    print(f"Epoch {epoch}: train loss :: {train_loss :.3f}.")
    print('Generating several output sonnets...')
    # model.eval()
    # for batch in held_out_sonnet_dataset:
    #   encoding = model.tokenizer(batch[1], return_tensors='pt', padding=True, truncation=True).to(device)
    #   output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)
    #   print(f'{batch[1]}{output[1]}\n\n')

    # TODO: consider a stopping condition to prevent overfitting on the small dataset of sonnets.
    save_model(model, optimizer, args, f'{epoch}_{args.filepath}')

@torch.no_grad()
def generate_submission_sonnets(args):
  # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")
  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)

  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset(args.held_out_sonnet_path)

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))

    # generated_ids, decoded_output = model.generate(encoding['input_ids'], do_sample=False, max_new_tokens=96, no_repeat_ngram_size=3, repetition_penalty=1.1)
    # full_sonnet = f'{decoded_output[0]}\n\n'
    # generated_sonnets.append((sonnet_id, full_sonnet))
    print(f'{decoded_output}\n\n')

  with open(args.sonnet_out, "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument("--sonnet_path", type=str, default="data/sonnets.txt")
  parser.add_argument("--held_out_sonnet_path", type=str, default="data/sonnets_held_out.txt")
  parser.add_argument("--sonnet_out", type=str, default="predictions/generated_sonnets.txt")

  parser.add_argument("--seed", type=int, default=11711)
  parser.add_argument("--epochs", type=int, default=20)
  parser.add_argument("--use_gpu", action='store_true')

  # Generation parameters.
  parser.add_argument("--temperature", type=float, help="softmax temperature.", default=1.2)
  parser.add_argument("--top_p", type=float, help="Cumulative probability distribution for nucleus sampling.",
                      default=0.9)

  parser.add_argument("--batch_size", help='The training batch size.', type=int, default=12)
  parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)
  parser.add_argument("--model_size", type=str, help="The model size as specified on hugging face.",
                      choices=['gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'], default='gpt2')

  args = parser.parse_args()
  return args


def add_arguments(args):
  """Add arguments that are deterministic on model size."""
  if args.model_size == 'gpt2':
    args.d = 768
    args.l = 12
    args.num_heads = 12
  elif args.model_size == 'gpt2-medium':
    args.d = 1024
    args.l = 24
    args.num_heads = 16
  elif args.model_size == 'gpt2-large':
    args.d = 1280
    args.l = 36
    args.num_heads = 20
  else:
    raise Exception(f'{args.model_size} is not supported.')
  return args

def test_dev_sonnets(args):
  # device = torch.device('cuda') if args.use_gpu else torch.device('cpu')
  if torch.backends.mps.is_available():
      device = torch.device("mps")
  else:
      device = torch.device("cpu")

  saved = torch.load(f'{args.epochs-1}_{args.filepath}', weights_only=False)
  model = SonnetGPT(saved['args'])
  model.load_state_dict(saved['model'])
  model = model.to(device)
  model.eval()

  # Create the held-out dataset: these only have the first 3 lines. Your job is to fill in the rest!
  held_out_sonnet_dataset = SonnetsDataset('data/sonnets_held_out_dev.txt')

  generated_sonnets = []
  for batch in held_out_sonnet_dataset:
    sonnet_id = batch[0]
    encoding = model.tokenizer(batch[1], return_tensors='pt', padding=False, truncation=True).to(device)
    output = model.generate(encoding['input_ids'], temperature=args.temperature, top_p=args.top_p)[0][0]
    decoded_output = model.tokenizer.decode(output)
    full_sonnet = f'{decoded_output}\n\n'
    generated_sonnets.append((sonnet_id, full_sonnet))
    
    # generated_ids, decoded_output = model.generate(encoding['input_ids'], do_sample=False, max_new_tokens=96, no_repeat_ngram_size=3, repetition_penalty=1.1)
    # full_sonnet = f'{decoded_output[0]}\n\n'
    # generated_sonnets.append((sonnet_id, full_sonnet))

    print(f'{decoded_output}\n\n')

  with open('predictions/generated_sonnets_dev.txt', "w+") as f:
    f.write(f"--Generated Sonnets-- \n\n")
    for sonnet in generated_sonnets:
      f.write(f"\n{sonnet[0]}\n")
      f.write(sonnet[1])

  print(test_sonnet('predictions/generated_sonnets_dev.txt', 'data/TRUE_sonnets_held_out_dev.txt'))
  



if __name__ == "__main__":
  args = get_args()
  args.filepath = f'{args.epochs}-{args.lr}-sonnet.pt'  # Save path.
  seed_everything(args.seed)  # Fix the seed for reproducibility.
  train(args)
  test_dev_sonnets(args)
  generate_submission_sonnets(args)