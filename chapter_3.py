# Attention mechanisms
#### A simple self-attention mechanism without trainable weights ####
# Goal: calculate a context vector for each x -> enriched embedding vector
# 1. dot product of the query with every other input token
import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]  # 2nd input token is the query (journey)

attn_scores_2 = torch.empty(inputs.shape[0]) # 6 rows/words
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) # dot product (transpose not necessary here since they are 1-dim vectors)

print(attn_scores_2)

# dot product is a measure of similarity
# because it quantifies how closely two vectors are aligned: a higher dot product indi-
# cates a greater degree of alignment or similarity between the vectors.

# 2. Normalize each attention score
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()

print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# Softmax normalization
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)

print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# 3. context vector = sum(embedded input tokens * attention weight)
query = inputs[1] # 2nd input token is the query

context_vec_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i

print(context_vec_2)

# simple self-attention mechanism without trainable weights for all input tokens (words)
# 1. compute attention scores
attn_scores = inputs @ inputs.T
print(attn_scores)

# 2. compute attention weights
attn_weights = torch.softmax(attn_scores, dim=-1)# -1 to normalize across rows (values from each column)
print(attn_weights)

# 3. compute context vectors
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

#### A self-attention mechanism wit trainable weights ####
# scaled dot product attention
x_2 = inputs[1] # second input element
d_in = inputs.shape[1] # the input embedding size, d=3
d_out = 2 # the output embedding size, d=2

# 3 weight matrices
torch.manual_seed(123)

W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

query_2 = x_2 @ W_query # _2 because it's with respect to the 2nd input element
key_2 = x_2 @ W_key 
value_2 = x_2 @ W_value

print(query_2)