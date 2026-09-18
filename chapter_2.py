import torch
import re
from torch.utils.data import Dataset, DataLoader
import importlib
import tiktoken

# Tokenizer
class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    # string to integer
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # integer to string
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text


# Tokenizer
class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s, i in vocab.items()}

    # string to integer
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)

        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.str_to_int 
            else "<|unk|>" for item in preprocessed
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    # integer to string
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must at least be equal to max_length+1"

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size=4, max_length=256, 
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):

    # Initialize the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Create dataset
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# ====================================================================
# EXECUTION GUARD: Everything below this line ONLY runs when you 
# execute chapter_2.py directly, not when you import it into other scripts
# ====================================================================
if __name__ == "__main__":
    # I. Prepare the input text for LLM training
    # 1. Load data (this will be my Neurotech Primer)
    with open("the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    print("Total number of character:", len(raw_text))

    # 2. Build a vocabulary / tokenizer (for my project I will use BPE tokenizer)
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
    preprocessed = [item.strip() for item in preprocessed if item.strip()]
    all_words = sorted(set(preprocessed))

    vocab = {token:integer for integer,token in enumerate(all_words)}

    tokenizer = SimpleTokenizerV1(vocab)

    text = """"It's the last he painted, you know," 
               Mrs. Gisburn said with pardonable pride."""
    ids = tokenizer.encode(text)
    print(ids)

    # Add tokens
    all_tokens = sorted(list(set(preprocessed)))
    all_tokens.extend(["<|endoftext|>", "<|unk|>"])
    vocab = {token:integer for integer,token in enumerate(all_tokens)}

    print(len(vocab.items()))

    tokenizer_v2 = SimpleTokenizerV2(vocab)

    text1 = "Hello, do you like tea?"
    text2 = "In the sunlit terraces of the palace."

    text = " <|endoftext|> ".join((text1, text2))

    print(text)
    tokenizer_v2.encode(text)
    tokenizer_v2.decode(tokenizer_v2.encode(text))

    # Byte pair encoding (Maybe I will use this tokenizer)
    #print("tiktoken version:", importlib.metadata.version("tiktoken"))

    # instantiate the BPE tokenizer
    tokenizer_bpe = tiktoken.get_encoding("gpt2")
    text_bpe = (
        "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
         "of someunknownPlace."
    )

    integers = tokenizer_bpe.encode(text_bpe, allowed_special={"<|endoftext|>"})

    print(integers)
    strings = tokenizer_bpe.decode(integers)

    print(strings)

    ### BPE algorithm: if the tokenizer
    #encounters an unfamiliar word during tokenization, it can represent it as a sequence
    #of subword tokens or characters

    # Data sampling with a sliding window
    enc_text = tokenizer_bpe.encode(raw_text)
    print(len(enc_text))

    enc_sample = enc_text[50:]

    # Since we want the model to predict the next word, the targets are the inputs shifted by one position to the right
    context_size = 4

    x = enc_sample[:context_size]
    y = enc_sample[1:context_size+1]

    print(f"x: {x}")
    print(f"y:      {y}")

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]

        print(context, "---->", desired)

    for i in range(1, context_size+1):
        context = enc_sample[:i]
        desired = enc_sample[i]

        print(tokenizer_bpe.decode(context), "---->", tokenizer_bpe.decode([desired]))

    #########
    dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
    )

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)

    second_batch = next(data_iter)
    print(second_batch)

    # 3. Convert the token IDs into embedding vectors
    input_ids = torch.tensor([2, 3, 5, 1])

    vocab_size = 6
    output_dim = 3

    torch.manual_seed(123)
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    print(embedding_layer.weight)
    print(embedding_layer(torch.tensor([3])))
    print(embedding_layer(input_ids))

    vocab_size = 50257
    output_dim = 256

    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

    max_length = 4
    dataloader = create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length,
        stride=max_length, shuffle=False
    )
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print("Token IDs:\n", inputs)
    print("\nInputs shape:\n", inputs.shape)

    token_embeddings = token_embedding_layer(inputs)
    print(token_embeddings.shape)

    print(token_embeddings)

    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

    print(pos_embedding_layer.weight)
    pos_embeddings = pos_embedding_layer(torch.arange(max_length))
    print(pos_embeddings.shape)

    print(pos_embeddings)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)

    print(input_embeddings)

    # Input processing pipeline: input text => tokens => token IDs => token embeddings + positional embeddings => input embeddings
    """
    ##### tokenize only one pdf text
    import fitz  # PyMuPDF
    import re
    import tiktoken
    import torch

    def extract_text_from_pdf(pdf_path):
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            # Extract raw text from each page
            page_text = page.get_text("text")
            text += page_text + "\n"
        
        # Basic ETL cleaning: remove multiple spaces, erratic newlines, and page artifacts
        text = re.sub(r'\n+', '\n', text) 
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    # Extract your book
    pdf_text = extract_text_from_pdf("primer.pdf")

    # Use your existing function to create the DataLoader
    # For pretraining, stride typically matches max_length to avoid overlap redundancy
    max_length = 1024 # Matches GPT_CONFIG_124M context_length
    dataloader = create_dataloader_v1(
        txt=pdf_text, 
        batch_size=4, 
        max_length=max_length, 
        stride=max_length, 
        shuffle=True
    )

    # Test the pipeline
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f"Inputs shape: {inputs.shape}")   # Expected: [4, 1024]
    print(f"Targets shape: {targets.shape}") # Expected: [4, 1024]



    ####### Hugging Face implementation ABSTRACTS ONLY #################
    from datasets import load_dataset
    import tiktoken
    import torch

    # 1. Load dataset in streaming mode (streams data over the network iteratively)
    dataset = load_dataset("ncbi/pubmed", split="train", streaming=True)

    # 2. Create a filter for computational neuroscience
    def is_neuro(example):
        try:
            # The HF dataset preserves the original XML hierarchy as nested JSON
            abstract = example['MedlineCitation']['Article']['Abstract']['AbstractText']
            if abstract:
                text = abstract.lower()
                return any(kw in text for kw in ["eeg", "neuro", "brain-computer", "fmri"])
        except KeyError:
            pass
        return False

    # 3. Apply the filter to the stream
    neuro_stream = dataset.filter(is_neuro)

    # 4. Tokenize on the fly
    tokenizer = tiktoken.get_encoding("gpt2")

    def tokenize_stream(example):
        abstract = example['MedlineCitation']['Article']['Abstract']['AbstractText']
        tokens = tokenizer.encode(abstract, allowed_special={"<|endoftext|>"})
        return {"token_ids": tokens}

    tokenized_stream = neuro_stream.map(tokenize_stream)

    # Test the stream (fetches and processes only the first valid article)
    first_article = next(iter(tokenized_stream))
    print(f"Token count: {len(first_article['token_ids'])}")

    import torch
    from torch.utils.data import IterableDataset, DataLoader

    class StreamPackingDataset(IterableDataset):
        def __init__(self, tokenized_stream, max_length):
            self.stream = tokenized_stream
            self.max_length = max_length

        def __iter__(self):
            buffer = []
            for example in self.stream:
                # Pour the newly downloaded tokens into the buffer
                buffer.extend(example["token_ids"])
                
                # When the buffer has enough tokens to form an input and a target
                while len(buffer) >= self.max_length + 1:
                    # Slice out exactly what the model needs
                    chunk = buffer[:self.max_length + 1]
                    
                    # Keep the leftover tokens in the buffer for the next round
                    # Using stride = max_length to avoid redundant overlap during pretraining
                    buffer = buffer[self.max_length:] 
                    
                    # Create the shifted inputs and targets (just like Chapter 2)
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    
                    yield x, y

    # 1. Instantiate the buffer using your tokenized Hugging Face stream
    iterable_dataset = StreamPackingDataset(tokenized_stream, max_length=1024)

    # 2. Wrap it in the standard PyTorch DataLoader
    # Note: drop_last and shuffle are not used with IterableDatasets in this way
    dataloader = DataLoader(iterable_dataset, batch_size=4)

    # 3. Test the pipeline
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f"Inputs shape: {inputs.shape}")   # Will perfectly output [4, 1024]
    print(f"Targets shape: {targets.shape}") # Will perfectly output [4, 1024]

    #################### FULL-TEXT DATASET ##################
    import torch
    from torch.utils.data import IterableDataset, DataLoader
    from datasets import load_dataset
    import tiktoken

    # 1. Stream the PMC Full-Text Dataset
    # We use the 'commercial' split which contains ~3.84 million papers
    dataset = load_dataset("TomTBT/pmc_open_access_xml", split="commercial", streaming=True)

    # 2. Filter for Neurotechnology
    def is_neuro(example):
        # The 'front' field contains the abstract and title strings
        # We join them and use keyword matching to quickly filter the stream
        try:
            front_text = " ".join(example.get("front", [])).lower()
            return any(kw in front_text for kw in [
                "electroencephalography", "eeg", "brain-computer", 
                "neural decoding", "fmri"
            ])
        except TypeError:
            return False

    neuro_stream = dataset.filter(is_neuro)

    # 3. Extract and Tokenize Full Text
    tokenizer = tiktoken.get_encoding("gpt2")

    def extract_and_tokenize(example):
        # The 'body' field is already a clean list of paragraph strings. 
        # References are safely excluded by the dataset structure.
        body_paragraphs = example.get("body", [])
        
        # Stitch the paragraphs together into one massive document
        full_text = "\n\n".join(body_paragraphs)
        
        # Tokenize the entire paper (this could be 15,000+ tokens)
        token_ids = tokenizer.encode(full_text, allowed_special={"<|endoftext|>"})
        
        # Append the End-Of-Text token so the model learns when a paper finishes
        token_ids.append(tokenizer.eot_token)
        
        return {"token_ids": token_ids}

    tokenized_stream = neuro_stream.map(extract_and_tokenize)

    # 4. The Packing IterableDataset
    class StreamPackingDataset(IterableDataset):
        def __init__(self, tokenized_stream, max_length):
            self.stream = tokenized_stream
            self.max_length = max_length

        def __iter__(self):
            buffer = []
            for example in self.stream:
                # Pour the massive full-text token list into the buffer
                buffer.extend(example["token_ids"])
                
                # Continuously slice exactly 1,025 tokens from the buffer
                while len(buffer) >= self.max_length + 1:
                    chunk = buffer[:self.max_length + 1]
                    
                    # Pop the used tokens off the front. 
                    # This naturally bridges paper boundaries: the end of Paper A 
                    # and the start of Paper B will sit in the same context window, 
                    # separated only by the <|endoftext|> token.
                    buffer = buffer[self.max_length:] 
                    
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    
                    yield x, y

    # 5. Initialize the DataLoader
    max_length = 1024 # Matches your GPT_CONFIG_124M context_length
    iterable_dataset = StreamPackingDataset(tokenized_stream, max_length=max_length)
    dataloader = DataLoader(iterable_dataset, batch_size=4)

    # 6. Test the Pipeline
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
        
    print(f"Inputs shape: {inputs.shape}")   # Expected: [4, 1024]
    print(f"Targets shape: {targets.shape}") # Expected: [4, 1024]

    """