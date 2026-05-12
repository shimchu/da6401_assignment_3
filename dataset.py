import torch
import spacy
from collections import Counter
from datasets import load_dataset

class Multi30kDataset:
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        self.dataset = load_dataset("bentrevett/multi30k")[split]

        self.de_tokenizer = spacy.load("de_core_news_sm")
        self.en_tokenizer = spacy.load("en_core_web_sm")

        self.src_vocab = None
        self.tgt_vocab = None
        self.src_itos = None
        self.tgt_itos = None


    def tokenize_de(self, text):
        return [tok.text.lower() for tok in self.de_tokenizer(text)]

    def tokenize_en(self, text):
        return [tok.text.lower() for tok in self.en_tokenizer(text)]
        
    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        specials = ["<pad>", "<sos>", "<eos>", "<unk>"]

        src_counter = Counter()
        tgt_counter = Counter()

        for example in self.dataset:
            src_counter.update(self.tokenize_de(example["de"]))
            tgt_counter.update(self.tokenize_en(example["en"]))

        self.src_vocab = {tok: i for i, tok in enumerate(specials)}
        self.tgt_vocab = {tok: i for i, tok in enumerate(specials)}

        for tok in src_counter:
            if tok not in self.src_vocab:
                self.src_vocab[tok] = len(self.src_vocab)

        for tok in tgt_counter:
            if tok not in self.tgt_vocab:
                self.tgt_vocab[tok] = len(self.tgt_vocab)

        self.src_itos = {i: t for t, i in self.src_vocab.items()}
        self.tgt_itos = {i: t for t, i in self.tgt_vocab.items()}


    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        src_data = []
        tgt_data = []

        sos_id = self.src_vocab["<sos>"]
        eos_id = self.src_vocab["<eos>"]

        tgt_sos = self.tgt_vocab["<sos>"]
        tgt_eos = self.tgt_vocab["<eos>"]

        for example in self.dataset:
            src_tokens = self.tokenize_de(example["de"])
            tgt_tokens = self.tokenize_en(example["en"])

            src_indices = [sos_id] + [
                self.src_vocab.get(tok, self.src_vocab["<unk>"])
                for tok in src_tokens
            ] + [eos_id]

            tgt_indices = [tgt_sos] + [
                self.tgt_vocab.get(tok, self.tgt_vocab["<unk>"])
                for tok in tgt_tokens
            ] + [tgt_eos]

            src_data.append(torch.tensor(src_indices, dtype=torch.long))
            tgt_data.append(torch.tensor(tgt_indices, dtype=torch.long))

        return src_data, tgt_data