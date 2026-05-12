"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional
from lr_scheduler import NoamScheduler
import torch.optim as optim
from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler
from nltk.translate.bleu_score import corpus_bleu
import wandb

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1
        log_probs = torch.log_softmax(logits, dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.pad_idx] = 0
            mask = target == self.pad_idx
            true_dist[mask] = 0

        loss = torch.sum(-true_dist * log_probs, dim=1)
        non_pad_mask = target != self.pad_idx
        loss = loss[non_pad_mask]

        return loss.mean()
        


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """
    total_loss = 0
    model.train() if is_train else model.eval()
    loop = tqdm(data_iter, desc="Train" if is_train else "Val", leave=False)

    for src, tgt in loop:
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]

        src_mask = make_src_mask(src)
        tgt_mask = make_tgt_mask(tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask)
        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1)
        )
        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_iter)


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    model.eval()
    src = src.to(device)
    src_mask = src_mask.to(device)

    memory = model.encode(src, src_mask)
    ys = torch.tensor([[start_symbol]], dtype=torch.long).to(device)

    for _ in range(max_len - 1):
        tgt_mask = make_tgt_mask(ys).to(device)
        out = model.decode(memory, src_mask, ys, tgt_mask)
        prob = out[:, -1, :]
        next_word = torch.argmax(prob, dim=-1).item()

        ys = torch.cat(
            [ys, torch.tensor([[next_word]], dtype=torch.long).to(device)],
            dim=1
        )

        if next_word == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    from nltk.translate.bleu_score import corpus_bleu

    model.eval()
    references = []
    hypotheses = []
    itos = {v: k for k, v in tgt_vocab.items()}

    for src, tgt in test_dataloader:
        src = src.to(device)
        src_mask = make_src_mask(src).to(device)

        for i in range(src.size(0)):
            pred_tokens = greedy_decode(
                model,
                src[i:i+1],
                src_mask[i:i+1],
                max_len,
                tgt_vocab["<sos>"],
                tgt_vocab["<eos>"],
                device
            ).squeeze().tolist()

            pred_tokens = pred_tokens[1:]
            if tgt_vocab["<eos>"] in pred_tokens:
                pred_tokens = pred_tokens[:pred_tokens.index(tgt_vocab["<eos>"])]

            tgt_tokens = tgt[i].tolist()
            tgt_tokens = tgt_tokens[1:]
            if tgt_vocab["<eos>"] in tgt_tokens:
                tgt_tokens = tgt_tokens[:tgt_tokens.index(tgt_vocab["<eos>"])]

            pred_sentence = [itos.get(tok, "<unk>") for tok in pred_tokens]
            tgt_sentence = [itos.get(tok, "<unk>") for tok in tgt_tokens]

            hypotheses.append(pred_sentence)
            references.append([tgt_sentence])

    return corpus_bleu(references, hypotheses) * 100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "model_config": {
        "src_vocab_size": model.src_embed.num_embeddings,
        "tgt_vocab_size": model.tgt_embed.num_embeddings,
        "d_model": model.d_model
    }
    }, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint["epoch"]


# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment
    import wandb
    from dataset import Multi30kDataset
    from torch.utils.data import DataLoader
    import torch
    import torch.optim as optim

    device = "cuda" if torch.cuda.is_available() else "cpu"

    config = {
        "d_model": 512,
        "N": 6,
        "num_heads": 8,
        "d_ff": 2048,
        "dropout": 0.1,
        "batch_size": 32,
        "epochs": 10,
        "warmup_steps": 4000
    }

    wandb.init(project="da6401-a3", config=config)

    train_data = Multi30kDataset(split="train")
    val_data = Multi30kDataset(split="validation")
    test_data = Multi30kDataset(split="test")

    train_data.build_vocab()
    val_data.src_vocab = train_data.src_vocab
    val_data.tgt_vocab = train_data.tgt_vocab
    test_data.src_vocab = train_data.src_vocab
    test_data.tgt_vocab = train_data.tgt_vocab

    print("Starting dataset processing...")
    train_src, train_tgt = train_data.process_data()
    
    
    val_src, val_tgt = val_data.process_data()


    
    test_src, test_tgt = test_data.process_data()
   

    pad_idx = train_data.src_vocab["<pad>"]

    def collate_fn(batch):
        src_batch, tgt_batch = zip(*batch)

        src_lens = [len(x) for x in src_batch]
        tgt_lens = [len(x) for x in tgt_batch]

        max_src = max(src_lens)
        max_tgt = max(tgt_lens)

        padded_src = torch.full((len(batch), max_src), pad_idx)
        padded_tgt = torch.full((len(batch), max_tgt), pad_idx)

        for i in range(len(batch)):
            padded_src[i, :src_lens[i]] = src_batch[i]
            padded_tgt[i, :tgt_lens[i]] = tgt_batch[i]

        return padded_src.long(), padded_tgt.long()

    train_loader = DataLoader(list(zip(train_src, train_tgt)),
                              batch_size=config["batch_size"],
                              shuffle=True,
                              collate_fn=collate_fn)

    val_loader = DataLoader(list(zip(val_src, val_tgt)),
                            batch_size=config["batch_size"],
                            shuffle=False,
                            collate_fn=collate_fn)

    test_loader = DataLoader(list(zip(test_src, test_tgt)),
                             batch_size=1,
                             shuffle=False,
                             collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(train_data.src_vocab),
        tgt_vocab_size=len(train_data.tgt_vocab),
        d_model=config["d_model"],
        N=config["N"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        dropout=config["dropout"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9)

    scheduler = NoamScheduler(
        optimizer,
        d_model=config["d_model"],
        warmup_steps=config["warmup_steps"]
    )

    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_data.tgt_vocab),
        pad_idx=pad_idx,
        smoothing=0.1
    )

    for epoch in range(config["epochs"]):
        print(f" Starting Epoch {epoch+1}/{config['epochs']}")
        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            is_train=True,
            device=device
        )

        val_loss = run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            is_train=False,
            device=device
        )
        print(f"Epoch {epoch+1}|Train Loss: {train_loss:.4f}| Val Loss: {val_loss:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        save_checkpoint(model, optimizer, scheduler, epoch)

    bleu = evaluate_bleu(model, test_loader, train_data.tgt_vocab, device=device)

    wandb.log({"test_bleu": bleu})


if __name__ == "__main__":
    run_training_experiment()
