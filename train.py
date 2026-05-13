"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask


# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need".

    Smoothed distribution:
        y_smooth[correct] = 1 - eps + eps / (vocab_size - 1)
        y_smooth[others]  = eps / (vocab_size - 1)
        y_smooth[pad]     = 0

    Args:
        vocab_size : Number of output classes.
        pad_idx    : Index of <pad> token — receives 0 probability.
        smoothing  : Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_idx    = pad_idx
        self.smoothing  = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : [batch * tgt_len, vocab_size]  (raw model output)
            target : [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        log_probs = torch.log_softmax(logits, dim=-1)  # [N, V]

        # Build smooth target distribution
        smooth_val = self.smoothing / (self.vocab_size - 2)  # exclude pad & correct
        with torch.no_grad():
            smooth_dist = torch.full_like(log_probs, smooth_val)
            smooth_dist[:, self.pad_idx] = 0.0
            smooth_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            # Zero out rows where target == pad
            pad_mask = (target == self.pad_idx)
            smooth_dist[pad_mask] = 0.0

        # KL-divergence / cross-entropy with smooth targets
        loss = -(smooth_dist * log_probs).sum(dim=-1)

        # Average only over non-pad positions
        non_pad = (~pad_mask).sum().float()
        if non_pad == 0:
            return loss.sum() * 0.0
        return loss.sum() / non_pad


# ══════════════════════════════════════════════════════════════════════
#  TRAINING LOOP
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

    Returns:
        avg_loss : Average loss over the epoch.
    """
    model.train() if is_train else model.eval()

    total_loss  = 0.0
    total_tokens = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (src, tgt) in enumerate(data_iter):
            src = src.to(device)  # [batch, src_len]
            tgt = tgt.to(device)  # [batch, tgt_len]

            # Teacher-forcing: decoder input = tgt[:-1], target = tgt[1:]
            tgt_input  = tgt[:, :-1]   # [batch, tgt_len-1]
            tgt_output = tgt[:, 1:]    # [batch, tgt_len-1]

            # Build masks (pad_idx = 1 by default)
            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_input).to(device)

            # Forward pass
            logits = model(src, tgt_input, src_mask, tgt_mask)
            # logits: [batch, tgt_len-1, vocab_size]

            # Reshape for loss
            batch_size, seq_len, vocab_size = logits.size()
            logits_flat  = logits.contiguous().view(-1, vocab_size)
            targets_flat = tgt_output.contiguous().view(-1)

            loss = loss_fn(logits_flat, targets_flat)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping for stability
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            # Count non-pad tokens for reporting
            non_pad = (tgt_output != 1).sum().item()
            total_loss   += loss.item() * non_pad
            total_tokens += non_pad

            if batch_idx % 50 == 0:
                mode = "TRAIN" if is_train else "EVAL"
                print(f"[{mode}] Epoch {epoch_num} | Batch {batch_idx} | "
                      f"Loss: {loss.item():.4f}")

    avg_loss = total_loss / max(total_tokens, 1)
    return avg_loss


# ══════════════════════════════════════════════════════════════════════
#  GREEDY DECODING
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
    Generate translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : [1, src_len]
        src_mask     : [1, 1, 1, src_len]
        max_len      : Maximum tokens to generate.
        start_symbol : <sos> vocab index.
        end_symbol   : <eos> vocab index.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : [1, out_len]  — includes start_symbol; stops at end_symbol or max_len.
    """
    model.eval()
    src      = src.to(device)
    src_mask = src_mask.to(device)

    with torch.no_grad():
        memory = model.encode(src, src_mask)  # [1, src_len, d_model]

    # Start with <sos>
    ys = torch.tensor([[start_symbol]], dtype=torch.long, device=device)  # [1, 1]

    for _ in range(max_len - 1):
        with torch.no_grad():
            tgt_mask = make_tgt_mask(ys).to(device)
            logits   = model.decode(memory, src_mask, ys, tgt_mask)
            # logits: [1, cur_len, vocab_size]

        # Take the last position
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # [1, 1]
        ys = torch.cat([ys, next_token], dim=1)

        if next_token.item() == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#  BLEU EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader yielding (src, tgt) batches.
        tgt_vocab       : Vocabulary with .itos[] or .lookup_token().
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (0–100).
    """
    from torchtext.data.metrics import bleu_score as torchtext_bleu

    # Resolve vocab lookup method
    if hasattr(tgt_vocab, 'itos'):
        idx_to_token = lambda i: tgt_vocab.itos[i]
    elif hasattr(tgt_vocab, 'lookup_token'):
        idx_to_token = lambda i: tgt_vocab.lookup_token(i)
    elif isinstance(tgt_vocab, dict):
        itos = {v: k for k, v in tgt_vocab.items()}
        idx_to_token = lambda i: itos.get(i, '<unk>')
    else:
        raise ValueError("Unrecognised tgt_vocab type.")

    special = {'<sos>', '<eos>', '<pad>'}

    # Determine special token indices
    if hasattr(tgt_vocab, 'stoi'):
        sos_idx = tgt_vocab.stoi.get('<sos>', 2)
        eos_idx = tgt_vocab.stoi.get('<eos>', 3)
        pad_idx = tgt_vocab.stoi.get('<pad>', 1)
    elif isinstance(tgt_vocab, dict):
        sos_idx = tgt_vocab.get('<sos>', 2)
        eos_idx = tgt_vocab.get('<eos>', 3)
        pad_idx = tgt_vocab.get('<pad>', 1)
    else:
        sos_idx, eos_idx, pad_idx = 2, 3, 1

    model.eval()
    all_hypotheses  = []
    all_references  = []

    for src, tgt in test_dataloader:
        src = src.to(device)
        tgt = tgt.to(device)

        for i in range(src.size(0)):
            src_i      = src[i].unsqueeze(0)       # [1, src_len]
            src_mask_i = make_src_mask(src_i, pad_idx).to(device)

            ys = greedy_decode(
                model, src_i, src_mask_i,
                max_len=max_len,
                start_symbol=sos_idx,
                end_symbol=eos_idx,
                device=device,
            )

            # Hypothesis: drop <sos>/<eos>/<pad>
            hyp_tokens = [
                idx_to_token(tok.item())
                for tok in ys.squeeze()
                if idx_to_token(tok.item()) not in special
            ]

            # Reference: drop <sos>/<eos>/<pad>
            ref_tokens = [
                idx_to_token(tok.item())
                for tok in tgt[i]
                if idx_to_token(tok.item()) not in special
            ]

            all_hypotheses.append(hyp_tokens)
            all_references.append([ref_tokens])  # BLEU expects list of refs per sentence

    bleu = torchtext_bleu(all_hypotheses, all_references, max_n=4) * 100.0
    return bleu


# ══════════════════════════════════════════════════════════════════════
#  CHECKPOINT UTILITIES
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimizer + scheduler state to disk.

    Saved keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'
    """
    torch.save({
        'epoch':                epoch,
        'model_state_dict':     model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'model_config':         model.model_config,
    }, path)
    print(f"Checkpoint saved to {path} (epoch {epoch})")


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) from disk.

    Returns:
        epoch : The epoch at which the checkpoint was saved.
    """
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch


# ══════════════════════════════════════════════════════════════════════
#  EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def run_training_experiment() -> None:
    """
    Full training experiment with W&B logging.
    """
    import wandb
    from dataset import Multi30kDataset
    from torch.utils.data import DataLoader
    from lr_scheduler import NoamScheduler

    # ── Config ────────────────────────────────────────────────────────
    config = {
        'd_model':      256,
        'N':            3,
        'num_heads':    8,
        'd_ff':         512,
        'dropout':      0.1,
        'batch_size':   128,
        'num_epochs':   20,
        'warmup_steps': 4000,
        'smoothing':    0.1,
    }

    wandb.init(project="da6401-a3", config=config)
    cfg = wandb.config
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────
    train_ds = Multi30kDataset(split='train')
    val_ds   = Multi30kDataset(split='valid')
    test_ds  = Multi30kDataset(split='test')

    train_ds.build_vocab()
    train_ds.process_data()

    # Share vocabs
    val_ds.src_vocab  = train_ds.src_vocab
    val_ds.tgt_vocab  = train_ds.tgt_vocab
    test_ds.src_vocab = train_ds.src_vocab
    test_ds.tgt_vocab = train_ds.tgt_vocab

    val_ds.process_data()
    test_ds.process_data()

    def collate_fn(batch):
        """Pad sequences in a batch."""
        src_batch, tgt_batch = zip(*batch)
        src_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(s) for s in src_batch], batch_first=True, padding_value=1)
        tgt_padded = nn.utils.rnn.pad_sequence(
            [torch.tensor(t) for t in tgt_batch], batch_first=True, padding_value=1)
        return src_padded, tgt_padded

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    src_vocab_size = len(train_ds.src_vocab)
    tgt_vocab_size = len(train_ds.tgt_vocab)
    pad_idx        = train_ds.tgt_vocab['<pad>']

    # ── Model ─────────────────────────────────────────────────────────
    model = Transformer(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=cfg.d_model,
        N=cfg.N,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Optimizer & Scheduler ─────────────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1.0, betas=(0.9, 0.98), eps=1e-9
    )
    scheduler = NoamScheduler(optimizer, d_model=cfg.d_model,
                              warmup_steps=cfg.warmup_steps)

    # ── Loss ──────────────────────────────────────────────────────────
    loss_fn = LabelSmoothingLoss(
        vocab_size=tgt_vocab_size,
        pad_idx=pad_idx,
        smoothing=cfg.smoothing,
    )

    # ── Training Loop ─────────────────────────────────────────────────
    best_val_loss = float('inf')
    for epoch in range(cfg.num_epochs):
        train_loss = run_epoch(
            train_loader, model, loss_fn, optimizer, scheduler,
            epoch_num=epoch, is_train=True, device=device,
        )
        val_loss = run_epoch(
            val_loader, model, loss_fn, None, None,
            epoch_num=epoch, is_train=False, device=device,
        )

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")
        wandb.log({
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'lr':         optimizer.param_groups[0]['lr'],
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, path='best_checkpoint.pt')

    # ── Final BLEU ────────────────────────────────────────────────────
    load_checkpoint('best_checkpoint.pt', model)
    bleu = evaluate_bleu(model, test_loader, train_ds.tgt_vocab, device=device)
    print(f"Test BLEU: {bleu:.2f}")
    wandb.log({'test_bleu': bleu})
    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()