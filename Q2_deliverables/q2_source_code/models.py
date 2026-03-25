"""
Model Implementations – Task 1
CSL 7640 - NLU Assignment 2, Problem 2
Author: B23CS1061

Three character-level generative models implemented from scratch:

  1. VanillaRNN       – standard Elman RNN language model (unidirectional)
  2. BidirectionalLSTM – ELMo-style BiLM: forward LSTM (next-char) +
                         backward LSTM (prev-char) trained jointly; generation
                         uses forward LSTM only — no train/generation mismatch
  3. AttentionRNN     – autoregressive RNN with self-attention over all past
                         hidden states (Bahdanau-style); no encoder, fully
                         left-to-right — no train/generation mismatch

All models share the same generate() interface for model-agnostic evaluation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import SOS_TOKEN, EOS_TOKEN, PAD_TOKEN


# ══════════════════════════════════════════════════════════════════════════════
# 1. Vanilla RNN Language Model
# ══════════════════════════════════════════════════════════════════════════════

class VanillaRNN(nn.Module):
    """
    Character-level name generator using a plain Elman RNN language model.

    Architecture:
        Embedding(vocab, embed_dim)
        → RNN(embed_dim, hidden_size, num_layers)
        → Linear(hidden_size, vocab_size)
        → softmax (during generation)

    At each step the model reads the previous character and produces logits
    over the full vocabulary. During training, teacher-forcing is applied
    (ground-truth characters are always fed as input). During generation,
    the model samples autoregressively.

    Hyperparameters:
        vocab_size   = 29    (26 letters + PAD/SOS/EOS)
        embed_dim    = 64
        hidden_size  = 256
        num_layers   = 2
        dropout      = 0.3
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Character embedding table
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)

        # Elman RNN (tanh nonlinearity)
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Project hidden state to vocabulary logits
        self.fc  = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x      – (batch, seq_len) token indices (teacher-forced input)
            hidden – previous hidden state, or None for zero initialisation
        Returns:
            logits – (batch, seq_len, vocab_size)
            hidden – updated hidden state
        """
        emb = self.drop(self.embed(x))       # (B, T, E)
        out, hidden = self.rnn(emb, hidden)  # (B, T, H)
        return self.fc(out), hidden           # (B, T, V)

    @torch.no_grad()
    def generate(
        self,
        device: torch.device,
        max_len: int = 20,
        temperature: float = 0.8,
    ) -> list[int]:
        """
        Autoregressively sample one name.
        Returns list of token indices (SOS and EOS excluded).
        """
        self.eval()
        x = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
        hidden = None
        chars = []
        for _ in range(max_len):
            logits, hidden = self.forward(x, hidden)
            probs  = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            tok    = torch.multinomial(probs, num_samples=1).item()
            if tok == EOS_TOKEN:
                break
            # Skip PAD and SOS if somehow sampled
            if tok not in (PAD_TOKEN, SOS_TOKEN):
                chars.append(tok)
            x = torch.tensor([[tok]], dtype=torch.long, device=device)
        return chars


# ══════════════════════════════════════════════════════════════════════════════
# 2. Bidirectional LSTM (ELMo-style BiLM)
# ══════════════════════════════════════════════════════════════════════════════

class BidirectionalLSTM(nn.Module):
    """
    Bidirectional LSTM language model (ELMo / BiLM style).

    Architecture:
        Shared embedding layer

        Forward LSTM:
            reads x_1, x_2, ..., x_T  (left → right)
            predicts x_{t+1} at each step
            loss: cross-entropy on forward logits

        Backward LSTM:
            reads x_T, x_{T-1}, ..., x_1  (right → left, reversed sequence)
            predicts x_{t-1} at each step
            loss: cross-entropy on backward logits

        Total loss = forward_CE + backward_CE (during training)

    Generation:
        Only the forward LSTM is used. This avoids the train/generation
        mismatch that arises in encoder-decoder designs where the encoder
        reads the full target sequence during training but cannot do so
        at generation time.

    Why bidirectional?
        Training the backward LSTM simultaneously forces the shared embedding
        to encode information about BOTH preceding and following context, making
        the representations richer than a purely unidirectional model. This
        improves the forward LSTM's generation quality even though the backward
        LSTM is unused at inference.

    Hyperparameters:
        vocab_size   = 29
        embed_dim    = 64
        hidden_size  = 128   (reduced from 256 for fair param-count comparison)
        num_layers   = 2
        dropout      = 0.3

    Parameter budget note:
        Two independent LSTMs (forward + backward) are inherently ~2× a single
        LSTM. Setting hidden_size=128 keeps total BLSTM params (~472K) in the
        same order of magnitude as VanillaRNN (223K) and AttentionRNN (~362K),
        enabling a fair architectural comparison.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 128,   # 128 keeps param count comparable to other models
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # Shared embedding: both LSTMs use the same character representations
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.drop  = nn.Dropout(dropout)

        # Forward LSTM: left-to-right language model
        self.fwd_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Backward LSTM: right-to-left language model (trained on reversed seqs)
        self.bwd_lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Independent projection heads for forward and backward
        self.fwd_fc = nn.Linear(hidden_size, vocab_size)
        self.bwd_fc = nn.Linear(hidden_size, vocab_size)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Bidirectional language model forward pass.

        Args:
            x – (batch, seq_len) full padded sequence with SOS and EOS
        Returns:
            fwd_logits – (batch, T-1, vocab_size)  predicts x[1..T]   from x[0..T-1]
            bwd_logits – (batch, T-1, vocab_size)  predicts x[T-2..0] from x[T-1..1]
        """
        emb = self.drop(self.embed(x))           # (B, T, E)

        # Forward: input = x[:-1], target = x[1:]
        fwd_in  = emb[:, :-1, :]                 # (B, T-1, E)
        fwd_out, _ = self.fwd_lstm(fwd_in)       # (B, T-1, H)
        fwd_logits = self.fwd_fc(fwd_out)        # (B, T-1, V)

        # Backward: reverse the sequence, input = reversed x[1:], target = reversed x[:-1]
        # i.e. we give the model x[T-1], x[T-2], ..., x[1] and it predicts x[T-2], ..., x[0]
        bwd_in  = emb[:, 1:, :].flip(dims=[1])   # (B, T-1, E) reversed
        bwd_out, _ = self.bwd_lstm(bwd_in)        # (B, T-1, H)
        bwd_logits = self.bwd_fc(bwd_out)         # (B, T-1, V)

        return fwd_logits, bwd_logits

    @torch.no_grad()
    def generate(
        self,
        device: torch.device,
        max_len: int = 20,
        temperature: float = 0.8,
    ) -> list[int]:
        """
        Generate a name using only the forward LSTM (left-to-right).
        Backward LSTM is not used during generation.
        """
        self.eval()
        x      = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
        hidden = None
        chars  = []
        for _ in range(max_len):
            emb = self.drop(self.embed(x))          # (1, 1, E)
            out, hidden = self.fwd_lstm(emb, hidden) # (1, 1, H)
            logits = self.fwd_fc(out[:, -1, :])      # (1, V)
            probs  = F.softmax(logits / temperature, dim=-1)
            tok    = torch.multinomial(probs, num_samples=1).item()
            if tok == EOS_TOKEN:
                break
            if tok not in (PAD_TOKEN, SOS_TOKEN):
                chars.append(tok)
            x = torch.tensor([[tok]], dtype=torch.long, device=device)
        return chars


# ══════════════════════════════════════════════════════════════════════════════
# 3. RNN with Self-Attention over Past Hidden States
# ══════════════════════════════════════════════════════════════════════════════

class AttentionRNN(nn.Module):
    """
    Autoregressive Elman RNN language model with Bahdanau-style additive
    self-attention over all past hidden states.

    Architecture (at each generation step t):
        1. Embed current token x_t
        2. RNN step: h_t = tanh( W_ih · embed(x_t)  +  W_hh · h_{t-1} )
              — plain Elman RNN, no gates (no LSTM/GRU)
              — hidden state shape: (num_layers, batch, hidden_size)
        3. Self-attention over past hidden states H = [h_1, ..., h_{t-1}]:
               score_s = v · tanh( W_q · h_t  +  W_k · h_s )   for s < t
               α = softmax(scores)
               context_t = Σ_s α_s · h_s
           (if t=1 there are no past states; context = zeros)
        4. Predict next token: fc( concat(h_t, context_t) ) → vocab logits

    Unlike encoder-decoder attention (which needs a separate source sequence),
    this self-attention is fully autoregressive: at training and generation time
    the model attends only to characters it has already processed. There is no
    train/generation mismatch.

    Why attention helps over plain RNN:
        The RNN hidden state must compress all past context into a fixed-size
        vector; long-range information decays due to the vanishing gradient
        problem. Attention gives the model a direct, trainable retrieval path
        to earlier representations — e.g. remembering the first character when
        generating the final character of a long name.

    Hyperparameters:
        vocab_size   = 29
        embed_dim    = 64
        hidden_size  = 256
        num_layers   = 2
        dropout      = 0.3
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN)
        self.drop  = nn.Dropout(dropout)

        # Plain Elman RNN — tanh nonlinearity, NO gates (not GRU/LSTM).
        # Hidden state: h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
        # Weight shapes per layer: W_ih (H×E), W_hh (H×H) — no gate expansion.
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity="tanh",   # standard Elman RNN activation
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bahdanau additive attention parameters
        # W_q: projects current hidden (query)
        # W_k: projects past hidden states (keys)
        # v: scores the combined representation
        self.attn_Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_Wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_v  = nn.Linear(hidden_size, 1, bias=False)

        # Output: project concat(h_t, context_t) → vocab
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def _attend(
        self,
        query: torch.Tensor,   # (B, H) — current hidden (last layer)
        keys: torch.Tensor,    # (B, S, H) — all past hidden states
    ) -> torch.Tensor:
        """
        Compute Bahdanau additive attention.
        Returns context vector of shape (B, H).
        If keys is empty (first step), returns zeros.
        """
        if keys.size(1) == 0:
            return torch.zeros_like(query)   # no past context at t=0

        q = self.attn_Wq(query).unsqueeze(1)    # (B, 1, H)
        k = self.attn_Wk(keys)                  # (B, S, H)
        scores = self.attn_v(torch.tanh(q + k)).squeeze(-1)  # (B, S)
        alpha  = F.softmax(scores, dim=-1)       # (B, S)
        context = (alpha.unsqueeze(-1) * keys).sum(dim=1)    # (B, H)
        return context

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Teacher-forced forward pass with self-attention over past hidden states.

        Args:
            x – (batch, seq_len) token sequence (includes SOS, excludes nothing)
        Returns:
            logits – (batch, seq_len-1, vocab_size)
                     predicts x[1..T] from x[0..T-1], step by step with attention
        """
        inp = x[:, :-1]                           # (B, T-1)  input
        emb = self.drop(self.embed(inp))          # (B, T-1, E)
        T   = emb.size(1)

        # Run full RNN to get all hidden states at once
        all_hidden, _ = self.rnn(emb)             # (B, T-1, H)

        logits_list = []
        for t in range(T):
            h_t = all_hidden[:, t, :]             # (B, H)
            # Attend over all hidden states BEFORE position t
            past = all_hidden[:, :t, :]           # (B, t, H) — empty at t=0
            ctx  = self._attend(h_t, past)        # (B, H)
            out  = self.fc(torch.cat([h_t, ctx], dim=-1))  # (B, V)
            logits_list.append(out)

        return torch.stack(logits_list, dim=1)    # (B, T-1, V)

    @torch.no_grad()
    def generate(
        self,
        device: torch.device,
        max_len: int = 20,
        temperature: float = 0.8,
    ) -> list[int]:
        """
        Autoregressive generation with self-attention over past hidden states.
        Maintains a growing buffer of past hidden states.
        """
        self.eval()
        x       = torch.tensor([[SOS_TOKEN]], dtype=torch.long, device=device)
        hidden  = None
        past_hs = []   # buffer of past last-layer hidden states (B, H) each
        chars   = []

        for _ in range(max_len):
            emb = self.drop(self.embed(x))           # (1, 1, E)
            out, hidden = self.rnn(emb, hidden)      # (1, 1, H)
            h_t = out[:, 0, :]                       # (1, H) — last-layer output

            # Self-attention over all past hidden states collected so far
            if past_hs:
                past_tensor = torch.stack(past_hs, dim=1)  # (1, S, H)
                ctx = self._attend(h_t, past_tensor)        # (1, H)
            else:
                ctx = torch.zeros_like(h_t)

            past_hs.append(h_t)

            logits = self.fc(torch.cat([h_t, ctx], dim=-1))  # (1, V)
            probs  = F.softmax(logits / temperature, dim=-1)
            tok    = torch.multinomial(probs, num_samples=1).item()
            if tok == EOS_TOKEN:
                break
            if tok not in (PAD_TOKEN, SOS_TOKEN):
                chars.append(tok)
            x = torch.tensor([[tok]], dtype=torch.long, device=device)

        return chars
