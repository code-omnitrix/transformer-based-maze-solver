# %% [Cell 3 - Updated] 2D Spatial Looped Transformer Architecture
import math

class SinusoidalLoopEncoding(nn.Module):
    """
    Generates continuous sinusoidal embeddings for loop iteration index k.
    Allows test-time compute extrapolation (K_test > K_train).
    """
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        self.register_buffer("div_term", div_term)

    def forward(self, k_idx, device):
        # k_idx: integer loop step (1-indexed)
        pe = torch.zeros(1, 1, self.d_model, device=device)
        k_tensor = torch.tensor([float(k_idx)], device=device)
        pe[0, 0, 0::2] = torch.sin(k_tensor * self.div_term)
        pe[0, 0, 1::2] = torch.cos(k_tensor * self.div_term)
        return pe  # (1, 1, d_model)


class LoopedTransformerEncoder(nn.Module):
    """
    Weight-tied Transformer Encoder that iteratively refines the 2D maze state
    for `num_loops` steps with Input Injection (boundary condition anchoring).
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_loops):
        super().__init__()
        self.num_loops = num_loops
        # Single shared Encoder block (Pre-LN for stable deep recurrence)
        self.shared_block = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.loop_emb = SinusoidalLoopEncoding(d_model)
        self.input_gate = nn.Parameter(torch.ones(1))  # Learnable anchor weight alpha
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, src, num_loops=None):
        K = num_loops if num_loops is not None else self.num_loops
        h_init = src  # H^(0): Static maze boundary conditions
        h = src

        for k in range(1, K + 1):
            # Inject initial state H^(0) + sinusoidal loop-step encoding tau(k)
            step_enc = self.loop_emb(k, src.device)
            h_in = h + (self.input_gate * h_init) + step_enc
            h = self.shared_block(h_in)

        return self.final_norm(h)


class LoopedTransformerDecoder(nn.Module):
    """
    Weight-tied Causal Transformer Decoder that iteratively refines action
    representations across `num_loops` recurrent steps.
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout, num_loops):
        super().__init__()
        self.num_loops = num_loops
        # Single shared Decoder block (Pre-LN)
        self.shared_block = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.loop_emb = SinusoidalLoopEncoding(d_model)
        self.input_gate = nn.Parameter(torch.ones(1))
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, tgt_mask=None, tgt_key_padding_mask=None, num_loops=None):
        K = num_loops if num_loops is not None else self.num_loops
        h_init = tgt  # Initial embedded action prefix
        h = tgt

        for k in range(1, K + 1):
            step_enc = self.loop_emb(k, tgt.device)
            h_in = h + (self.input_gate * h_init) + step_enc
            h = self.shared_block(
                tgt=h_in,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask
            )

        return self.final_norm(h)


class MazeTransformerSolver(nn.Module):
    def __init__(
        self,
        grid_size=GRID_SIZE,
        d_model=128,
        nhead=8,
        num_layers=6,        # Treated as default `num_loops` (K)
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=128
    ):
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model
        self.num_loops = num_layers

        # Encoder Embeddings: Token + 2D Spatial (Row + Column)
        self.maze_token_emb = nn.Embedding(len(MAZE_VOCAB), d_model)
        self.row_emb = nn.Embedding(grid_size, d_model)
        self.col_emb = nn.Embedding(grid_size, d_model)

        # Decoder Embeddings: Action Token + 1D Step Position
        self.action_token_emb = nn.Embedding(len(ACTION_VOCAB), d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Looped Encoder-Decoder Container (compatible with self.transformer.encoder / .decoder)
        self.transformer = nn.ModuleDict({
            "encoder": LoopedTransformerEncoder(
                d_model, nhead, dim_feedforward, dropout, num_loops=self.num_loops
            ),
            "decoder": LoopedTransformerDecoder(
                d_model, nhead, dim_feedforward, dropout, num_loops=self.num_loops
            )
        })

        # Output projection to action vocabulary logits
        self.fc_out = nn.Linear(d_model, len(ACTION_VOCAB))

        # Precompute 2D coordinate indices for the flattened HxW grid
        rows = torch.arange(grid_size).unsqueeze(1).repeat(1, grid_size).view(-1)
        cols = torch.arange(grid_size).unsqueeze(0).repeat(grid_size, 1).view(-1)
        self.register_buffer("grid_rows", rows)
        self.register_buffer("grid_cols", cols)

    def encode_maze(self, grid, num_loops=None):
        """
        Maps (B, H, W) grid into (B, H*W, d_model) via K recurrent encoder loops.
        """
        B = grid.size(0)
        flat_grid = grid.view(B, -1)  # (B, H*W)

        tok_e = self.maze_token_emb(flat_grid)
        row_e = self.row_emb(self.grid_rows).unsqueeze(0)  # (1, H*W, d_model)
        col_e = self.col_emb(self.grid_cols).unsqueeze(0)  # (1, H*W, d_model)

        src = tok_e + row_e + col_e
        return self.transformer["encoder"](src, num_loops=num_loops)

    def forward(self, grid, tgt_seq, num_loops=None):
        """
        Forward pass for training with teacher forcing.
        grid: (B, H, W)
        tgt_seq: (B, T)
        num_loops: Optional override for recurrent depth K
        """
        B, T = tgt_seq.shape

        # 1. Encode 2D Maze via Looped Encoder
        memory = self.encode_maze(grid, num_loops=num_loops)

        # 2. Embed Target Action Sequence with 1D Positional Encoding
        positions = torch.arange(0, T, device=tgt_seq.device).unsqueeze(0)
        tgt = self.action_token_emb(tgt_seq) + self.pos_emb(positions)

        # 3. Causal Mask (upper triangular True = masked out) & Padding Mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt_seq.device, dtype=torch.bool), diagonal=1
        )
        tgt_pad_mask = (tgt_seq == ACTION_VOCAB["PAD"])

        # 4. Decode via Looped Decoder
        out = self.transformer["decoder"](
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            num_loops=num_loops
        )
        return self.fc_out(out)