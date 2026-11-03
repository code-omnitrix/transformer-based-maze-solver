class MazeTransformerSolver(nn.Module):
    def __init__(
        self,
        grid_size=GRID_SIZE,
        d_model=128,
        nhead=8,
        num_layers=4,
        dim_feedforward=256,
        dropout=0.1,
        max_seq_len=64
    ):
        super().__init__()
        self.grid_size = grid_size
        self.d_model = d_model

        # Encoder Embeddings: Token + 2D Spatial (Row + Column)
        self.maze_token_emb = nn.Embedding(len(MAZE_VOCAB), d_model)
        self.row_emb = nn.Embedding(grid_size, d_model)
        self.col_emb = nn.Embedding(grid_size, d_model)

        # Decoder Embeddings: Action Token + 1D Step Position
        self.action_token_emb = nn.Embedding(len(ACTION_VOCAB), d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Core Encoder-Decoder Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        # Output projection to action vocabulary logits
        self.fc_out = nn.Linear(d_model, len(ACTION_VOCAB))

        # Precompute 2D coordinate indices for the flattened HxW grid
        rows = torch.arange(grid_size).unsqueeze(1).repeat(1, grid_size).view(-1)
        cols = torch.arange(grid_size).unsqueeze(0).repeat(grid_size, 1).view(-1)
        self.register_buffer("grid_rows", rows)
        self.register_buffer("grid_cols", cols)

    def encode_maze(self, grid):
        """
        Maps (B, H, W) grid into (B, H*W, d_model) with 2D spatial encodings.
        """
        B = grid.size(0)
        flat_grid = grid.view(B, -1)  # (B, H*W)

        tok_e = self.maze_token_emb(flat_grid)
        row_e = self.row_emb(self.grid_rows).unsqueeze(0)  # (1, H*W, d_model)
        col_e = self.col_emb(self.grid_cols).unsqueeze(0)  # (1, H*W, d_model)

        src = tok_e + row_e + col_e
        return self.transformer.encoder(src)

    def forward(self, grid, tgt_seq):
        """
        Forward pass for training with teacher forcing.
        grid: (B, H, W)
        tgt_seq: (B, T)
        """
        B, T = tgt_seq.shape

        # 1. Encode 2D Maze
        memory = self.encode_maze(grid)

        # 2. Embed Target Action Sequence with 1D Positional Encoding
        positions = torch.arange(0, T, device=tgt_seq.device).unsqueeze(0)
        tgt = self.action_token_emb(tgt_seq) + self.pos_emb(positions)

        # 3. Causal Mask (upper triangular True = masked out) & Padding Mask
        causal_mask = torch.triu(
            torch.ones(T, T, device=tgt_seq.device, dtype=torch.bool), diagonal=1
        )
        tgt_pad_mask = (tgt_seq == ACTION_VOCAB["PAD"])

        # 4. Decode
        out = self.transformer.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_pad_mask
        )
        return self.fc_out(out)
