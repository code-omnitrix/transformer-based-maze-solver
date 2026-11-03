model = MazeTransformerSolver().to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=ACTION_VOCAB["PAD"])

EPOCHS = 30 

for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0

    for grids, seqs in train_loader:
        grids, seqs = grids.to(DEVICE), seqs.to(DEVICE)

        # Teacher Forcing: Input is [BOS, a_1, ..., a_T], Target is [a_1, ..., a_T, EOS]
        tgt_input = seqs[:, :-1]
        tgt_labels = seqs[:, 1:]

        optimizer.zero_grad()
        logits = model(grids, tgt_input)

        loss = criterion(logits.reshape(-1, len(ACTION_VOCAB)), tgt_labels.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    # Calculate validation token accuracy
    model.eval()
    correct_tokens, total_tokens = 0, 0
    with torch.no_grad():
        for grids, seqs in val_loader:
            grids, seqs = grids.to(DEVICE), seqs.to(DEVICE)
            tgt_input, tgt_labels = seqs[:, :-1], seqs[:, 1:]
            logits = model(grids, tgt_input)
            preds = logits.argmax(dim=-1)
            mask = tgt_labels != ACTION_VOCAB["PAD"]
            correct_tokens += (preds[mask] == tgt_labels[mask]).sum().item()
            total_tokens += mask.sum().item()

    val_acc = (correct_tokens / total_tokens) * 100
    print(f"Epoch {epoch:02d}/{EPOCHS} | Train Loss: {total_loss/len(train_loader):.4f} | Val Token Acc: {val_acc:.2f}%")
