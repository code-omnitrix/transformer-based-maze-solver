@torch.no_grad()
def solve_maze_autoregressive(model, grid, max_steps=40):
    """
    Takes a raw 2D maze grid and autoregressively predicts ['U', 'L', 'R', 'D']
    one token at a time until EOS is emitted.
    """
    model.eval()
    grid_tensor = torch.tensor(grid, dtype=torch.long, device=DEVICE).unsqueeze(0)
    memory = model.encode_maze(grid_tensor)

    generated = [ACTION_VOCAB["BOS"]]

    for _ in range(max_steps):
        tgt_seq = torch.tensor([generated], dtype=torch.long, device=DEVICE)
        T = tgt_seq.size(1)
        positions = torch.arange(0, T, device=DEVICE).unsqueeze(0)

        tgt = model.action_token_emb(tgt_seq) + model.pos_emb(positions)
        causal_mask = torch.triu(torch.ones(T, T, device=DEVICE, dtype=torch.bool), diagonal=1)

        out = model.transformer.decoder(tgt=tgt, memory=memory, tgt_mask=causal_mask)
        next_token_logits = model.fc_out(out[:, -1, :])
        next_token = next_token_logits.argmax(dim=-1).item()

        if next_token == ACTION_VOCAB["EOS"]:
            break
        generated.append(next_token)

    return [IDX_TO_ACTION[idx] for idx in generated[1:]]


def visualize_predictions(model, num_tests=4):
    fig, axes = plt.subplots(1, num_tests, figsize=(4 * num_tests, 4))
    if num_tests == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        grid, true_path, start, goal = generate_solvable_maze()
        pred_actions = solve_maze_autoregressive(model, grid)

        # Trace predicted path coordinates
        curr_r, curr_c = start
        path_coords = [(curr_r, curr_c)]
        valid_trace = True

        for act in pred_actions:
            if act not in MOVE_DELTAS:
                continue
            dr, dc = MOVE_DELTAS[act]
            nr, nc = curr_r + dr, curr_c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE and grid[nr, nc] != MAZE_VOCAB["WALL"]:
                curr_r, curr_c = nr, nc
                path_coords.append((curr_r, curr_c))
            else:
                valid_trace = False
                break

        reached_goal = (curr_r, curr_c) == goal and valid_trace

        # Render grid colors
        color_map = np.ones((GRID_SIZE, GRID_SIZE, 3))
        color_map[grid == MAZE_VOCAB["WALL"]] = [0.15, 0.15, 0.2]   # Dark Slate Walls
        color_map[start] = [0.2, 0.6, 1.0]                          # Blue Start
        color_map[goal] = [0.2, 0.85, 0.4]                          # Green Goal

        ax.imshow(color_map)

        # Plot predicted trajectory line
        ys = [c for r, c in path_coords]
        xs = [r for r, c in path_coords]
        ax.plot(ys, xs, color="crimson", linewidth=3, marker="o", markersize=5, label="Pred Path")

        status = "SOLVED" if reached_goal else "FAILED"
        ax.set_title(
            f"Test {idx+1}: {status}\nPred: {'-'.join(pred_actions)}\nTrue: {'-'.join(true_path)}",
            fontsize=9
        )
        ax.set_xticks(range(GRID_SIZE))
        ax.set_yticks(range(GRID_SIZE))
        ax.grid( color="gray", linestyle=":", linewidth=0.5)

    plt.tight_layout()
    plt.show()

# Run visual evaluation on 4 unseen mazes
visualize_predictions(model, num_tests=8)