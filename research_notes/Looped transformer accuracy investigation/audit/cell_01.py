def generate_solvable_maze(size=GRID_SIZE, wall_prob=WALL_PROB, min_path_len=4):
    """
    Generates a random 2D grid maze and uses Breadth-First Search (BFS)
    to guarantee a valid shortest path of (U, L, R, D) actions exists.
    """
    while True:
        # 1. Initialize random walls and free spaces
        grid = np.random.choice(
            [MAZE_VOCAB["FREE"], MAZE_VOCAB["WALL"]],
            size=(size, size),
            p=[1 - wall_prob, wall_prob]
        )

        # 2. Pick distinct random coordinates for Start and Goal
        start = (random.randint(0, size - 1), random.randint(0, size - 1))
        goal = (random.randint(0, size - 1), random.randint(0, size - 1))
        if start == goal:
            continue

        grid[start] = MAZE_VOCAB["START"]
        grid[goal] = MAZE_VOCAB["GOAL"]

        # 3. Breadth-First Search (BFS) for shortest action sequence
        queue = deque([(start[0], start[1], [])])
        visited = {start}

        while queue:
            r, c, path = queue.popleft()
            if (r, c) == goal:
                if len(path) >= min_path_len:
                    return grid, path, start, goal
                break

            # Explore neighbors in canonical order: U, L, R, D
            for action, (dr, dc) in MOVE_DELTAS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if grid[nr, nc] != MAZE_VOCAB["WALL"] and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append((nr, nc, path + [action]))


class MazeDataset(Dataset):
    def __init__(self, num_samples=12000, size=GRID_SIZE):
        self.samples = []
        self.max_seq_len = size * size  # Upper bound on path length

        print(f"Generating {num_samples} solvable mazes via BFS...")
        for _ in range(num_samples):
            grid, path, start, goal = generate_solvable_maze(size=size)
            # Tokenize action sequence: [BOS, ...actions..., EOS]
            seq = [ACTION_VOCAB["BOS"]] + [ACTION_VOCAB[a] for a in path] + [ACTION_VOCAB["EOS"]]
            self.samples.append((grid, seq))

        self.max_len = max(len(s[1]) for s in self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        grid, seq = self.samples[idx]
        # Pad sequence to uniform length for batch processing
        padded_seq = seq + [ACTION_VOCAB["PAD"]] * (self.max_len - len(seq))
        return (
            torch.tensor(grid, dtype=torch.long),
            torch.tensor(padded_seq, dtype=torch.long)
        )


# Create Training & Validation Datasets
train_dataset = MazeDataset(num_samples=50000)
val_dataset = MazeDataset(num_samples=5000)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
