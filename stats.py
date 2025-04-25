# stats.py
import subprocess, random, time, os
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def is_solvable(board, N):
    # Count inversions
    inv = 0
    flat = [x for x in board if x != 0]
    for i in range(len(flat)):
        for j in range(i+1, len(flat)):
            if flat[i] > flat[j]:
                inv += 1
    if N % 2 == 1:
        return inv % 2 == 0
    else:
        row_from_bottom = N - (board.index(0) // N)
        # goal parity = 1 for even N
        return (inv + row_from_bottom) % 2 == 1

def random_board(N, max_shuffle=None):
    goal = list(range(1, N*N)) + [0]
    if max_shuffle is not None:
        while True:
            board = goal[:]
            zero = N*N - 1
            swap_count = 0
            # keep track of every board state we've visited during shuffling
            seen = { tuple(board) }

            dr = (-1, 1, 0, 0)
            dc = (0, 0, -1, 1)

            for _ in range(max_shuffle):
                r, c = divmod(zero, N)
                # all possible moves of the blank
                candidates = []
                for dr_, dc_ in zip(dr, dc):
                    nr, nc = r + dr_, c + dc_
                    if 0 <= nr < N and 0 <= nc < N:
                        candidates.append(nr*N + nc)

                # filter out any move that would revisit a seen state
                next_moves = []
                for swap_idx in candidates:
                    new_board = board[:]
                    new_board[zero], new_board[swap_idx] = new_board[swap_idx], new_board[zero]
                    if tuple(new_board) not in seen:
                        next_moves.append((swap_idx, new_board))

                if not next_moves:
                    # no unseen neighbors — stop early
                    break

                # pick one at random
                swap_idx, new_board = random.choice(next_moves)
                board = new_board
                zero = swap_idx
                seen.add(tuple(board))
                swap_count += 1
            if swap_count == max_shuffle:
                return board

    # fully random
    while True:
        board = list(range(N*N))
        random.shuffle(board)
        # ensure blank at end for consistency
        if board[-1] != 0:
            zi = board.index(0)
            board[zi], board[-1] = board[-1], board[zi]
        if is_solvable(board, N):
            return board

def run_once(board, heuristic):
    cmd = ['./solver', heuristic, str(int(len(board)**0.5))] + list(map(str, board))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    start = time.perf_counter()
    stats = {'visited': 0, 'generated': 0, 'path_length': 0, 'time_s': 0.0}

    # stream solver-logs in real time
    while True:
        line = proc.stderr.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            print(line.rstrip())  # real-time solver log

    out, _ = proc.communicate()
    elapsed = time.perf_counter() - start

    # parse final solver output
    for line in out.splitlines():
        print(line)
        if line.startswith('visited:'):
            stats['visited'] = int(line.split(':',1)[1])
        elif line.startswith('generated:'):
            stats['generated'] = int(line.split(':',1)[1])
        elif line.startswith('path_length:'):
            stats['path_length'] = int(line.split(':',1)[1])
        elif line.startswith('time_s:'):
            stats['time_s'] = float(line.split(':',1)[1])

    return stats

if __name__=='__main__':
    trials = 50
    outdir = 'plots'
    os.makedirs(outdir, exist_ok=True)

    scenarios = [
        (3, None, ['hamming','manhattan']),
        (4, 20,   ['hamming','manhattan']),
        (4, None, ['manhattan'])
    ]

    for N, shuf, heus in scenarios:
        kind = f"{N}x{N}_{'shuffle' if shuf else 'random'}"
        data = {h: {'visited': [], 'generated': [], 'path': [], 'time': []} for h in heus}

        logging.info(f"=== {kind} ({trials} trials) ===")
        for i in range(1, trials + 1):
            # Generate one board for this trial
            board = random_board(N, max_shuffle=shuf)

            for h in heus:
                print(f"\n>>> Run {i}/{trials} — {h.upper()} on {kind}")
                stats = run_once(board, h)
                data[h]['visited'].append(stats['visited'])
                data[h]['generated'].append(stats['generated'])
                data[h]['path'].append(stats['path_length'])
                data[h]['time'].append(stats['time_s'])

        # compute & log averages
        for h, vals in data.items():
            avg_v = sum(vals['visited'])/len(vals['visited'])
            avg_g = sum(vals['generated'])/len(vals['generated'])
            avg_p = sum(vals['path'])/len(vals['path'])
            avg_t = sum(vals['time'])/len(vals['time'])
            logging.info(
                f"Avg {h} on {kind}: visited={avg_v:.1f}, gen={avg_g:.1f}, "
                f"path={avg_p:.1f}, time={avg_t:.3f}s"
            )

        # overlayed scatter: visited vs path
        plt.figure()
        for h, vals in data.items():
            plt.scatter(vals['path'], vals['visited'], alpha=0.6, label=h.capitalize())
        plt.title(f"Visited vs Path — {kind}")
        plt.xlabel("Path Length"); plt.ylabel("Visited"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"{outdir}/visited_vs_path_{kind}.png"); plt.close()

        # overlayed scatter: ratio vs path
        plt.figure()
        for h, vals in data.items():
            ratio = [v/g if g else 0 for v,g in zip(vals['visited'], vals['generated'])]
            plt.scatter(vals['path'], ratio, alpha=0.6, label=h.capitalize())
        plt.title(f"Visited/Generated vs Path — {kind}")
        plt.xlabel("Path Length"); plt.ylabel("Visited/Generated"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"{outdir}/ratio_vs_path_{kind}.png"); plt.close()

        # solution-length histogram
        plt.figure()
        bins = range(min(min(vals['path']) for vals in data.values()),
             max(max(vals['path']) for vals in data.values())+2)
        for h, vals in data.items():
            plt.hist(vals['path'], bins=bins, alpha=0.6, label=h.capitalize())

        plt.title(f"Solution Length Dist — {kind}")
        plt.xlabel("Path Length")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(True)
        plt.xticks(bins)  # <-- force all integer bins to show
        plt.tight_layout()
        plt.savefig(f"{outdir}/hist_path_{kind}.png")
        plt.close()


        # bar: avg visited vs gen
        labels = list(data.keys())
        x = range(len(labels))
        avg_v = [sum(data[h]['visited'])/trials for h in labels]
        avg_g = [sum(data[h]['generated'])/trials for h in labels]
        plt.figure()
        plt.bar(x,     avg_v, width=0.4, label='Visited')
        plt.bar([xi+0.4 for xi in x], avg_g, width=0.4, label='Generated')
        plt.xticks([xi+0.2 for xi in x], [h.capitalize() for h in labels])
        plt.title(f"Avg Visited vs Generated — {kind}")
        plt.ylabel("Count"); plt.legend(); plt.grid(True)
        plt.tight_layout(); plt.savefig(f"{outdir}/bar_vis_gen_{kind}.png"); plt.close()

    logging.info("All plots saved under 'plots/'.")
