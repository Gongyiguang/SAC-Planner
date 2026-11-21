"""Run ComponentAStar on a synthetic one-passage map.

The component construction, graph adjacency, state representation, and A* cost
follow path_planner/SEA_pathPlanner.py and ComponentAstar_pathPlanner.py.
SA-Planner resolution selection and CB-Refiner are outside this demo.
"""
import argparse
import heapq
from pathlib import Path

import numpy as np


def normalize(values):
    low, high = np.percentile(values, [5, 95])
    if low == high:
        return np.full_like(values, 0.5, dtype=np.float32)
    return np.clip((np.clip(values, low, high) - low) / (high - low),
                   0.0, 1.0).astype(np.float32)


def build_components(passable, block_size):
    """Mirror SEAPlanner._build_components(enable_gid_split=True)."""
    height, width = passable.shape
    labels = np.full((height, width), -1, dtype=np.int32)
    module_of, local_label, centroids = {}, {}, {}
    next_local = {}
    gid = 0
    offsets = [(dr, dc) for dr in (-1, 0, 1)
               for dc in (-1, 0, 1) if dr or dc]
    for r0 in range(0, height, block_size):
        for c0 in range(0, width, block_size):
            r1, c1 = min(r0 + block_size, height), min(c0 + block_size, width)
            tile = passable[r0:r1, c0:c1]
            seen = np.zeros(tile.shape, dtype=bool)
            for rr, cc in zip(*np.nonzero(tile)):
                if seen[rr, cc]:
                    continue
                stack, cells = [(rr, cc)], []
                seen[rr, cc] = True
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dr, dc in offsets:
                        ny, nx = y + dr, x + dc
                        if (0 <= ny < tile.shape[0] and 0 <= nx < tile.shape[1]
                                and tile[ny, nx] and not seen[ny, nx]):
                            seen[ny, nx] = True
                            stack.append((ny, nx))
                component = np.asarray(cells)
                labels[r0 + component[:, 0], c0 + component[:, 1]] = gid
                module_of[gid] = (r0 // block_size, c0 // block_size)
                module = module_of[gid]
                next_local[module] = next_local.get(module, 0) + 1
                local_label[gid] = next_local[module]
                cells = np.asarray(cells)
                centroids[gid] = (float(cells[:, 0].mean() + r0),
                                  float(cells[:, 1].mean() + c0))
                gid += 1

    adjacency = {g: set() for g in module_of}
    height, width = labels.shape
    for dr, dc in [(-1, 0), (0, -1), (-1, -1), (-1, 1)]:
        ra, rb = max(0, -dr), height - max(0, dr)
        ca, cb = max(0, -dc), width - max(0, dc)
        a = labels[ra:rb, ca:cb]
        b = labels[ra + dr:rb + dr, ca + dc:cb + dc]
        valid = (a >= 0) & (b >= 0) & (a != b)
        for u, v in zip(a[valid], b[valid]):
            adjacency[int(u)].add(int(v))
            adjacency[int(v)].add(int(u))
    return labels, module_of, local_label, centroids, adjacency


def astar(start_gid, goal_gid, goal_pixel, module_of, local_label, centroids,
          adjacency, block_size, dem, slope, landcover):
    """ComponentAStar's state, terrain edge cost, heuristic, and stopping rule."""
    slopenorm, lccnorm = normalize(slope), normalize(landcover)
    rowdiff = np.abs(np.diff(dem, axis=0)).mean()
    coldiff = np.abs(np.diff(dem, axis=1)).mean()
    dem_scale = max(float((rowdiff + coldiff) / 2), 1e-3)
    h_factor = max(1.0, min(1.0 + 5 * slopenorm.mean() + 5 * lccnorm.mean(), 8.0))

    def state(g):
        i, j = module_of[g]
        return i, j, local_label[g]

    def heuristic(node):
        g = state_to_gid[node]
        cr, cc = centroids[g]
        gr, gc = goal_pixel
        return h_factor * np.hypot(cr - gr, cc - gc) / max(block_size, 1)

    def edge_cost(a, b):
        ia, ja = module_of[a]
        ib, jb = module_of[b]
        dx, dy = abs(ia - ib), abs(ja - jb)
        za = dem[min(ia * block_size, dem.shape[0] - 1),
                 min(ja * block_size, dem.shape[1] - 1)]
        zb = dem[min(ib * block_size, dem.shape[0] - 1),
                 min(jb * block_size, dem.shape[1] - 1)]
        distance = np.sqrt(dx * dx + dy * dy + ((za - zb) / dem_scale) ** 2)
        sa = slopenorm[ia, ja]
        sb = slopenorm[ib, jb]
        la = lccnorm[ia, ja]
        lb = lccnorm[ib, jb]
        return distance * (1 + 5 * (sa + sb) / 2 + 5 * (la + lb) / 2)

    gid_to_state = {g: state(g) for g in module_of}
    state_to_gid = {v: k for k, v in gid_to_state.items()}
    start, goal = gid_to_state[start_gid], gid_to_state[goal_gid]
    frontier = [(0.0, 0, start)]
    counter, parent, gscore, fscore, closed = 0, {}, {start: 0.0}, {start: 0.0}, set()
    while frontier:
        priority, _, current = heapq.heappop(frontier)
        if priority != fscore.get(current):
            continue
        if current in closed:
            continue
        closed.add(current)
        if current == goal:
            path = [current]
            while path[-1] in parent:
                path.append(parent[path[-1]])
            path.reverse()
            return [state_to_gid[node] for node in path], len(closed)
        current_gid = state_to_gid[current]
        for neighbour_gid in sorted(adjacency[current_gid]):
            neighbour = gid_to_state[neighbour_gid]
            if neighbour in closed:
                continue
            tentative = gscore[current] + edge_cost(current_gid, neighbour_gid)
            if tentative < gscore.get(neighbour, float('inf')):
                parent[neighbour] = current
                gscore[neighbour] = tentative
                fscore[neighbour] = tentative + heuristic(neighbour)
                counter += 1
                heapq.heappush(frontier,
                               (fscore[neighbour], counter, neighbour))
    return None, len(closed)


def recover_grid_path(passable, labels, allowed_gids, start, goal):
    """Prefer a straight free-cell route inside the components selected by A*."""
    allowed = np.isin(labels, list(allowed_gids)) & passable
    offsets = [(dr, dc) for dr in (-1, 0, 1)
               for dc in (-1, 0, 1) if dr or dc]
    dr_goal, dc_goal = goal[0] - start[0], goal[1] - start[1]
    length_sq = dr_goal * dr_goal + dc_goal * dc_goal or 1
    frontier = [(0, 0.0, 0, start)]
    serial = 0
    parent = {start: None}
    best = {start: (0, 0.0)}
    while frontier:
        steps, deviation, _, cell = heapq.heappop(frontier)
        if best.get(cell) != (steps, deviation):
            continue
        row, col = cell
        if (row, col) == goal:
            path = []
            cell = goal
            while cell is not None:
                path.append(cell)
                cell = parent[cell]
            return np.asarray(path[::-1], dtype=np.int32)
        for dr, dc in offsets:
            nxt = row + dr, col + dc
            nr, nc = nxt
            if (0 <= nr < allowed.shape[0] and 0 <= nc < allowed.shape[1]
                    and allowed[nr, nc]):
                projection = ((nr - start[0]) * dr_goal
                              + (nc - start[1]) * dc_goal) / length_sq
                ideal_row = start[0] + projection * dr_goal
                ideal_col = start[1] + projection * dc_goal
                next_cost = (steps + 1, deviation
                             + abs(nr - ideal_row) + abs(nc - ideal_col))
                if next_cost < best.get(nxt, (float('inf'), float('inf'))):
                    best[nxt] = next_cost
                    parent[nxt] = cell
                    serial += 1
                    heapq.heappush(frontier, (*next_cost, serial, nxt))
    raise RuntimeError('Could not recover a free-cell route within the A* components')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--R', type=int, default=10)
    parser.add_argument('--map', type=Path, default=Path(__file__).with_name('sample_map.npz'))
    parser.add_argument('--output', type=Path, default=Path('demo_component_graph.png'))
    args = parser.parse_args()
    if args.R < 1:
        parser.error('--R must be positive')
    with np.load(args.map, allow_pickle=False) as data:
        dem, slope, lcc, obstacles = (data[k] for k in
                                      ('dem', 'slope', 'landcovercost', 'obstaclemask'))
    passable = ~obstacles
    height, width = passable.shape
    start_pixel = (height // 2, width // 2 - 32)
    goal_pixel = (height // 2, width // 2 + 32)
    labels, modules, local, centers, adjacency = build_components(passable, args.R)
    start_gid, goal_gid = int(labels[start_pixel]), int(labels[goal_pixel])
    if start_gid < 0 or goal_gid < 0:
        parser.error('start or goal falls on an obstacle')
    path, visited = astar(start_gid, goal_gid, goal_pixel, modules, local, centers,
                          adjacency, args.R, dem, slope, lcc)
    if path is None:
        raise SystemExit('CGA* component-graph search failed')
    path_gids = set(path)
    pixel_path = recover_grid_path(passable, labels, path_gids,
                                   start_pixel, goal_pixel)
    if (tuple(pixel_path[0]) != start_pixel
            or tuple(pixel_path[-1]) != goal_pixel
            or not passable[pixel_path[:, 0], pixel_path[:, 1]].all()):
        raise RuntimeError('Recovered route endpoints or traversability check failed')
    steps = np.diff(pixel_path, axis=0)
    if len(steps) and np.any(np.max(np.abs(steps), axis=1) != 1):
        raise RuntimeError('Recovered route contains non-adjacent grid cells')
    gate_slice = slice(width // 2 - 2, width // 2 + 2)
    if not np.any((pixel_path[:, 1] >= gate_slice.start)
                  & (pixel_path[:, 1] < gate_slice.stop)):
        raise SystemExit('Search succeeded but its route does not pass through the gate')
    if np.any(obstacles[pixel_path[:, 0], pixel_path[:, 1]]):
        raise RuntimeError('Recovered route intersects an obstacle')
    print(f'block size R       : {args.R}')
    print(f'start / goal       : {start_pixel} -> {goal_pixel}')
    print(f'coarse search      : success')
    print(f'path components    : {len(path)}')
    print(f'visited components : {visited}')
    print(f'grid path cells    : {len(pixel_path)}')
    print(f'start / goal reached: yes')
    print(f'gate used          : yes')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rgb = np.empty((*labels.shape, 3), dtype=float)
    rgb[passable] = (0.98, 0.98, 0.96)
    rgb[obstacles] = (0.42, 0.43, 0.43)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)
    axes[0].imshow(rgb, interpolation='nearest')
    axes[0].plot(pixel_path[:, 1], pixel_path[:, 0], color='#159447', linewidth=2)
    axes[0].scatter(start_pixel[1], start_pixel[0], marker='o', c='#2563eb',
                    s=42, edgecolors='white', zorder=5)
    axes[0].scatter(goal_pixel[1], goal_pixel[0], marker='*', c='white',
                    s=105, edgecolors='#159447', linewidths=1, zorder=5)
    axes[0].annotate('Start', (start_pixel[1], start_pixel[0]), xytext=(5, -13),
                     textcoords='offset points', color='#1d4ed8', fontsize=9)
    axes[0].annotate('Goal', (goal_pixel[1], goal_pixel[0]), xytext=(5, -13),
                     textcoords='offset points', color='#13783a', fontsize=9)
    axes[0].set_title('Coarse search path on the full map')
    axes[0].set_xticks(np.arange(-.5, width, args.R), minor=True)
    axes[0].set_yticks(np.arange(-.5, height, args.R), minor=True)
    axes[0].grid(which='minor', color='black', alpha=.2, linewidth=.4)
    axes[1].imshow(rgb, interpolation='nearest')
    axes[1].plot(pixel_path[:, 1], pixel_path[:, 0], color='#159447', linewidth=2.2)
    axes[1].scatter(start_pixel[1], start_pixel[0], marker='o', c='#2563eb',
                    s=48, edgecolors='white', zorder=5)
    axes[1].scatter(goal_pixel[1], goal_pixel[0], marker='*', c='white',
                    s=115, edgecolors='#159447', linewidths=1.2, zorder=5)
    half = 44
    rlo, rhi = max(0, height // 2 - half), min(height, height // 2 + half)
    clo, chi = max(0, width // 2 - half), min(width, width // 2 + half)
    axes[1].set_title('Passage and nearby obstacles')
    axes[1].set_xlim(clo - .5, chi - .5)
    axes[1].set_ylim(rhi - .5, rlo - .5)
    axes[1].set_xticks(np.arange(clo - .5, chi, args.R), minor=True)
    axes[1].set_yticks(np.arange(rlo - .5, rhi, args.R), minor=True)
    axes[1].grid(which='minor', color='black', alpha=.3, linewidth=.5)
    for ax in axes:
        ax.set(xlabel='Column', ylabel='Row', xlim=(-.5, width-.5),
               ylim=(height-.5, -.5))
        ax.set_aspect('equal')
    axes[1].set_xlim(clo - .5, chi - .5)
    axes[1].set_ylim(rhi - .5, rlo - .5)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)


if __name__ == '__main__':
    main()
