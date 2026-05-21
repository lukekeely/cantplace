from dataclasses import dataclass

import numpy as np
import torch
from macro_place.benchmark import Benchmark


@dataclass
class Settings:
    steps: int = 80
    attract: float = 0.10
    repel: float = 1.0
    edge: float = 1.0
    corner: float = 0.30
    gap: float = 1.0e-3
    max_disp_fraction: float = 1.0 / 8.0
    min_disp_fraction: float = 1.0 / 140.0
    legalize_passes: int = 80


@dataclass
class Problem:
    canvas: np.ndarray
    pos0: np.ndarray
    size: np.ndarray
    fixed: np.ndarray
    soft_pos: np.ndarray
    soft_size: np.ndarray
    nets: list

    def __post_init__(self):
        self.n = len(self.pos0)
        self.movable = ~self.fixed
        self.width = self.size[:, 0]
        self.height = self.size[:, 1]
        self.area = np.maximum(self.width * self.height, 1e-12)
        self.lo = 0.5 * self.size
        self.hi = self.canvas - self.lo

        if np.any(self.movable):
            mass_ref = max(float(np.mean(self.area[self.movable])), 1e-12)
        else:
            mass_ref = 1.0
        mass = np.clip(self.area / mass_ref, 0.25, 6.0)
        self.inv_mass = np.where(self.movable, 1.0 / mass, 0.0)

        # Reuse macro pairs across force and legalization passes.
        self.pair_i, self.pair_j = np.triu_indices(self.n, 1)
        live = self.movable[self.pair_i] | self.movable[self.pair_j]
        self.live_i = self.pair_i[live]
        self.live_j = self.pair_j[live]
        self.live_sep_x = 0.5 * (self.width[self.live_i] + self.width[self.live_j])
        self.live_sep_y = 0.5 * (self.height[self.live_i] + self.height[self.live_j])
        inv = self.inv_mass[self.live_i] + self.inv_mass[self.live_j]
        self.share_i = self.inv_mass[self.live_i] / np.maximum(inv, 1e-12)
        self.share_j = self.inv_mass[self.live_j] / np.maximum(inv, 1e-12)


def as_numpy(value, dtype=np.float64):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def tensor_items(value):
    return value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else list(value)


def build_problem(benchmark: Benchmark) -> Problem:
    full_pos = as_numpy(benchmark.macro_positions)
    full_size = as_numpy(benchmark.macro_sizes)
    fixed = as_numpy(benchmark.macro_fixed, bool)
    n_full = int(benchmark.num_macros)
    n_hard = int(benchmark.num_hard_macros)
    canvas = np.array([float(benchmark.canvas_width), float(benchmark.canvas_height)])

    ports = as_numpy(getattr(benchmark, "port_positions", np.zeros((0, 2))))
    offsets_by_macro = getattr(benchmark, "macro_pin_offsets", [])
    offsets_by_macro = [as_numpy(offsets) for offsets in offsets_by_macro]
    net_weights = as_numpy(getattr(benchmark, "net_weights", np.ones(0)))
    hard_start = grid_start(full_size[:n_hard, :2], fixed[:n_hard], canvas)

    nets = []

    def net_weight(net_id):
        if net_id < len(net_weights):
            weight = abs(float(net_weights[net_id]))
            if weight > 0.0:
                return weight
        return 1.0

    def pin_offset(owner, slot):
        if 0 <= owner < len(offsets_by_macro):
            offsets = offsets_by_macro[owner]
            if offsets.ndim == 1 and len(offsets) >= 2:
                return offsets[:2]
            if offsets.ndim >= 2 and 0 <= slot < offsets.shape[0]:
                return offsets[slot, :2]
        return np.zeros(2)

    def fixed_point(owner, offset):
        if 0 <= owner < n_full:
            return full_pos[owner, :2] + offset
        port_id = owner - n_full
        if 0 <= port_id < len(ports):
            return ports[port_id, :2] + offset
        return np.zeros(2)

    def add_net(net_id, pins):
        movable = []
        fixed_points = []
        for owner, offset in pins:
            if 0 <= owner < n_hard and not fixed[owner]:
                movable.append((owner, offset))
            else:
                fixed_points.append(fixed_point(owner, offset))

        if not movable or len(movable) + len(fixed_points) < 2:
            return

        nets.append(
            (
                net_weight(net_id),
                np.asarray([i for i, _ in movable], dtype=np.int32),
                np.vstack([offset for _, offset in movable]),
                np.vstack(fixed_points) if fixed_points else np.zeros((0, 2)),
            )
        )

    if len(getattr(benchmark, "net_pin_nodes", [])) > 0:
        for net_id, net in enumerate(benchmark.net_pin_nodes):
            add_net(
                net_id,
                [
                    (int(owner), pin_offset(int(owner), int(slot)))
                    for owner, slot in tensor_items(net)
                ],
            )
    else:
        for net_id, net in enumerate(benchmark.net_nodes):
            owners = as_numpy(net, np.int64).reshape(-1)
            add_net(net_id, [(int(owner), np.zeros(2)) for owner in owners])

    return Problem(
        canvas=canvas,
        pos0=hard_start,
        size=full_size[:n_hard, :2].copy(),
        fixed=fixed[:n_hard].copy(),
        soft_pos=full_pos[n_hard:n_full, :2].copy(),
        soft_size=full_size[n_hard:n_full, :2].copy(),
        nets=nets,
    )


def grid_start(size, fixed, canvas):
    pos = np.zeros((len(size), 2), dtype=np.float64)
    movable = np.flatnonzero(~fixed)
    if len(movable) == 0:
        return pos

    aspect = float(canvas[0]) / max(float(canvas[1]), 1e-12)
    cols = max(1, int(np.ceil(np.sqrt(len(movable) * aspect))))
    rows = max(1, int(np.ceil(len(movable) / cols)))
    cell = canvas / np.array([cols, rows], dtype=np.float64)
    order = sorted(
        movable,
        key=lambda idx: (-float(size[idx, 0] * size[idx, 1]), idx),
    )

    for rank, idx in enumerate(order):
        row = rank // cols
        col = rank % cols
        center = np.array([(col + 0.5) * cell[0], (row + 0.5) * cell[1]])
        pos[idx] = np.minimum(np.maximum(center, 0.5 * size[idx]), canvas - 0.5 * size[idx])
    return pos


class Placer:
    def __init__(self, settings=None):
        self.settings = settings or Settings()

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        problem = build_problem(benchmark)
        pos = problem.pos0.copy()
        fixed_hard = as_numpy(benchmark.macro_positions)[: problem.n, :2]
        pos[problem.fixed] = fixed_hard[problem.fixed]

        span = max(float(problem.canvas[0]), float(problem.canvas[1]))
        max_disp0 = self.settings.max_disp_fraction * span
        min_disp = self.settings.min_disp_fraction * span

        for step in range(max(0, int(self.settings.steps))):
            t = step / max(1, self.settings.steps - 1)
            max_disp = min_disp + (max_disp0 - min_disp) * (1.0 - t) ** 1.6

            # Pull by net connections, then resolve overlaps.
            force = self._attract(problem, pos)
            force += self._repel(problem, pos, max_disp)
            force += self._boundary_force(problem, pos, max_disp)
            force[~problem.movable] = 0.0

            for axis in (0, 1):
                live = force[problem.movable, axis]
                peak = float(np.max(np.abs(live))) if len(live) else 0.0
                if peak > 1e-12:
                    pos[problem.movable, axis] += live / peak * max_disp

            self._clamp(problem, pos)
            if step % 10 == 9:
                # Keep overlaps from building up too much.
                pos = self._legalize(problem, pos, 8)

        final = self._legalize(problem, pos, self.settings.legalize_passes)
        if not self._is_legal(problem, final):
            final = self._repair_legalize(problem, final)
        if not self._is_legal(problem, final):
            raise RuntimeError("failed to construct a legal hard-macro placement")

        out = benchmark.macro_positions.clone()
        out[: problem.n] = torch.as_tensor(final, dtype=out.dtype, device=out.device)
        out[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]
        return out

    def _attract(self, problem, pos):
        force = np.zeros_like(pos)
        for weight, ids, offsets, fixed in problem.nets:
            points = [pos[ids] + offsets]
            if len(fixed):
                points.append(fixed)
            center = np.mean(np.vstack(points), axis=0)
            np.add.at(force, ids, self.settings.attract * weight * (center - (pos[ids] + offsets)))
        return force

    def _repel(self, problem, pos, max_disp):
        force = np.zeros_like(pos)
        if len(problem.live_i):
            i, j = problem.live_i, problem.live_j
            delta = pos[j] - pos[i]
            ox = problem.live_sep_x + self.settings.gap - np.abs(delta[:, 0])
            oy = problem.live_sep_y + self.settings.gap - np.abs(delta[:, 1])
            hit = (ox > 0.0) & (oy > 0.0)
            move_x = hit & (ox <= oy)
            if np.any(move_x):
                sign = np.where(delta[move_x, 0] >= 0.0, 1.0, -1.0)
                push = self.settings.repel * max_disp * sign
                np.add.at(force[:, 0], i[move_x], -push * problem.share_i[move_x])
                np.add.at(force[:, 0], j[move_x], push * problem.share_j[move_x])
            move_y = hit & ~move_x
            if np.any(move_y):
                sign = np.where(delta[move_y, 1] >= 0.0, 1.0, -1.0)
                push = self.settings.repel * max_disp * sign
                np.add.at(force[:, 1], i[move_y], -push * problem.share_i[move_y])
                np.add.at(force[:, 1], j[move_y], push * problem.share_j[move_y])

        for i in np.flatnonzero(problem.movable):
            delta = problem.soft_pos - pos[i]
            ox = (
                0.5 * (problem.width[i] + problem.soft_size[:, 0])
                + self.settings.gap
                - np.abs(delta[:, 0])
            )
            oy = (
                0.5 * (problem.height[i] + problem.soft_size[:, 1])
                + self.settings.gap
                - np.abs(delta[:, 1])
            )
            hit = (ox > 0.0) & (oy > 0.0)
            if not np.any(hit):
                continue
            move_x = hit & (ox <= oy)
            if np.any(move_x):
                force[i, 0] -= (
                    self.settings.repel * max_disp * float(np.sum(np.sign(delta[move_x, 0])))
                )
            move_y = hit & ~move_x
            if np.any(move_y):
                force[i, 1] -= (
                    self.settings.repel * max_disp * float(np.sum(np.sign(delta[move_y, 1])))
                )
        return force

    def _boundary_force(self, problem, pos, max_disp):
        force = np.zeros_like(pos)
        ids = np.flatnonzero(problem.movable)
        if len(ids) == 0:
            return force

        if self.settings.edge <= 0.0 and self.settings.corner <= 0.0:
            return force

        # Add a small bias toward the boundary.
        dist = np.column_stack(
            (
                pos[ids, 0] - problem.lo[ids, 0],
                problem.hi[ids, 0] - pos[ids, 0],
                pos[ids, 1] - problem.lo[ids, 1],
                problem.hi[ids, 1] - pos[ids, 1],
            )
        )

        if self.settings.edge > 0.0:
            nearest_edge = np.argmin(dist, axis=1)
            strength = self.settings.edge * max_disp
            force[ids[nearest_edge == 0], 0] -= strength
            force[ids[nearest_edge == 1], 0] += strength
            force[ids[nearest_edge == 2], 1] -= strength
            force[ids[nearest_edge == 3], 1] += strength

        if self.settings.corner <= 0.0:
            return force

        targets = np.stack(
            (
                np.column_stack((problem.lo[ids, 0], problem.lo[ids, 1])),
                np.column_stack((problem.lo[ids, 0], problem.hi[ids, 1])),
                np.column_stack((problem.hi[ids, 0], problem.lo[ids, 1])),
                np.column_stack((problem.hi[ids, 0], problem.hi[ids, 1])),
            ),
            axis=1,
        )
        delta = targets - pos[ids, None, :]
        nearest_corner = np.argmin(np.sum(delta * delta, axis=2), axis=1)
        direction = delta[np.arange(len(ids)), nearest_corner]
        norm = np.linalg.norm(direction, axis=1)
        live = norm > 1e-12
        force[ids[live]] += self.settings.corner * max_disp * direction[live] / norm[live, None]
        return force

    def _clamp(self, problem, pos):
        pos[problem.movable] = np.minimum(
            np.maximum(pos[problem.movable], problem.lo[problem.movable]),
            problem.hi[problem.movable],
        )

    def _legalize(self, problem, pos, passes):
        pos = pos.copy()
        i, j = problem.live_i, problem.live_j
        for _ in range(passes):
            self._clamp(problem, pos)
            delta = pos[j] - pos[i]
            ox = problem.live_sep_x + self.settings.gap - np.abs(delta[:, 0])
            oy = problem.live_sep_y + self.settings.gap - np.abs(delta[:, 1])
            hit = (ox > 0.0) & (oy > 0.0)
            if not np.any(hit):
                break

            # Separate each overlap on the smaller axis.
            move_x = hit & (ox <= oy)
            if np.any(move_x):
                sign = np.where(delta[move_x, 0] >= 0.0, 1.0, -1.0)
                step = 0.88 * ox[move_x]
                np.add.at(pos[:, 0], i[move_x], -sign * step * problem.share_i[move_x])
                np.add.at(pos[:, 0], j[move_x], sign * step * problem.share_j[move_x])
            move_y = hit & ~move_x
            if np.any(move_y):
                sign = np.where(delta[move_y, 1] >= 0.0, 1.0, -1.0)
                step = 0.88 * oy[move_y]
                np.add.at(pos[:, 1], i[move_y], -sign * step * problem.share_i[move_y])
                np.add.at(pos[:, 1], j[move_y], sign * step * problem.share_j[move_y])
        self._clamp(problem, pos)
        return pos

    def _repair_legalize(self, problem, target):
        pos = target.copy()
        self._clamp(problem, pos)
        placed_idx = np.flatnonzero(problem.fixed).astype(np.int32)
        placed_pos = pos[placed_idx].copy() if len(placed_idx) else np.zeros((0, 2))
        order = sorted(
            np.flatnonzero(problem.movable),
            key=lambda idx: (-problem.area[idx], idx),
        )

        for i in order:
            wanted = pos[i].copy()
            if len(placed_idx):
                x_sep = 0.5 * (problem.width[i] + problem.width[placed_idx]) + self.settings.gap
                y_sep = 0.5 * (problem.height[i] + problem.height[placed_idx]) + self.settings.gap
                xs = np.concatenate(
                    (
                        [wanted[0], problem.lo[i, 0], problem.hi[i, 0]],
                        placed_pos[:, 0] - x_sep,
                        placed_pos[:, 0] + x_sep,
                    )
                )
                ys = np.concatenate(
                    (
                        [wanted[1], problem.lo[i, 1], problem.hi[i, 1]],
                        placed_pos[:, 1] - y_sep,
                        placed_pos[:, 1] + y_sep,
                    )
                )
            else:
                xs = np.array([wanted[0], problem.lo[i, 0], problem.hi[i, 0]])
                ys = np.array([wanted[1], problem.lo[i, 1], problem.hi[i, 1]])

            # Try candidate centers close to the target first.
            x_candidates = np.unique(np.round(xs, 9))
            y_candidates = np.unique(np.round(ys, 9))
            grid = np.array(np.meshgrid(x_candidates, y_candidates, indexing="ij")).reshape(2, -1).T
            for center in grid[np.argsort(np.sum((grid - wanted) ** 2, axis=1))]:
                if self._fits(problem, i, center, placed_idx, placed_pos):
                    pos[i] = center
                    break
            placed_idx = np.append(placed_idx, i)
            placed_pos = np.vstack((placed_pos, pos[i]))
        return self._legalize(problem, pos, self.settings.legalize_passes)

    def _fits(self, problem, i, center, placed_idx, placed_pos):
        if np.any(center < problem.lo[i] - 1e-9) or np.any(center > problem.hi[i] + 1e-9):
            return False
        if len(placed_idx) == 0:
            return True
        delta = np.abs(placed_pos - center)
        min_x = 0.5 * (problem.width[i] + problem.width[placed_idx]) + self.settings.gap
        min_y = 0.5 * (problem.height[i] + problem.height[placed_idx]) + self.settings.gap
        return not np.any((delta[:, 0] < min_x - 1e-9) & (delta[:, 1] < min_y - 1e-9))

    def _is_legal(self, problem, pos):
        if np.any(pos < problem.lo - 1e-6) or np.any(pos > problem.hi + 1e-6):
            return False
        i, j = problem.pair_i, problem.pair_j
        delta = np.abs(pos[j] - pos[i])
        ox = 0.5 * (problem.width[i] + problem.width[j]) - delta[:, 0]
        oy = 0.5 * (problem.height[i] + problem.height[j]) - delta[:, 1]
        return not np.any((ox > 1e-8) & (oy > 1e-8))
