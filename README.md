# Can't Place

This reporitory contains a force-directed hard macro placer for the [Partcl/HRT Macro Placement Challenge](https://github.com/partcleda/macro-place-challenge-2026).

<p align="center">
  <img src="assets/ibm01_fd_steps.gif" width="48%">
  <img src="assets/ibm01_fd_final.gif" width="48%">
</p>

## Problem

Macro placement is a chip physical design problem. Large fixed-size blocks, called macros, must be placed inside a chip outline.

A valid placement requires all hard macros inside the chip boundary and do not overlap.
Following this a good placement insists strongly connected macros are placed close together while maintaining space for routing and soft logic.
A good placement is measured according to a low proxy cost, which considers the wirelength, density and congestion in a weighted sum.

Ultimately the problem appears simple until considering the large number of legal placements and wells of 'low' proxy cost.

## Method

While many methods were experimented with. Overall I chose to submit a clean and simple placer which uses a force-directed loop followed by legalizations.


### Initialization

Movable hard macros are sorted by area and placed on a deterministic coarse grid. This gives a stable starting point and avoids random initialization.

### Net attraction

For each net, connected pins are pulled toward the net centroid. This acts as a simple wirelength-reduction heuristic.

### Overlap repulsion

Overlapping hard macros repel each other along the axis of smaller overlap. Larger macros are assigned larger effective mass, so smaller macros move more during separation.

### Soft macro avoidance

Soft macros are not actively placed. Their benchmark positions are treated as occupied regions that repel hard macros, giving the placer a weak model of soft-area blockage.

### Boundary bias

Movable hard macros are weakly biased toward nearby chip edges and corners. This heuristic helps free central area for soft logic and routing.

### Legalization

A pairwise legalizer is applied periodically during optimization and again at the end. It separates overlapping macro pairs along the smaller overlap axis. If needed, a repair pass places macros one by one into nearby legal candidate positions.

## Algorithm

```text
Build benchmark problem
Initialize movable hard macros on a grid

For each force-directed step:
    compute net attraction
    compute hard macro overlap repulsion
    compute soft macro avoidance
    compute boundary/corner bias
    move movable hard macros
    clamp to chip boundary
    periodically legalize overlaps

Run final legalization
Repair if needed
Return legal hard macro positions
```

## Results

Evaluation on the IBM benchmark set:

| Metric                |               Value |
| --------------------- | ------------------: |
| Average proxy cost    |              1.8074 |
| Average SA proxy      |              2.1251 |
| Average RePlAce proxy |              1.4578 |
| Improvement vs SA     |              +15.0% |
| Gap vs RePlAce        |              -24.0% |
| Hard macro overlaps   |                   0 |
| Total runtime         |            737.12 s |
| Average runtime       | 43.36 s / benchmark |

The placer improves over the simulated annealing baseline on average and produces legal hard macro placements on all IBM benchmarks.

It remains behind RePlAce, which is expected. RePlAce uses a stronger analytical placement formulation with more advanced global wirelength and density optimization. FD placer uses local force heuristics and only indirect congestion handling.

The best relative results were on `ibm17`, `ibm18`, `ibm12`, and `ibm06`. The weakest results were on `ibm03`, `ibm04`, and `ibm08`, where the local force model was not enough to match stronger global optimization.

### Note
During development, I found that the initial positions given in the benchmarks files already contained a lot of information, in some cases, preserving much of the provided placement and mainly applying legalization produced results that were faster and stronger than the force-directed method, and sometimes even competitive with or better than the RePlAce reference.

For the submitted method, movable hard macros are reset to a deterministic grid initialization instead. This gives a cleaner test of the force-directed algorithm itself, rather than using on benchmark-provided starting coordinates.
