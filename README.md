# SAC-Planner

Scale-adaptive hierarchical path planning with corridor-based local refinement
for off-road navigation on large terrain cost maps.

SAC-Planner combines hierarchical guidance generation with complete-route
corridor refinement.

## How it works

**SA-Planner.** For each planning region, SA-Planner selects a resolution factor
using its spatial scale and an empirical runtime model, subject to an aggregation
limit. It repeatedly extracts local submaps along the current path and refines
individual segments until the stopping criteria are met, producing a complete
start-to-goal guidance path.

**CGA\*.** CGA\* performs coarse search within SA-Planner. It represents
block-local eight-connected traversable components as separate graph vertices and
connects them through valid original-grid adjacencies. This preserves
original-grid reachability, including narrow-passage connectivity, in the coarse
graph.

**CB-Refiner.** Using the guidance path produced by SA-Planner, CB-Refiner
constructs a two-dimensional corridor and replans the entire route at the original
resolution within it. Intermediate guidance waypoints are not fixed constraints,
allowing terrain costs and connections across segment boundaries to be jointly
re-evaluated. The corridor is progressively widened if the search fails.

---

## Narrow-passage connectivity

![Coarse-graph search success on synthetic narrow-passage maps](figures/fig14_narrow_passage.png)

**Fig. 14.** Synthetic narrow-passage search outcomes. (a) CGA\*; (b) CGA\* non-split; (c) first observed failure factor versus passage width. Check marks and crosses indicate successful and failed coarse searches, respectively. The dashed red indicator denotes no observed failure for CGA\* among the tested factors and does not represent a measured critical value.

*Success refers to the coarse-graph search only; it is not the success rate of final route planning.*

## Scale adaptability

![Planning time, path cost, visited nodes and peak memory across map scales](figures/fig16_scale_adaptability.png)

**Fig. 16.** Scale adaptability. (a) Median paired planning-time changes relative to FA\*. (b) Median paired composite-cost changes relative to FA\*. (c) Median visited-vertex counts. (d) Median peak memory. Error bars in (a) and (b) indicate bootstrap 95% confidence intervals; open markers denote the single matched pair at scale 10000. Panel (c) and (d) summarize successful runs.

---

## Demonstration

`demo/demo_component_graph.py` illustrates the coarse representation used by
CGA\* on a synthetic 192 x 192 map that contains a one-cell-wide passage. It
splits each `R x R` block into its eight-connected passable components, connects
components of neighbouring blocks through original-grid adjacency, and compares
the outcome with a whole-block rule that keeps a block only if it is fully
passable.

```bash
pip install -r requirements.txt
cd demo
python demo_component_graph.py --R 8
```

The script prints the number of vertices and edges, the number of connected
groups obtained with the two representations, and writes
`demo_component_graph.png`.

With the default settings, the component graph keeps the map connected, whereas
the whole-block rule splits it into two groups because the one-cell passage lies
inside a mixed block. The demonstration does not run the planner; it only shows
how connectivity is preserved during aggregation.

The bundled sample is synthetic demonstration data, not an experimental dataset. The whole-block baseline below is a conservative illustration, not the manuscript's CGA* non-split implementation.

## Status

Currently available: a project overview, selected experimental result
visualizations (Figs. 14 and 16 of the manuscript), and the component-graph
demonstration.

Not yet released: the full SAC-Planner implementation and the scripts needed to
reproduce the experiments.

## Contact

Yiguang Gong — yiguang-gong@nuist.edu.cn
Nanjing University of Information Science and Technology
