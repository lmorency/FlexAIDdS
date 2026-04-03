# ML Rescoring Bridge

FlexAIDΔS provides featurization tools to bridge physics-based docking with machine learning rescoring.

## Architecture

1. **Voronoi Graph Extraction**: Parse contact type-pair adjacency matrices from results
2. **Shannon Profile Extraction**: Extract entropy trajectories from GA logs
3. **Feature Building**: Combine graphs + entropy + thermodynamic properties
4. **ML Rescoring**: Apply user-provided models (scikit-learn, PyTorch)

## Usage

```python
from flexaidds import VoronoiGraphExtractor, ShannonProfileExtractor, FeatureBuilder

graph_ext = VoronoiGraphExtractor(n_types=40)
graph = graph_ext.extract("result.pdb")

shannon_ext = ShannonProfileExtractor()
profile = shannon_ext.extract("docking.log")

builder = FeatureBuilder(n_types=40, max_generations=500)
features = builder.build(graph, profile)
```
