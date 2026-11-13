# damn

Distributed Adaptive Mesh Numerics: the data infrastructure under [plamn](https://github.com/PhilipDeegan/plamn), in the role SAMRAI plays for PHARE.

damn moves data around: index space (boxes, points), n-dimensional storage, tiling, execution on CPU/GPU, and input handling, with the patch hierarchy and decomposition to come. It does not know what the data is; users decide its shape.

```sh
mkn clean build test -p test_damn_core -dKO -t 8
```
