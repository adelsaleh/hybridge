# T600 stage: GPU element-local assembly and architecture autotuning

## Why this stage was added

The T600 experiments show that kernel fusion is not automatically faster.
For the 64x64, p=1 case, the two-kernel additive-Schwarz path reduced the
standalone application from about 0.103 ms to 0.071 ms and reduced the
ASM-polynomial solve from about 83.5 ms to 68.6 ms.  At larger block sizes,
especially p=4, the fused local kernel can instead be slower than the separate
restriction and dense-product path.  The matrix-vector operator has the same
pattern: `raw_fused` saves its gathered workspace but does not win for every
mesh/order pair.

The production code must therefore retain numerically equivalent alternatives
and select them with a short benchmark on each GPU architecture.

The earlier `validate_face_dense_gpu_fused_asm.py` also called
`solve_diffusion_face_dense_direct`.  That helper materializes a dense scalar
matrix and is deliberately intended only for small correctness cases.  For
64x64, p=4, the scalar matrix alone would require roughly 27.5 GiB in float64,
before factorization workspace.  The script now builds the face-dense system
without a direct dense solve.

## GPU element-local diffusion assembly

The new module is:

```text
hdgfem/backends/cupy_diffusion_assembly.py
```

For identity diffusion it moves the following numerical pipeline to the GPU:

```text
reference/geometry arrays
        |
        v
physical derivative matrices d0, d1
        |
        v
reaction + stabilization + normal boundary blocks
        |
        v
scalar condensed local matrix E^{-1}
        |
        v
full mixed local inverse [u, qx, qy]
        |
        v
trace lift and element-boundary coupling
        |
        v
oriented local Schur trace blocks
        |
        v
complete elemental face blocks
        |
        v
deterministic global face-dense assembly
```

The scalar local matrices are inverted by either:

```python
inverse_backend="cublas_inverse"
inverse_backend="gpu_inverse"
```

The first path uses the validated `getrfBatched/getriBatched` implementation.

### Host work that remains

Arbitrary Python coefficient functions are evaluated on the CPU.  This is an
interface limitation rather than a dense-algebra limitation: an arbitrary
Python callable cannot run directly inside a CUDA kernel.  The resulting
reaction mass matrices are transferred once.  Mesh topology and reference
basis tables are also setup metadata prepared on the host.

The expensive per-element dense algebra is device-resident.

## Ping-pong buffers

`CuPyDiffusionAssemblyWorkspace` owns two arrays:

```text
ping[NE, N, N]
pong[NE, N, N]
```

They alternate through chains such as:

```text
(m_n - d) @ M^{-1} -> ping
ping @ d           -> pong
```

and through the full mixed-inverse block formulas.  This avoids retaining a
separate temporary for every matrix product.  Output arrays that must survive
the stage—local inverse, trace lift, trace blocks, and element blocks—remain
separate by necessity.

Use `retain_intermediates=False` when only elemental face blocks are needed.
The returned object then releases references to the full local inverse, trace
lift, and boundary-coupling tensor.

## Example

```python
assembler = CuPyDiffusionLocalAssembler.from_space(
    reaction,
    stabilization,
    space,
    dtype=np.float64,
    inverse_backend="cublas_inverse",
)

local = assembler.assemble(retain_intermediates=True)

global_blocks = assembler.assemble_global_blocks(
    local,
    loc2glob_face=space.mesh.loc2glob_edge,
    topology=topology,
    active_row_faces=space.mesh.interior_face_mask,
)

interior_rhs = assembler.assemble_interior_rhs(
    source_rhs,
    local,
    topology=topology,
)
```

No elemental or global face block is copied back to the CPU in this path.

## Local-assembly validation

Run:

```bash
PYTHONPATH=. pytest -q \
    tests/test_cupy_diffusion_assembly.py
```

Then benchmark p=1 and p=4:

```bash
PYTHONPATH=. python scripts/validate_face_dense_gpu_local_assembly.py \
    --mesh 64 --order 1 --dtype float64 \
    --inverse-backend cublas_inverse --warmup 2 --repeats 5

PYTHONPATH=. python scripts/validate_face_dense_gpu_local_assembly.py \
    --mesh 64 --order 4 --dtype float64 \
    --inverse-backend cublas_inverse --warmup 2 --repeats 5
```

The script compares:

- full mixed local inverse;
- complete elemental blocks;
- global face-dense blocks;
- unpenalized global trace RHS;
- scalar local inverse residual;
- CPU and GPU setup times;
- ping-pong workspace size.

## Architecture-specific autotuning

The new autotuner validates and times:

```text
operator: raw, raw_fused
ASM:      raw, fused
```

Run it for each important order on the T600:

```bash
for P in 1 2 3 4; do
    PYTHONPATH=. python scripts/autotune_face_dense_gpu.py \
        --mesh 64 --order "$P" \
        --warmup 20 --repeats 100 \
        --output "results/t600_autotune_64_p${P}.json"
done
```

At the mesocentre, repeat the same short command before the large benchmark
campaign.  The V100 may choose different kernels from the T600 because launch
overhead, cache capacity, memory bandwidth, and occupancy differ.

The autotuner refuses to select a candidate whose output differs from the
reference candidate beyond the configured precision tolerance.

## Next development stages

1. Device-side direct Dirichlet elimination and penalty row replacement.
2. Persistent GPU maps for the global RHS, avoiding setup-map transfer on each
   call.
3. Chunked element batches for meshes whose complete local tensors do not fit
   in GPU memory.
4. Tensor-diffusion local matrices.
5. Source and coefficient evaluation from device-ready quadrature values.
6. End-to-end GPU reconstruction of the volume field and flux.
7. Nsight Compute tuning of the selected p-dependent kernels.
