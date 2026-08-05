# Advection-Reaction Experiments

This directory contains research harnesses that are not part of the supported
`hdgfem` API. Production runners live one directory above. Each experiment must
state its reference implementation and retain a focused regression test when it
introduces reusable numerical logic.

- `check_upwind_block_gs_on_the_fly.py` compares on-the-fly upwind block
  Gauss-Seidel implementations with the maintained CSR reference path.
