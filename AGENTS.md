# Repository ownership and run policy

`hdgfem/` is the general DG/HDG library. The diocotron application lives in
`projects/diocotron/`, with separate DOLFINx, HDG, and FreeFEM implementations.
The HDG application may import HDGFEM; neither the library nor the DOLFINx
implementation imports the other application's solver code.

For diocotron execution, follow
[projects/diocotron/AGENTS.md](projects/diocotron/AGENTS.md), including the
measured MUMPS rank map and one numerical-library thread per MPI rank.
Application reports, configurations, and tests belong to that project.
Generated run results belong in its ignored `runs/` directory and LaTeX build
products in its ignored `build/` directory.
