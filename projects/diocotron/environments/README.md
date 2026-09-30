# Numerical environments

`dolfinx.yml` is the existing tested FEniCSx environment specification,
relocated from the equiband examples. It applies to the DOLFINx implementation
and its numerical/MPI tests. Gmsh generation remains isolated in subprocesses
where the existing workflows require it.

The HDG implementation uses the library environment described in
[the installation guide](../../../docs/getting_started/installation.md).
FreeFEM programs use the separately installed FreeFEM executable and plugins.
Only optional field-transfer verification needs both Python solver stacks.
