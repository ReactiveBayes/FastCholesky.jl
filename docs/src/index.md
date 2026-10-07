```@meta
CurrentModule = FastCholesky
```

# FastCholesky

This package exports `fastcholesky` function, which works exactly like the `cholesky` from the Julia, but faster!

## Static arrays

With StaticArrays loaded, a static matrix gives a static factorisation, and `cholinv` of it a static matrix.
As for a dense matrix, an input symmetric within `1e-8` is factorised from one triangle, and one that is not is reported and symmetrised.
StaticArrays' own `cholesky` accepts only exactly symmetric input, which a product such as `A * Σ * A'` rarely is.

## Reactant

With [Reactant](https://github.com/EnzymeAD/Reactant.jl) loaded, `fastcholesky`, `cholinv`, `chollogdet`, `cholinv_logdet` and `cholsqrt` accept a matrix traced by `@compile` or `@jit`, so code calling them compiles.
The factorisation of a traced matrix supports `\`, `/`, `inv`, `logdet`, `det`, `.U` and `.L`.
Solves, inverses and determinants go through the LU factorisation of the matrix: on CPU, Reactant's `cholesky` is currently 50–170× slower than LAPACK, while its `lu` calls LAPACK ([Reactant.jl#3411](https://github.com/EnzymeAD/Reactant.jl/issues/3411)).
The Cholesky factor itself is computed only where `.U`, `.L` or `cholsqrt` use it.
A traced matrix is assumed symmetric and positive definite, with neither the symmetry check nor the fallback of a dense matrix, since neither can branch on a traced value, and `issuccess` reports `true`.

```@index
```

```@autodocs
Modules = [FastCholesky]
```
