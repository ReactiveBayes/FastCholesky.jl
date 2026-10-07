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
The factorisation of a traced matrix supports `\`, `/`, `inv`, `logdet`, `det`, `.U` and `.L`, each through triangular solves.
A traced matrix is assumed symmetric and positive definite: its upper triangle is factorised, with neither the symmetry check nor the fallback of a dense matrix, since neither can branch on a traced value, and `issuccess` reports `true`.

```@index
```

```@autodocs
Modules = [FastCholesky]
```
