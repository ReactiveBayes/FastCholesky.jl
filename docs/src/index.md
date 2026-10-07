```@meta
CurrentModule = FastCholesky
```

# FastCholesky

This package exports `fastcholesky` function, which works exactly like the `cholesky` from the Julia, but faster!

## Static arrays

With StaticArrays loaded, a static matrix gives a static factorisation, and `cholinv` of it a static matrix.
As for a dense matrix, an input symmetric within `1e-8` is factorised from one triangle, and one that is not is reported and symmetrised.
StaticArrays' own `cholesky` accepts only exactly symmetric input, which a product such as `A * Σ * A'` rarely is.

```@index
```

```@autodocs
Modules = [FastCholesky]
```
