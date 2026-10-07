module StaticArraysCoreExt # Should be same name as the file (just like a normal package)

using FastCholesky, PositiveFactorizations, StaticArraysCore, LinearAlgebra

# As the dense `fastcholesky!` with its defaults: an input symmetric within `1e-8` is factorised
# from one triangle, one that is not is reported and symmetrised. StaticArrays' own `cholesky` of a
# plain static matrix rejects anything not exactly symmetric, even with `check = false`, so a
# product such as `A * Σ * A'` would fail on rounding alone.
function FastCholesky.fastcholesky(input::StaticArraysCore.StaticArray)
    symmetric_tol = 1e-8
    A = input
    if !FastCholesky._issymmetric(A; tol=symmetric_tol)
        FastCholesky._report_non_symmetric(symmetric_tol)
        A = (A + A') / 2
    end
    return static_cholesky(A)
end

# StaticArrays factorises a `Hermitian` static matrix from its parent's upper triangle whatever its
# `uplo`, so the full matrix is materialised first.
function FastCholesky.fastcholesky(input::Hermitian{<:Real,<:StaticArraysCore.StaticMatrix})
    return static_cholesky(typeof(parent(input))(input))
end

function static_cholesky(A::StaticArraysCore.StaticMatrix)
    C = cholesky(Hermitian(A); check=false)
    LinearAlgebra.issuccess(C) && return C
    C_ = cholesky(Positive, Hermitian(Matrix(A)); tol=PositiveFactorizations.default_δ(A))
    return Cholesky(typeof(C.factors)(C_.factors), typeof(C.uplo)(C_.uplo), typeof(C.info)(C_.info))
end

end # module
