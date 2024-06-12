module StaticArraysCoreExt # Should be same name as the file (just like a normal package)

using FastCholesky, PositiveFactorizations, StaticArraysCore, LinearAlgebra

function FastCholesky.fastcholesky(input::StaticArraysCore.StaticArray)
    C = cholesky(input, check = false)
    f = C.factors
    u = C.uplo
    c = C.info
    if !LinearAlgebra.issuccess(C)
        C_ = cholesky(Positive, Matrix(C), tol = PositiveFactorizations.default_δ(C))
        f = typeof(C.factors)(C_.factors)
        u = typeof(C.uplo)(C_.uplo)
        c = typeof(C.info)(C_.info)
    end
    return Cholesky(f, u, c)
end

end # module