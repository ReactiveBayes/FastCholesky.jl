module FastCholesky

using LinearAlgebra
using PositiveFactorizations

import LinearAlgebra: BlasInt, BlasFloat

export fastcholesky, fastcholesky!, cholinv, cholsqrt, chollogdet, cholinv_logdet

"""
    fastcholesky(input)

Calculate the Cholesky factorization of the input matrix `input`. 
This function provides a more efficient implementation for certain input matrices compared to the standard `LinearAlgebra.cholesky` function. 
By default, it falls back to using `LinearAlgebra.cholesky(PositiveFactorizations.Positive, input)`, which means it does not require the input matrix to be perfectly symmetric.

!!! note 
    This function assumes that the input matrix is nearly positive definite, and it will attempt to make the smallest possible adjustments 
    to the matrix to ensure it becomes positive definite. Note that the magnitude of these adjustments may not necessarily be small, so it's important to use 
    this function only when you expect the input matrix to be nearly positive definite.
    
```jldoctest; setup = :(using FastCholesky, LinearAlgebra)
julia> C = fastcholesky([ 1.0 0.5; 0.5 1.0 ]);

julia> C.L * C.L' ≈ [ 1.0 0.5; 0.5 1.0 ]
true
```
"""
function fastcholesky(input::AbstractMatrix)
    # The `tol` argument here is tricky, it basically defines the threshold for a diagonal entry to be zero
    # The `PositiveFactorizations` replaces zero diagonal entries under the hood and simply uses `1` instead
    # The `PositiveFactorizations.default_δ` should small enough in majority of the cases
    return cholesky(PositiveFactorizations.Positive, Hermitian(input), tol = PositiveFactorizations.default_δ(input))
end

fastcholesky(input::Number) = cholesky(input)
fastcholesky(input::Diagonal) = cholesky(input)
fastcholesky(input::Hermitian) = cholesky(PositiveFactorizations.Positive, input)

function fastcholesky(x::UniformScaling)
    return error(
        "`fastcholesky` is not defined for `UniformScaling`. The shape is not determined."
    )
end
function fastcholesky!(x::UniformScaling)
    return error(
        "`fastcholesky!` is not defined for `UniformScaling`. The shape is not determined."
    )
end

function fastcholesky(input::Matrix{<:BlasFloat})
    C = fastcholesky!(copy(input))
    return isposdef(C) ? C : cholesky(PositiveFactorizations.Positive, Hermitian(input))
end

"""
    fastcholesky!(input)

Calculate the Cholesky factorization of the input matrix `input` in-place. This function is an in-place version of `fastcholesky`, 
and it does not check the positive-definiteness of the input matrix or throw errors. You can use `LinearAlgebra.issuccess` to check if the result is a proper Cholesky factorization.

!!! note
    This function does not verify the positive-definiteness of the input matrix and does not throw errors. Ensure that the input matrix is appropriate for Cholesky factorization before using this function.

```jldoctest; setup = :(using FastCholesky, LinearAlgebra)
julia> C = fastcholesky!([ 1.0 0.5; 0.5 1.0 ]);

julia> C.L * C.L' ≈ [ 1.0 0.5; 0.5 1.0 ]
true
```
"""
function fastcholesky! end

function fastcholesky!(A::Matrix{<:BlasFloat})
    n = LinearAlgebra.checksquare(A)
    # Heuristics, the built-in version is faster for large inputs
    # Perhaps due to the better cache usage, the Base version fallbacks to LAPACK
    return n < 20 ? _fastcholesky!(n, A) : cholesky!(A; check=false)
end

function _fastcholesky!(n, A::AbstractMatrix)
    @inbounds @fastmath for col in 1:n
        @simd for idx in 1:(col - 1)
            A[col, col] -= A[col, idx]^2
        end
        if A[col, col] <= 0
            return Cholesky(A, 'L', convert(BlasInt, -1))
        end
        A[col, col] = sqrt(A[col, col])
        invAcc = inv(A[col, col])
        for row in (col + 1):n
            @simd for idx in 1:(col - 1)
                A[row, col] -= A[row, idx] * A[col, idx]
            end
            A[row, col] *= invAcc
        end
    end
    return Cholesky(A, 'L', convert(BlasInt, 0))
end

"""
    cholinv(input)

Calculate the inverse of the input matrix `input` using Cholesky factorization. This function is an alias for `inv(fastcholesky(input))`.

```jldoctest; setup = :(using FastCholesky, LinearAlgebra)
julia> A = [4.0 2.0; 2.0 5.0];

julia> A_inv = cholinv(A);

julia> A_inv ≈ inv(A)
true
```
"""
cholinv(input::AbstractMatrix) = inv(fastcholesky(input))

cholinv(input::UniformScaling) = inv(input.λ) * I
cholinv(input::Diagonal) = inv(input)
cholinv(input::Number) = inv(input)

function cholinv(input::Matrix{<:BlasFloat})
    C = fastcholesky(input)
    LinearAlgebra.inv!(C)
    return C.factors
end

"""
    cholsqrt(input)

Calculate the square root of the input matrix `input` using Cholesky factorization. This function is an alias for `fastcholesky(input).L`.

```jldoctest 
julia> A = [4.0 2.0; 2.0 5.0];

julia> A_sqrt = cholsqrt(A);

julia> isapprox(A_sqrt * A_sqrt', A)
true
```
"""
cholsqrt(input) = fastcholesky(input).L

cholsqrt(input::UniformScaling) = sqrt(input.λ) * I
cholsqrt(input::Diagonal) = Diagonal(sqrt.(diag(input)))
cholsqrt(input::Number) = sqrt(input)

"""
    chollogdet(input)

Calculate the log-determinant of the input matrix `input` using Cholesky factorization. This function is an alias for `logdet(fastcholesky(input))`.

```jldoctest; setup = :(using FastCholesky, LinearAlgebra)
julia> A = [4.0 2.0; 2.0 5.0];

julia> logdet_A = chollogdet(A);

julia> isapprox(logdet_A, log(det(A)))
true
```
"""
chollogdet(input) = logdet(fastcholesky(input))

chollogdet(input::UniformScaling) = logdet(input)
chollogdet(input::Diagonal) = logdet(input)
chollogdet(input::Number) = logdet(input)

"""
    cholinv_logdet(input)

Calculate the inverse and the natural logarithm of the determinant of the input matrix `input` simultaneously using Cholesky factorization.

```jldoctest; setup = :(using FastCholesky, LinearAlgebra)
julia> A = [4.0 2.0; 2.0 5.0];

julia> A_inv, logdet_A = cholinv_logdet(A);

julia> isapprox(A_inv * A, I)
true

julia> isapprox(logdet_A, log(det(A)))
true
```
"""
function cholinv_logdet(input)
    C = fastcholesky(input)
    return inv(C), logdet(C)
end

function cholinv_logdet(input::Matrix{<:BlasFloat})
    C = fastcholesky(input)
    lC = logdet(C)
    LinearAlgebra.inv!(C)
    return C.factors, lC
end

cholinv_logdet(input::UniformScaling) = inv(input), logdet(input)
cholinv_logdet(input::Diagonal) = inv(input), logdet(input)
cholinv_logdet(input::Number) = inv(input), log(abs(input))

# Extensions 
@static if !isdefined(Base, :get_extension)
    include("../ext/StaticArraysCoreExt.jl")
end

end
