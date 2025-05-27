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

# Environment Variables

The behavior of this function regarding non-symmetric matrices can be controlled through environment variables:

- `JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC=1`: Set this to suppress warnings about non-symmetric input matrices
- `JULIA_FASTCHOLESKY_THROW_ERROR_NON_SYMMETRIC=1`: Set this to make the function error instead of warn when encountering non-symmetric matrices
    
```jldoctest; setup = :(using FastCholesky, LinearAlgebra)
julia> C = fastcholesky([ 1.0 0.5; 0.5 1.0 ]);

julia> C.L * C.L' ≈ [ 1.0 0.5; 0.5 1.0 ]
true
```
"""
function fastcholesky(input::AbstractMatrix)
    A = zeros(eltype(input), size(input))
    copyto!(A, input)
    return fastcholesky!(A)
end

fastcholesky(input::Number) = cholesky(input)
fastcholesky(input::Diagonal) = cholesky(input)
fastcholesky(input::Hermitian) = cholesky(PositiveFactorizations.Positive, input)

function fastcholesky(x::UniformScaling)
    return error("`fastcholesky` is not defined for `UniformScaling`. The shape is not determined.")
end
function fastcholesky!(x::UniformScaling)
    return error("`fastcholesky!` is not defined for `UniformScaling`. The shape is not determined.")
end

"""
    fastcholesky!(input; fallback_gmw81=true, symmetrize_input=true, gmw81_tol=PositiveFactorizations.default_δ(input), symmetric_tol=1e-8)

Calculate the Cholesky factorization of the input matrix `input` in-place. This function is an in-place version of `fastcholesky`.
It first checks if the input matrix is symmetric, and if it is, it will use the built-in Cholesky factorization by wrapping the input in a `Hermitian` matrix.
If the input matrix is not symmetric and `symmetrize_input=true`, it will symmetrize the input matrix and try again recursively.
If the input matrix is not symmetrics, not positive definite, and `fallback_gmw81=true`, it will use the `GMW81` algorithm as a fallback.
In other cases, it will throw an error.

# Keyword arguments
- `fallback_gmw81::Bool=true`: If true, the function will use the `GMW81` algorithm as a fallback if the input matrix is not positive definite.
- `symmetrize_input::Bool=true`: If true, the function will symmetrize the input matrix before retrying the Cholesky factorization in case if the first attempt failed.
- `gmw81_tol::Real=PositiveFactorizations.default_δ(input)`: The tolerance for the positive-definiteness of the input matrix for the `GMW81` algorithm.
- `symmetric_tol::Real=1e-8`: The tolerance for the symmetry of the input matrix.

# Environment Variables

The behavior of this function regarding non-symmetric matrices can be controlled through environment variables:

- `JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC=1`: Set this to suppress warnings about non-symmetric input matrices
- `JULIA_FASTCHOLESKY_THROW_ERROR_NON_SYMMETRIC=1`: Set this to make the function error instead of warn when encountering non-symmetric matrices

!!! note
    Use this function only when you expect the input matrix to be nearly positive definite.

```jldoctest; setup = :(using FastCholesky, LinearAlgebra)
julia> C = fastcholesky!([ 1.0 0.5; 0.5 1.0 ]);

julia> C.L * C.L' ≈ [ 1.0 0.5; 0.5 1.0 ]
true
```
"""
function fastcholesky! end

const NO_WARN_NON_SYMMETRIC_ENV = "JULIA_FASTCHOLESKY_NO_WARN_NON_SYMMETRIC"
const THROW_ERROR_NON_SYMMETRIC_ENV = "JULIA_FASTCHOLESKY_THROW_ERROR_NON_SYMMETRIC"

function fastcholesky!(
    A::AbstractMatrix;
    fallback_gmw81::Bool=true,
    symmetrize_input::Bool=true,
    gmw81_tol=PositiveFactorizations.default_δ(A),
    symmetric_tol=1e-8,
)
    n = LinearAlgebra.checksquare(A)

    is_almost_symmetric = _issymmetric(A; tol=symmetric_tol)

    if is_almost_symmetric
        C = n < 20 ? _fastcholesky!(n, A) : cholesky!(Hermitian(A); check=false)
        if issuccess(C)
            return C
        else
            if fallback_gmw81
                RetryC = cholesky(PositiveFactorizations.Positive, Hermitian(A); tol=gmw81_tol)
                if issuccess(RetryC)
                    return RetryC
                else
                    throw(ArgumentError("Cholesky factorization failed, the input matrix is not positive definite"))
                end
            else
                throw(ArgumentError("Cholesky factorization failed, the input matrix is not positive definite"))
            end
        end
    else
        if haskey(ENV, THROW_ERROR_NON_SYMMETRIC_ENV)
            error(
                lazy"The input matrix to `FastCholesky` was not symmetric and `$(THROW_ERROR_NON_SYMMETRIC_ENV)` environment variable was set. The tolerance threshold was `$symmetric_tol`. Unset the environment variable to suppress this error and turn it to a warning.",
            )
        end
        if !haskey(ENV, NO_WARN_NON_SYMMETRIC_ENV)
            @warn lazy"The input matrix to `FastCholesky` is not symmetric. The tolerance threshold is `$symmetric_tol`. Set `$(NO_WARN_NON_SYMMETRIC_ENV)=1` environment variable to suppress this warning. Set `$(THROW_ERROR_NON_SYMMETRIC_ENV)=1` to throw an error instead of a warning."
        end
        if symmetrize_input
            A = (A + A') / 2
            return fastcholesky!(A; fallback_gmw81=fallback_gmw81, symmetrize_input=false, gmw81_tol=gmw81_tol, symmetric_tol=symmetric_tol)
        else
            throw(
                ArgumentError(
                    "The input matrix is not symmetric, and `symmetrize_input` was set to `false`. Set `symmetrize_input=true` to symmetrize the input matrix even if it is not symmetric.",
                ),
            )
        end
    end

    # this is actually unreachable
    # the first `is_almost_symmetric` branch either returns or errors
    # the second `!is_almost_symmetric` branch either returns or errors
    # this statement to make the compiler happy and infer that the function is returning a `Cholesky` object
    return cholesky!(Hermitian(A); check=false)
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

function _issymmetric(A::AbstractMatrix; tol=PositiveFactorizations.default_δ(A))
    for i in axes(A, 1)
        for j in 1:i
            if abs(A[i, j] - A[j, i]) > tol
                return false
            end
        end
    end
    return true
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

Calculate the Cholesky square root of the input matrix `input`. This function is an alias for `fastcholesky(input).L`.
NOTE: This is not equal to the standard matrix square root used in literature, which requires the result to be symmetric.

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
