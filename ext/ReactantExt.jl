module ReactantExt

using FastCholesky, LinearAlgebra, Reactant

const TracedMatrix = Reactant.AnyTracedRMatrix

"""
    TracedCholesky(matrix, factor)

The Cholesky factorisation of a symmetric positive definite matrix traced by Reactant.

Solves, inverses and determinants go through the matrix's LU factorisation rather than its Cholesky
factor: on CPU, Reactant lowers `cholesky` to a loop that is 50–170× slower than LAPACK, while `lu`
calls LAPACK (https://github.com/EnzymeAD/Reactant.jl/issues/3411). Reactant's own factorisation
objects support only `\\` with a vector, and the generic `Cholesky` methods index scalars and branch
on values, which a trace cannot. The upper factor itself, for `.U`, `.L` and `cholsqrt`, is computed
with `cholesky`; the compiled program keeps that computation only where the factor is used.
"""
struct TracedCholesky{T, M <: AbstractMatrix{T}, F <: AbstractMatrix{T}} <: Factorization{T}
    matrix::M
    factor::F
end

# Neither the dense path's symmetry check nor its fallback can branch on a traced value: the input
# is taken as symmetric and positive definite. The factor is computed here, not in `getproperty`,
# which Reactant does not trace into.
function FastCholesky.fastcholesky(input::TracedMatrix)
    return TracedCholesky(input, triu(cholesky(Hermitian(input)).factors))
end

function Base.getproperty(C::TracedCholesky, name::Symbol)
    name === :U && return UpperTriangular(getfield(C, :factor))
    name === :L && return LowerTriangular(copy(adjoint(getfield(C, :factor))))
    return getfield(C, name)
end

Base.size(C::TracedCholesky, dims...) = size(getfield(C, :matrix), dims...)
LinearAlgebra.issuccess(::TracedCholesky) = true

# The matrix is symmetric, so B / A = (A \ Bᵀ)ᵀ
Base.:\(C::TracedCholesky, B::AbstractVecOrMat) = getfield(C, :matrix) \ B
Base.:/(B::AbstractMatrix, C::TracedCholesky) = permutedims(getfield(C, :matrix) \ permutedims(B))
Base.:/(B::Adjoint{<:Any, <:AbstractMatrix}, C::TracedCholesky) = permutedims(getfield(C, :matrix) \ permutedims(B))

Base.inv(C::TracedCholesky) = inv(getfield(C, :matrix))

# Positive definite, so the determinant is positive
LinearAlgebra.logdet(C::TracedCholesky) = first(logabsdet(getfield(C, :matrix)))
LinearAlgebra.det(C::TracedCholesky) = det(getfield(C, :matrix))

FastCholesky._cholsqrt_lower(C::TracedCholesky) = C.L

end # module
