module ReactantExt

using FastCholesky, LinearAlgebra, Reactant

const TracedMatrix = Reactant.AnyTracedRMatrix

"""
    TracedCholesky(factor)

The Cholesky factorisation `A = U'U` of a matrix traced by Reactant, holding the upper factor `U`.
Reactant's own `cholesky` of a traced matrix supports only `\\` with a vector, and the generic
`Cholesky` methods index scalars and branch on values, which a trace cannot; every operation here
is a triangular solve on `U`.
"""
struct TracedCholesky{T, M <: AbstractMatrix{T}} <: Factorization{T}
    factor::M
end

# Neither the dense path's symmetry check nor its fallback can branch on a traced value: the input
# is factorised from its upper triangle, as a symmetric matrix, and is assumed positive definite.
function FastCholesky.fastcholesky(input::TracedMatrix)
    F = cholesky(Hermitian(input))
    return TracedCholesky(triu(F.factors))
end

upper(C::TracedCholesky) = UpperTriangular(getfield(C, :factor))

function Base.getproperty(C::TracedCholesky, name::Symbol)
    name === :U && return upper(C)
    name === :L && return LowerTriangular(copy(adjoint(getfield(C, :factor))))
    return getfield(C, name)
end

Base.size(C::TracedCholesky, dims...) = size(getfield(C, :factor), dims...)
LinearAlgebra.issuccess(::TracedCholesky) = true

# A \ B = U⁻¹ U⁻ᵀ B and B / A = B U⁻¹ U⁻ᵀ
Base.:\(C::TracedCholesky, B::AbstractVecOrMat) = upper(C) \ (upper(C)' \ B)
Base.:/(B::AbstractMatrix, C::TracedCholesky) = (B / upper(C)) / upper(C)'
Base.:/(B::Adjoint{<:Any, <:AbstractMatrix}, C::TracedCholesky) = (B / upper(C)) / upper(C)'

function Base.inv(C::TracedCholesky)
    Ui = inv(upper(C))
    return Ui * Ui'
end

LinearAlgebra.logdet(C::TracedCholesky) = 2 * sum(log, diag(getfield(C, :factor)))
LinearAlgebra.det(C::TracedCholesky) = prod(diag(getfield(C, :factor)))^2

FastCholesky._cholsqrt_lower(C::TracedCholesky) = C.L

end # module
