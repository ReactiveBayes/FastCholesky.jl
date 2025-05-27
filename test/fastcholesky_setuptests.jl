using Test, FastCholesky, LinearAlgebra, StaticArrays, StaticArraysCore, JET, BlockArrays

const SupportedTypes = (Float32, Float64, BigFloat)

function make_rand_diagonal(::Type{T}, size::Number) where {T}
    return Diagonal(10rand(T, size))
end

function make_rand_posdef(::Type{T}, size::Number) where {T}
    return collect(make_rand_hermitian(T, size))
end

function make_rand_hermitian(::Type{T}, size::Number) where {T}
    return make_rand_hermitian(T, make_rand_lowertriangular(T, size))
end

function make_rand_hermitian(::Type{T}, L::LowerTriangular) where {T}
    return Hermitian(convert(AbstractMatrix{T}, L * L' + (size(L, 1) + 1) * I))
end

function make_rand_lowertriangular(::Type{T}, size) where {T}
    return LowerTriangular(10rand(T, size, size))
end

function make_rand_block_matrix(::Type{T}, size) where {T}
    return BlockedArray(make_rand_posdef(T, 2size), [size, size], [size, size])
end

function make_rand_inputs(::Type{T}, size) where {T}
    inputs = [
        make_rand_diagonal(T, size),
        make_rand_posdef(T, size),
        make_rand_hermitian(T, size),
        make_rand_block_matrix(T, size),
        view(make_rand_hermitian(T, size), 1:size, 1:size),
        oneunit(T) * I(size),
    ]
    # StaticArrays do not work efficiently for large inputs anyway
    if size < 32
        push!(inputs, SMatrix{size,size}(make_rand_posdef(T, size)))
    end
    return inputs
end