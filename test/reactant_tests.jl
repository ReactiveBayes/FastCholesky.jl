# Reactant supports Linux and macOS only
@testitem "Traced matrices: every operation compiles and agrees with the dense one" skip = Sys.iswindows() begin
    using Reactant, LinearAlgebra

    A = [4.0 1.0 0.5; 1.0 3.0 0.2; 0.5 0.2 2.0]
    B = [1.0 2.0 3.0; 0.0 1.0 -1.0]
    b = [1.0, 2.0, 3.0]
    rA, rB, rb = Reactant.to_rarray(A), Reactant.to_rarray(B), Reactant.to_rarray(b)

    # Each factor is multiplied by `A` rather than compared alone: `.U` and `.L` are triangular wrappers
    cases = [
        ((A, B, b) -> fastcholesky(A).U * A, cholesky(A).U * A),
        ((A, B, b) -> fastcholesky(A).L * A, cholesky(A).L * A),
        ((A, B, b) -> fastcholesky(A) \ b, A \ b),
        ((A, B, b) -> fastcholesky(A) \ permutedims(B), A \ permutedims(B)),
        ((A, B, b) -> B / fastcholesky(A), B / A),
        ((A, B, b) -> permutedims(B)' / fastcholesky(A), B / A),
        ((A, B, b) -> inv(fastcholesky(A)), inv(A)),
        ((A, B, b) -> cholinv(A), inv(A)),
        ((A, B, b) -> chollogdet(A), logdet(A)),
        ((A, B, b) -> cholinv_logdet(A)[1], inv(A)),
        ((A, B, b) -> cholinv_logdet(A)[2], logdet(A)),
        ((A, B, b) -> cholsqrt(A) * A, cholesky(A).L * A),
    ]
    for (f, expected) in cases
        result = @jit f(rA, rB, rb)
        @test (result isa Number ? Float64(result) : Array(result)) ≈ expected
    end

    # A traced factorisation is not checked, so it reports success
    traced_success(A) = (C = fastcholesky(A); issuccess(C) ? sum(A) : -sum(A))
    @test Float64(@jit traced_success(rA)) ≈ sum(A)
end
