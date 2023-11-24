@testitem "General functionality" begin
    include("fastcholesky_setuptests.jl")

    # General properties
    for size in 1:20:1000
        for Type in SupportedTypes

            # `BigFloat` tests are too slow
            if size > 101 && Type === BigFloat
                continue
            end

            for input in make_rand_inputs(Type, size)

                # We check only posdef inputs
                @test isposdef(input)

                let input = input
                    C = fastcholesky(input)

                    @test issuccess(C)
                    @test isposdef(C)

                    @test collect(fastcholesky(input).L) ≈ collect(cholesky(input).L)
                    @test cholinv(input) ≈ inv(input)
                    @test cholsqrt(input) ≈ cholesky(input).L
                    @test chollogdet(input) ≈ logdet(input)
                    @test all(cholinv_logdet(input) .≈ (inv(input), logdet(input)))
                    @test cholsqrt(input) * cholsqrt(input)' ≈ input

                    if Type <: LinearAlgebra.BlasFloat && input isa Matrix 
                        @test collect(fastcholesky(input).L) ≈ collect(fastcholesky!(deepcopy(input)).L)
                    end

                    # `sqrt` does not work on `BigFloat` matrices
                    if Type !== BigFloat
                        @test cholsqrt(input) * cholsqrt(input)' ≈ sqrt(input) * sqrt(input)'
                    end

                    # Check that we do not lose the static type in the process for example
                    @test typeof(cholesky(input)) === typeof(fastcholesky(input))
                end
            end
        end

        nonposdefmatrix = Matrix(Diagonal(-ones(size)))

        @test !issuccess(fastcholesky!(nonposdefmatrix))
    end
end

@testitem "UniformScaling support" begin 
    include("fastcholesky_setuptests.jl")

    for Type in SupportedTypes
        two = Type(2)
        invtwo = Type(inv(2))
        zero = Type(0)
        one = Type(1)

        @test cholinv(two * I) ≈ invtwo * I
        @test cholsqrt(two * I) ≈ sqrt(two) * I
        @test chollogdet(I) ≈ zero
        @test all(cholinv_logdet(one * I) .≈ (one * I, zero))
        @test_throws ArgumentError chollogdet(two * I)
        @test fastcholesky(I) ≈ I
        @test fastcholesky!(I) ≈ I
    end
end

@testitem "A number support" begin 
    include("fastcholesky_setuptests.jl")

    for Type in SupportedTypes
        let number = rand(Type)
            @test fastcholesky(number) === number
            @test cholinv(number) ≈ inv(number)
            @test cholsqrt(number) ≈ sqrt(number)
            @test chollogdet(number) ≈ logdet(number)
            @test all(cholinv_logdet(number) .≈ (inv(number), logdet(number)))
        end
    end
end

@testitem "special case #1 (found in ExponentialFamily.jl)" begin
    include("fastcholesky_setuptests.jl")

    # This is a very bad matrix, but should be solveable
    F = [
        42491.1429254459 1.0544416413649244e6 64.9016820609457 1712.2779951809016
        1.0544416413649244e6 2.616823794441869e7 1610.468694700484 42488.422800411565
        64.9016820609457 1610.468694700484 0.10421453600353446 2.6155294717625517
        1712.2779951809016 42488.422800411565 2.6155294717625517 69.0045838263577
    ]
    @test inv(fastcholesky(F)) * F ≈ Diagonal(ones(4)) rtol=1e-4
    @test cholinv(F) * F ≈ Diagonal(ones(4)) rtol=1e-4
    @test fastcholesky(F).L ≈ cholesky(F).L
end
