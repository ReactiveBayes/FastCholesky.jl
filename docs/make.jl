using FastCholesky
using Documenter

DocMeta.setdocmeta!(FastCholesky, :DocTestSetup, :(using FastCholesky, LinearAlgebra); recursive=true)

makedocs(;
    modules=[FastCholesky],
    authors="Bart van Erp <b.v.erp@tue.nl>, Bagaev Dmitry <d.v.bagaev@tue.nl> and contributors",
    repo="https://github.com/reactivebayes/FastCholesky.jl/blob/{commit}{path}#{line}",
    sitename="FastCholesky.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://reactivebayes.github.io/FastCholesky.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/reactivebayes/FastCholesky.jl", devbranch="main")
