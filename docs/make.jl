#!/usr/bin/env julia
#
#

if "--help" ∈ ARGS
    println(
        """
docs/make.jl

Render the `GeometricKalman.jl` documentation with optional arguments

Arguments
* `--exclude-tutorials` - exclude the tutorials from the menu of Documenter,
  This can be used if not all tutorials are rendered and you want to therefore exclude links
  to these, especially the corresponding menu. This option should not be set on CI.
  Locally this is also set if `--quarto` is not set and not all tutorials are rendered.
* `--help`              - print this help and exit without rendering the documentation
* `--prettyurls`        – toggle the pretty urls part to true, which is always set on CI
* `--quarto`            – (re)run the Quarto notebooks from the `tutorials/` folder before
  generating the documentation. If they are generated once they are cached accordingly.
  Then you can spare time in the rendering by not passing this argument.
  If quarto is not run, some tutorials are generated as empty files, since they
  are referenced from within the documentation.
""",
    )
    exit(0)
end

run_quarto = "--quarto" in ARGS
run_on_CI = (get(ENV, "CI", nothing) == "true")
tutorials_in_menu = !("--exclude-tutorials" ∈ ARGS)
#
#
# (a) setup the tutorials menu – check whether all files exist
tutorials_menu =
    "How to..." => [
    ]
# Check whether all tutorials are rendered, issue a warning if not (and quarto if not set)
all_tutorials_exist = true
for (name, file) in tutorials_menu.second
    fn = joinpath(@__DIR__, "src/", file)
    if !isfile(fn) || filesize(fn) == 0 # nonexistent or empty file
        global all_tutorials_exist = false
        if !run_quarto
            @warn "Tutorial $name does not exist at $fn."
            if (!isfile(fn)) && (endswith(file, "getstarted.md"))
                @warn "Generating empty file, since this tutorial is linked to from the documentation."
                touch(fn)
            end
        end
    end
end
if !all_tutorials_exist && !run_quarto && !run_on_CI
    @warn """
        Not all tutorials exist. Run `make.jl --quarto` to generate them. For this run they are excluded from the menu.
    """
    tutorials_in_menu = false
end
if !tutorials_in_menu
    @warn """
    You are either explicitly or implicitly excluding the tutorials from the documentation.
    You will not be able to see their menu entries nor their rendered pages.
    """
    run_on_CI &&
        (@error "On CI, the tutorials have to be either rendered with Quarto or be cached.")
end
#
# (b) if docs is not the current active environment, switch to it
# (from https://github.com/JuliaIO/HDF5.jl/pull/1020/) 
if Base.active_project() != joinpath(@__DIR__, "Project.toml")
    using Pkg
    Pkg.activate(@__DIR__)
    Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
    Pkg.resolve()
    Pkg.instantiate()
end

# (c) If quarto is set, or we are on CI, run quarto
if run_quarto || run_on_CI
    using CondaPkg
    CondaPkg.withenv() do
        @info "Rendering Quarto"
        tutorials_folder = (@__DIR__) * "/../tutorials"
        # instantiate the tutorials environment if necessary
        Pkg.activate(tutorials_folder)
        # For a breaking release -> also set the tutorials folder to the most recent version
        Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
        Pkg.resolve()
        Pkg.instantiate()
        Pkg.build("IJulia") # build `IJulia` to the right version.
        Pkg.activate(@__DIR__) # but return to the docs one before
        run(`quarto render $(tutorials_folder)`)
        return nothing
    end
end

# (d) load necessary packages for the docs
using Plots, RecipesBase, Manifolds, GeometricKalman, ManifoldsBase, Documenter, PythonPlot
using DocumenterCitations, DocumenterInterLinks

ENV["GKSwstype"] = "100"

# (e) add CONTRIBUTING.md and NEWS.md to docs
generated_path = joinpath(@__DIR__, "src", "misc")
base_url = "https://github.com/JuliaManifolds/GeometricKalman.jl/blob/main/"
isdir(generated_path) || mkdir(generated_path)
for fname in ["NEWS.md"]
    open(joinpath(generated_path, fname), "w") do io
        # Point to source license file
        println(
            io,
            """
            ```@meta
            EditURL = "$(base_url)$(fname)"
            ```
            """,
        )
        # Write the contents out below the meta block
        for line in eachline(joinpath(dirname(@__DIR__), fname))
            println(io, line)
        end
    end
end

# (f) final step: render the docs
bib = CitationBibliography(joinpath(@__DIR__, "src", "references.bib"); style=:alpha)
links = InterLinks(
    "ManifoldsBase" => ("https://juliamanifolds.github.io/ManifoldsBase.jl/stable/"),
)
modules = [GeometricKalman]
if modules isa Vector{Union{Nothing,Module}}
    error("At least one module has not been properly loaded: ", modules)
end
makedocs(;
    format=Documenter.HTML(
        prettyurls=(get(ENV, "CI", nothing) == "true") || ("--prettyurls" ∈ ARGS),
        assets=["assets/favicon.ico", "assets/citations.css", "assets/link-icons.css"],
        size_threshold_warn=200 * 2^10, # raise slightly from 100 to 200 KiB
        size_threshold=300 * 2^10,      # raise slightly 200 to 300 KiB
    ),
    modules=modules,
    authors="Mateusz Baran, Ronny Bergmann, and contributors.",
    sitename="GeometricKalman.jl",
    pages=[
        "Home" => "index.md",
        (tutorials_in_menu ? [tutorials_menu] : [])...,
        "Interface" => "interface.md",
        "Miscellanea" => [
            "Changelog" => "misc/NEWS.md",
            "References" => "misc/references.md",
        ],
    ],
    plugins=[bib, links],
    warnonly=[:missing_docs],
)
deploydocs(repo="github.com/JuliaManifolds/GeometricKalman.jl.git", push_preview=true)
