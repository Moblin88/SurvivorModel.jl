FROM julia:1.13

WORKDIR /app
ENV JULIA_DEPOT_PATH=/root/.julia

COPY Project.toml /app/Project.toml
COPY test/Project.toml /app/test/Project.toml

RUN julia --startup-file=no --project=/app -e 'using Pkg, TOML; Pkg.instantiate(; workspace=false, allow_autoprecomp=false); Pkg.precompile(collect(keys(TOML.parsefile("Project.toml")["deps"])); workspace=false, strict=true)'

COPY docker/blas/Project.toml /opt/julia-blas/Project.toml
RUN if [ "$(uname -m)" = "x86_64" ]; then \
        julia --startup-file=no --project=/opt/julia-blas -e 'using Pkg; Pkg.instantiate(; allow_autoprecomp=false); Pkg.precompile(; strict=true)' \
        && julia --startup-file=no --project=/opt/julia-blas -e 'using AOCL' \
        && julia --startup-file=no --project=/opt/julia-blas -e 'using MKL'; \
    fi

COPY src/ /app/src/

RUN julia --startup-file=no --project=/app -e 'using Pkg; Pkg.precompile(["SurvivorModel"]; workspace=false, strict=true); Pkg.Apps.develop(path="/app")' \
    && /root/.julia/bin/survivor --help

ENV JULIA_NUM_THREADS=auto

COPY docker/startup.jl /root/.julia/config/startup.jl

ENTRYPOINT ["/root/.julia/bin/survivor"]
