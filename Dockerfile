FROM julia:1.13

WORKDIR /app
ENV JULIA_DEPOT_PATH=/root/.julia

RUN apt-get update \
    && apt-get install -y --no-install-recommends xvfb xauth \
    && rm -rf /var/lib/apt/lists/*

COPY Project.toml /app/Project.toml
COPY test/Project.toml /app/test/Project.toml

RUN xvfb-run -a julia --startup-file=no --project=/app -e 'using Pkg, TOML; Pkg.instantiate(; workspace=false, allow_autoprecomp=false); Pkg.precompile(collect(keys(TOML.parsefile("Project.toml")["deps"])); workspace=false, strict=true)'

COPY src/ /app/src/

RUN xvfb-run -a julia --startup-file=no --project=/app -e 'using Pkg; Pkg.precompile(["SurvivorModel"]; workspace=false, strict=true); Pkg.Apps.develop(path="/app")' \
    && /root/.julia/bin/survivor --help

ENTRYPOINT ["/root/.julia/bin/survivor"]
