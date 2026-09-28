FROM julia:1.13

WORKDIR /app
ENV JULIA_DEPOT_PATH=/root/.julia

COPY Project.toml /app/Project.toml
COPY test/Project.toml /app/test/Project.toml
COPY src/ /app/src/

RUN julia -e 'using Pkg; Pkg.Apps.develop(path="/app")'

ENTRYPOINT ["/root/.julia/bin/survivor"]
