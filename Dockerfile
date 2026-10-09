FROM julia:1.13

ARG TARGETARCH

WORKDIR /app
ENV JULIA_DEPOT_PATH=/root/.julia \
    JULIA_NUM_THREADS=auto \
    OPENBLAS_NUM_THREADS=1

COPY Project.toml /app/Project.toml
COPY test/Project.toml /app/test/Project.toml
COPY src/ /app/src/
COPY docker/blas/Project.toml /opt/survivormodel-blas/Project.toml
COPY docker/blas_backend.jl /opt/survivormodel/docker/blas_backend.jl
COPY docker/smoke.jl /opt/survivormodel/docker/smoke.jl
COPY docker/startup.jl /root/.julia/config/startup.jl

RUN if [ "$TARGETARCH" = "amd64" ]; then \
        JULIA_NUM_THREADS=1 julia --startup-file=no --project=/opt/survivormodel-blas \
          -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; \
    fi

RUN JULIA_NUM_THREADS=1 julia --startup-file=no \
    -e 'using Pkg; Pkg.Apps.develop(path="/app")'

RUN \
    grep -F -- '--startup-file=no --startup-file=yes' /root/.julia/bin/survivor && \
    julia --project=/app /opt/survivormodel/docker/smoke.jl 1 && \
    JULIA_NUM_THREADS=2 SURVIVORMODEL_BLAS_THREADS=2 \
      julia --project=/app /opt/survivormodel/docker/smoke.jl 2 2 && \
    /root/.julia/bin/survivor --help >/dev/null

ENTRYPOINT ["/root/.julia/bin/survivor"]
