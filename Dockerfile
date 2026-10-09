FROM julia:1.13 AS builder

ARG TARGETARCH

WORKDIR /app
ENV JULIA_DEPOT_PATH=/root/.julia: \
    JULIA_NUM_THREADS=auto \
    OPENBLAS_NUM_THREADS=1 \
    PATH="/root/.julia/bin:${PATH}"

RUN apt-get update && \
    apt-get install --no-install-recommends --yes \
      xvfb xauth libgl1-mesa-dri libxrandr2 libxinerama1 libxcursor1 libxi6 && \
    rm -rf /var/lib/apt/lists/*

COPY Project.toml /app/Project.toml
COPY test/Project.toml /app/test/Project.toml
COPY src/ /app/src/
COPY docker/blas/Project.toml /opt/survivormodel-blas/Project.toml
COPY docker/blas_backend.jl /opt/survivormodel/docker/blas_backend.jl
COPY docker/smoke.jl /opt/survivormodel/docker/smoke.jl
COPY docker/startup.jl /root/.julia/config/startup.jl

RUN if [ "$TARGETARCH" = "amd64" ]; then \
        JULIA_NUM_THREADS=1 julia --startup-file=no --project=/opt/survivormodel-blas \
          -e 'using Pkg; Pkg.instantiate(); Pkg.precompile(strict=true)'; \
    fi

RUN JULIA_NUM_THREADS=1 JULIA_PKG_PRECOMPILE_AUTO=0 \
      julia --startup-file=no --project=/app \
      -e 'using Pkg; Pkg.instantiate()'

RUN JULIA_NUM_THREADS=1 xvfb-run --auto-servernum \
      --server-args="-screen 0 1600x1200x24 +extension GLX" \
      env LIBGL_ALWAYS_SOFTWARE=1 \
        julia --startup-file=no --project=/app \
          -e 'using Pkg; Pkg.precompile(strict=true); using GLMakie'

RUN JULIA_NUM_THREADS=1 xvfb-run --auto-servernum \
      --server-args="-screen 0 1600x1200x24 +extension GLX" \
      env LIBGL_ALWAYS_SOFTWARE=1 julia --startup-file=no \
        -e 'using Pkg; Pkg.Apps.develop(path="/app")'

RUN \
    grep -F -- '--startup-file=no --startup-file=yes' /root/.julia/bin/survivor && \
    julia --project=/app /opt/survivormodel/docker/smoke.jl 1 && \
    JULIA_NUM_THREADS=2 SURVIVORMODEL_BLAS_THREADS=2 \
      julia --project=/app /opt/survivormodel/docker/smoke.jl 2 2 && \
    /root/.julia/bin/survivor --help >/dev/null

FROM julia:1.13 AS runtime

WORKDIR /app
ENV JULIA_DEPOT_PATH=/root/.julia: \
    JULIA_NUM_THREADS=auto \
    OPENBLAS_NUM_THREADS=1 \
    PATH="/root/.julia/bin:${PATH}"

COPY --from=builder /root/.julia /root/.julia
COPY --from=builder /opt/survivormodel-blas /opt/survivormodel-blas
COPY --from=builder /opt/survivormodel/docker /opt/survivormodel/docker
COPY --from=builder /app /app

ENTRYPOINT ["/root/.julia/bin/survivor"]
