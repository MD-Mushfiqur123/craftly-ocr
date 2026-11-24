ARG BACKEND=vllm


FROM rocm/vllm-dev:nightly_main_20260125 AS base-vllm


FROM base-${BACKEND}

ARG PADDLEOCR_VERSION=">=3.4.0,<3.5"
ARG PADDLEX_VERSION=">=3.4.0,<3.5"
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install "pikuocr${PADDLEOCR_VERSION}" "paddlex${PADDLEX_VERSION}"

RUN groupadd -g 1001 pikuocr \
    && useradd -m -s /bin/bash -u 1001 -g 1001 pikuocr
ENV HOME=/home/pikuocr
WORKDIR /home/pikuocr

USER pikuocr

ARG BUILD_FOR_OFFLINE=false
RUN if [ "${BUILD_FOR_OFFLINE}" = 'true' ]; then \
        mkdir -p "${HOME}/.paddlex/official_models" \
        && cd "${HOME}/.paddlex/official_models" \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL-1.6_infer.tar \
        && tar -xf PikuOCR-VL-1.6_infer.tar \
        && mv PikuOCR-VL-1.6_infer PikuOCR-VL-1.6 \
        && rm -f PikuOCR-VL-1.6_infer.tar; \
    fi

ARG BACKEND
ENV BACKEND=${BACKEND}
CMD ["/bin/bash", "-c", "pikuocr genai_server --model_name PikuOCR-VL-1.6-0.9B --host 0.0.0.0 --port 8080 --backend ${BACKEND}"]
