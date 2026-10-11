# ponytail: deps baked so Vast GPU bills only train time, never pip time.
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install -U pip && pip install \
    "transformers>=5.0.0" \
    accelerate trl datasets unsloth \
    huggingface_hub hf_transfer \
    lm-eval \
    jupyterlab

EXPOSE 8080

COPY train_pilot.py eval_gate.py prep_data.py train.ipynb /workspace/lfm350m-vi/
WORKDIR /workspace/lfm350m-vi

# ponytail: pilot corpus baked at build (~200MB, free local time, zero GPU billing).
# Full-scale 1B-token corpus stays outside image — rebuild per data change
# costs more than one upload.
RUN python prep_data.py
