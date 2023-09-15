# Base Docker image.
FROM 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-sagemaker

RUN git clone https://github.com/mosaicml/llm-foundry.git \
 && cd llm-foundry \
 && pip install -e ".[gpu]"
 
RUN pip install hydra-core