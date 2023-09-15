# mosaic-llm-foundry-sagemaker


Repository to train MPT model using LLM foundry on Amazon SageMaker


## Build the docker image 

The training scripts are located in the scripts folder. The scripts/yamls folder contains yaml files for different configuration. We have added files for MPT-7B, MPT-13B and MPT-70B. We can modify these files as per the required configuration and build the docker image.

Launch the build/docker_build.sh file which will build the docker image and tag it with the repo name provided. The repo details are maintained the .env file. The dockerfile extends from SageMaker Deep Learning container and installs the required dependencies for LLM foundry. The script later pushes the docker image to ECR.

## Launch the training job

Train.ipynb is the main entry point notebook that helps to interactively download data, upload to S3 and then launch the training job. Please follow the notebook to launch the training. 