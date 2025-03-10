FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
RUN apt-get update \
    && pip install gpustat \
    && apt-get -y install tmux \
    && apt-get -y install git \
    && apt-get -y install gcc \
    && apt-get -y install libsm6 \
    && apt-get -y install libxext-dev \
    && apt-get -y install libxrender1 \
    && apt-get -y install libglib2.0-dev \
    && apt-get -y install default-jre \
    && pip install scipy \
    && pip install datasets==1.2.1\
    && pip install pandas\
    && pip install numpy\
    && pip install scikit-learn==0.24.0\
    && pip install prettytable==2.1.0\
    && pip install gradio\
    && pip install tqdm \
    && pip install setuptools==49.3.0\
    && pip install pyarrow \
    && pip install lmdb \
    && conda install -y -c pytorch faiss-gpu \
    && pip install torchvision \
    && pip install fairscale==0.4.4 \
    && pip install timm==0.4.12 \
    && pip install git+https://github.com/openai/CLIP.git \
    && pip install git+https://github.com/jmhessel/pycocoevalcap.git \
    && python -c 'import pycocoevalcap.spice.get_stanford_models as f; f.get_stanford_models()' \
    && python -c 'import pycocoevalcap.clipscore.clipscore as f; f.ClipScore()' \
    && pip install einops==0.3.0 \
    && pip install fairscale==0.4.4 \
    && pip install opencv-python==4.1.2.30\
    && pip install mmcv==1.7.1 \
    && pip install ftfy \
    && pip install regex \
    && pip install transformers==4.15.0 \
    && pip install loralib