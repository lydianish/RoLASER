# Use a base image with Conda installed
FROM continuumio/miniconda3:latest
FROM python:3.9.18

WORKDIR /app

# Clone the RoLASER repository into the container
RUN apt-get update && \
    apt-get install -y git && \
    git clone https://github.com/lydianish/RoLASER.git /app

# Create conda environment with the required Python and GCC versions
RUN conda create -n "rolaser_env" python=3.9.18 gxx=11.2.0=h702ea55_10 -c conda-forge && \
    echo "source activate rolaser_env" > ~/.bashrc
ENV PATH /opt/conda/envs/rolaser_env/bin:$PATH

# Install PyTorch 1.10.1
RUN pip3 install --no-cache-dir torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# Install Fairseq and dependencies
RUN git clone https://github.com/lydianish/fairseq.git /app
WORKDIR /app/fairseq
RUN git checkout rolaser
RUN pip3 install --no-cache-dir --editable .
RUN python3 setup.py build_ext --inplace
RUN pip3 install --no-cache-dir numpy==1.21.6 fairscale==0.4.6
ENV FAIRSEQ /app/fairseq

# Install LASER and dependencies
WORKDIR /app
RUN git clone https://github.com/lydianish/LASER.git /app
WORKDIR /app/LASER
RUN git checkout rolaser
ENV LASER /app/LASER
RUN bash ./install_external_tools.sh
RUN pip3 install --no-cache-dir faiss-gpu JapaneseTokenizer jieba transliterate tabulate

# Install other RoLASER dependencies
WORKDIR /app/RoLASER 
RUN pip3 install --no-cache-dir transformers pandas tensorboardX matplotlib seaborn streamlit

# Download models
WORKDIR /app/RoLASER/models
RUN wget https://zenodo.org/api/records/10864557/files-archive
RUN unzip files-archive

RUN mkdir $LASER/models
RUN mv laser* $LASER/models
RUN mkdir RoLASER
RUN mv rolaser* RoLASER/
RUN mkdir c-RoLASER
RUN mv c-rolaser* c-RoLASER/
RUN rm files-archive

COPY RoLASER/* /app/RoLASER/models/RoLASER
COPY c-RoLASER/* /app/RoLASER/models/c-RoLASER

# User
RUN useradd -m -u 1000 user
USER user
ENV HOME /home/user
ENV PATH $HOME/.local/bin:$PATH

WORKDIR $HOME
RUN mkdir app
WORKDIR $HOME/app
COPY . $HOME/app

EXPOSE 8501
CMD streamlit run app.py \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection false \
    --server.fileWatcherType none