FROM mambaorg/micromamba:latest

WORKDIR /app

COPY environment.txt .

RUN micromamba create --name myenv --file environment.txt && \
    micromamba clean --all --yes

# when using conda use this and change the From to miniconda version (all that are commented)    
# RUN echo "conda activate myenv" > ~/.bashrc
# ENV PATH /opt/conda/envs/myenv/bin:$PATH

# RUN pip install streamlit

COPY  . .

EXPOSE 8501

CMD [ "streamlit", "run", "whatsapp.py" ]
