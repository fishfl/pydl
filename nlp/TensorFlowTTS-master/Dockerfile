FROM fishfl/sys_base:v9

COPY . /root/TensorFlowTTS
RUN mv /root/TensorFlowTTS/nltk_data /root/nltk_data
RUN apt-get install libsndfile1 -y

RUN cd /root/TensorFlowTTS&& \
    source activate py38 && \
    pip3 install git+https://github.com/repodiac/german_transliterate.git#egg=german_transliterate && \
    pip3 install -r requirements.txt && \
    pip3 install flask

CMD cd /root/TensorFlowTTS/ && \
    source activate py38 && \
    python3 predict_web.py
