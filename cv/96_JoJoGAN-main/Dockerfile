FROM fishfl/sys_base:v10

COPY . /root/jojogan

RUN cd /root/jojogan && \
    source activate py38 && \
    pip3 install -r requirements.txt && \
    pip3 install flask

CMD cd /root/jojogan/ && \
    source /opt/conda/bin/activate py38 && \
    python3 predict_web.py



