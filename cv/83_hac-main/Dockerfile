FROM fishfl/sys_base:v4

RUN apt-get -y update
RUN apt-get install -y libx11-xcb1
RUN apt-get install -y libgl1-mesa-glx

COPY . /root/hac

# rm /bin/sh && ln -s /bin/bash /bin/sh
# RUN source activate py38 && \

RUN cd /root/hac && \
    pip3 install -r requirements.txt && \
    python3 setup.py install && \
    pip3 install flask

CMD cd /root/hac/ && \
#     source activate py38 && \
    python3 web/pose_holistic_web_rest.py




