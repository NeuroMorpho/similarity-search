# our base image
FROM ubuntu:bionic

# Install python and pip
LABEL maintainer="bljungqu@gmu.edu"

RUN apt-get update -y && \
    apt-get install -y python3-pip python3-dev libopenblas-base libomp-dev   
RUN apt-get install -y apt-utils debconf-utils dialog
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
RUN echo "resolvconf resolvconf/linkify-resolvconf boolean false" | debconf-set-selections
RUN apt-get update
RUN apt-get install -y resolvconf

# install Python modules needed by the Python app
COPY requirements.txt /app/
    
WORKDIR /app

RUN python3 -m pip install -U pip

RUN pip3 install --no-cache-dir -r /app/requirements.txt

# copy files required for the app to run
COPY search.py /app/
COPY pvec_cache.pkl /app/
COPY pvecmes_cache.pkl /app/
COPY sum_cache.pkl /app/
COPY meta_cache.pkl /app/
COPY detailed_cache.pkl /app/
COPY detailedpvec_cache.pkl /app/
COPY sis/cfg.py /app/sis/
COPY sis/com.py /app/sis/
COPY sis/datamgmt.py /app/sis/

ENTRYPOINT [ "flask" ]

ENV FLASK_APP=search.py
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8


# tell the port number the container should expose
EXPOSE 5000

# run the application
CMD ["run", "--host", "0.0.0.0"]