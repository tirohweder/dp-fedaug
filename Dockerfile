FROM alpine:latest

RUN apk add --no-cache git bash openssh-client

WORKDIR /workspace/dp-fl2

COPY . /workspace/dp-fl2/

RUN git config --global --add safe.directory /workspace/dp-fl2

CMD ["/bin/bash"]
