docker build --build-arg UID=$UID -t mmflow-image .
docker run --name mmflow-container --gpus all --shm-size=11g -dt \
           -v $(pwd)/../:/codebase/ \
           -v $(readlink -f ../data/):/codebase/data/ \
           -w /codebase/ \
           mmflow-image

