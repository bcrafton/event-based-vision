
# Event Based Vision

This repository contains template code for event based vision using an augmented YOLO object detector

We have used Prophesee's wonderful dataset available here:
https://www.prophesee.ai/2020/01/24/prophesee-gen1-automotive-detection-dataset/

And their data visualization and extraction code available here:
https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox

## Getting Started

It is intended to run on our lab server: icsrl-exxact1.ece.gatech.edu

You will likely want to setup a virtual environment (virtualenv) for this. And you will need to append your .bashrc with:

```
if [ -d "/usr/local/cuda-10.1/bin/" ]; then
    export PATH=/usr/local/cuda-10.1/bin${PATH:+:${PATH}}
    export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
fi
```
Because the default CUDA on the server is 10.2, but we need 10.1 for Tensorflow 2.2

## Affiliation 

```
Georgia Institute of Technology, ICSRL (http://icsrl.ece.gatech.edu/)
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
