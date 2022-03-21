## Learning from Noisy Data with Robust Representation Learning (ICCV 2021)

This is the PyTorch implementation of the ICCV paper [link].

### Requirements:
* PyTorch = 1.4
* pip install tensorboard_logger torchnet faiss-gpu

### Configuration:

Hyper-parameters and model configurations are located in ./config

### Dataset:

In order to run experiments, please download the corresponding dataset and place it at the location specified in the config file. 

### Execution:
<pre>python main.py --exp [config_file]</pre> 

For example, run the following command to reproduce the paper's result on CIFAR-10:

1. 50% symmetric noise: <pre>python main.py --exp cifar10_sym</pre> 
2. 40% asymmetric noise: <pre>python main.py --exp cifar10_asym</pre> 


### Citation
If you find this code to be useful for your research, please consider citing.
<pre>
@inproceedings{RRL,
      title={Learning from Noisy Data with Robust Representation Learning}, 
      author={Junnan Li and Caiming Xiong and Steven Hoi},
      year={2021},
      booktitle = {{ICCV}},
}</pre>
