# MiCRO: Near-Zero Cost Gradient Sparsification for Scaling and Accelerating Distributed DNN Training

## Description
> ***MiCRO*** is abbreviation of 'Minimizing compression ratio error on-the-fly'. MiCRO sparsifies gradients in partitioned gradient vectors, and estimates accurate threshold for sparsification condition by minimizing compression ratio error. Benchmarks include: 1) image classification using CNNs, 2) language modelling using LSTMs `cnn-lstm`, and 3) recommendation using NCF `ncf`.

## How to setup

To install the necessary dependencies, create Conda environment using `environment.yml` by running the following commands. ***Note***: check compatibility of each package version such as `cudatoolkit` and `cudnn` with your device, e.g., NVIDIA Tesla V100 GPU is compatible.

```bash
$ conda env create --file environment.yml
$ conda activate micro
$ python -m spacy download en
$ conda deactivate micro
```

## How to start

The scripts to run code are written for SLURM workload manager. The source code supports distributed training with **multi-node and multi-GPU**. In `run.sh`, you can specify *model*, *dataset*, ***reducer***, and *world_size*.

### Overview of shell scripts

 - If you use **SLURM**, use `pararun` and modify it for your configuration. The script `pararun` executes `run.sh` in parallel. The script `run.sh` includes setup for distributed training.
 - If you do not use SLURM, you do not need to use `pararun`. Instead, run `run.sh` on your nodes, then rendezvous of pytorch allows processes are connected.

### CNN and LSTM benchmarks

 - If you use **SLURM**, use following command.
```bash
$ sbatch pararun
```
 - If you do not use SLURM, use following command on each node.
```bash
$ hostip=<ip> port=<port> mpirun -np <world_size> run.sh
```

### Neural Collaborative Filtering (NCF) benchmarks

#### 1. Prepare dataset

 - To download dataset, use following command.
```bash
$ ./prepare_dataset.sh
```

#### 2. Run training script

 - If you use **SLURM**, use following command.
```bash
$ sbatch pararun
```
 - If you do not use SLURM, use following command on each node.
```bash
$ hostip=<ip> port=<port> mpirun -np <world_size> run.sh
```

## Acknowledgements

Most of code except [MiCRO](https://github.com/kljp/micro) implementation is provided by previous works. If you use the code, please cite the following papers also.

**PowerSGD** \[[Paper](https://arxiv.org/abs/1905.13727)\] \[[Code](https://github.com/epfml/powersgd)\] (`cnn-lstm`)

    @inproceedings{vkj2019powerSGD,
      author = {Vogels, Thijs and Karimireddy, Sai Praneeth and Jaggi, Martin},
      title = "{{PowerSGD}: Practical Low-Rank Gradient Compression for Distributed Optimization}",
      booktitle = {NeurIPS 2019 - Advances in Neural Information Processing Systems},
      year = 2019,
      url = {https://arxiv.org/abs/1905.13727}
    }
**Rethinking-sparsification** \[[Paper](https://arxiv.org/abs/2108.00951)\] \[[Code](https://github.com/sands-lab/rethinking-sparsification)\] (`cnn-lstm` and `ncf`)

    @inproceedings{sda+2021rethinking-sparsification,
      author = {Sahu, Atal Narayan and Dutta, Aritra and Abdelmoniem, Ahmed M. and Banerjee, Trambak and Canini, Marco and Kalnis, Panos},
      title = "{Rethinking gradient sparsification as total error minimization}",
      booktitle = {NeurIPS 2021 - Advances in Neural Information Processing Systems},
      year = 2021,
      url = {https://arxiv.org/abs/2108.00951}
    }

## Publication

Under review.

## Contact

If you have any questions about this project, contact me by one of the followings:
- slashxp@naver.com
- kljp@ajou.ac.kr
