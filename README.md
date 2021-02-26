# TIDIGITS recipe

[![DOI](https://zenodo.org/badge/342497960.svg)](https://zenodo.org/badge/latestdoi/342497960)

This repository contains a _recipe_ for training an automatic speech recognition (ASR) system using the [TIDIGITS database](https://catalog.ldc.upenn.edu/LDC93S10).
The recipe is entirely Julia-flavoured and uses following packages (among others):
* [Flux](https://github.com/FluxML/Flux.jl) as ML library
* [FiniteStateTransducers](https://github.com/idiap/FiniteStateTransducers.jl) for WFST compositions
* [HMMGradients](https://github.com/idiap/HMMGradients.jl) for maximum likelihood training

Currently the training runs only on CPU and employs a simple greedy decoder.

## Installation

Run `julia --project -e 'using Pkg; Pkg.instantiate()'` to install all the dependencies.
For the live demo install [sox](http://sox.sourceforge.net/).

## Live Demo

Open a Julia terminal with `julia --project` and type `include("demo.jl")` to try out the ASR with your own voice. A model trained with configuration `2b` (see below) is already present in this repository. 

## Training

### Configuration

Specify your current configuration in the folder `conf`.
The configuration files are loaded from the folder `conf/mysetup/`.
This folder must contain the following files:
* `feat_conf.jl` for feature extraction
* `model_conf.jl` for model and optimisation parameters (hyperparameters)
A couple of setups are present in this repository for reference in the folder `conf`.
Currently a TDNN/ConvNet is used as acoustic model.

### Data preparation

Set in your shell environment the path `TIDIGITS_PATH=\your\path\to\tidigits`.
If you're using SGE set the command flags in `CPU_CMD`, i.e. the queue options.

This can be done e.g. by running `source env.sh` before lunching Julia, where `env.sh` is a script that export these variables. 
Alternatively, the environment variables can be specified [directly in the REPL](https://docs.julialang.org/en/v1/manual/environment-variables/). 

Run `julia --project prepare_data.jl --conf 2a` to extract feature and prepare training data using the configuration `2a`.
Features and transcriptions will be saved in the folder `data/uuid/`.
Here `uuid` is linked to `feat_conf.jl` file, meaning that if you create a new `model_conf.jl` without modifying feature extraction you don't need to run data preparation twice. 
If SGE grid is available add the flag `--nj N` to split the work into `N` jobs.

For the moment HMM configuration is fixed in `wfsts.jl` with a phone based 2-state HMM.

### Training

Training is performed running the script `julia --project prepare_data.jl --conf 2a`.
Notice that if you're just experimenting it is more convenient to run the experiment from Julia's REPL.
```julia
$ julia --project

julia> include("train.jl")

```
Modify the `conf` by changing the default in the `ArgParse` table.

### Evaluation

Run the script `eval.jl` to calculate Word Error Rates (WER) and Phone Error Rate (PER).
