# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>
#
#  This script is mainly for testing the model input/output works correctly
#
using HMMGradients, FiniteStateTransducers
using Random, Statistics, FileIO
using BSON

setup="2a"

include("WFSTs.jl")
include("Models.jl")
include("Utils.jl")
include("conf/$(setup)/feat_conf.jl")
include("conf/$(setup)/model_conf.jl")

# get transition matrix
lexicon, ilexicon = get_lexicon()
H, L = get_HL(lexicon)
a, A = get_aA(H)
Ns = size(A,1)

# init model
modely = get_convnet(Nf,Ns; 
                     Nks=Nks,
                     Nhs=Nhs,
                     strides=strides,
                     dilations=dilations,
                     dropout=dropout,
                     fout=fout)

Nt,Nb = rand(500:1000),4
x = zeros(Float32,Nt,Nf,Nb)
t,b = 500,1 
x[t,:,b] .= 1.0

y = modely(x)
z = sum(y[:,:,b],dims=2)
Nt2 = ceil(Int,Nt/3)
@assert Nt2 == size(y,1) 
println("Setup = $setup")
println("Num of parameters = $(sum(prod.(size.(params(modely)))))")
println("Context bins = $(subsample*sum( (!).(z .≈ z[100]) ))")
