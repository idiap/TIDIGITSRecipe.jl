# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>
#
# This script can be used to check the 
# output of the acoustic model and its decoding
# using two random utterances taken from the test set

using HMMGradients, Flux, Zygote
using Random, Statistics, LinearAlgebra
using FiniteStateTransducers
using DSP, MFCC
using BSON, JLD2, LibSndFile, FileIO
include("WFSTs.jl")
include("Models.jl")
include("Utils.jl")

setup="2a"

include("conf/$(setup)/feat_conf.jl")
include("conf/$(setup)/model_conf.jl")

# get transition matrix
lexicon, ilexicon = get_lexicon()
H, L = get_HL(lexicon)
a, A = get_aA(H)
ippsym = get_iisym(H)

BSON.@load "models/$(setup)/current_modely.bson" best_modely
Flux.testmode!(best_modely)

feat_dir = get_feat_dir(setup)
data = load(joinpath(feat_dir,"test.jld2"))
uttID2feats, uttID2text = data["uttID2feats"], data["uttID2text"]

uttIDs = [keys(uttID2text)...]
i,j = rand(uttIDs),rand(uttIDs)

xi,xj = feats_post(uttID2feats[i]), feats_post(uttID2feats[j])
yi,yj = best_modely(Flux.unsqueeze(xi,3)),best_modely(Flux.unsqueeze(xj,3))

gammai = logposterior(size(yi,1),a,A,yi[:,:])
gammaj = logposterior(size(yj,1),a,A,yj[:,:])

outi = posterior2phones(ippsym,gammai)
outj = posterior2phones(ippsym,gammaj)
outi[outi .== "<SIL>"] .= " "
outj[outj .== "<SIL>"] .= " "

using Plots
pyplot()
psi = prod([prod(lexicon[w]) for w in split(uttID2text[i])].*" ")
p1i = heatmap(xi', title=uttID2text[i])
p2i = heatmap(yi[:,:]', clims = (maximum(yj)-20,maximum(yj)), title=psi)
p3i = heatmap(gammai',  clims = (-20,0), title=prod(outi))

psj = prod([prod(lexicon[w]) for w in split(uttID2text[j])].*" ")
p1j = heatmap(xj', title=uttID2text[j])
p2j = heatmap(yj[:,:]', clims = (maximum(yj)-20,maximum(yj)), title=psj)
p3j = heatmap(gammaj',  clims = (-20,0), title=prod(outj))

plot(p1i,p1j,p2i,p2j,p3i,p3j,layout=(3,2))
