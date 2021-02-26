# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

using HMMGradients, Flux
using Random, Statistics, LinearAlgebra
using FiniteStateTransducers
using BSON, JLD2, LibSndFile, FileIO, UUIDs
using MFCC, DSP
include("WFSTs.jl")
include("Models.jl")
include("Utils.jl")

setup="2a"
duration=5 # duration of recoring in seconds
plot_stuff=false

println("
        TIDIGIT demo

        $duration seconds will be recorded

        The following digits can be recognized:
        ZERO OH ONE TWO THREE FOUR FIVE SIX SEVEN EIGHT NINE
        (requires sox)
        ")


include("conf/$(setup)/feat_conf.jl")
# get transition matrix
lexicon, ilexicon = get_lexicon()
H, L = get_HL(lexicon)
a, A = get_aA(H)
ippsym = get_iisym(H)

model_folder = joinpath("models","$setup")
BSON.@load joinpath(model_folder,"best_modely_final.bson") best_modely
Flux.testmode!(best_modely)

mkpath("data")
file = "data/test.wav"
run(`sox -d -r 16k -c 1 --clobber $file trim 0 $duration`)
x = get_feats(file)
x = feats_post(x)

y = best_modely(Flux.unsqueeze(x,3))
gamma  = logposterior(size(y,1),a,A,y[:,:])
phones = posterior2phones(ippsym, gamma)
dec    = phones2words_greedy(ilexicon,phones; min_dist=2)

println("\nDecoded Phones")
println(strip(prod(phones.*" ")))
println("\nDecoded Digits")
println(strip(prod(dec.*" ")))

if plot_stuff
  using Plots
  p1 = heatmap(x', title="Input feats")
  p2 = heatmap(y[:,:]', clims = (-10,0), title=strip(prod(phones.*" ")))
  p3 = heatmap(gamma',  clims = (-10,0), title=strip(prod(dec.*" ")))
  plot(p1,p2,p3,layout=(3,1))
end
