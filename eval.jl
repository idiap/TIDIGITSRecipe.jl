# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--conf"
        help = "configuration setup"
        arg_type = String
        default = "2a"
end
parsed_args = parse_args(ARGS, s)
setup = parsed_args["conf"]

using HMMGradients, Flux, Zygote
using Random, Statistics, LinearAlgebra
using FiniteStateTransducers
using DSP, MFCC
using BSON, JLD2, LibSndFile, FileIO, UUIDs
include("WFSTs.jl")
include("Models.jl")
include("Utils.jl")

include("conf/$(setup)/feat_conf.jl")

# get transition matrix
lexicon, ilexicon = get_lexicon()
H, L = get_HL(lexicon)
a, A = get_aA(H)
ippsym = get_iisym(H)

BSON.@load "models/$setup/best_modely_final.bson" best_modely
Flux.testmode!(best_modely)

feat_dir = get_feat_dir(setup)
data = load(joinpath(feat_dir,"train.jld2"))
uttID2feats, uttID2text, uttID2phones = data["uttID2feats"], data["uttID2text"], data["uttID2phones"]

uttID2text_dec   = Dict()
uttID2phones_dec = Dict()
min_dist=2

for uttID in keys(uttID2feats)
  x = uttID2feats[uttID]
  x = feats_post(x)
  y = best_modely(Flux.unsqueeze(x,3))
  gamma = logposterior(size(y,1),a,A,y[:,:])
  ps  = posterior2phones(ippsym,gamma)
  ws  = phones2words_greedy(ilexicon,ps; min_dist=min_dist)
  uttID2phones_dec[uttID] = ps
  uttID2text_dec[uttID] = strip(prod(ws.*" "))
end

uttID2err_textdec = Dict{String,Tuple{String,String}}()
for uttID in keys(uttID2text)
  text, dec = uttID2text[uttID], uttID2text_dec[uttID]
  if text != dec 
    uttID2err_textdec[uttID] = (text,dec)
  end
end

accuracy = 1-length(uttID2err_textdec) / length(uttID2text)
wer = get_error_rate(uttID2text, uttID2text_dec)
per = get_error_rate(uttID2phones, uttID2phones_dec; is_split=true)
println("# Setup $setup")
println("* Phone Error Rate (PER): $(round(per * 100, digits=3)) %")
println("* Word Error Rate (WER) : $(round(wer * 100, digits=3)) %")
println("* Accuracy: $(round(accuracy,digits=3))")
