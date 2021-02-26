# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
    "--conf"
        help = "configuration setup"
        arg_type = String
        default = "1a"
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

# load training data
feat_dir = get_feat_dir(setup)
data = load(joinpath(feat_dir,"train.jld2"))
uttID2feats, uttID2tr, uttID2text, uttID2phones = 
data["uttID2feats"], data["uttID2tr"], data["uttID2text"], data["uttID2phones"]
# curriculum data (isolated words)
uttID2text_cur = filter(x->length(split(x.second))==1,uttID2text)

# load dev data
data = load(joinpath(feat_dir,"dev.jld2"))
uttID2feats_dev, uttID2tr_dev, uttID2text_dev, uttID2phones_dev =
data["uttID2feats"], data["uttID2tr"], data["uttID2text"], data["uttID2phones"]
# curriculum data (isolated words)
uttID2text_dev_cur = filter(x->length(split(x.second))==1,uttID2text_dev)

# dataloaders
Xs_cur  = [uttID2feats[uttID]     for uttID in keys(uttID2text_cur) ]
Ys_cur  = [uttID2tr[uttID]        for uttID in keys(uttID2text_cur) ]
Xs      = [uttID2feats[uttID]     for uttID in keys(uttID2feats)    ]
Ys      = [uttID2tr[uttID]        for uttID in keys(uttID2tr)       ]

Xs_test_cur = [uttID2feats_dev[uttID] for uttID in keys(uttID2text_dev_cur)]
Ys_test_cur = [uttID2phones_dev[uttID]  for uttID in keys(uttID2text_dev_cur)]
Xs_test = [uttID2feats_dev[uttID] for uttID in keys(uttID2feats_dev)]
Ys_test = [uttID2phones_dev[uttID]  for uttID in keys(uttID2feats_dev)]

N_cur  = length(Xs_cur)
N      = length(Xs)
N_test = length(Xs_test)

cur_data   = Flux.Data.DataLoader((Xs_cur ,Ys_cur ), batchsize=Nb, shuffle=false)
train_data = Flux.Data.DataLoader((Xs     ,Ys     ), batchsize=Nb, shuffle=true)
test_data_cur  = Flux.Data.DataLoader((Xs_test_cur,Ys_test_cur), batchsize=Nb)
test_data  = Flux.Data.DataLoader((Xs_test,Ys_test), batchsize=Nb)

# define maximum likelihood function
function loss(Nt,t2tr,A,x,λ1) 
  y  = modely(x)
  yp = exp.(y)
  f  = nlogMLlog(Nt,t2tr,A,y) +  λ1 * norm(yp,1)
  return f
end

function test(modely,a,A,ippsym,test_data)
  Flux.testmode!(modely)
  Nw = 0
  err = 0
  for (x,ps) in test_data
    x = feats_post.(x)
    Nts = ceil.(Int,size.(x,1)/3)
    xb = zeropad(x)
    y  = modely(xb)
    for i in eachindex(Nts)
      gamma = logposterior(Nts[i],a,A,view(y,:,:,i))
      ps_dec  = posterior2phones(ippsym,gamma)
      Nw  += length(ps[i])
      err += levenshtein(ps[i],ps_dec)
    end
  end
  per = err / Nw  
  Flux.trainmode!(modely)
  return per
end

function train!(modely,a,A,H,opt,λ1,epochs,train_data,test_data)
  Flux.trainmode!(modely)
  N = length(train_data.data[1]) 
  ps = Flux.params(modely)
  best_per = Inf
  best_modely = deepcopy(modely)
  ippsym = get_iisym(H)
  for e in 1:epochs
    cost = 0
    for (x,t2trs) in train_data
      x = feats_post.(x)
      Nts = length.(t2trs) .+ 1
      xb = zeropad(x)
      train_loss, back = 
      Zygote.pullback(() -> loss(Nts,t2trs,A,xb,λ1), ps)
      if isnan(train_loss) | isinf(train_loss)
        error("Nan/Inf cost function!!")
      end
      cost += train_loss
      gs = back(one(Float32))
      Flux.update!(opt, ps, gs)
    end
    per = test(modely,a,A,ippsym,test_data) 
    save_best = per <= best_per
    if save_best 
      best_modely = deepcopy(modely)
      best_per = per
      BSON.@save "models/$setup/current_modely.bson" best_modely
    end
    println("epoch: $e cost: $(round(cost/N,digits=4)) PER: $(round(per*100,digits=3))" * (save_best ? " ⋆ " : ""))
  end
  Flux.testmode!(best_modely)
  Flux.testmode!(modely)
  return best_modely, modely
end

model_folder = joinpath("models","$setup")
mkpath(model_folder)
println("Using setup: $setup")
println(read("conf/$setup/model_conf.jl",String))
if curriculum_training
  println("Curriculum training with $N_cur isolated words")
  best_modely, modely = train!(modely,a,A,H,opt,λ1,epochs_cur,cur_data,test_data_cur)
  BSON.@save joinpath(model_folder,"best_modely_curriculum.bson") best_modely
  modely = deepcopy(best_modely)
end
println("Training with $N utterances")
best_modely, modely = train!(modely,a,A,H,opt,λ1,epochs,train_data,test_data)
BSON.@save joinpath(model_folder,"best_modely_final.bson") best_modely
BSON.@save joinpath(model_folder,"modely.bson") modely
