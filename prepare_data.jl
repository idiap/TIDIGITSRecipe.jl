# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

using ArgParse
using Distributed, ClusterManagers
include("Utils.jl")

function parse_commandline()
  s = ArgParseSettings()
  @add_arg_table! s begin
    "--nj"
    help = "number of jobs"
    arg_type = Int
    default = 1
    "--conf"
    help = "configuration setup"
    arg_type = String
    default = "2a"
  end
  return parse_args(ARGS, s)
end

# parse command line and add workers
parsed_args = parse_commandline()
nj, setup = parsed_args["nj"], parsed_args["conf"]
feat_dir = get_feat_dir(setup)
check_env()
if ispath(feat_dir)
  error("Data already processed for this feature conf in $(feat_dir). Remove this folder to re-run feature extraction from scratch.")
end

if nj > 1
  addprocs_sge(nj;
               qsub_flags=split(ENV["CPU_CMD"]), 
               wd=mktempdir(pwd()),
               exeflags="--project"
              )
end

@everywhere begin
  setup = $setup
  using DSP, MFCC, HMMGradients, FiniteStateTransducers
  using JLD2, LibSndFile, FileIO
  include("Utils.jl")
  include("WFSTs.jl")
  include("DatasetParsers.jl")
end

@everywhere function process_data(dataset_path,lexicon,L,H,Fs,subsample,
    uttID2file,uttID2text,feat_dir,set,nj)
  T = Float32

  uttID2feats  = Dict()
  uttID2phones = Dict()
  uttID2tr     = Dict{String,Vector{Pair{Vector{Int},Vector{Int}}}}()
  wsym = get_osym(L)

  for uttID in keys(uttID2file)
    # process audio
    x = uttID2file[uttID]
    x = get_feats(x) 
    uttID2feats[uttID] = x
    x = feats_post(x) 
    Nt = size(x,1)

    # process text
    text = uttID2text[uttID]
    uttID2phones[uttID] = text2phones(lexicon,text)
    text = split(text;keepempty=false)
    text = String.(vcat("<SIL>",[[ti,"<SIL>"] for ti in text]...)) #silence between every word
    S = linearfst(text,text, ones(typeofweight(L),length(text)), wsym, wsym)
    HLS = rm_eps!(H∘(L∘S))
    Nt2 = subsample == 1 ? Nt : ceil(Int,Nt/3)
    time2tr = wfst2tr(HLS,Nt2)
    uttID2tr[uttID] = HMMGradients.t2tr2t2IJ(time2tr)
  end
  if nj > 1
    q = joinpath(feat_dir,"q_split_$set")
    mkpath(q)
    JLD2.@save joinpath(q,"$(myid()).jld2") uttID2feats uttID2tr uttID2phones
  else
    return uttID2feats, uttID2tr, uttID2text, uttID2phones
  end
end

function prepare_data(dataset_path,lexicon,L,H,Fs,subsample,set,feat_dir,nj)
  println("Processing $set set with $nj jobs")
  uttID2file  = get_uttID2file(dataset_path,set)
  uttID2text  = get_uttID2text(uttID2file)
  if nj > 1
    # split utterances
    uttIDs = [keys(uttID2text)...]
    Nu = length(uttIDs)
    delta = div(Nu,nj)
    uttIDss = [uttIDs[1+(i-1)*delta:(i==nj ? Nu : i*delta)] for i = 1:nj]
    uttID2files = [filter(x -> x.first in uttIDs, uttID2file) for uttIDs in uttIDss]
    uttID2texts = [filter(x -> x.first in uttIDs, uttID2text) for uttIDs in uttIDss]
    pmap(
         uttID2filetext ->
         process_data(dataset_path,lexicon,L,H,Fs,subsample,
                      uttID2filetext[1],uttID2filetext[2],
                      feat_dir,set,nj),
         zip(uttID2files,uttID2texts)
        )
    uttID2feats  = Dict()
    uttID2phones = Dict()
    uttID2tr     = Dict{String,Vector{Pair{Vector{Int},Vector{Int}}}}()
    println("Merging files")
    q = joinpath(feat_dir,"q_split_$set")
    for id in workers()
      data = load(joinpath(q,"$id.jld2"))
      uttID2feats_nj, uttID2tr_nj, uttID2phones_nj = 
      data["uttID2feats"], data["uttID2tr"], data["uttID2phones"]
      merge!(uttID2feats , uttID2feats_nj )
      merge!(uttID2phones, uttID2phones_nj)
      merge!(uttID2tr    , uttID2tr_nj    )     
    end
    rm(q;recursive=true)
    return uttID2feats, uttID2tr, uttID2text, uttID2phones
  else
    process_data(dataset_path,lexicon,L,H,Fs,subsample,uttID2file,uttID2text,feat_dir,set,nj)
  end
end

###
@everywhere begin
  include("conf/$(setup)/feat_conf.jl")
  lexicon, ilexicon = get_lexicon()
  H, L = get_HL(lexicon)
end

dataset_path = ENV["TIDIGITS_PATH"]
T = @elapsed uttID2feats_train, uttID2tr_train, uttID2text_train, uttID2phones_train =
prepare_data(dataset_path,lexicon,L,H,Fs,subsample,"train",feat_dir,nj)
println("Done in $T sec")
T = @elapsed uttID2feats_test, uttID2tr_test, uttID2text_test, uttID2phones_test =
prepare_data(dataset_path,lexicon,L,H,Fs,subsample,"test",feat_dir,nj)
println("Done in $T sec")

if nj > 1
  t = rmprocs(workers())
  wait(t)
end

# test data in TIDIGITS has same size of train, so we repartition it
uttID2feats_all = merge(uttID2feats_train , uttID2feats_test )
uttID2tr_all    = merge(uttID2tr_train    , uttID2tr_test    )
uttID2text_all  = merge(uttID2text_train  , uttID2text_test  )
uttID2phones_all= merge(uttID2phones_train, uttID2phones_test)
uttIDs_all = [keys(uttID2text_all)...]
Nu = length(uttIDs_all)
idx_train, idx_test = round(Int,Nu*0.7), round(Int,Nu*0.9)
set2uttID  = Dict()
set2uttID["train"], set2uttID["test"], set2uttID["dev"] = uttIDs_all[1:idx_train], uttIDs_all[idx_train+1:idx_test], uttIDs_all[idx_test+1:end] 

mkpath(feat_dir)
for set in ("train","test","dev")
  filename = set
  uttID2feats   = filter(x -> x.first in set2uttID[set], uttID2feats_all )
  uttID2tr      = filter(x -> x.first in set2uttID[set], uttID2tr_all    )
  uttID2text    = filter(x -> x.first in set2uttID[set], uttID2text_all  )
  uttID2phones  = filter(x -> x.first in set2uttID[set], uttID2phones_all)
  JLD2.@save joinpath(feat_dir,"$set.jld2") uttID2feats uttID2tr uttID2text uttID2phones
end
