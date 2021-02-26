# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

using UUIDs

export zeropad
function zeropad(x::Vector{Matrix{T}}) where {T}
  Nt_max = maximum(size.(x,1))
  Nf = size(x[1],2)
  Nb = length(x)
  x_batched = zeros(T,Nt_max,Nf,Nb)
  for (i,xi) in enumerate(x)
    for f = 1:Nf, t = 1:size(xi,1)
      x_batched[t,f,i] = xi[t,f]
    end
  end
  return x_batched
end

export posterior2phones
function posterior2phones(ippsym, gamma)
  #z = [replace(ippsym[argmax(gamma[t,:])], r"[0-9\-]"=>"") for t in 1:size(gamma,1)]
  p = [ippsym[argmax(gamma[t,:])] for t in 1:size(gamma,1)] # all phones symbols
  k = [p[1]] # phones without repetitions
  for t in 2:length(p)
    if p[t-1] != p[t]
      push!(k,p[t])
    end
  end
  k2 = String[]
  for ki in k
    if ki[end] == '1' # emitting symbol
      push!(k2,replace(ki, r"[0-9\-]"=>""))
    end
  end
  return k2
end

export levenshtein
function levenshtein(s,t)
  n,m = length(s),length(t)
  D = zeros(Int,n+1,m+1)

  D[:,1] = 0:n # this is the cost we would have for insertion only
  D[1,:] = 0:m # this is the cost we would have for deletion only
  for i = 2:n+1, j = 2:m+1
    # check substition is needed
    cost = s[i-1] == t[j-1] ? 0 : 1

    D[i,j] = min(
                 D[i-1,j] + 1,        # del
                 D[i,j-1] + 1,        # ins
                 D[i-1,j-1] + cost,   # subs / ok
                )
  end
  return D[n+1,m+1]
end

export text2phones
function text2phones(lexicon,text; add_sil=true)
  if add_sil
    phones = [[lexicon[t]...,"<SIL>"] for t in split(text)]
  else
    phones = [lexicon[t] for t in split(text)]
  end
  phones = vcat(phones...)
  if add_sil
    phones = ["<SIL>",phones...]
  end
  return phones
end

export get_error_rate
function get_error_rate(uttID2seq::Dict,
    uttID2seq_dec::Dict; kwargs...)
  seqs, seq_decs = [], []
  for uttID in keys(uttID2seq)
    push!(seqs,uttID2seq[uttID])
    push!(seq_decs,uttID2seq_dec[uttID])
  end
  get_error_rate(seqs,seq_decs; kwargs...)
end

function get_error_rate(seqs::Vector,seq_decs::Vector; is_split=false)
  Nw = 0
  err = 0
  for i in eachindex(seqs)
    seq, seq_dec = seqs[i], seq_decs[i]
    if is_split == false
      seq     = split(seq;keepempty=false)
      seq_dec = split(seq_dec; keepempty=false)
    end
    Nw += length(seq)
    err += levenshtein(seq,seq_dec)
  end
  er = err/Nw
end

export min_dist_word
function min_dist_word(prons,min_dist,word_phones)
  d = [levenshtein(word_phones,pr) for pr in prons]
  idxs = findall(d .<= min_dist)
  if isempty(idxs)
    return "<?>"
  else
    return ilexicon[prons[idxs[argmin(d[idxs])]]]
  end
end

function add_word!(dec,prons,word_phones; min_dist=2) 
  t = try
    ilexicon[word_phones]
  catch
    if min_dist == 0
      "<?>"
    else
      min_dist_word(prons,min_dist,word_phones)
    end
  end
  push!(dec,t)
end

function phones2words_greedy(ilexicon,phones; min_dist=2)
  prons = [keys(ilexicon)...]
  word_phones = String[]
  dec = String[] 
  for (i,p) in enumerate(phones)
    if i == 1
      word_phones = String[]
      if p != "<SIL>"
        push!(word_phones,p)
      end
    elseif (p == "<SIL>") && i > 1
      if !isempty(word_phones)
        add_word!(dec,prons,word_phones; min_dist=min_dist)
        word_phones = String[]
      end
    else
      if p != "<SIL>"
        push!(word_phones,p)
      end
    end
  end
  if !isempty(word_phones)
    add_word!(dec,prons,word_phones; min_dist=min_dist)
  end
  return dec
end

export check_env
function check_env()
  if !("TIDIGITS_PATH" in keys(ENV))
    @warn "ENV[\"TIDIGITS_PATH\"] not exisitng: `export TIDIGITS_PATH=path/to/dataset` to your env."
  end
  if !("CPU_CMD" in keys(ENV))
    @warn "ENV[\"CPU_CMD\"] not exisitng: `export CPU_CMD='...'` to your env first. Only needed for SGE."
  end
end

export get_feat_dir
function get_feat_dir(setup; root="data")
  uuid_folder = UUID("04a07b93-95e4-4b85-94b9-d3516eb06ea2")
  conf = read("conf/$(setup)/feat_conf.jl",String)
  return joinpath("data", string(uuid5(uuid_folder,conf)))
end
