# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

using FiniteStateTransducers

export get_L
# builds the L transducer
function get_L(lexicon::Dict{S,Vector{S}}) where {S<:AbstractString}
  phones = sort!(unique!(vcat(values(lexicon)...)))
  words = sort!([keys(lexicon)...])
  psym = Dict(p => i for (i,p) in enumerate(phones))
  wsym = Dict(w => i for (i,w) in enumerate(words ))

  L = WFST(psym,wsym)
  add_states!(L,2)
  initial!(L,1)
  final!(L,2)
  ϵ = get_eps(S)
  c = 3
  for w in keys(lexicon)
    pron=lexicon[w]
    for (i,p) in enumerate(pron)
      if i == 1 && (length(pron) != 1)
        add_arc!(L,1,c,p,w,1)
      elseif i == length(pron)
        if i == 1
          add_arc!(L,1,2,p,w)
        else
          add_arc!(L,c,2,p,ϵ)
          c += 1
        end
      else
        add_arc!(L,c,c+1,p,ϵ)
        c += 1
      end
    end
  end
  add_arc!(L,2,1,ϵ,ϵ)
  return L
end

export get_H
# builds the H transucer, 2 state phone per HMM
function get_H(psym; selfloop_prob=0.4)
  isym = Dict{String,Int}()
  c=1
  for p in sort([keys(psym)...])
    if p == "<SIL>"
      isym["$(p)1"] = c
      isym["$(p)2"] = c+1
      c +=2
    else
      for i=1:2
        isym["$p$i"] = c
        c+=1
      end
    end
  end
  Ns = length(psym)

  H = WFST(isym, psym)
  add_states!(H,Ns+1)
  initial!(H,1)
  ϵ = get_eps(String)
  for p in sort([keys(psym)...])
    if p == "<SIL>"
      # from initial state, assume silence
      add_arc!(H, 1  ,  isym["<SIL>1"]+1, "$(p)1", p)
    end
    # this avoids trivial solution of always staying in the same state
    add_arc!(H, isym["$(p)1"]+1, isym["$(p)1"]+1, "$(p)1", ϵ,-log(selfloop_prob))
    # prob of transistion to other state unknown, set to 1
    add_arc!(H, isym["$(p)1"]+1, isym["$(p)2"]+1, "$(p)2", ϵ)
    # to final state
    final!(H,isym["$(p)2"]+1)
  end
  for s in keys(get_final(H))
    for p in keys(psym)
      # prob of transistion to other phone unknown, set to 1
      add_arc!(H, s, isym["$(p)1"]+1, "$(p)1", p) # emitting state
    end
  end
  return H
end

export Hfst2trans
# convert the H transducer into transition matrix
function Hfst2trans(H::WFST)
  Ns = length(get_isym(H)) 
  A = zeros(Float32,Ns,Ns)
  state2outtr=Dict(i => (get_ilabel.(s),get_weight.(s)) for (i,s) in enumerate(H))
  for (p,s,n,d,e,a) in FiniteStateTransducers.DFS(H,1)
    if d
      intr = get_ilabel(a)
      outtr,w = state2outtr[n]
      for i in eachindex(outtr)
        A[intr,outtr[i]] = exp(-get(w[i]))
      end
    end
  end
  return A
end

export get_lexicon
function get_lexicon()
  lexicon = Dict(
                 "<SIL>" => ["<SIL>"],
                 "OH"  => ["OW"],
                 "ZERO" =>  ["Z", "IH", "R", "OW"],
                 "ONE" =>  ["W", "AH", "N"],
                 "TWO" =>  ["T", "UW"],
                 "THREE" => ["TH", "R", "IY"],
                 "FOUR"  => ["F", "AO", "R"],
                 "FIVE" =>  ["F", "AY", "V"],
                 "SIX" => ["S", "IH", "KS"],
                 "SEVEN"  => ["S", "EH", "V", "AH", "N"],
                 "EIGHT" => ["EY", "T"],
                 "NINE" =>  ["N", "AY", "N"]
                )
  ilexicon = Dict(lexicon[w] => w for w in keys(lexicon))
  return lexicon, ilexicon
end

export get_HL
function get_HL(lexicon)
  L = get_L(lexicon)
  H = get_H(get_isym(L))
  return H,L
end

export get_aA
function get_aA(H; use_log=true)
  A = Hfst2trans(H)
  Ns = size(A,1)
  a = zeros(Float32,Ns) # initial state prob
  a[H.isym["<SIL>1"]] = one(Float32)

  if use_log
    A .= log.(A)
    a .= log.(a)
  end
  return a,A
end
