# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

using Flux

struct FullyConnected{T<:AbstractFloat}
  M::Matrix{T}
  b::Matrix{T}
end

init_bias(Ny,Nx) = 
2/sqrt(Float32(first(Flux.nfan(Ny,Nx)))) * Float32.(rand(Ny) .- 0.5)

function FullyConnected(Nx::Int,Ny::Int)
  M = Flux.kaiming_uniform(Ny,Nx)'[:,:]
  b = reshape(init_bias(Ny,Nx),1,Ny)
  return FullyConnected(M,b)
end

# TODO: in future Flux versions this can be replaced by Dense
function (model::FullyConnected{T})(X::AbstractArray{T,3}) where {T}
  Nt, Nx, Nb = size(X)
  Ny = size(model.M,2)

  X = permutedims(X,(1,3,2))
  X = reshape(X,Nt*Nb,Nx)

  Y = X*model.M .+ model.b

  Y = reshape(Y,Nt,Nb,Ny)
  Y = permutedims(Y,(1,3,2))
  return Y
end
Flux.@functor FullyConnected

export get_convnet
function get_convnet(Nf,Ns;
                     Nhs=128*ones(Int,2),
                     Nks=[3,3],
                     strides=[1,3],
                     dilations=[1,2],
                     dropout=[0.0,0.0],
                     fout = x->logsoftmax(x,dims=2)
                    )
  T = Float32
  Nl = length(Nhs)
  @assert length(Nhs) == length(Nks) == length(strides) == length(dilations)
  convs = [Conv((Nks[i],), (i == 1 ? Nf : Nhs[i-1]) => Nhs[i],
                stride=strides[i],
                dilation=dilations[i],
                pad=SamePad(),
                init = Flux.kaiming_uniform,
                bias = init_bias(Nhs[i], i==1 ? Nf : Nhs[i-1]) 
               ) for i=1:Nl]
  bns = [BatchNorm(Nhs[i],relu) for i=1:Nl]
  dro =[Dropout(dropout[i],dims=2) for i=1:Nl]
  out = FullyConnected(Nhs[end],Ns)
  layers = []
  for i=1:Nl
    push!(layers,convs[i])
    push!(layers,bns[i])
    push!(layers,dro[i])
  end
  push!(layers,out)
  push!(layers,fout)
  return Chain(layers...)
end
