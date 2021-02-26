
# Model Configuration
## obs likelihood model configuration
Random.seed!(1)
Nhs = [256,256,256,256] # hidden layer dims
dilations = [1,3,5,7]
Nks = 3 .*ones(Int,length(Nhs))         # conv kernel dims
strides = ones(Int,length(Nhs))
strides[end] = subsample                # output subsampling
dropout = zeros(length(Nhs))
fout = x -> logsoftmax(x,dims=2)        # function in last layer

# training opts
λ1 = 0f-5      # l1 output regularization
lr = 1f-4      # learning rate
Nb = 16        # batch size
epochs_cur = 5
epochs     = 15
opt = Flux.Optimise.Optimiser(WeightDecay(1e-5),ADAM(lr))
curriculum_training = true
