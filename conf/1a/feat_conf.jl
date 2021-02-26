Fs = 16000
Nf = 40 # input feat dim
# feature extraction (saved to data)
get_feats(x) = identity(x)
# on the fly feature processing
function feats_post(x)
  Fs_original=20000
  Fs=16000
  x = load(x)
  if typeof(x) <: Tuple
    x = x[1]
  else
    x = x.data
  end
  x = x[:]
  x = resample(x,Fs//Fs_original)
  x = mfcc(x, float(Fs); 
    dither=true, numcep=40, nbands=40, minfreq=20.0, maxfreq=7600.0)[1]
  return Float32.(x)
end
subsample = 3 # out subsampling
