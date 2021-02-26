# Copyright (c) 2021 Idiap Research Institute, http://www.idiap.ch/
#  Niccolò Antonello <nantonel@idiap.ch>

export get_uttID2file
function get_uttID2file(dataset_path,folder::String)
  uttID2file = Dict{String,String}()
  for (root,dir,files) in walkdir(joinpath(dataset_path,folder);follow_symlinks=true)
    wavs = files[findall(contains.(files,".wav"))]
    folders = split(root, "/")
    spkID = folders[end]
    type = folders[end-1]
    for f in wavs
      sentenceID = split(f, "."; limit=2)[1]
      uttID = "$(spkID)-$(type)-$(sentenceID)"
      uttID2file[uttID] = joinpath(root,f)
    end
  end
  return uttID2file
end

export get_uttID2text
function get_uttID2text(uttID2file::Dict)
  d = Dict(
           'z' => "ZERO", '3' => "THREE", '7' => "SEVEN",
           'o' => "OH",   '4' => "FOUR",  '8' => "EIGHT",
           '1' => "ONE",  '5' => "FIVE",  '9' => "NINE",
           '2' => "TWO",  '6' => "SIX",   'a' =>"",  'b'=>"")
  uttID2text = Dict{String,String}()
  for uttID in keys(uttID2file)
    text = split(uttID,"-")[3]
    try
      uttID2text[uttID] = strip(prod([d[t] for t in text].*" "))
    catch
      error("$text is an invalid filename, invalid dataset!")
    end
  end
  return uttID2text
end

function get_uttID2wav(uttID2file::Dict; T=Float32)
  uttID2wav = Dict(uttID => T.(load(uttID2file[uttID]).data)[:]
                   for uttID in keys(uttID2file))
end
