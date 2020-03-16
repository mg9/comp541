using Dates, YAML, Knet
include("data.jl")
include("probe.jl")


function choose_probe(args)
  maximum_rank = args["probe"]["maximum_rank"] 
  embed_dim = args["model"]["hidden_dim"]
  if args["probe"]["task_signature"] == "word"
    return OneWordPSDProbe
  elseif args["probe"]["task_signature"] == "word_pair"
    return TwoWordPSDProbe(embed_dim, maximum_rank)
  end
end



function loss(dataset, batchsize)
  loss = 0.0
  numwords = 0
  for (id, sent) in collect(Iterators.take(dataset, batchsize))
    gold  = convert(_atype, sent.distances)
    embed = convert(_atype, sent.embeddings)
    pred  = probe(embed)  
    numwords += length(gold)
    loss  += sum(abs.(pred - gold))
  end
  loss /= numwords
  println("loss: ", loss)
  return -loss
end



function train(probe, trn, dev)
  println("trn length: ", length(trn))
  batchsize = 5
  strn = sort!(collect(trn), by = x -> length(trn[x[1]].observations))
  strn = Iterators.Stateful(strn)
  iter = 0
  while iter < 100
    J = @diff loss(strn, batchsize)
    for par in params(probe)
      g = grad(J, par)
      if isnothing(g) return; end
      update!(value(par), g, eval(Meta.parse("Adam()")))
    end
    iter += 1
  end
end



CONFIG_PATH = "example/config/prd_en_ewt-ud-sample.yaml"
args = YAML.load(open(CONFIG_PATH))
dataset = Dataset(args)
probe = choose_probe(args)
train(probe, dataset.trn, dataset.dev)
