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
  batchloss = 0.0
  numwords = 0
  for (id, sent) in collect(Iterators.take(dataset, batchsize))
    gold  = convert(_atype, sent.distances)
    embed = convert(_atype, sent.embeddings)
    pred  = probe(embed)  
    sent_numwords= size(gold,1)
    sent_loss  = sum(abs.(pred - gold))
    squared_length = abs2.(sent_numwords)
    normalized_sent_loss = sent_loss / squared_length
    batchloss += normalized_sent_loss
  end
  batchloss = batchloss / batchsize
  return batchloss
end



function train(probe, trn, dev)
  batchsize = 100
  trn = sort(collect(trn), by=x->x[1])
  iter = 0
  while iter < 100
    strn = Iterators.Stateful(trn)
    J = @diff loss(strn, batchsize)
    lossvalue = value(J)
    println("iteration: $iter, loss: $lossvalue")
    for par in params(probe)
      g = grad(J, par)
      update!(value(par), g, eval(Meta.parse("Adam()")))
    end
    iter += 1
  end
end



CONFIG_PATH = "config/prd_en_ewt-ud-sample.yaml"
args = YAML.load(open(CONFIG_PATH))
dataset = Dataset(args)
probe = choose_probe(args)
train(probe, dataset.trn, dataset.dev)
