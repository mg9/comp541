using Knet

_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})

struct Regimen
  max_epochs
  params_path
  optimizer
  scheduler
end


function Regimen(max_epochs::Int, params_path, optimizer, scheduler)
    return Regimen(max_epochs, params_path, optimizer, scheduler)
end

""" Trains a probe until a convergence criterion is met.

Trains until loss on the development set does not improve by more than epsilon
for 5 straight epochs.

Writes parameters of the probe to disk, at the location specified by config.

Args:
  probe: An instance of probe.Probe, transforming model outputs to predictions
  model: An instance of model.Model, transforming inputs to word reprs
  loss: An instance of loss.Loss, computing loss between predictions and labels
  train_dataset: a torch.DataLoader object for iterating through training data
  dev_dataset: a torch.DataLoader object for iterating through dev data
"""

function train_until_convergence(probe, model, loss, trn, dev)
  #println("trn length: ", length(trn))
  #println("dev length: ", length(dev))
  batchsize = 5
  strn = sort!(collect(trn), by = x -> length(trn[x[1]].observations))
  strn = Iterators.Stateful(strn)
  iter = 0
  while iter < 100
    J = @diff calc_loss(strn, batchsize)
    for par in params(probe)
      g = grad(J, par)
      if isnothing(g) return; end
      update!(value(par), g, eval(Meta.parse("Adam()")))
    end
    iter += 1
  end
end

function calc_loss(dataset, batchsize)
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


function distancelength(sent::Pair{Any,Any})
    return size(sent[2].distances,1)
end


