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
  println("train until convergence... ")
  println("probe: ", probe)
  println("model: ", model)
  println("loss: ", loss)
  println("trn length: ", length(trn))
  println("dev length: ", length(dev))
end