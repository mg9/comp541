using Dates, YAML
include("data.jl")
include("task.jl")
include("loss.jl")
include("reporter.jl")
include("probe.jl")
include("model.jl")
include("regimen.jl")


"""
Chooses which reporesentation learner to use based on config.
Args:
  args: the global config dictionary built by yaml.
Returns:
  A class to be instantiated as a model to supply word representations.
"""
function choose_model(args)
  if args["model"]["model_type"] == "ELMo-disk" || args["model"]["model_type"] == "BERT-disk"
    return DiskModel
  elseif args["model"]["model_type"] == "ELMo-random-projection"
    return ProjectionModel
  elseif args["model"]["model_type"] == "ELMo-decay"
    return DecayModel
  end
end



"""
Chooses which probe to use based on config.
Args:
  args: the global config dictionary built by yaml.
Returns:
  A probe_class to be instantiated.
"""
function choose_probe(args)
  maximum_rank = args["probe"]["maximum_rank"] 
  embed_dim = args["model"]["hidden_dim"]
  if args["probe"]["task_signature"] == "word"
    return OneWordPSDProbe
  elseif args["probe"]["task_signature"] == "word_pair"
    return TwoWordPSDProbe(embed_dim, maximum_rank)
  end
end




"""
Chooses which task and loss  to use based on config.
Args:
  args: the global config dictionary built by yaml.
Returns:
  A class to be instantiated as a task specification.
"""
function choose_task(args)
  
  if args["probe"]["task_name"] == "parse-distance"
    task = ParseDistanceTask
    #reporter_class = WordPairReporter
    if args["probe_training"]["loss"] == "L1"
      word_pair_dims = (1,2)
      loss = L1DistanceLoss(word_pair_dims)
    end
  #=  
  elseif args["probe"]["task_name"] == "parse-depth"
    task_class = ParseDepthTask
    #reporter_class = WordReporter
    if args["probe_training"]["loss"] == "L1"
      loss_class = L1DepthLoss
    end
  =#
  end
  return task, loss
end


"""
Trains a structural probe according to args.
Args:
  args: the global config dictionary built by yaml.
        Describes experiment settings.
  probe: An instance of probe.Probe or subclass.
        Maps hidden states to linguistic quantities.
  dataset: An instance of data.SimpleDataset or subclass.
        Provides access to DataLoaders of corpora. 
  model: An instance of model.Model
        Provides word representations.
  reporter: An instance of reporter.Reporter
        Implements evaluation and visualization scripts.
Returns:
  None; causes probe parameters to be written to disk.
"""

function run_train_probe(args, probe, dataset, model, loss)
  train_until_convergence(probe, model, loss,
      dataset.trn, dataset.dev)
end


  """
    execute_experiment(args, train_probe, report_results)

Execute an experiment as determined by the configuration in args.
# Arguments
- `train_probe`: Boolean whether to train the probe
- `report_results`: Boolean whether to report results
"""
function execute_experiment(args, train_probe, report_results)

  dataset = Dataset(args)
  task, loss = choose_task(args)
  probe = choose_probe(args)
  model = choose_model(args)

  if train_probe == 1
    print("Training probe...")
    run_train_probe(args, probe, dataset, model, loss) 
  end
end


  """
    setup_new_experiment_dir(args, yaml_args)

  Constructs a directory in which results and params will be stored.
  # Arguments
  - `args`: the command-line arguments
  - `yaml_args`: the global config dictionary loaded from yaml
  """

function setup_new_experiment_dir(args, yaml_args)
    date_suffix = Dates.now() 
    model_suffix = join([yaml_args["model"]["model_type"] "-" yaml_args["probe"]["task_name"]])
    root = join([yaml_args["reporting"]["root"] "/" model_suffix "-" date_suffix])
    yaml_args["reporting"]["root"] = root
    mkpath(root)
    println("Constructed new results directory at $root")
end


## RUN experiment
CONFIG_PATH = "example/config/prd_en_ewt-ud-sample.yaml"
args = YAML.load(open(CONFIG_PATH))
execute_experiment(args, 1, 1)
