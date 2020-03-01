using Dates, YAML

CONFIG_PATH = "example/config/prd_en_ewt-ud-sample.yaml"


"""
    choose_model_class(args)

Chooses which reporesentation learner class to use based on config
and returns a class to be instantiated as a model to supply word reporesentations.
# Arguments:
- 'args': the global config dictionary built by yaml.
"""

function choose_model_class(args)
  modeltype = args["model"]["model_type"]
  
  if modeltype == "ELMo-disk"
    return model.DiskModel
  elseif modeltype == "BERT-disk"
    return model.DiskModel
  elseif modeltype == "ELMo-random-projection"
    return model.ProjectionModel
  elseif modeltype == "ELMo-decay"
    return model.DecayModel
  end

end



"""
    choose_probe_class(args)
Chooses which probe and reporter classes to use based on config
and returns a probe_class to be instantiated.
# Arguments:
- 'args': the global config dictionary built by yaml.
"""
function choose_probe_class(args)
  tasksignature = args["probe"]["task_signature"]
  psdparams = args["probe"]["psd_parameters"]

  if tasksignature == "word"
    return probe.OneWordPSDProbe
  elseif tasksignature == "word_pair"
    return probe.TwoWordPSDProbe
  end
  
end




"""
    choose_task_classes(args)

Chooses which task class to use based on config 
and returns a class to be instantiated as a task specification.
# Arguments
- 'args' : the global config dictionary built by yaml.
"""
function choose_task_classes(args)
  taskname = args["probe"]["task_name"]
  lossname = args["probe_training"]["loss"]
  
  if lossname == "L1"
    loss_class = loss.L1DistanceLoss
  end

  if taskname == "parse-distance"
    task_class = task.ParseDistanceTask
    reporter_class = reporter.WordPairReporter
  elseif taskname == "parse-depth"
    task_class = task.ParseDepthTask
    reporter_class = reporter.WordReporter
  end
  return task_class, reporter_class, loss_class
end 




"""
    choose_dataset_class(args)

Chooses which dataset class to use based on config 
and returns a class to be instantiated as dataset.
# Arguments
- 'args': the global config dictionary built by yaml.
"""
function choose_dataset_class(args)
  modeltype = args["model"]["model_type"]
 
  if modeltype in ["ELMo-disk", "ELMo-random-projection", "ELMo-decay"]
    dataset_class = data.ELMoDataset
  elseif modeltype == "BERT-disk"
    dataset_class = data.BERTDataset
  end
  return dataset_class
end




"""
    execute_experiment(args, train_probe, report_results)

Execute an experiment as determined by the configuration in args.
# Arguments
- `train_probe`: Boolean whether to train the probe
- `report_results`: Boolean whether to report results
"""
function execute_experiment(args, train_probe, report_results)

  dataset_class = choose_dataset_class(args)
  task_class, reporter_class, loss_class = choose_task_classes(args)
  probe_class = choose_probe_class(args)
  model_class = choose_model_class(args)
  regimen_class = regimen.ProbeRegimen

  task = task_class()
  expt_dataset = dataset_class(args, task)
  expt_reporter = reporter_class(args)
  expt_probe = probe_class(args)
  expt_model = model_class(args)
  expt_regimen = regimen_class(args)
  expt_loss = loss_class(args)

  if train_probe == 1
    print("Training probe...")
    run_train_probe(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)
  end

  if report_results
    print("Reporting results of trained probe...")
    run_report_results(args, expt_probe, expt_dataset, expt_model, expt_loss, expt_reporter, expt_regimen)
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
yaml_args = YAML.load(open(CONFIG_PATH))
execute_experiment(yaml_args, 1, 1)
