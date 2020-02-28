using Dates;


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

