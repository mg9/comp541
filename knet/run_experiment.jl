using Dates;

function sayhello(name)
    println("helloo to $name")
end

function setup_new_experiment_dir(args, yaml_args, reuse_results_path)
  #=	
      Constructs a directory in which results and params will be stored.
      If reuse_results_path is not None, then it is reused; no new
      directory is constructed.     
    Args:
    args: the command-line arguments:
    yaml_args: the global config dictionary loaded from yaml
    reuse_results_path: the (optional) path to reuse from a previous run.
  =#

    date_suffix = Dates.now() 
    model_suffix = join([yaml_args["model"]["model_type"] "-" yaml_args["probe"]["task_name"]])
    
    if isempty(reuse_results_path)
      new_root = reuse_results_path
      new_root = join([yaml_args["reporting"]["root"] "/" model_suffix "-" date_suffix])
      println("Constructing new results directory at $new_root")
    end
    yaml_args["reporting"]["root"] = new_root
    mkpath(new_root)

end

