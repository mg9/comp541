include("run_experiment.jl")

#import Pkg; Pkg.add("ArgParse")
#import Pkg; Pkg.add("YAML")

using HDF5
using ArgParse
using YAML



function report_on_stdin(yaml_args)
#=Runs a trained structural probe on sentences piped to stdin.

  Sentences should be space-tokenized.
  A single distance image and depth image will be printed for each line of stdin.

  Args:
    args: the yaml config dictionary
=#

  read_representations()
end


function read_representations()
  reps = h5open("example/data/reps.h5", "r") do file
    read(file)
  end
  
  sent_rep = reps["dataset_1"]
  println("sent_rep: ", size(sent_rep))
end




yaml_args = YAML.load(open("example/demo-bert.yaml"))
cli_args = Dict()
results_dir = ""
setup_new_experiment_dir(cli_args, yaml_args, results_dir)
report_on_stdin(yaml_args)

