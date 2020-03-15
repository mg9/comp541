include("probe.jl")

#import Pkg; Pkg.add("ArgParse"); Pkg.add("YAML")
using HDF5, ArgParse, YAML


SENTREPS_PATH = "example/data/sentreps.h5"
CONFIG_PATH = "example/demo-bert.yaml"
TWO_WORD_PROBE_PARAMS_PATH = "example/data/distanceprobe.h5"
ONE_WORD_PROBE_PARAMS_PATH = "example/data/depthprobe.h5"


"""
    report_on_stdin(yaml_args)

Runs a trained structural probe on sentences piped to stdin.
Sentences should be space-tokenized.
A single distance image and depth image will be printed for each line of stdin.
# Arguments
- `args`: the yaml config dictionary
"""
function report_on_stdin(yaml_args)
  distance_probe = loadTwoWordPSDProbe(loadparams(TWO_WORD_PROBE_PARAMS_PATH))
  depth_probe = loadOneWordPSDProbe(loadparams(ONE_WORD_PROBE_PARAMS_PATH))
  sentrep = loadsentencerepresentations()
  
  # Run BERT token vectors through the trained probes
  distance_predictions = distance_probe(sentrep)
  depth_predictions = depth_probe(sentrep)

  println("size(distance_predictions): ", size(distance_predictions))
  #println("distance_predictions: ", distance_predictions)
  println("depth_predictions: ", depth_predictions)
end


function loadparams(path)
  mparams = h5open(path, "r") do file
    read(file)
  end
  return mparams["proj"]
end


function loadsentencerepresentations()
  sentreps = h5open(SENTREPS_PATH, "r") do file
    read(file)
  end
  return sentreps["dataset_1"]
end



## RUN demo
yaml_args = YAML.load(open(CONFIG_PATH))
#setup_new_experiment_dir(Dict(), yaml_args)
report_on_stdin(yaml_args)

