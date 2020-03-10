include("data.jl")

mutable struct ParseDistanceTask
    SentenceObservations
    distances
end

function ParseDistanceTask(sent_obs::SentenceObservations)

    sentence_length = length(sent_obs.Observations) #All observation fields must be of same length
    distances = zeros(sentence_length, sentence_length)
    
    for i in range(sentence_length)
      for j in range(i,sentence_length)
        i_j_distance = distance_between_pairs(sent_obs, i, j)
        distances[i][j] = i_j_distance
        distances[j][i] = i_j_distance
      end
    end

    ParseDistanceTask(sent_obs, distances)

end


function distance_between_pairs(sent_obs, i, j, head_indices=None)
    if i == j
      return 0
    end
    
    if observation
      head_indices = []
      number_of_underscores = 0
      for elt in observation.head_indices
        if elt == "_"
          head_indices.append(0)
          number_of_underscores += 1
        else
          head_indices.append(int(elt) + number_of_underscores)
        end
      end
    end

    i_path = [i+1]
    j_path = [j+1]
    i_head = i+1
    j_head = j+1

    while true
      if ! (i_head == 0 && (i_path == [i+1] || i_path[-1] == 0))
        i_head = head_indices[i_head - 1]
        i_path.append(i_head)
      end
      if ! (j_head == 0 && (j_path == [j+1] || j_path[-1] == 0))
        j_head = head_indices[j_head - 1]
        j_path.append(j_head)
      end
      if i_head in j_path
        j_path_length = j_path.index(i_head)
        i_path_length = length(i_path) - 1
        break
      elseif j_head in i_path
        i_path_length = i_path.index(j_head)
        j_path_length = length(j_path) - 1
        break
      elseif i_head == j_head
        i_path_length = length(i_path) - 1
        j_path_length = length(j_path) - 1
        break
      end
    end
    
    total_length = j_path_length + i_path_length
    return total_length
end