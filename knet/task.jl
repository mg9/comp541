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

