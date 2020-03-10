
struct L1DistanceLoss
  word_pair_dims
end


""" Computes L1 loss on distance matrices.

Ignores all entries where label_batch=-1
Normalizes first within sentences (by dividing by the square of the sentence length)
and then across the batch.

Args:
  predictions:  A  batch of predicted distances
  label_batch:  A  batch of true distances
  length_batch: A  batch of sentence lengths

Returns:
  A tuple of:
    batch_loss: average loss in the batch
    total_sents: number of sentences in the batch
"""

function L1DistanceLoss(label_batch, predictions, length_batch)
    #print("label_batch: ", label_batch)
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0
      loss_per_sent = sum(abs(predictions_masked - labels_masked), dim=word_pair_dims)
      normalized_loss_per_sent = loss_per_sent / squared_lengths
      batch_loss = sum(normalized_loss_per_sent) / total_sents
    else
      batch_loss = 0.0
        #batch_loss = torch.tensor(0.0, device=self.args['device'])
    end
    return batch_loss, total_sents
end