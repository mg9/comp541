"""
A class for providing pre-computed word representations.
Assumes the batch is constructed of loaded-from-disk
embeddings.
"""
struct DiskModel
    batch  # a batch of pre-computed embeddings loaded from disk.
end

