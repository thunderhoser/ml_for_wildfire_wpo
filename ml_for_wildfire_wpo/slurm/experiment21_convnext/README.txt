Sampling strategy during training:
 - Each microbatch contains 1 patch
 - Each minibatch contains 88 patches.  On average, these 88 patches come from 10 different days.  I sample patches with a sampling rate of 0.2, so 80% are skipped.
 - Each training epoch contains 10 minibatches
 - Each validation epoch contains 3 minibatches (still 264 patches, from an average of 30 different days)
