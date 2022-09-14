# Deep Cluster

Deep cluster is a clustering technique that combines the Autoencoder and K-Means algorithms.

K-Means clustering is performed using a low dimensional latent-vector extracted from AE.

This provides better performance than K-Means clustering raw data.

Below are the results of benchmarking using the uncategorized MNIST and Fasion-MNIST datasets.

### MNIST Benchmark
| Class | KMeans | Deep Cluster |
|-|-|-|
0	| 0.89 | 0.89
1	| 0.99 | 0.89
2	| 0.70 | 0.87
3	| 0.63 | 0.82
4	| 0.54 | 0.96
5	| 0.00 | 0.39
6	| 0.83 | 0.90
7	| 0.60 | 0.84
8	| 0.59 | 0.77
9	| 0.00 | 0.00
avg	| 0.58 | 0.73
<br>

### Fashion MNIST Benchmark
| Class | KMeans | Deep Cluster |
|-|-|-|
0	| 0.56 | 0.83
1	| 0.90 | 0.90
2	| 0.00 | 0.57
3	| 0.00 | 0.87
4	| 0.59 | 0.64
5	| 0.62 | 0.00
6	| 0.34 | 0.00
7	| 0.78 | 0.80
8	| 0.77 | 0.91
9	| 0.93 | 0.97
avg	| 0.55 | 0.65

## Things we tried that didn't work

- Convolutional Autoencoder
	- CAE was used to extract a better latent vector, but it was not as good as Dense AE.

- Convolutional Encoder with Dense Decoder
	- Convolution and pooling were repeated to expand the filter's receptive field, and eventually the latent vector was extracted through Global Average Pooling.
	Dense Decoder was used to restore the extracted late vector to the original image.
	But the result was not as good as Dense AE.

- Dense Encoder with Convolutional Decoder
	- Likewise, the result was worse than Dense AE.

- Leaky or Parametric ReLU activation
	- First, we used an active function with a negative gradient throughout the model.
	- Second, we used an active function that had a negative gradient only at the output of the latent vector.
	- The above two attempts resulted in worse results than using the Vanilla ReLU active function on all layers.
