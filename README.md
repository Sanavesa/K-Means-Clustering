# K-Means Clustering Applications
This project showcases some examples of using K-Means to classify hand-written digits or to perform lossy image compression using NumPy. While this clustering algorithm is simple, it has a wide range of applications!

## Classify Hand-Written Digits
K-means is a clustering algorithm, so it is not obvious on how it may classify hand-written digits. Having said that, it can be used as a faster nearest neighbor algorithm. Similar to K-Nearest Neighbors, classifying with K-means compresses the entire dataset into K centroids, which in turn speeds up the algorithm since finding the nearest neighbor is reduced from <code>O(N)</code> to <code>O(K)</code>. For more clarification, please refer to the pseudocode below:

All in all, the results of this classifier were compared to other baseline classifiers such as logistic regression classifier and nearest neighbor classifier.

| Classifier | Accuracy |
| ---------- | -------- |
| K-means | 79% |
| Logistic Regression | 97% |
| K-Nearest Neighbors | 99% |

At first glance, the K-means classifier is severely underperforming; however, I believe with more optimization, it is promising route as it is relatively faster than the other classifiers in prediction.

## Image Compression with K-Means
This is a perfect application of K-means, even though it may be obscure. This is done by treating each pixel of an image as a 3D point, then performing K-means algorithm to cluster these points. Next, you replace each pixel with its nearest centroid, which is also a 3D point. The great thing about this way of compression, is that you can clearly control the rate of compression: the less the clusters, the greater the compression. Here is a compressed baboon image, using 16 clusters:

| Before (612 KB) | After (168 KB) |
| --------------- | -------------- |
| ![baboon](https://user-images.githubusercontent.com/31108136/149611147-22599ac6-a169-46a0-8a75-475984c1ffbb.png) | ![compressed_baboon](https://user-images.githubusercontent.com/31108136/149611158-8c8b8597-36fd-4559-9dcd-9257ea66d854.png) |

Can you tell the difference? It is minor, most notably on the bottom left region is where the biggest impact is; however, the image was compressed by roughly 75%!
