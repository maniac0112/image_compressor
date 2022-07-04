import numpy as np
from skimage import io
from sklearn.cluster import KMeans
image = io.imread("img.jpg")
rows=image.shape[0]
cols=image.shape[1]
#flatten the image for input-ing it to kmeans 
image=image.reshape(rows*cols,3)
k=8 #number of Clusters
kmeans=KMeans(n_clusters=k) 
kmeans.fit(image)
img_compressed=kmeans.cluster_centers_[kmeans.labels_] #swapping actual rgb values of image of each pixel to its nearest cluster rgb value
img_compressed=np.clip(img_compressed.astype("uint8"),0,255) #converting the intensity of rgb all to int type (clusters have floating point co-ordinates)
img_compressed=img_compressed.reshape(rows,cols,3) #un-flatten the image
io.imsave("compressed_k_"+str(k)+".jpg", img_compressed)
