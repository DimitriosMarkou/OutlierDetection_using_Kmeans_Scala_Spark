# OutlierDetection-using-Kmeans (Scala-Spark)

This project utilizes a K-Means model to detect outliers in data.
After cleaning and scaling the dataset of 2D points, it creates clusters using a K-Means model with a large value of K.
For each point, it calculates the Euclidean distance from its center, and based on a threshold it determines which points are outliers.

According to the following plots, the value of K is set to 300.
<img src="plots_of_readme/sl.png" width="500">

<img src="plots_of_readme/sse.png" width="500">

Additionally with the threshold to be set to 0.2 the results are:

<img src="plots_of_readme/outliers.png" width="500">


