import numpy as np
import pandas as pd

class KMeans(object):

    def __init__(self, k, start_var, end_var, num_observations, data):
        """Class constructor for KMeans
        Arguments:
            k {int} -- number of clusters to create from the data
            start_var {int} -- starting index of the variables (columns) to
            consider in creating clusters from the dataset. This is
            useful for excluding some columns for clustering.
            end_var {int} -- ending index of the variables (columns) to
            consider in creating clusters from the dataset. This is
            useful for excluding some columns for clustering.
            num_observations {int} -- total number of observations (rows) in
            the dataset.
            data {DataFrame} -- the dataset to cluster
        """
        np.random.seed(1)
        self.k = k
        self.start_var = start_var
        self.end_var = end_var
        self.num_observations = num_observations
        self.columns = [i for i in data.columns[start_var:end_var]]
        self.centroids = pd.DataFrame(columns=self.columns)


    def initialize_centroids(self, data):
        """Returns initial centroids. This function picks a random point from
        the dataset as the first centroid, then iteratively picks points that
        are farthest from the current set of centroids.

        The algorithm for this initialization is as follows:
        1. Randomly select the first centroid from the data points in the
        dataset.
        2. For each data point, compute its distance from each centroid in the
        current set of centroids. For each distance computed from each
        centroid, retain only the shortest distance for each data point. In
        other words, we are computing the distance of each data point from
        the nearest centroid.
        3. Select the data point with the maximum distance from the nearest
        centroid as the next centroid.
        4. Repeat steps 2 and 3 until we have k number of centroids.

        Arguments:
            data {DataFrame} -- dataset to cluster
        Returns:
            DataFrame -- contains the values of the initial location of the
            centroids.
        """
        index = np.random.randint(low=0, high=self.num_observations)
        point = data.iloc[index, self.start_var:self.end_var]
        self.centroids = self.centroids.append(point, ignore_index=True)
        sliced_data = data.iloc[:, self.start_var:self.end_var]

        for i in range(1, self.k):
            distances = pd.DataFrame()
            for j in range(len(self.centroids)):
                distance = self.get_euclidean_distance(sliced_data, self.centroids.iloc[j])
                distances = pd.concat([distances, distance], axis=1)

            distances = distances.min(axis=1)
            index = distances.idxmax()
            point = data.iloc[index, self.start_var:self.end_var]
            self.centroids = self.centroids.append(point, ignore_index=True)

        return self.centroids

    def get_euclidean_distance(self, point1, point2):
        """Returns the Euclidean distance between two data points. These
        data points can be represented as 2 Series objects. This function can
        also compute the Euclidean distance between a list of data points
        (represented as a DataFrame) and a single data point (represented as
        a Series), using broadcasting.

        The Euclidean distance can be computed by getting the square root of
        the sum of the squared difference between each variable of each data
        point.

        For the arguments point1 and point2, you can only pass these
        combinations of data types:
        - Series and Series -- returns np.float64
        - DataFrame and Series -- returns pd.Series

        For a DataFrame and a Series, if the shape of the DataFrame is
        (3, 2), the shape of the Series should be (2,) to enable broadcasting.
        This operation will result to a Series of shape (3,)

        Arguments:
            point1 {Series or DataFrame} - data point
            point2 {Series or DataFrame} - data point
        Returns:
            np.float64 or pd.Series -- contains the Euclidean distance
            between the data points.
        """
        diff = point1 - point2
        squared = diff ** 2
        if len(squared.shape) == 1:
            axis = 0
        else:
            axis = 1
        summed = squared.sum(axis=axis)
        return np.sqrt(summed)

    def group_observations(self, data):
        """Returns the clusters of each data point in the dataset given
        the current set of clustroid. Suppose this function is given 100 data
        points to cluster into 3 groups, the function returns a Series of
        shape (100,), where each value is between 0 to 2.

        Arguments:
            data {DataFrame} -- dataset to cluster
        Returns:
            Series -- represents the cluster of each data point in the dataset.
        """
        distances = pd.DataFrame()
        sliced_data = data.iloc[:, self.start_var:self.end_var]
        for i in range(self.k):
            distance = self.get_euclidean_distance(sliced_data, self.centroids.iloc[i])
            distances[str(i)] = distance
        groups = distances.idxmin(axis=1)

        return groups.astype('int32')

    def adjust_centroids(self, data, groups):
        """Returns the new values for each centroid. This function adjusts
        the location of centroids based on the average of the values of the
        data points in their corresponding clusters.

        Arguments:
            data {DataFrame} -- dataset to cluster
            groups {Series} -- represents the cluster of each data point in the
            dataset.
        Returns:
            DataFrame -- contains the values of the adjusted location of the
            centroids.
        """
        grouped_data = pd.concat([data, groups.rename('group')], axis=1)
        grouped_data = grouped_data.groupby(['group'], as_index=False).mean()
        centroids = grouped_data.iloc[:, self.start_var+1:self.end_var+1]

        return centroids

    def train(self, data, iters):
        """Returns a Series which represents the final clusters of each data
        point in the dataset. This function stops clustering if one of the
        following is met:
        - The values of the centroids do not change.
        - The clusters of each data point do not change.
        - The maximum number of iterations is met.

        Arguments:
            data {DataFrame} -- dataset to cluster
            iters {int} -- maximum number of iterations before the clustering
            stops
        Returns:
            Series -- represents the final clusters of each data point in the
            dataset.
        """
        cur_groups = pd.Series(-1, index=[i for i in range(self.num_observations)])
        i = 0
        flag_groups = False
        flag_centroids = False

        while i < iters and not flag_groups and not flag_centroids:
            groups = self.group_observations(data)
            centroids = self.adjust_centroids(data, groups)

            diff_groups = pd.Series(np.where(groups != cur_groups, 1, 0)).sum()
            flag_groups = (diff_groups == 0)
            diff_centroids = (self.centroids - centroids)
            flag_centroids = (diff_centroids == 0).all(axis=None)

            cur_groups = groups
            self.centroids = centroids

            i+=1
            print('Iteration', i)

        print('Done clustering!')
        return cur_groups
