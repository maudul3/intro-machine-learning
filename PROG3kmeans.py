from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Coordinate:

    def __init__(self, x, y):
        self.x = x
        self.y = y

class Centroid:

    def __init__(self):
        self.mean = Coordinate(2*random() - 1, 2*random() - 1)
        self.datapoints_idx = []

    def reset_datapoints(self):
        self.datapoints_idx = []

    def calculate_new_mean(self, data):
        sum_x = 0
        sum_y = 0
        length = len(self.datapoints_idx)

        for idx in self.datapoints_idx:
            sum_x += data[idx].x
            sum_y += data[idx].y

        if length != 0:
            self.mean = Coordinate(sum_x/length, sum_y/length)

    def distance(self, coord):
        return (coord.x - self.mean.x)**2 + (coord.y - self.mean.y)**2

    def calculate_mean_squared_error(self, data):
        mse = 0
        for idx in self.datapoints_idx:
            mse += (data[idx].x - self.mean.x)**2 + (data[idx].y - self.mean.y)**2 
        return mse
            
        
if __name__ == '__main__':

    '''K-means'''
    r = int(input("Enter the r value: "))
    k = int(input("Enter the k value: ")) 

    centroids = [Centroid() for _ in range(k)]

    data = []
    with open(
        "/Users/drewmahler/Desktop/School/CS545/CS545Code/545_cluster_dataset programming 3.txt", 'r'
    ) as f:
        for line in f.readlines():
            line = line.split()
            x, y = line
            data.append(Coordinate(float(x),float(y)))
    
    for iteration in range(r):
        
        for centroid in centroids:
            centroid.reset_datapoints()

        for data_idx, coordinate in enumerate(data):
            min_distance = 10000
            min_idx = -1
            for centroid_idx, centroid in enumerate(centroids):
                distance = centroid.distance(coordinate)
                if (distance < min_distance):
                    min_distance = distance
                    min_idx = centroid_idx
            centroids[min_idx].datapoints_idx.append(data_idx)
        
        total_mse = 0
        for centroid in centroids:
            centroid.calculate_new_mean(data)
            total_mse += centroid.calculate_mean_squared_error(data)
        
        colors = cm.rainbow(np.linspace(0, 1, len(centroids)))

        for centroid_idx, (centroid, color) in enumerate(zip(centroids, colors)):
            current_x = []
            current_y = []
            for data_idx in centroid.datapoints_idx:
                current_x.append(data[data_idx].x)
                current_y.append(data[data_idx].y)

            plt.scatter(x=current_x, y=current_y, label=centroid_idx, s=2, c=color)

        mean_x = []
        mean_y = []

        for centroid in centroids:
            mean_x.append(centroid.mean.x)
            mean_y.append(centroid.mean.y)

        plt.scatter(x=mean_x, y=mean_y, label="means", s=12, c="black")
        plt.title("Iteration #{} with MSE: {}".format(iteration + 1, total_mse))

        plt.legend()
        plt.show()
        plt.clf()
        