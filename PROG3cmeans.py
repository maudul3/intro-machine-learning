from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class Coordinate:
    # Coordinate class to store x,y values for gaussian coordinates and means
    # Includes the weights for each centroid

    def __init__(self, x, y, c):
        self.x = x
        self.y = y
        self.weights = [random() for _ in range(c)]

    # Update the weights associated with each centroid
    def update_weights(self, centroids):
        for jth_weight_idx, jth_centroid in enumerate(centroids):
            
            denominator = 0
            for kth_centroid in centroids:
                denominator +=  ( 
                     np.sqrt(( self.x - jth_centroid.mean.x )**2 + ( self.y - jth_centroid.mean.y )**2 )
                     / 
                    (np.sqrt(( self.x - kth_centroid.mean.x )**2 + ( self.y - kth_centroid.mean.y )**2 ) ) 
                ) ** ( 2 / ( m - 1 ) )
            
            self.weights[jth_weight_idx] = 1 / denominator


class Centroid:
    # Centroid class that keeps track of a mean and indexes of its datapoints
    def __init__(self):
        self.mean = Coordinate(2*random() - 1, 2*random() - 1, 0)
        self.datapoints_idx = []

    def reset_datapoints(self):
        self.datapoints_idx = []

    # Calculate the new mean of the centroid
    def calculate_new_mean(self, data, centroid_position_in_list):
        numerator_x = 0
        numerator_y = 0
        denominator_x = 0
        denominator_y = 0

        for idx in self.datapoints_idx:
            numerator_x += data[idx].weights[centroid_position_in_list] * data[idx].x ** (m + 1)
            numerator_y += data[idx].weights[centroid_position_in_list] * data[idx].y ** (m + 1) 
            denominator_x += data[idx].weights[centroid_position_in_list] * data[idx].x ** (m) 
            denominator_y += data[idx].weights[centroid_position_in_list] * data[idx].y ** (m) 

        length = len(self.datapoints_idx)
        if length != 0:
            self.mean = Coordinate(numerator_x/denominator_x, numerator_y/denominator_y, 0)

    def distance(self, coord):
        '''Determine the distance of a coordinate from centroid mean'''
        return (coord.x - self.mean.x)**2 + (coord.y - self.mean.y)**2

    def calculate_mean_squared_error(self, data):
        '''Calculate the MSE for this centroids'''
        mse = 0
        for idx in self.datapoints_idx:
            mse += (data[idx].x - self.mean.x)**2 + (data[idx].y - self.mean.y)**2 
        return mse
            
        
if __name__ == '__main__':

    '''C-means'''
    m = 2 # fuzziness parameter
    r = int(input("Enter the r value: ")) # number of iterations
    c = int(input("Enter the c value: ")) # number of centroids

    '''Initialize a list of centroids'''
    centroids = [Centroid() for _ in range(c)]

    '''Read the data into a list of coordinates'''
    data = []
    with open(
        "/Users/drewmahler/Desktop/School/CS545/CS545Code/545_cluster_dataset programming 3.txt", 'r'
    ) as f:
        for line in f.readlines():
            line = line.split()
            x, y = line
            data.append(Coordinate(float(x),float(y), c))
    
    # Run the c-means algorithm r times
    for iteration in range(r):
        
        # At beginning of each run reset datapoints associated with centroid
        for centroid in centroids:
            centroid.reset_datapoints()

        # Determine the weight values for each datapoint and assign max val to centroid
        for data_idx, coordinate in enumerate(data):
            coordinate.update_weights(centroids)
            max_weight = -1
            max_idx = -1
            for idx, weight in enumerate(coordinate.weights):
                if (weight > max_weight):
                    max_weight = weight 
                    max_idx = idx
            
            centroids[max_idx].datapoints_idx.append(data_idx)
        
        # Calculate the total mean squared error 
        total_mse = 0
        for position_in_list, centroid in enumerate(centroids):
            centroid.calculate_new_mean(data, position_in_list)
            total_mse += centroid.calculate_mean_squared_error(data)
        
        # Plot the data for this run
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

        plt.scatter(x=mean_x, y=mean_y, label="means", s=12, c="black", marker="x")
        plt.title("Iteration #{} with MSE: {}".format(iteration + 1, total_mse))

        plt.legend()
        filename = "cmeans_{}_clusters_at_iteration_{}".format(c, iteration)
        plt.savefig("/Users/drewmahler/Desktop/{}".format(filename))
        plt.clf()
        