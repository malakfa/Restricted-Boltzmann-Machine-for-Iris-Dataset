import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
import pandas as pd
import copy

class RBM:
    def __init__(self, num_visible, num_hidden):
        self.num_hidden = num_hidden
        self.num_visible = num_visible

        # Initialize weights matrix and bias vectors
        self.weights = np.random.rand(num_visible, num_hidden)
        self.a = np.random.rand(num_visible)
        self.b = np.random.rand(num_hidden)

        # Initialize arrays for visible and hidden neurons with zeros
        self.visible_neurons = np.zeros(num_visible)
        self.hidden_neurons = np.zeros(num_hidden)

    def update_hidden_neurons(self, T):
        num_neurons_changed = 0

        for i in range(self.num_hidden):
            delt_E = self.b[i] + sum(self.weights[j][i] * self.visible_neurons[j] for j in range(self.num_visible))
            probability = 1 / (1 + math.exp(-delt_E / T))

            # Check if the random number is less than pk
            if random.random() < probability:
                if self.hidden_neurons[i] == 0:
                    num_neurons_changed += 1
                self.hidden_neurons[i] = 1
            else:
                if self.hidden_neurons[i] == 1:
                    num_neurons_changed += 1
                self.hidden_neurons[i] = 0

        return num_neurons_changed, delt_E

    def update_visible_neurons(self, input, T):
        num_neurons_changed = 0

        for i in range(self.num_visible):
            # The visible neurons that are not contiguous to input
            if i not in input.keys():
                delt_E = self.a[i] + sum(self.weights[i][j] * self.hidden_neurons[j] for j in range(self.num_hidden))
                probability = 1 / (1 + math.exp(-delt_E / T))

                # Check if the random number is less than pk
                if random.random() < probability:
                    if self.visible_neurons[i] == 0:
                        num_neurons_changed += 1
                    self.visible_neurons[i] = 1
                else:
                    if self.visible_neurons[i] == 1:
                        num_neurons_changed += 1
                    self.visible_neurons[i] = 0


        return num_neurons_changed ,  delt_E

    # the input is from the shape -> dictinary : {[index : value]} the index in the array visible_neurons
    def conclusion_algo(self , input):

        self.initialization(input)

        T = 5
        stopping_condition = False
        iterations = 0
        max_iterations = 1000 
        previous_total_energy = float('inf')

        while not stopping_condition and iterations < max_iterations:
            iterations += 1
            num_neurons_changed, delt_E = self.update_hidden_neurons(T)


            num_neurons , energy = self.update_visible_neurons(input, T)
            num_neurons_changed += num_neurons
            delt_E += energy


            # Check stopping conditions
            if num_neurons_changed < 0.01 * (self.num_hidden + self.num_visible):
                stopping_condition = True
            
            # Check stopping conditions
            if abs(delt_E) < 0.001 :
                stopping_condition = True

            # Update temperature parameter T 
            T *= 0.99
  
    def initialization(self,input):
        for i in range(self.num_visible) :
            if i in input.keys():
                self.visible_neurons[i] = input[i]
            else : 
                self.visible_neurons[i] = random.choice([0, 1])
            
        for i in range(self.num_hidden):
            self.hidden_neurons[i] = random.choice([0, 1])

    def learning_algo(self):
        n = 1
        T = 5
        stopping_condition = False
        iterations = 0
        max_iterations = 1000 

        while not stopping_condition and iterations < max_iterations:
            iterations += 1 

            # Choose a random sample from the training data
            random_line_number = random.randint(1, 100)
            df = pd.read_csv('train_dataset.csv')

            # Extract the line corresponding to the random number 
            self.visible_neurons = df.loc[random_line_number - 1].to_numpy()
            print(self.visible_neurons)
        
            #The activation probability of each neuron in the hidden layer
            probabilities = self.calculate_prob()
        
            prev_visible_neurons = copy.deepcopy(self.visible_neurons)

            #A single iteration of the conclusion algorithm
            self.update_hidden_neurons(T)
            self.update_visible_neurons({}, T)

            # Updating the parameters
            count = self.update_par(prev_visible_neurons , n , probabilities)

            # Update temperature parameter T 
            T *= 0.99

            # Check stopping conditions
            if count == (self.num_visible + self.num_hidden + self.num_visible*self.num_hidden) :
                stopping_condition = True

    def update_par(self,prev_visible_neurons,n,probabilities):
        count = 0
        for i in range(self.num_visible):
            prev_a = self.a[i]
            self.a[i] = self.a[i] + n * (prev_visible_neurons[i] - self.visible_neurons[i])
            if abs(prev_a - self.a[i]) < 0.001:
                count += 1

        for j in range(self.num_hidden):
            prev_b = self.b[j]
            self.b[j] = self.b[j] + n * (probabilities[j] - self.hidden_neurons[j])
            if abs(prev_b - self.b[j]) < 0.001:
                count += 1

        for i in range(self.num_visible):
            for j in range(self.num_hidden):
                prev_w = self.weights[i][j]
                num = prev_visible_neurons[i] * probabilities[j] - self.visible_neurons[i] * self.hidden_neurons[j]
                self.weights[i][j] = self.weights[i][j] + n * num
                if abs(prev_w - self.weights[i][j]) < 0.001:
                    count += 1
        return count

    def calculate_prob(self):
        #The activation probability of each neuron in the hidden layer
        probabilities = []
        for k in range(self.num_hidden):

            s = sum(self.weights[i][k]*self.visible_neurons[i] for i in range(self.num_visible))
            p = 1 / (1 + math.exp(-self.b[k] - s))

            probabilities.append(p)
        
        return probabilities

def split_data():

    # Load the dataset
    df = pd.read_csv('binarize_data.csv')

    # Drop the 'Number' column
    df = df.drop(columns=['Dataset order'])

    # Split the data into training (2/3) and testing (1/3) sets
    train_df, test_df = train_test_split(df, test_size=0.33, random_state=42)

    # Save the new datasets to CSV files
    train_df.to_csv('train_dataset.csv', index=False)
    test_df.to_csv('test_dataset.csv', index=False)

if __name__ == '__main__':
    rbm = RBM(15 , 10)
    # F question , first row in data 
    print("before training")
    rbm.conclusion_algo({0 : 1, 1 : 0, 2 : 0, 3 : 1, 4 : 0, 5 : 0, 6 : 1, 7 : 0, 8 : 0, 9 : 1, 10 : 0, 11 : 0})
    print(rbm.visible_neurons)
    #rbm.conclusion_algo({0 : 1, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 1, 6 : 1, 7 : 0, 8 : 0, 9 : 1, 10 : 0, 11 : 0})
    #print(rbm.visible_neurons)
    split_data()

    rbm.learning_algo()

    print("after training")
    rbm.conclusion_algo({0 : 1, 1 : 0, 2 : 0, 3 : 1, 4 : 0, 5 : 0, 6 : 1, 7 : 0, 8 : 0, 9 : 1, 10 : 0, 11 : 0})
    print(rbm.visible_neurons)
    #rbm.conclusion_algo({0 : 1, 1 : 0, 2 : 0, 3 : 0, 4 : 0, 5 : 1, 6 : 1, 7 : 0, 8 : 0, 9 : 1, 10 : 0, 11 : 0})
    #print(rbm.visible_neurons)



