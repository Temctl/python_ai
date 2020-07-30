from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt # graph the model

class Regression_model:


    def __init__(self, dataset, x, y):
        self.dataset = dataset
        self.x = dataset[x].values.reshape(-1, 1)
        self.y = dataset[y].values
        self.lin_reg = LinearRegression().fit(self.x, self.y)

    def graph(self):
        plt.scatter(self.x, self.y, color = "blue")
        plt.plot(self.x, self.lin_reg.predict(self.x), color = "red", linewidth = 3)
        plt.show()

    def predict(self):
        done = False
        print("type in quit to stop")
        while(not done):
            user_input = input("predict y, type in the x: ")
            if(user_input == "done"):
                done = True
            else:
                print(self.lin_reg.predict(user_input))

        
