def main():

    vehicle, car, train = Vehicle(capacity=100), Car(200, "red", 4), Train(100, "gray", 100)

    object_lst = [vehicle, car, train]

    for i in object_lst:

        print('{0}\ngeneral property -> {1}\nfare -> {2}\n'.format(str(i),
                        i.general_property, i.get_fare()))
                        
    # 6

    if isinstance(car, Car): print("car is instance of Car")

    else: print("car is not instance of Car")

    if isinstance(train, Train): print("train is instance of Train")

    else: print("train is not instance of Train")




class Vehicle():

    # 4

    general_property = "white"

    # 1
    
    def __init__(self, maxspeed=0, color="black",
                 capacity=0):

        self.maxspeed = maxspeed

        self.color = color
        
        self.capacity = capacity
    
    # 2

    def __str__(self):

        return str(self.color) + " vehicle with " + str(self.maxspeed) + " max speed"


    def get_fare(self): return self.capacity * 35

# 3

class Car(Vehicle):

    pass

# 5

class Train(Vehicle):

    def get_fare(self): return self.capacity * 35 * 1.1
    

if __name__ == "__main__": main()

