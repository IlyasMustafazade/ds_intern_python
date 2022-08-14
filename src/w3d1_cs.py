def main(): 

    car = Car()

    print(car.year)

    car.set_year(2001)

    print(car.year)

    Car.year = 2012

    print(car.year)

    car_2 = Car()

    print(car_2.year)

    car.accelerate(4)

    print(car.speed)

    print(car)

    bus = Bus()

    print(bus)

    bus.accelerate(100)

    # 6 (changed self.mpg -> self._mpg)

    bus._mpg = 10

    print(bus._mpg)

    print(bus)
    
# 4

class Vehicle():

    # 2

    def __init__(self, year=2000, mpg=0, speed=0):
        
        self.year = year

        self._mpg = mpg

        self.speed = speed


    def accelerate(self, change): self.speed += change

    def decelerate(self, change): self.speed -= change
     
    def set_year(self, year): self.year = year

    def set_mpg(self, mpg): self._mpg = mpg

    def set_speed(self, speed): self.speed = speed

    def __str__(self):

        return "vehicle of {0} with {1} mpg and {2} speed".format(self.year,
                 self._mpg, self.speed)


class Car(Vehicle):
    
    # 1
    
    year = 2000

    mpg = 0

    speed = 0

    def __init__(self):
        
        # 3

        self.year = Car.year

        self._mpg = Car.mpg

        self.speed = Car.speed

    def __str__(self):

        return "car of {0} with {1} mpg and {2} speed".format(self.year,
                 self._mpg, self.speed)

# 5

class Bus(Vehicle):

    def __str__(self):

        return "bus of {0} with {1} mpg and {2} speed".format(self.year,
                 self._mpg, self.speed)


if __name__ == "__main__": main()

