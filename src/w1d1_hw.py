def main():

    # 1

    age = 32
    print(type(age))

    # 2

    height = 1.85
    print(type(height))

    # 3

    your_name, your_surname = 'Ilyas', 'Mustafazade'
    print(your_name, your_surname)
    
    # 4

    id_ = your_name + your_surname
    print(id_)

    # 5

    print(your_name[-1])

    # 6

    print(your_surname[1:3])

    # 7

    new_list = [3, 4, 5, 6, 7]
    print(new_list)

    # 8 

    new_list.remove(5)
    print(new_list)

    # 9

    del new_list[1]
    print(new_list)
    
    # 10

    new_tuple = (3, 4, 5, 6, 7)
    print(new_tuple)

    # 11

    as_list = list(new_tuple)
    as_list[new_tuple.index(4)] = 8
    new_tuple = tuple(as_list)
    print(new_tuple)

    # 12

    print(square_difference(42),
          square_difference(44),
          square_difference(41))

    # 13

    print(get_sign(13), get_sign(-13), get_sign(13.),
          get_sign(-13.), get_sign(0), get_sign(0.))


def square_difference(age):
    if not isinstance(age, int): raise Exception("age must be int")
    SUBTRACTED, LIMIT = 25, 17
    diff = age - SUBTRACTED
    if diff > LIMIT: return diff * diff


def get_sign(num):
    if not (isinstance(num, int) or isinstance(num, float)):
        raise Exception("argument must be numeric")
    
    # return 1 for positive, -1 for negative, None for zero 
    # Thought this way it would be easier to use returned values

    if num > 0: return 1
    elif num < 0: return -1


if __name__ == "__main__": main()
