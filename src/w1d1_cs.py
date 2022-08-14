def main(): calculator()

# 1

def addition_2number():
    n_1 = float(input("Enter first number: "))
    n_2 = float(input("Enter second number: "))
    return n_1 + n_2, 2

# 2

def addition():
    count, sum_ = 0, 0
    inp = float(input("Enter number: "))
    while inp != 0:
        count += 1
        sum_ += inp
        inp = float(input("Enter number: "))
    return sum_, count

# 3

def subtraction():
    count = 0
    inp = float(input("Enter number: "))
    result = inp
    while inp != 0:
        count += 1
        inp = float(input("Enter number: "))
        result -= inp
    return result, count

# 4

def multiplication():
    count = 0
    inp = float(input("Enter number: "))
    result = inp
    while inp != 0:
        count += 1
        inp = float(input("Enter number: "))
        if inp: result *= inp
    return result, count

# 5

def average():
    sum_, count = addition()
    if count == 0: return 0
    return round(sum_ / count, 3)

# 6

def calculator():
    resp = 'y'
    result = 0
    while resp != 'q':
        if resp == 'a': result = addition()[0]
        elif resp == 's': result = subtraction()[0]
        elif resp == 'm': result =  multiplication()[0]
        elif resp == 'av': result = average()
        elif resp == 'a2': result = addition_2number()[0]
        if not result is None: print("Result -> ", result)
        resp = input("Choose operation (type 'a' for addition, \
's' for subtraction, 'm' for multiplication, \
\n'av' for average, 'a2' for addition of two numbers, 'q' to quit): ")

# 7

def arr_intersection(arr_1, arr_2):
    if not (isinstance(arr_1, list) or isinstance(arr_1, tuple)) \
       or not (isinstance(arr_2, list) or isinstance(arr_2, tuple)):
       raise Exception("arguments must be of array-like type")
    result, result_copy = list(filter(lambda x: x in arr_1, arr_2)), list()
    result_reduced = list(set(result))
    arr_1_len, arr_2_len, result_len = len(arr_1), len(arr_2), len(result)
    for i in result_reduced:
        result_count_i, arr_1_count_i, arr_2_count_i = \
            result.count(i), arr_1.count(i), arr_2.count(i)
        if result_count_i > arr_1_count_i or result_count_i > arr_2_count_i:
            copy_count = min([arr_1_count_i, arr_2_count_i])
        else: copy_count = result_count_i
        result_copy.extend([i for j in range(copy_count)])
    return result_copy

# 8

def add_list(lst_1, lst_2):
    if len(lst_1) <= len(lst_2): lst_2 = lst_2[:len(lst_1)]
    else: lst_1 = lst_1[:len(lst_2)]
    return list(map(lambda a, b: a + b, lst_1, lst_2))

# 9

def get_divisible_sublist(lst, divisor_1 = 13, divisor_2 = 19):
    return [i for i in lst if ((i % divisor_1 == 0) or (i % divisor_2 == 0))]

if __name__ == "__main__": main()
