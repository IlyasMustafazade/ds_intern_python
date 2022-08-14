def main(): pass

# 1

def print_natural(start = 1, count = 10):
    if start < 1: raise Exception("natural number can not be less than 1")
    if not isinstance(start, int): raise Exception("natural number must be of type int")
    for i in range(start, start + count): print(i)

# 2

def print_negative():
    for i in range(-10, 0, 1): print(i)

# 3

def print_subset():
    for i in range(0, 7):
        if not (i == 3 or i == 6): print(i)

# 4

def natural_sum(lst = [i for i in range(2, 111)]): return sum(lst)

# 5 

def reverse_lst(lst = [10, 20, 30, 40, 50]):
    temp_lst = list()
    for i in range(len(lst) - 1, -1, -1): temp_lst.append(lst[i])
    return temp_lst

# 6

def filter_divisible(d_1 = 5, d_2 = 7,
                     lst = [i for i in range(1500, 2701)]):
    return list(filter(lambda x: \
        (x % d_1 == 0 and x % d_2 == 0), lst))

# 7 ?

def print_done():
    for i in range(10): pass
    print("Done")

# 8

add_scalar, multiply = lambda x: x + 15, lambda x, y: x * y

# 9

def power_lst(lst, power): return list(map(lambda x: (x ** power), lst))

def square_lst(lst): return power_lst(lst, 2)

def cube_lst(lst): return power_lst(lst, 3)

# 10

L = lambda x: x + 2

# 11

f = lambda x, y: x * y

# 12

def add_lst(lst_1, lst_2): return list(map(lambda x, y: x + y, lst_1, lst_2))

# 13

def add_scalar(lst, scalar = 5): return list(map(lambda x: x + scalar, lst))

# 14

def add_str(lst, str = "Hello, "): return list(map(lambda x: str + x, lst))

# 15

def len_lst(lst): return list(map(lambda x: len(x), lst))

# 16 same as 12 ?

# 17

def char_count_list(lst, char = 'a'): return list(map(lambda x: x.count(char), lst))

# 18

def insensitive_char_count_list(lst, char = 'a'): return list(map(lambda x: x.lower().count(char), lst))

# 19

def derive_negative(lst): return list(filter(lambda x: x < 0, lst)) 

# 20

def derive_odd(lst): return list(filter(lambda x: x % 2 == 1, lst)) 

# 21

def derive_vowel(str_): return list(filter(lambda x: x.lower() in ['a', 'e', 'i', 'o', 'u'], list(str_)))

# 22 

def derive_pos_int(str_): return list(filter(lambda x: str(x).isdigit(), list(str_)))

# 23

def add_scalar_if(lst, scalar = 2000): return list(map(lambda x: x + scalar if (x < 8000) else x, lst))

# 24

def count_even_odd(lst):
    lst_len, odd_lst = len(lst), list(filter(lambda x: x % 2 == 1, lst))
    odd_lst_len = len(odd_lst)
    return lst_len - odd_lst_len, odd_lst_len

# 25

def filter_by_len(lst): return list(filter(lambda x: len(x) == 6, lst))

# 26

def filter_by_divisibility(lst, d_1 = 13, d_2 = 19): return list(filter(lambda x: x % d_1 == 0 or x % d_2 == 0, lst))

# 27

def generate_merged_list(lst_1, lst_2): return list(zip(lst_1, lst_2))

# 28

def merge_list_range(lst): return list(zip(lst, range(1, 9)))

# 29

def generate_dict(lst_1, lst_2): return dict(zip(lst_1, lst_2))

# 30

def generate_sorted_merged_list(lst_1, lst_2): return list(zip(sorted(lst_1), sorted(lst_2)))

if __name__ == "__main__": main()
