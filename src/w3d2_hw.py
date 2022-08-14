import pandas as pd

def main(): pass

# 1

FIFA = pd.read_csv("fifa19.csv")

# 2

def print_head(): print(FIFA.head(12))

# 3

def print_info(): FIFA.info()

# 4

def print_n_observation(): print(len(FIFA))

# 5

def print_col_names(): print(list(FIFA.columns))

# 6

def print_observation(): print(FIFA.iloc[4999])

# 7

def select_col(col_name='ID'): return FIFA[col_name]

# 8

def sort_by_col(): return FIFA.sort_values(by=['Club', 'Name'])

# 9

def filter_high_score(): return FIFA[FIFA['Overall'] > 90]

# 10

def remove_col(): return FIFA.drop(labels='Unnamed: 0', axis=1)

# 11

def count_uniq(col_name='Position'): return len(list(FIFA[col_name].unique()))

# 12

def show_null(): return FIFA.iloc[FIFA[(FIFA.isnull().sum(axis=1) >= 2)].index]

# 13

def fill_empty(col_name='Release Clause'): 

    fifa_copy = FIFA

    fifa_copy[[col_name]] =  \
         fifa_copy[[col_name]].fillna(value='unknown')

    return fifa_copy

# 14

def comp_mean_col(col_name='Age'): return FIFA[col_name].mean()

# 15

def find_max(col_name='ShotPower'): return FIFA[col_name].max()

# 16

def find_max_row(col_name='ShotPower'): 

    return FIFA.iloc[FIFA[FIFA[col_name] == find_max()].index]

# 17

def describe_data(): return FIFA.describe()


if __name__ == "__main__": main()


