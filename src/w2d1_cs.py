import re

def main(): pass

# 1

def create_dict(str_):

	keys, vals = re.findall(r'[A-Z]\w*', str_), re.findall(r'\d+', str_)

	return {re.findall(r'[A-Z]\w*', str_)[i]: re.findall(r'\d+', str_)[i] for i in range(len(vals))}

# 2

def verify_email(str_):

	return False if len(re.findall(r'[A-Za-z0-9]{3,10}@[A-Za-z0-9]{2,10}\.[A-Za-z]{2,4}', str_)) < 1 \
	    else True

# 3

def verify_number(str_): return False if re.search(r'\(\d{3,3}\) \d+-\d', str_) is None else True

if __name__ == "__main__": main()

