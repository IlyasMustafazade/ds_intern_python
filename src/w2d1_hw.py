import re

def main(): pass

# 1

def match_vowel_inside(str_):
    # [bcdfghjklmnpqrstvwxyz] -> match consonant
    # {1,} -> at least one in front of word
    # [aeiou] -> match vowel
    # {1,} -> can be multiple
    # \w* -> after matching, pass over 0 or more alpha num chars
    # because \w+ does not match "cec" but should match it
    # [bcdfghjklmnpqrstvwxyz]{1}\b ->
    #     match 1 consonant at last character of first word
    pattern = \
        re.compile(
        r"[bcdfghjklmnpqrstvwxyz]{1,}[aeiou]{1,}\w*[bcdfghjklmnpqrstvwxyz]{1}\b"
        )
    return re.match(pattern, str_)

# 2

def match_vowel_inside(str_): return re.match(r'\w+[aeiou]{1,}\w+', str_)

# 3 

def check_number_at_end(str_): return False if re.search(r'\d$', str_) is None else True

# 4

def match_word_at_end(str_): return re.search(r'\w+[\.!]?$', str_)

# 5

def change_tab_ws(str_):

    str_copy = list(str_)

    ws_str, str_copy = re.finditer(r' ', str_), list(re.sub('\t', ' ', str_))

    for i in ws_str: str_copy[i.span()[0]] = '\t'

    return ''.join(str_copy)

# 6

def match_starting_with(str_): return re.findall(r'\b[əü]\w+', str_)

# 7

def remove_ws(str_): return re.sub(r'\s', '', str_)

# 8

def find_long_word(str_): return re.findall(r'[\w]{4,}', str_)

# 9

def seperate_capital(str_): return re.sub(r'\B([A-Z])', r' \1', str_)

if __name__ == "__main__": main()

