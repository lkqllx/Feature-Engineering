
def isMatch(s: str, p: str) -> bool:
    max_str_loc = len(s) - 1
    str_loc = 0
    p_iter = iter(range(len(p)))
    '''Not empty scenario '''
    # for p_loc in range(len(p)):
    next_value = 0
    while next_value != -1 and str_loc <= max_str_loc:
        p_loc = next(p_iter, -1)
        if p_loc == -1:
            break

        if p_loc == len(p) - 1:
            if p[p_loc] == s[str_loc] or p[p_loc] == '.':
                str_loc += 1
                break
            else:
                return False

        '''Handle str_loc <= max_str_loc '''
        if p[p_loc + 1] != '*':
            if p[p_loc] == s[str_loc] or p[p_loc] == '.':
                str_loc += 1 # skip '*'
            else:
                return False
        elif p[p_loc + 1] == '*':

            if p[p_loc] == '.':
                str_loc = max_str_loc + 1
                _ = next(p_iter, -1)  # skip '*'
                continue
            elif p[p_loc] != '.':
                if p[p_loc] != s[str_loc]:
                    _ = next(p_iter, -1)  # skip '*'
                    continue
                else:
                    _ = next(p_iter, -1)  # skip '*'
                    str_loc += 1
                    while str_loc <= max_str_loc - (len(p)-(p_loc+1) -1):
                        if s[str_loc] == p[p_loc]:
                            str_loc += 1
                        else:
                            break

    p_list = list(p_iter)
    if not p_list and str_loc > max_str_loc:
        return True
    elif not p_list and str_loc <= max_str_loc:
        return False
    elif p_list:
        if len(p_list) % 2 != 0:
            return False

        p_list = p_list[1::2]
        for idx in p_list:
            if p[idx] != '*':
                return False
        return True

print(isMatch('bbbba', '.*a*a'))






