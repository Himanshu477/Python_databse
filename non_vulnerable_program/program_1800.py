from splitline import string_replace_map

def parse_expr(line, lower=False):
    newline, repmap = string_replace_map(line, lower=lower)
    if repmap:
        raise NotImplementedError,`newline,repmap`




