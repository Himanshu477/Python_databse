import base_info
class inline_info(base_info.base_info):
    _support_code = [get_variable_support_code, py_to_raw_dict_support_code]


# should re-write compiled functions to take a local and global dict
# as input.
