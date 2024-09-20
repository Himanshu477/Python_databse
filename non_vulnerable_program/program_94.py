    from standard_array_spec import array_specification
    default_type_factories.append(array_specification())
except: 
    pass    

try: 
    # this is currently safe because it doesn't import wxPython.
