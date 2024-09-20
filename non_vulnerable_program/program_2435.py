        import types
        f.seek = types.MethodType(seek, f)
        f.tell = types.MethodType(tell, f)
    else:
