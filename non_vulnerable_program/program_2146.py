    import nose
except ImportError:
    fine_nose = False
else:
    nose_version = nose.__versioninfo__
    if nose_version[0] < 1 and nose_version[1] < 10:
        fine_nose = False

if fine_nose:
