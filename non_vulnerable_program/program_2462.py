import util

def _path(*a):
    return os.path.join(*((os.path.dirname(__file__),) + a))

class TestMixed(util.F2PyTest):
    sources = [_path('src', 'assumed_shape', 'foo_free.f90')]

    @dec.slow
    def test_all(self):
        print self.module.__doc__
        r = self.module.fsum([1,2])
        assert r==3,`r`
        #r = self.module.sum([1,2])
        #assert r==3,`r`

if __name__ == "__main__":
