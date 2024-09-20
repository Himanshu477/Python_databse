from readfortran import FortranFileReader, FortranStringReader
from block import Block

class FortranParser:

    def __init__(self, reader):
        self.reader = reader

    def get_item(self):
        try:
            return self.reader.next(ignore_comments = True)
        except StopIteration:
            pass

    def put_item(self, item):
        self.reader.fifo_item.insert(0, item)

    def parse(self):
        main = Block(self)
        main.fill()
        return main

def test_pyf():
    string = """
python module foo
  interface
    subroutine bar
    real r
    end subroutine bar
  end interface
end python module
"""
    reader = FortranStringReader(string, True, True)
    reader = FortranFileReader(filename)
    parser = FortranParser(reader)
    block = parser.parse()
    print block

def simple_main():
    import sys
    for filename in sys.argv[1:]:
        print 'Processing',filename
        reader = FortranFileReader(filename)
        parser = FortranParser(reader)
        block = parser.parse()
        print block

if __name__ == "__main__":
    #test_pyf()
    simple_main()



