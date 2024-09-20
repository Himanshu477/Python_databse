import weave
from RandomArray import *
from Numeric import *
from wxPython.wx import *

"""
const int n_pts = _Nline[0];
const int bunch_size = 100;
const int bunches = n_pts / bunch_size;
const int left_over = n_pts % bunch_size;

for (int i = 0; i < bunches; i++)
{
    Polyline(hdc,(POINT*)p_data,bunch_size);
    p_data += bunch_size*2; //*2 for two longs per point
}
Polyline(hdc,(POINT*)p_data,left_over);
"""

def polyline(dc,line,xoffset=0,yoffset=0):
    #------------------------------------------------------------------------
    # Make sure the array is the correct size/shape 
    #------------------------------------------------------------------------
    shp = line.shape
    assert(len(shp)==2 and shp[1] == 2)

    #------------------------------------------------------------------------
    # Offset data if necessary
    #------------------------------------------------------------------------
    if xoffset or yoffset:
        line = line + array((xoffset,yoffset),line.typecode())
    
    #------------------------------------------------------------------------
    # Define the win32 version of the function
    #------------------------------------------------------------------------        
    if sys.platform == 'win32':
        # win32 requires int type for lines.
        if (line.typecode() != Int or not line.iscontiguous()):
            line = line.astype(Int)   
        code = """
               HDC hdc = (HDC) dc->GetHDC();                    
               Polyline(hdc,(POINT*)line,Nline[0]);
               """
    else:
        if (line.typecode() != UInt16 or 
            not line.iscontiguous()):
            line = line.astype(UInt16)   
        code = """
               GdkWindow* win = dc->m_window;                    
               GdkGC* pen = dc->m_penGC;
               gdk_draw_lines(win,pen,(GdkPoint*)line,Nline[0]);         
               """
    weave.inline(code,['dc','line'])

    
    #------------------------------------------------------------------------
    # Find the maximum and minimum points in the drawing list and add
    # them to the bounding box.    
    #------------------------------------------------------------------------
    max_pt = maximum.reduce(line,0)
    min_pt = minimum.reduce(line,0)
    dc.CalcBoundingBox(max_pt[0],max_pt[1])
    dc.CalcBoundingBox(min_pt[0],min_pt[1])    

#-----------------------------------------------------------------------------
# Define a new version of DrawLines that calls the optimized
# version for Numeric arrays when appropriate.
#-----------------------------------------------------------------------------
def NewDrawLines(dc,line):
    """
    """
    if (type(line) is ArrayType):
        polyline(dc,line)
    else:
        dc.DrawLines(line)            

#-----------------------------------------------------------------------------
# And attach our new method to the wxPaintDC class
# !! We have disabled it and called polyline directly in this example
# !! to get timing comparison between the old and new way.
#-----------------------------------------------------------------------------
#wxPaintDC.DrawLines = NewDrawLines
        
if __name__ == '__main__':
