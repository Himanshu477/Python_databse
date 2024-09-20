    from wxPython.wx import *
    import time

    class Canvas(wxWindow):
        def __init__(self, parent, id = -1, size = wxDefaultSize):
            wxWindow.__init__(self, parent, id, wxPoint(0, 0), size,
                              wxSUNKEN_BORDER | wxWANTS_CHARS)
            self.calc_points()
            EVT_PAINT(self, self.OnPaint)
            EVT_SIZE(self, self.OnSize)

        def calc_points(self):
            w,h = self.GetSizeTuple()            
            #x = randint(0+50, w-50, self.point_count)
            #y = randint(0+50, h-50, len(x))
            x = arange(0,w,typecode=Int32)
            y = h/2.*sin(x*2*pi/w)+h/2.
            y = y.astype(Int32)
            self.points = concatenate((x[:,NewAxis],y[:,NewAxis]),-1)

        def OnSize(self,event):
            self.calc_points()
            self.Refresh()

        def OnPaint(self,event):
            w,h = self.GetSizeTuple()            
            print len(self.points)
            dc = wxPaintDC(self)
            dc.BeginDrawing()

            # This first call is slow because your either compiling (very slow)
            # or loading a DLL (kinda slow)
            # Resize the window to get a more realistic timing.
            pt_copy = self.points.copy()
            t1 = time.clock()
            offset = array((1,0))
            mod = array((w,0))
            x = pt_copy[:,0];
            ang = 2*pi/w;
            
            size = 1
            red_pen = wxPen('red',size)            
            white_pen = wxPen('white',size)
            blue_pen = wxPen('blue',size)
            pens = iter([red_pen,white_pen,blue_pen])
            phase = 10
            for i in range(1500):
                if phase > 2*pi:
                    phase = 0                
                    try:
                        pen = pens.next()
                    except:
                        pens = iter([red_pen,white_pen,blue_pen])
                        pen = pens.next()
                    dc.SetPen(pen)
                polyline(dc,pt_copy)            
                next_y = (h/2.*sin(x*ang-phase)+h/2.).astype(Int32)            
                pt_copy[:,1] = next_y
                phase += ang
            t2 = time.clock()
            print 'Weave Polyline:', t2-t1

            t1 = time.clock()
            pt_copy = self.points.copy()
            pens = iter([red_pen,white_pen,blue_pen])
            phase = 10
            for i in range(1500):
                if phase > 2*pi:
                    phase = 0                
                    try:
                        pen = pens.next()
                    except:
                        pens = iter([red_pen,white_pen,blue_pen])
                        pen = pens.next()
                    dc.SetPen(pen)
                dc.DrawLines(pt_copy)
                next_y = (h/2.*sin(x*ang-phase)+h/2.).astype(Int32)            
                pt_copy[:,1] = next_y
                phase += ang
            t2 = time.clock()
            dc.SetPen(red_pen)
            print 'wxPython DrawLines:', t2-t1

            dc.EndDrawing()

    class CanvasWindow(wxFrame):
        def __init__(self, id=-1, title='Canvas',size=(500,500)):
            parent = NULL
            wxFrame.__init__(self, parent,id,title, size=size)
            self.canvas = Canvas(self)
            self.Show(1)

    class MyApp(wxApp):
        def OnInit(self):
            frame = CanvasWindow(title="Speed Examples",size=(500,500))
            frame.Show(true)
            return true

    app = MyApp(0)
    app.MainLoop()
    

# h:\wrk\scipy\weave\examples>python object.py
# initial val: 1
# inc result: 2
# after set attr: 5

