    import string,os,stat
    import pwd, grp
    
    hertz = 100. #standard value for jiffies (in seconds) on Linux.
    states = {'R':'RUNNING','S':'SLEEPING','Z':'ZOMBIE',
              'T':'TRACED','D':'DEEPSLEEP'}
    
