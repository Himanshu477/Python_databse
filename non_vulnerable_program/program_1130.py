     imported.


     Outputs:

       The function returns a tuple with all the names of the modules which
       were actually imported. [NotImplemented]

     """
        frame = self.parent_frame
        self.info_modules = {}
        if options.get('force',False):
            self.imported_packages = []
        self.verbose = verbose = options.get('verbose',False)
        postpone = options.get('postpone',False)

        self._init_info_modules(packages or None)

        self.log('Imports to %r namespace\n----------------------------'\
                 % self.parent_name)

        for package_name in self._get_sorted_names():
            if package_name in self.imported_packages:
                continue
            info_module = self.info_modules[package_name]
            global_symbols = getattr(info_module,'global_symbols',[])            
            if postpone and not global_symbols:
                self.log('__all__.append(%r)' % (package_name))
                if '.' not in package_name:
                    self.parent_export_names.append(package_name)
                continue
            
            old_object = frame.f_locals.get(package_name,None)

            cmdstr = 'import '+package_name
            if self._execcmd(cmdstr):
                continue
            self.imported_packages.append(package_name)
    
            if verbose!=-1:
                new_object = frame.f_locals.get(package_name)
                if old_object is not None and old_object is not new_object:
                    self.warn('Overwriting %s=%s (was %s)' \
                              % (package_name,self._obj2str(new_object),
                                 self._obj2str(old_object)))

            if '.' not in package_name:
                self.parent_export_names.append(package_name)

            for symbol in global_symbols:
                if symbol=='*':
                    symbols = eval('getattr(%s,"__all__",None)'\
                                   % (package_name),
                                   frame.f_globals,frame.f_locals)
                    if symbols is None:
                        symbols = eval('dir(%s)' % (package_name),
                                       frame.f_globals,frame.f_locals)
                        symbols = filter(lambda s:not s.startswith('_'),symbols)
                else:
                    symbols = [symbol]

                if verbose!=-1:
                    old_objects = {}
                    for s in symbols:
                        if frame.f_locals.has_key(s):
                            old_objects[s] = frame.f_locals[s]

                cmdstr = 'from '+package_name+' import '+symbol
                if self._execcmd(cmdstr):
                    continue

                if verbose!=-1:
                    for s,old_object in old_objects.items():
                        new_object = frame.f_locals[s]
                        if new_object is not old_object:
                            self.warn('Overwriting %s=%s (was %s)' \
                                      % (s,self._obj2repr(new_object),
                                         self._obj2repr(old_object)))

                if symbol=='*':
                    self.parent_export_names.extend(symbols)
                else:
                    self.parent_export_names.append(symbol)

        return

    def _execcmd(self,cmdstr):
        """ Execute command in parent_frame."""
        frame = self.parent_frame
        try:
            exec (cmdstr, frame.f_globals,frame.f_locals)
        except Exception,msg:
            self.error('%s -> failed: %s' % (cmdstr,msg))
            return True
        else:
            self.log('%s -> success' % (cmdstr))                            
        return

    def _obj2repr(self,obj):
        """ Return repr(obj) with"""
        module = getattr(obj,'__module__',None)
        file = getattr(obj,'__file__',None)
        if module is not None:
            return repr(obj) + ' from ' + module
        if file is not None:
            return repr(obj) + ' from ' + file
        return repr(obj)

    def log(self,mess):
        if self.verbose>1:
            print >> sys.stderr, str(mess)
    def warn(self,mess):
        if self.verbose>=0:
            print >> sys.stderr, str(mess)
    def error(self,mess):
        if self.verbose!=-1:
            print >> sys.stderr, str(mess)

try:
