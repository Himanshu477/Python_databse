        from waflib.Tools import ccroot

        # TODO see get_uselib_vars from ccroot.py
        _vars = set([])
        for x in kw['features']:
            if x in ccroot.USELIB_VARS:
                _vars |= ccroot.USELIB_VARS[x]

        for k in _vars:
            lk = k.lower()
            if k == 'INCLUDES': lk = 'includes'
            if k == 'DEFKEYS': lk = 'defines'
            if lk in kw:
                val = kw[lk]
                # remove trailing slash
                if isinstance(val, str):
                    val = val.rstrip(os.path.sep)
                self.env.append_unique(k + '_' + kw['uselib_store'], val)
    return is_success

@waflib.Configure.conf
def define_with_comment(conf, define, value, comment=None, quote=True):
    if comment is None:
        return conf.define(define, value, quote)

    assert define and isinstance(define, str)

    comment_tbl = conf.env[DEFINE_COMMENTS] or {}
    comment_tbl[define] = comment
    conf.env[DEFINE_COMMENTS] = comment_tbl

    return conf.define(define, value, quote)

@waflib.Configure.conf
def get_comment(self, key):
    assert key and isinstance(key, str)

    if key in self.env[DEFINE_COMMENTS]:
        return self.env[DEFINE_COMMENTS][key]
    return None

@waflib.Configure.conf
def define_cond(self, name, value, comment):
    """Conditionally define a name.
    Formally equivalent to: if value: define(name, 1) else: undefine(name)"""
    if value:
        self.define_with_comment(name, value, comment)
    else:
        self.undefine(name)

@waflib.Configure.conf
def get_config_header(self, defines=True, headers=False):
    """
    Create the contents of a ``config.h`` file from the defines and includes
    set in conf.env.define_key / conf.env.include_key. No include guards are added.

    :param defines: write the defines values
    :type defines: bool
    :param headers: write the headers
    :type headers: bool
    :return: the contents of a ``config.h`` file
    :rtype: string
    """
    tpl = self.env["CONFIG_HEADER_TEMPLATE"] or "%(content)s"

    lst = []
    if headers:
        for x in self.env[INCKEYS]:
            lst.append('#include <%s>' % x)

    if defines:
        for x in self.env[DEFKEYS]:
            if self.is_defined(x):
                val = self.get_define(x)
                cmt = self.get_comment(x)
                if cmt is not None:
                    lst.append(cmt)
                lst.append('#define %s %s\n' % (x, val))
            else:
                lst.append('/* #undef %s */\n' % x)
    return tpl % {"content": "\n".join(lst)}


