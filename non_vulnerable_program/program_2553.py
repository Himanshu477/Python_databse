    import jinja2
    def format_template(template, **kw):
        return jinja2.Template(template).render(**kw)
except ImportError:
