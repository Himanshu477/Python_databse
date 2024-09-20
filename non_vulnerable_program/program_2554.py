    import jinja
    def format_template(template, **kw):
        return jinja.from_string(template, **kw)

TEMPLATE = """
{{ source_code }}

{{ only_html }}

   {% if source_link or (html_show_formats and not multi_image) %}
   (
   {%- if source_link -%}
   `Source code <{{ source_link }}>`__
   {%- endif -%}
   {%- if html_show_formats and not multi_image -%}
     {%- for img in images -%}
       {%- for fmt in img.formats -%}
         {%- if source_link or not loop.first -%}, {% endif -%}
         `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
       {%- endfor -%}
     {%- endfor -%}
   {%- endif -%}
   )
   {% endif %}

   {% for img in images %}
   .. figure:: {{ build_dir }}/{{ img.basename }}.png
      {%- for option in options %}
      {{ option }}
      {% endfor %}

      {% if html_show_formats and multi_image -%}
        (
        {%- for fmt in img.formats -%}
        {%- if not loop.first -%}, {% endif -%}
        `{{ fmt }} <{{ dest_dir }}/{{ img.basename }}.{{ fmt }}>`__
        {%- endfor -%}
        )
      {%- endif -%}
   {% endfor %}

{{ only_latex }}

   {% for img in images %}
   .. image:: {{ build_dir }}/{{ img.basename }}.pdf
   {% endfor %}

"""

class ImageFile(object):
    def __init__(self, basename, dirname):
        self.basename = basename
        self.dirname = dirname
        self.formats = []

    def filename(self, format):
        return os.path.join(self.dirname, "%s.%s" % (self.basename, format))

    def filenames(self):
        return [self.filename(fmt) for fmt in self.formats]

def run(arguments, content, options, state_machine, state, lineno):
    if arguments and content:
        raise RuntimeError("plot:: directive can't have both args and content")

    document = state_machine.document
    config = document.settings.env.config

    options.setdefault('include-source', config.plot_include_source)

    # determine input
    rst_file = document.attributes['source']
    rst_dir = os.path.dirname(rst_file)

    if arguments:
        if not config.plot_basedir:
            source_file_name = os.path.join(rst_dir,
                                            directives.uri(arguments[0]))
        else:
            source_file_name = os.path.join(setup.confdir, config.plot_basedir,
                                            directives.uri(arguments[0]))
        code = open(source_file_name, 'r').read()
        output_base = os.path.basename(source_file_name)
    else:
        source_file_name = rst_file
        code = textwrap.dedent("\n".join(map(str, content)))
        counter = document.attributes.get('_plot_counter', 0) + 1
        document.attributes['_plot_counter'] = counter
        base, ext = os.path.splitext(os.path.basename(source_file_name))
        output_base = '%s-%d.py' % (base, counter)

    base, source_ext = os.path.splitext(output_base)
    if source_ext in ('.py', '.rst', '.txt'):
        output_base = base
    else:
        source_ext = ''

    # ensure that LaTeX includegraphics doesn't choke in foo.bar.pdf filenames
    output_base = output_base.replace('.', '-')

    # is it in doctest format?
    is_doctest = contains_doctest(code)
    if 'format' in options:
        if options['format'] == 'python':
            is_doctest = False
        else:
            is_doctest = True

    # determine output directory name fragment
    source_rel_name = relpath(source_file_name, setup.confdir)
    source_rel_dir = os.path.dirname(source_rel_name)
    while source_rel_dir.startswith(os.path.sep):
        source_rel_dir = source_rel_dir[1:]

    # build_dir: where to place output files (temporarily)
    build_dir = os.path.join(os.path.dirname(setup.app.doctreedir),
                             'plot_directive',
                             source_rel_dir)
    if not os.path.exists(build_dir):
        os.makedirs(build_dir)

    # output_dir: final location in the builder's directory
    dest_dir = os.path.abspath(os.path.join(setup.app.builder.outdir,
                                            source_rel_dir))

    # how to link to files from the RST file
    dest_dir_link = os.path.join(relpath(setup.confdir, rst_dir),
                                 source_rel_dir).replace(os.path.sep, '/')
    build_dir_link = relpath(build_dir, rst_dir).replace(os.path.sep, '/')
    source_link = dest_dir_link + '/' + output_base + source_ext

    # make figures
    try:
        results = makefig(code, source_file_name, build_dir, output_base,
                          config)
        errors = []
    except PlotError as err:
        reporter = state.memo.reporter
        sm = reporter.system_message(
            2, "Exception occurred in plotting %s: %s" % (output_base, err),
            line=lineno)
        results = [(code, [])]
        errors = [sm]

    # generate output restructuredtext
    total_lines = []
    for j, (code_piece, images) in enumerate(results):
        if options['include-source']:
            if is_doctest:
                lines = ['']
                lines += [row.rstrip() for row in code_piece.split('\n')]
            else:
                lines = ['.. code-block:: python', '']
                lines += ['    %s' % row.rstrip()
                          for row in code_piece.split('\n')]
            source_code = "\n".join(lines)
        else:
            source_code = ""

        opts = [':%s: %s' % (key, val) for key, val in list(options.items())
                if key in ('alt', 'height', 'width', 'scale', 'align', 'class')]

        only_html = ".. only:: html"
        only_latex = ".. only:: latex"

        if j == 0:
            src_link = source_link
        else:
            src_link = None

        result = format_template(
            TEMPLATE,
            dest_dir=dest_dir_link,
            build_dir=build_dir_link,
            source_link=src_link,
            multi_image=len(images) > 1,
            only_html=only_html,
            only_latex=only_latex,
            options=opts,
            images=images,
            source_code=source_code,
            html_show_formats=config.plot_html_show_formats)

        total_lines.extend(result.split("\n"))
        total_lines.extend("\n")

    if total_lines:
        state_machine.insert_input(total_lines, source=source_file_name)

    # copy image files to builder's output directory
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for code_piece, images in results:
        for img in images:
            for fn in img.filenames():
                shutil.copyfile(fn, os.path.join(dest_dir,
                                                 os.path.basename(fn)))

    # copy script (if necessary)
    if source_file_name == rst_file:
        target_name = os.path.join(dest_dir, output_base + source_ext)
        f = open(target_name, 'w')
        f.write(unescape_doctest(code))
        f.close()

    return errors


#------------------------------------------------------------------------------
# Run code and capture figures
#------------------------------------------------------------------------------

