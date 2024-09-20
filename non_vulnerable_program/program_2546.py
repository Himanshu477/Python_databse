       import matplotlib.pyplot as plt
       plt.plot([1,2,3], [4,5,6])

    .. plot::

       A plotting example:

       >>> import matplotlib.pyplot as plt
       >>> plt.plot([1,2,3], [4,5,6])

The content is interpreted as doctest formatted if it has a line starting
with ``>>>``.

The ``plot`` directive supports the options

    format : {'python', 'doctest'}
        Specify the format of the input

    include-source : bool
        Whether to display the source code. Default can be changed in conf.py
    
and the ``image`` directive options ``alt``, ``height``, ``width``,
``scale``, ``align``, ``class``.

Configuration options
---------------------

The plot directive has the following configuration options:

    plot_include_source
        Default value for the include-source option

    plot_pre_code
        Code that should be executed before each plot.

    plot_basedir
        Base directory, to which plot:: file names are relative to.
        (If None or empty, file names are relative to the directoly where
        the file containing the directive is.)

    plot_formats
        File formats to generate. List of tuples or strings::

            [(suffix, dpi), suffix, ...]

        that determine the file format and the DPI. For entries whose
        DPI was omitted, sensible defaults are chosen.

    plot_html_show_formats
        Whether to show links to the files in HTML.

TODO
----

* Refactor Latex output; now it's plain images, but it would be nice
  to make them appear side-by-side, or in floats.

"""

