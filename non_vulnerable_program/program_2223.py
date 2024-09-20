import sphinx
if sphinx.__version__ < "0.5":
    raise RuntimeError("Sphinx 0.5.dev or newer required")


# -----------------------------------------------------------------------------
# General configuration
# -----------------------------------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.pngmath', 'numpydoc',
              'phantom_import', 'autosummary', 'sphinx.ext.intersphinx',
              'sphinx.ext.coverage']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
#master_doc = 'index'

# General substitutions.
project = 'NumPy'
copyright = '2008, The Scipy community'

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
version = '1.2'
# The full version, including alpha/beta/rc tags.
release = '1.2.dev'

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
#today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
#unused_docs = []

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# List of directories, relative to source directories, that shouldn't be searched
# for source files.
exclude_dirs = []

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
#add_module_names = True

# If true, sectionauthor and moduleauthor directives will be shown in the
# output. They are ignored by default.
#show_authors = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -----------------------------------------------------------------------------
# HTML output
# -----------------------------------------------------------------------------

# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
html_style = 'scipy.css'

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
html_title = "%s v%s Manual (DRAFT)" % (project, version)

# The name of an image file (within the static path) to place at the top of
# the sidebar.
html_logo = 'scipyshiny_small.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = '%b %d, %Y'

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
#html_use_smartypants = True

# Custom sidebar templates, maps document names to template names.
html_sidebars = {
    'index': 'indexsidebar.html'
}

# Additional templates that should be rendered to pages, maps page names to
# template names.
html_additional_pages = {
    'index': 'indexcontent.html',
}

# If false, no module index is generated.
html_use_modindex = True

# If true, the reST sources are included in the HTML build as _sources/<name>.
#html_copy_source = True

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
#html_use_opensearch = ''

# If nonempty, this is the file name suffix for HTML files (e.g. ".html").
#html_file_suffix = '.html'

# Output file base name for HTML help builder.
htmlhelp_basename = 'NumPydoc'

# Pngmath should try to align formulas properly
pngmath_use_preview = True


# -----------------------------------------------------------------------------
# LaTeX output
# -----------------------------------------------------------------------------

# The paper size ('letter' or 'a4').
#latex_paper_size = 'letter'

# The font size ('10pt', '11pt' or '12pt').
#latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
_stdauthor = 'Written by the NumPy community'
latex_documents = [
  ('reference/index', 'numpy-ref.tex', 'NumPy Reference',
   _stdauthor, 'manual'),
  ('user/index', 'numpy-user.tex', 'NumPy User Guide',
   _stdauthor, 'manual'),
]

# The name of an image file (relative to this directory) to place at the top of
# the title page.
#latex_logo = None

# For "manual" documents, if this is true, then toplevel headings are parts,
# not chapters.
#latex_use_parts = False

# Additional stuff for the LaTeX preamble.
latex_preamble = r'''
\usepackage{amsmath}

% In the parameters section, place a newline after the Parameters
% header
\usepackage{expdlist}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}

% Make Examples/etc section headers smaller and more compact
\makeatletter
\titleformat{\paragraph}{\normalsize\py@HeaderFamily}%
            {\py@TitleColor}{0em}{\py@TitleColor}{\py@NormalColor}
\titlespacing*{\paragraph}{0pt}{1ex}{0pt}
\makeatother

% Fix footer/header
\renewcommand{\chaptermark}[1]{\markboth{\MakeUppercase{\thechapter.\ #1}}{}}
\renewcommand{\sectionmark}[1]{\markright{\MakeUppercase{\thesection.\ #1}}}
'''

# Documents to append as an appendix to all manuals.
#latex_appendices = []

# If false, no module index is generated.
latex_use_modindex = False


# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {'http://docs.python.org/dev': None}


# -----------------------------------------------------------------------------
# Numpy extensions
# -----------------------------------------------------------------------------

# If we want to do a phantom import from an XML file for all autodocs
phantom_import_file = 'dump.xml'

# Edit links
#numpydoc_edit_link = '`Edit </pydocweb/doc/%(full_name)s/>`__'

# -----------------------------------------------------------------------------
# Coverage checker
# -----------------------------------------------------------------------------
coverage_ignore_modules = r"""
    """.split()
coverage_ignore_functions = r"""
    test($|_) (some|all)true bitwise_not cumproduct pkgload
    generic\.
    """.split()
coverage_ignore_classes = r"""
    """.split()

coverage_c_path = []
coverage_c_regexes = {}
coverage_ignore_c_items = {}




********************
Using Python as glue
********************

|    There is no conversation more boring than the one where everybody
|    agrees. 
|    --- *Michel de Montaigne* 

|    Duct tape is like the force. It has a light side, and a dark side, and
|    it holds the universe together. 
|    --- *Carl Zwanzig* 

Many people like to say that Python is a fantastic glue language.
Hopefully, this Chapter will convince you that this is true. The first
adopters of Python for science were typically people who used it to
glue together large applicaton codes running on super-computers. Not
only was it much nicer to code in Python than in a shell script or
Perl, in addition, the ability to easily extend Python made it
relatively easy to create new classes and types specifically adapted
to the problems being solved. From the interactions of these early
contributors, Numeric emerged as an array-like object that could be
used to pass data between these applications. 

As Numeric has matured and developed into NumPy, people have been able
to write more code directly in NumPy. Often this code is fast-enough
for production use, but there are still times that there is a need to
access compiled code. Either to get that last bit of efficiency out of
the algorithm or to make it easier to access widely-available codes
written in C/C++ or Fortran. 

This chapter will review many of the tools that are available for the
purpose of accessing code written in other compiled languages. There
are many resources available for learning to call other compiled
libraries from Python and the purpose of this Chapter is not to make
you an expert. The main goal is to make you aware of some of the
possibilities so that you will know what to "Google" in order to learn more. 

The http://www.scipy.org website also contains a great deal of useful
information about many of these tools. For example, there is a nice
description of using several of the tools explained in this chapter at
http://www.scipy.org/PerformancePython. This link provides several
ways to solve the same problem showing how to use and connect with
compiled code to get the best performance. In the process you can get
a taste for several of the approaches that will be discussed in this
chapter. 


Calling other compiled libraries from Python
============================================

While Python is a great language and a pleasure to code in, its
dynamic nature results in overhead that can cause some code ( *i.e.*
raw computations inside of for loops) to be up 10-100 times slower
than equivalent code written in a static compiled language. In
addition, it can cause memory usage to be larger than necessary as
temporary arrays are created and destroyed during computation. For
many types of computing needs the extra slow-down and memory
consumption can often not be spared (at least for time- or memory-
critical portions of your code). Therefore one of the most common
needs is to call out from Python code to a fast, machine-code routine
(e.g. compiled using C/C++ or Fortran). The fact that this is
relatively easy to do is a big reason why Python is such an excellent
high-level language for scientific and engineering programming. 

Their are two basic approaches to calling compiled code: writing an
extension module that is then imported to Python using the import
command, or calling a shared-library subroutine directly from Python
using the ctypes module (included in the standard distribution with
Python 2.5). The first method is the most common (but with the
inclusion of ctypes into Python 2.5 this status may change). 

.. warning::

    Calling C-code from Python can result in Python crashes if you are not
    careful. None of the approaches in this chapter are immune. You have
    to know something about the way data is handled by both NumPy and by
    the third-party library being used. 


Hand-generated wrappers
=======================

Extension modules were discussed in Chapter `1
<#sec-writing-an-extension>`__ . The most basic way to interface with
compiled code is to write an extension module and construct a module
method that calls the compiled code. For improved readability, your
method should take advantage of the PyArg_ParseTuple call to convert
between Python objects and C data-types. For standard C data-types
there is probably already a built-in converter. For others you may
need to write your own converter and use the "O&" format string which
allows you to specify a function that will be used to perform the
conversion from the Python object to whatever C-structures are needed. 

Once the conversions to the appropriate C-structures and C data-types
have been performed, the next step in the wrapper is to call the
underlying function. This is straightforward if the underlying
function is in C or C++. However, in order to call Fortran code you
must be familiar with how Fortran subroutines are called from C/C++
using your compiler and platform. This can vary somewhat platforms and
compilers (which is another reason f2py makes life much simpler for
interfacing Fortran code) but generally involves underscore mangling
of the name and the fact that all variables are passed by reference
(i.e. all arguments are pointers). 

The advantage of the hand-generated wrapper is that you have complete
control over how the C-library gets used and called which can lead to
a lean and tight interface with minimal over-head. The disadvantage is
that you have to write, debug, and maintain C-code, although most of
it can be adapted using the time-honored technique of
"cutting-pasting-and-modifying" from other extension modules. Because,
the procedure of calling out to additional C-code is fairly
regimented, code-generation procedures have been developed to make
this process easier. One of these code- generation techniques is
distributed with NumPy and allows easy integration with Fortran and
(simple) C code. This package, f2py, will be covered briefly in the
next session. 


f2py
====

F2py allows you to automatically construct an extension module that
interfaces to routines in Fortran 77/90/95 code. It has the ability to
parse Fortran 77/90/95 code and automatically generate Python
signatures for the subroutines it encounters, or you can guide how the
subroutine interfaces with Python by constructing an interface-
defintion-file (or modifying the f2py-produced one). 

.. index::
   single: f2py

Creating source for a basic extension module
--------------------------------------------

Probably the easiest way to introduce f2py is to offer a simple
example. Here is one of the subroutines contained in a file named
:file:`add.f`:

.. code-block:: none

    C
          SUBROUTINE ZADD(A,B,C,N)
    C
          DOUBLE COMPLEX A(*)
          DOUBLE COMPLEX B(*)
          DOUBLE COMPLEX C(*)
          INTEGER N
          DO 20 J = 1, N
             C(J) = A(J)+B(J)
     20   CONTINUE
          END  

This routine simply adds the elements in two contiguous arrays and
places the result in a third. The memory for all three arrays must be
provided by the calling routine. A very basic interface to this
routine can be automatically generated by f2py::

    f2py -m add add.f

You should be able to run this command assuming your search-path is
set-up properly. This command will produce an extension module named
addmodule.c in the current directory. This extension module can now be
compiled and used from Python just like any other extension module. 


Creating a compiled extension module
------------------------------------

You can also get f2py to compile add.f and also compile its produced
extension module leaving only a shared-library extension file that can
be imported from Python::

    f2py -c -m add add.f

This command leaves a file named add.{ext} in the current directory
(where {ext} is the appropriate extension for a python extension
module on your platform --- so, pyd, *etc.* ). This module may then be
