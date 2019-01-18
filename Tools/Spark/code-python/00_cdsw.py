# # An Introduction to Cloudera Data Science Workbench (CDSW)

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# ## Entering Code

# Enter code as you normally would in a Python shell or script:

print("Hello, Data Scientists!")

2 + 2

import math
def expit(x):
    """Compute the expit function (aka inverse-logit function or logistic function)."""
    return 1.0 / (1.0 + math.exp(-x))
expit(2)

import seaborn as sns
iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species")


# ## Getting Help

# Use the standard Python and IPython commands to get help:

help(expit)

expit?

#expit??



# ## Accessing the Linux Command Line

# Run a Linux command by using the `!` prefix.

# Print the current working directory:
!pwd

# **Note:**  All CDSW users have the username `cdsw`.

# List the contents of the current directory:
!ls -l

# List the contents of the `/duocar` directory in the Hadoop Distributed File
# System (HDFS):
!hdfs dfs -ls /duocar

# You can also access the command line via the **Terminal access** menu item.


# ## Working with Python Packages

# Use `pip` in Python 2 or `pip3` in Python 3 to manage Python packages.

# **Important:** Packages are managed on a project-by-project basis.

# Use `!pip list` to get a list of the currently installed packages:
!pip list --format=columns

# Search for the [folium](https://github.com/python-visualization/folium)
# package in the Python package repository:
!pip search folium

# Install a new package:
!pip install folium

# **Note:** This package is now available in all future sessions launched in
# this project.

# Show details about the installed package:
!pip show folium

# **Note:**  This returns nothing if the package is not installed.

# Import the package:
import folium

# Use the package:
folium.Map(location=[46.8772222, -96.7894444])

# Uninstall the package:
!pip uninstall -y folium

# **Note:** Include the `-y` option to avoid an invisible prompt to confirm
# the uninstall.

# **Important:** This package is no longer available in all future sessions
# launched in this project.


# ## Formatting Session Output

# Use [Markdown](https://daringfireball.net/projects/markdown/syntax) text to
# format your session output.

# **Important:** Prefix the markdown text with the Python comment character
# `#`.

# ### Headings

# # Heading 1

# ## Heading 2

# ### Heading 3

# ### Text

# Plain text

# *Emphasized text* or _emphasized text_

# **Bold text** or __bold text__

# `Code text` (Note these are backtick quotes.)

# ### Mathematical Text

# Display an inline term like $\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$ using
# a [LaTeX](https://en.wikibooks.org/wiki/LaTeX/Mathematics) expression
# surrounded by dollar-sign characters.

# A math expression can be displayed set apart by surrounding the LaTeX
# shorthand with double dollar-signs, like so: $$f(x)=\frac{1}{1+e^{-x}}$$

# ### Lists

# Bulleted List
# * Item 1
#   * Item 1a
#   * Item 1b
# * Item 2
# * Item 3

# Numbered List
# 1. Item 1
# 2. Item 2
# 3. Item 3

# ### Links

# Link to [Cloudera](http://www.cloudera.com).

# ### Images

# Display a stored image file:
from IPython.display import Image
Image("resources/spark.png")

# **Note:** The image path is relative to `/home/cdsw/` regardless of script
# location.

# ### Code blocks

# To print a block of code in the output without running it, use a comment line
# with three backticks to begin the block, then the block of code with each
# line preceded with the comment character, then a comment line with three
# backticks to close the block. Optionally include the language name after the
# opening backticks:

# ``` python
# print("Hello, World!")
# ```

# You can omit the language name to print the code block in black text without
# syntax coloring, for example, to display a block of static data or output:

# ```
# Hello, World!
# ```

# ### Invisible comments

#[//]: # (To include a comment that will not appear in the)
#[//]: # (output at all, you can use this curious syntax.)

# Move along, nothing to see here.


# ## Exercises

# (1) Experiment with the CDSW command line.

# * Type `i` and then press the TAB key.  Use the UP and DOWN ARROW keys to
# navigate the tab completion options.  Select `import` and then press the
# RETURN key.

# * Type `import math` but do not press the RETURN key.  Use the LEFT and RIGHT
# ARROW keys to navigate along the command line.

# * Use the UP and DOWN ARROW keys to navigate the command line history.

# (2) Experiment with the CDSW file editor.

# * Create a new file called `random.py`.

# * Enter the following text and code into the file.

# ``` python
# # # Generate a Random Uniform Number
# 
# # Import the `random` module:
# import random
# 
# # Generate a random uniform variate in the interval [0, 1):
# random.random()
# ```

# * Select a line or block of code and run the selection.

# * Clear the console log and run the entire file.

# (3) Experiment with the Linux command line.

# * Click on **Terminal access** to open a terminal window.

# * Use the `hdfs dfs -ls` command to explore the `/duocar/raw/` directory in
# HDFS.


# ## References

# [Cloudera Data Science Workbench](https://www.cloudera.com/documentation/data-science-workbench/latest.html)

# [Markdown](https://daringfireball.net/projects/markdown/)

# [LaTeX](https://en.wikibooks.org/wiki/LaTeX/Mathematics)
