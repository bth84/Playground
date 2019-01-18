# # An Introduction to CDSW

# Copyright © 2010–2018 Cloudera. All rights reserved.
# Not to be reproduced or shared without prior written 
# consent from Cloudera.

# ## Entering Code

# Enter code as you normally would in the shell or a script:

print("Hello, CDSW!")

2 + 2

library(ggplot2)
ggplot(data = iris, aes(x = Sepal.Length, y = Sepal.Width, color = Species)) + 
	geom_point()


# ## Getting Help

help("geom_point")

?geom_point


# ## Accessing the Command Line

# You can enclose Linux commands in `system("")`:

system("pwd")

system("ls -l")

# But it's more common to use built-in R functions for tasks like these:

getwd()

base::print.data.frame(file.info(list.files()))


# ## Working with R Packages

# **Important:** Packages are managed on a project-by-project basis.

# Show what packages are loaded in this session of R:
search()
sessionInfo()

# Show the currently installed packages:
rownames(installed.packages())

# See if a particular package is installed:
"leaflet" %in% rownames(installed.packages())

# Install the current released version of a package from CRAN
# if it is not already installed:
#```r
#if(!"leaflet" %in% rownames(installed.packages())) {
#  install.packages("leaflet")
#}
#```

# Or install the latest development version from GitHub:
#```r
#devtools::install_github("rstudio/leaflet")
#```

# Installing leaflet from CRAN or GitHub on Linux can take 
# several minutes, because the package and its dependencies 
# must be built from source code.

# To save time, Cloudera has built the leaflet
# package and the other R packages it depends on into
# binary packages that work with CDSW running on RHEL
# or CentOS. Install these packages using this command:

if(!"leaflet" %in% rownames(installed.packages())) {
  install.packages(
    "leaflet",
    contriburl = "https://s3.amazonaws.com/r-pkgs/bin/linux/redhat"
  )
}

# Notice: Do not run the above command except on a CDSW
# instance provided by Cloudera for training purposes.
# These binary R packages are provided solely for Cloudera
# training purposes and are not intended for any other use.

# Note that this package is now installed for all sessions associated with 
# this project.

# See what version of the package is installed:
packageVersion("leaflet")

# Load the package:
library(leaflet)

# Use the package:
leaflet() %>%
  addTiles() %>%
  addMarkers(lng = -122.139869, lat = 37.425501, popup = "395 Page Mill Rd")

# Unload the package:
detach("package:leaflet", unload = TRUE)

# Uninstall the package:
# ``` r
# remove.packages("leaflet")
# ```


# ## Formatting Session Output

# Comments in your file can include
# [Markdown](https://daringfireball.net/projects/markdown/syntax) for
# presentation.  You will see some examples here, and you can see the
# References section below to find out more.

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

# Link to [Cloudera](http://www.cloudera.com)

# ### Images

# Display a stored image file:
library(png)
pic <- readPNG("resources/spark.png")
{plot.new(); rasterImage(pic, 0, 0, 1, 1)}

# **Note:** The image path is relative to `/home/cdsw/` regardless of script
# location.

# ### Code blocks

# To print a block of code in the output without running it, use a comment line
# with three backticks to begin the block, then the block of code with each
# line preceded with the comment character, then a comment line with three
# backticks to close the block. Optionally include the language name after the
# opening backticks:

# ``` r
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

# (1) Experiment with the CDSW command prompt.  Run some commands and note the
# following:
# * Tab completion is available
# * GNU Readline Library commands such as `C-a`, `C-e`, `C-f`, `C-b`, `C-d`,
# and `C-k` are available
# * The up and down arrows navigate the command history

# (2) Experiment with the CDSW editor.  Create a new file.  Enter Python code
# and Markdown text.  Run the file line by line (or chunk by chunk).  Clear the
# console log and run the entire file.

# (3) Open a terminal window.  Type `env | grep PYTHON` to explore the Python
# environmental variables.


# ## References

# [Cloudera Data Science Workbench](https://www.cloudera.com/documentation/data-science-workbench/latest.html)

# [Markdown](https://daringfireball.net/projects/markdown/)

# [LaTeX](https://en.wikibooks.org/wiki/LaTeX/Mathematics)

# [GNU Readline Library](http://cnswww.cns.cwru.edu/php/chet/readline/rluserman.html)

