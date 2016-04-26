# remove prior version if any
rm -rf sphinx_bootstrap_theme

# Download and untar
wget https://pypi.python.org/packages/source/s/sphinx-bootstrap-theme/sphinx-bootstrap-theme-0.4.0.tar.gz
tar -zxf sphinx-bootstrap-theme-0.4.0.tar.gz
rm sphinx-bootstrap-theme-0.4.0.tar.gz

# Move everything to sphinx_bootstrap_theme
mv sphinx-bootstrap-theme-0.4.0/sphinx_bootstrap_theme .
mv sphinx-bootstrap-theme-0.4.0/*.txt sphinx_bootstrap_theme
mv sphinx-bootstrap-theme-0.4.0/*.in sphinx_bootstrap_theme

# Clean theme that we don't want
 # rm -rf -ignore myfile.txt *
rm -rf sphinx_bootstrap_theme/bootstrap/static/bootstrap-2.*
rm -rf sphinx_bootstrap_theme/bootstrap/static/bootswatch-2.*

# remove all bootstwatch theme except one
mv sphinx_bootstrap_theme/bootstrap/static/bootswatch-3.1.0/lumen .
rm -rf sphinx_bootstrap_theme/bootstrap/static/bootswatch-3.1.0/
mkdir sphinx_bootstrap_theme/bootstrap/static/bootswatch-3.1.0
mv lumen sphinx_bootstrap_theme/bootstrap/static/bootswatch-3.1.0/



# Clean remaining files
rm -rf sphinx-bootstrap-theme-0.4.0
