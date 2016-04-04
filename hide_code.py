# http://stackoverflow.com/questions/27934885/how-to-hide-code-from-cells-in-ipython-notebook-visualized-with-nbviewer


# add toggle bar in a cell
from IPython.display import HTML

HTML('''<script>
code_show=true;
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
}
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')



# add toggle bar in a code cell

from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di # Example: di.display_html('<h3>%s:</h3>' % str, raw=True)

# This line will hide code by default when the notebook is exported as HTML
di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)

# This line will add a button to toggle visibility of code blocks, for use with the HTML export version
di.display_html('''<button onclick="jQuery('.input_area').toggle(); jQuery('.prompt').toggle();">Toggle code</button>''', raw=True)


# install jupyter extension
# https://github.com/ipython-contrib/IPython-notebook-extensions/wiki/Home-4.x-(Jupyter)
jupyter nbextension install nbextensions/usability/codefolding/main
jupyter nbextension install nbextensions/usability/hide_input_all
jupyter --paths
2. Activating an extension

jupyter nbextension enable <name of extension>

3. Deactivating extensions

jupyter nbextension disable <name of extension>


# hide code from ipython notebook

from IPython.display import HTML
HTML('''<script> $('div .input').hide()''')


# save the notebook in html filling up the whole page
from IPython.display import display
from IPython.display import HTML
import IPython.core.display as di

di.display_html('<script>jQuery(function() {if (jQuery("body.notebook_app").length == 0) { jQuery(".input_area").toggle(); jQuery(".prompt").toggle();}});</script>', raw=True)

CSS = """#notebook div.output_subarea {max-width:100%;}""" #changes output_subarea width to 100% (from 100% - 14ex)
HTML('<style>{}</style>'.format(CSS))
