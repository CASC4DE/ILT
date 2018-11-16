#!/usr/bin/env python
# encoding: utf-8
'''
Set of functionalities to create html tables, either from lists of list or csv files.

some css should be added to the html,
for instance:

    #peaks_table{ border: 2px; }                     <!-- table borders -->
    #peaks_line:nth-child(even) {background: #CCC;}  <!-- alternating line background color -->
    #peaks_line:nth-child(odd) {background: #FFF;}
    #peaks_line:nth-child(1) {font-weight: bold;}    <!-- first line -->

Author: Mélanie Proix jnmelanie@gmail.com - Oct 2018
'''
from __future__ import division, print_function
import csv
from jinja2 import Template

def list2html(liste, css=False):
    """
    from a list of lists, creates a html table.
    if css is activated, a default style will be applied
    """
    table = Template("""
<html>
<head><meta charset="utf-8"></head>
<body>
{{style}}
<table id='peaks_table'>
    <thead>
        <tr id='peaks_line_head'>{% for e in liste[0] %}<th>{{e}}</th>{% endfor %}</tr>
    </thead>
    <tbody>
        {% for elt in liste[1:] %}
            <tr id='peaks_line'>
            {% for e in elt %}<td>{{e}}</td>{% endfor %}
            </tr>
        {% endfor %}
    </tbody>
</table>
<script>
    var newTableObject = document.getElementById('peaks_table')
    sorttable.makeSortable(newTableObject);
</script>
</body>
</html>
""")
    if css:
        style = """
<style type="text/css">
#peaks_line:nth-child(even) {background: #CCC;}
#peaks_line:nth-child(odd) {background: #FFF;}
#peaks_line_head{
  background-color: rgba(34,141,204,1);
  color: white;
  font-weight: bold;
</style>"""
    else:
        style = ""
    return table.render(liste=liste, style=style)

def csv2html(fichiercsv, css=False, **kwargs):
    """from a csv file, creates a html rendering
    accepts any additional arguments for csv reading.
    """
    with open(fichiercsv) as csvfile:
        peaks = csv.reader(csvfile, **kwargs) #, delimiter=',', quotechar='|')
        peaks_list = list(peaks)
    return list2html(peaks_list, css=css)
                     