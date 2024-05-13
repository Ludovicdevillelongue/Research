import os
from html2image import Html2Image
hti = Html2Image()

directory = '../results_1y/CA3/papers graph'
for filename in os.listdir(directory):
    if filename.endswith('.html'):
        with open(os.path.join(directory, filename)) as f:
            filename=filename.split(".")[0]
            hti.screenshot(
                html_file='../results_1y/CA3/papers graph/{}.html'.format(filename), save_as='{}.png'.format(filename)
            )