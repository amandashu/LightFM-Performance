import nbformat
from nbconvert import HTMLExporter
import pandas as pd
import matplotlib.pyplot as plt
import os

def line_plot(figs_out_path, metrics_to_include):
    data = pd.read_csv('results/Metrics.csv').set_index('Recommender')

    if not os.path.exists(figs_out_path):
        os.makedirs(figs_out_path)

    for m in metrics_to_include:
        df = data[[c for c in data.columns if m in c]]
        df.columns = [c[c.find('@')+1:] for c in df.columns]

        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xlabel('Cutoff')
        ax.set_title(m+"@", pad=30)
        for i,r in df.iterrows():
            plt.plot(r,label=r.name)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),ncol=3, fancybox=True)
        plt.savefig(figs_out_path + m.lower()+'.png')
    return None


def convert_notebook(report_in_path, report_out_path):
    """
    code modified from https://github.com/DSC-Capstone/project-templates/blob/EDA/src/utils.py
    """

    curdir = os.path.abspath(os.getcwd())
    indir, _ = os.path.split(report_in_path)
    outdir, _ = os.path.split(report_out_path)
    os.makedirs(outdir, exist_ok=True)

    config = {
        "ExecutePreprocessor": {"enabled": True},
        "TemplateExporter": {"exclude_output_prompt": True, "exclude_input": True, "exclude_input_prompt": True},
    }

    nb = nbformat.read(open(report_in_path,encoding="utf-8"), as_version=4)
    html_exporter = HTMLExporter(config=config)

    os.chdir(indir)
    body, resources = html_exporter.from_notebook_node(nb)

    os.chdir(curdir)
    with open(report_out_path, 'w', encoding="utf-8") as fh:
        fh.write(body)

def report(figs_out_path, metrics_to_include, report_in_path, report_out_path):
    line_plot(figs_out_path, metrics_to_include)
    convert_notebook(report_in_path, report_out_path)
