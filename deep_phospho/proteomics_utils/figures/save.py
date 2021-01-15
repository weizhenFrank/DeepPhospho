import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.backends.backend_agg import FigureCanvasAgg


def save_pdf(path):
    with PdfPages(path) as pdf:
        pdf.savefig()


def to_array(figure=None):
    if figure is None:
        figure = plt.gcf()
    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    image_array = np.array(canvas.renderer.buffer_rgba())
    return image_array
