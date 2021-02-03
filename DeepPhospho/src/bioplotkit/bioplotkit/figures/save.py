from matplotlib.backends.backend_pdf import PdfPages


def save_pdf(path):
    with PdfPages(path) as pdf:
        pdf.savefig()

