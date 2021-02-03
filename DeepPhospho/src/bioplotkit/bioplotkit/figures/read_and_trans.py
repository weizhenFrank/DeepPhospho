import cv2


def read_img_to_rgb(img_path):
    fig = cv2.imread(img_path, cv2.IMREAD_LOAD_GDAL)
    bgr_ = cv2.split(fig)
    trans_fig = cv2.merge(bgr_[::-1]) if len(bgr_) == 3 else cv2.merge(bgr_[:3][::-1] + bgr_[-1])
    return trans_fig
