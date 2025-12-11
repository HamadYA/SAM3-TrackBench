import os
import numpy as np
import cv2


def overlay_rectangle(img, rect, color=(0, 255, 0), line_width=2):
    """Draw a rectangle [x, y, w, h] on img (in-place)."""
    if rect is None:
        return
    x, y, w, h = rect
    tl_ = (int(round(x)), int(round(y)))
    br_ = (int(round(x + w)), int(round(y + h)))
    cv2.rectangle(img, tl_, br_, color, line_width)


def overlay_mask(img, mask, color=(0, 255, 0), line_width=2, alpha=0.6):
    """Overlay binary/soft mask on img (in-place)."""
    if mask is None:
        return

    m = np.asarray(mask, dtype=np.float32)
    if m.ndim != 2:
        m = m.squeeze()
    if m.ndim != 2:
        return

    m_bin = m > 0.5

    if img.ndim != 3 or img.shape[2] != 3:
        print("overlay_mask: unexpected image shape:", img.shape)
        return

    img_r = img[:, :, 0]
    img_g = img[:, :, 1]
    img_b = img[:, :, 2]

    img_r[m_bin] = alpha * img_r[m_bin] + (1 - alpha) * color[0]
    img_g[m_bin] = alpha * img_g[m_bin] + (1 - alpha) * color[1]
    img_b[m_bin] = alpha * img_b[m_bin] + (1 - alpha) * color[2]

    # draw contour around mask
    M = m_bin.astype(np.uint8)
    if cv2.__version__[0] == '4':
        contours, _ = cv2.findContours(M, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    else:
        _, contours, _ = cv2.findContours(M, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, color, line_width)


class VisualizerSimple:
    """
    Instead of showing a window, this class saves every 'show' call
    into save_dir as PNG/JPG images.
    """
    def __init__(self, save_dir, prefix="frame"):
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(self.save_dir, exist_ok=True)
        self.counter = 0

    def show(self, img, mask=None, box=None):
        # image expected in RGB, HxWx3
        img_vis = img.copy()

        if mask is not None:
            overlay_mask(img_vis, mask, color=(255, 255, 0), line_width=1, alpha=0.7)

        if box is not None:
            overlay_rectangle(img_vis, box, color=(255, 255, 0))

        # optional resize for huge images
        if (img_vis.shape[0] * img_vis.shape[1]) > 1000000:
            img_vis = cv2.resize(img_vis, (0, 0), fx=0.5, fy=0.5)

        # convert RGB -> BGR for OpenCV saving
        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(self.save_dir, f"{self.prefix}_{self.counter:06d}.jpg")
        cv2.imwrite(out_path, img_vis)
        self.counter += 1


class VisualizerTracking:
    """
    Same idea as VisualizerSimple, but used in your tracking loop.
    """
    def __init__(self, save_dir, prefix="frame"):
        self.save_dir = save_dir
        self.prefix = prefix
        os.makedirs(self.save_dir, exist_ok=True)
        self.counter = 0

    def show(self, img, mask=None, box=None):
        # image expected in RGB, HxWx3
        img_vis = img.copy()

        if mask is not None:
            overlay_mask(img_vis, mask, color=(255, 255, 0), line_width=1, alpha=0.7)

        if box is not None:
            overlay_rectangle(img_vis, box, color=(255, 255, 0))

        if (img_vis.shape[0] * img_vis.shape[1]) > 1000000:
            img_vis = cv2.resize(img_vis, (0, 0), fx=0.5, fy=0.5)

        img_vis = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(self.save_dir, f"{self.prefix}_{self.counter:06d}.jpg")
        cv2.imwrite(out_path, img_vis)
        self.counter += 1
