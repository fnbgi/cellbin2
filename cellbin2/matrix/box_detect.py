import os
from cellbin2.contrib.param import ChipBoxInfo
import numpy as np
import cv2 as cv
import tifffile


class MatrixBoxDetector(object):
    def __init__(self, binsize=21, down_size=4, morph_size=9, gene_base_size=19992):
        self.gene_base_size = gene_base_size
        self.binsize = binsize
        self.down_size = down_size
        self.morph_size = morph_size
        self.image: np.ndarray = np.array([])

    @staticmethod
    def _walks_image(dst, box_size, begin_xy=None, end_xy=None):
        """

        Args:
            dst:
            box_size:

        Returns:

        """

        if begin_xy is not None:
            bx, by = begin_xy
        else:
            bx = by = 0

        if end_xy is not None:
            ex, ey = end_xy
        else:
            ey, ex = np.array(dst.shape) - box_size

        max_value = 0
        max_x = max_y = 0

        for i in range(by, ey):
            for j in range(bx, ex):
                _dst = dst[i: i + box_size, j: j + box_size]
                if np.sum(_dst) > max_value:
                    max_value = np.sum(_dst)
                    max_x = j
                    max_y = i

        return max_x, max_y

    def get_box4maximize_area(self, dst, box_size,
                              min_size=256, step=5):
        """

        Args:
            dst:
            box_size:
            min_size:
            step:

        Returns:

        """
        if min(dst.shape) < min_size:

            x, y = self._walks_image(dst, box_size)

            return x, y

        else:

            _dst = cv.pyrDown(dst)

            x, y = self.get_box4maximize_area(_dst, box_size=box_size // 2)

            x, y = map(lambda k: k * 2, (x, y))

            x, y = self._walks_image(dst, box_size=box_size,
                                begin_xy=[x - step, y - step], end_xy=[x + step, y + step])

            return x, y

    def detect(self, matrix: np.ndarray, chip_size: float = 1.0):
        gene_size = int(self.gene_base_size * chip_size)
        self.image = matrix
        gene_image = cv.filter2D(self.image, -1, np.ones((self.binsize, self.binsize), np.float32))
        _, gene_image = cv.threshold(gene_image, 0, 255, cv.THRESH_BINARY)

        gene_image_s = cv.resize(gene_image,
                                 (gene_image.shape[1] // self.down_size, gene_image.shape[0] // self.down_size))

        x, y = self.get_box4maximize_area(gene_image_s, box_size = gene_size // self.down_size)
        x, y = map(lambda k: k * self.down_size + gene_size / 2, (x, y))
        center = [x, y]

        new_box = [[center[0] + i * gene_size / 2, center[1] + j * gene_size / 2]
                   for i in [-1, 1] for j in [-1, 1]]
        new_box = np.array(new_box)[(0, 1, 3, 2), :]

        return new_box


def detect_chip_box(matrix: np.ndarray, chip_size: float = 1.0) -> ChipBoxInfo:
    mbd = MatrixBoxDetector()
    box = mbd.detect(matrix, chip_size)
    cbi = ChipBoxInfo(LeftTop=box[0], LeftBottom=box[1],
                      RightBottom=box[2], RightTop=box[3])

    return cbi


def main():
    matrix_path = r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch/A03599D1_gene.tif'
    matrix = tifffile.imread(matrix_path)
    box = detect_chip_box(matrix)
    print(box)


if __name__ == '__main__':
    main()
