import numpy as np
import cv2

from imgtopat import imread, imsave

COLORS_AROUND = 20
HALF_COLOR_AROUND = COLORS_AROUND // 2

class Color:
    def __init__(self, c):
        self.r = c[0]
        self.g = c[1]
        self.b = c[2]

    def __hash__(self):
        return hash((self.r, self.g, self.b))

    def __eq__(self, other):
        return (self.r, self.g, self.b) == (other.r, other.g, other.b)

    def __ne__(self, other):
        # Not strictly necessary, but to avoid having both x==y and x!=y
        # True at the same time
        return not (self == other)


dmc_number = np.loadtxt('DMCtoRGB.txt', delimiter=' , ', dtype=int, usecols=[0])
dmc_rgb = np.loadtxt('DMCtoRGB.txt', delimiter=' , ', dtype=int, usecols=[2, 3, 4]).astype(np.float32)
dmc_bgr = dmc_rgb[..., ::-1] / 255.
dmc_labels = np.loadtxt('DMCtoRGB.txt', delimiter=' , ', dtype=str, usecols=[1, 5])
TOTAL_COLORS = dmc_rgb.shape[0]


# img = cv2.imread('./input/pattern_l.png')
img = cv2.imread('./input/pattern.png')
Z = img.reshape((-1,3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 16
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))

cv2.imwrite("./output/kmeans.png", res2)
# cv2.imwrite("./output/kmeans_l.png", res2)

color_map = {}

img = imread('./output/kmeans.png')
# img = imread('./output/kmeans_l.png')

# img = imread('./input/pattern_l.png')
x_px = img.shape[0]
y_px = img.shape[1]
dmc = np.zeros(img.shape, dtype=np.float32)


img_bgr =img[..., ::-1].astype(np.float32) / 255.
img_cvt = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
dmc_cvt = cv2.cvtColor(np.reshape(dmc_bgr,(1,-1,3)), cv2.COLOR_BGR2LAB).reshape((-1, 3))

for x in range(x_px):
    for y in range(y_px):
        c = img[x, y]
        c_cvt = img_cvt[x,y]
        lumma = c_cvt[0]
        color = Color(c)
        if not color in color_map:

            # dmc_l = dmc_cvt[...,0].reshape((-1))
            # l_sorted_indexes  = dmc_l.argsort()
            # dmc_l_sorted = dmc_l[l_sorted_indexes]
            # idx = np.searchsorted(dmc_l_sorted, lumma)
            # start_idx = max(idx - HALF_COLOR_AROUND, 0)
            # end_idx = min(idx + HALF_COLOR_AROUND, TOTAL_COLORS)
            #
            # color_indices = l_sorted_indexes[start_idx:end_idx]
            # cvt_colors = dmc_cvt[color_indices]
            # c_ab = c_cvt[1:]
            # ab_colors = cvt_colors[...,1:]
            # dist = ab_colors - c_ab
            # dist = dist ** 2
            # dist = np.sum(dist, axis=1)
            # dist = np.sqrt(dist)
            # i = dist.argmin()
            # idx = color_indices[i]

            # dist = dmc_cvt - c_cvt
            dist = dmc_rgb - c
            dist = dist ** 2
            dist = np.sum(dist, axis=1)
            dist = np.sqrt(dist)
            idx = dist.argmin()


            color_map[color] = idx

        dmc_c = dmc_rgb[color_map[color]]
        dmc[x, y] = dmc_c
imsave('./output/dmc.png', dmc)
# imsave('./output/dmc_l.png', dmc)
print('Floss count: %d' % len(np.unique(np.array(list(color_map.values())))))
