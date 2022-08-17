import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize


class Canvas:
    def __init__(self,color,pixel):
        self.color = color
        self.pixel = pixel

        self.im = np.zeros((pixel,pixel,3))
        self.im[:,:] = color

    def show(self):
        plt.figure()

        plt.imshow(self.im)
        plt.axis("off")
        plt.show()

    def add_square(self, centroid, radius, color, theta):

        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s), (s, c)))

        x, y = np.meshgrid(np.arange(self.im.shape[0]), np.arange(self.im.shape[1]))
        tot = np.concatenate((np.expand_dims(x - centroid[0], axis=-1), np.expand_dims(y - centroid[1], axis=-1)), axis=-1)

        rotated = np.squeeze(np.matmul(R, np.expand_dims(tot, axis=-1)), axis=-1)
        circle_pixels = np.maximum(np.abs(rotated[..., 0]), np.abs(rotated[..., 1])) <= radius
        self.im[circle_pixels, ...] = color

    def get_img(self, res):
        resized = resize(self.im, res)
        return np.transpose(resized, (2,0,1))



N = 20000


equiv_data = []
equiv_lbls = []
haptic = []


blck = np.array([0., 0., 0.])
white = np.array([1., 1., 1.])
res = (64,64)

for i in range(N):
    print(i)
    c1 = Canvas(blck, 100)
    c2 = Canvas(blck, 100)
    angle1 = np.pi*np.random.random()
    angle2 = np.pi*np.random.random()

    c1.add_square((50, 50), 20, white, angle1)
    c2.add_square((50, 50), 20, white, angle2)

    img1 = c1.get_img(res)
    img2 = c2.get_img(res)
    angle = angle2 - angle1

    equiv_data.append([img1, img2])
    equiv_lbls.append(angle)



equiv_data = np.array(equiv_data)
equiv_lbls = np.array(equiv_lbls)
print(equiv_data.shape, equiv_lbls.shape)

np.save('equiv_data.npy', equiv_data)
np.save('equiv_lbls.npy', equiv_lbls)
