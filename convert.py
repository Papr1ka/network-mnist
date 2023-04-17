from PIL import Image
import numpy as np

im_frame = Image.open('test.png')
np_frame = np.array(im_frame.getdata())
t1 = np.full(784, 1)
# np_frame = (t1 - np_frame)
np_frame.shape
np.savetxt("image", np_frame)