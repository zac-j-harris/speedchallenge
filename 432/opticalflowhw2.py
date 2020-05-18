import numpy as np
from PIL import Image
from scipy import signal
#from scipy.misc import imread, imresize
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import cv2

# This function is used to help for plotting the flow diagram of optical flow
def plot_optic_flow(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros(flow.shape[:2]+(3,), dtype=np.uint8)
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,1] = 255
    hsv[...,2] = 12+ cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return flow_vis

#This function is used to help to plot the optical flow arrows
def plot_optic_flow_arrows(img, flow, filename, show=True):
    x = np.arange(0, img.shape[1], 1)
    y = np.arange(0, img.shape[0], 1)
    x, y = np.meshgrid(x, y)
    plt.figure()
    fig = plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    step = img.shape[0] // 50
    plt.quiver(x[::step, ::step], y[::step, ::step],
               flow[::step, ::step, 0], flow[::step, ::step, 1],
               color='r', pivot='middle', headwidth=2, headlength=3)
    plt.savefig(filename, bbox_inches='tight', pad_inches=0)

    if show:
        plt.show()


#Lucas Kanade method
def opticalflow(frame1, frame2, border=2):

    # calculate gradients in x, y and t dimensions
    Ix = np.zeros(frame1.shape, dtype=np.float32)
    Iy = np.zeros(frame1.shape, dtype=np.float32)
    It = np.zeros(frame1.shape, dtype=np.float32)
    Ix[1:-1, 1:-1] = cv2.subtract(frame1[1:-1, 2:], frame1[1:-1, :-2]) / 2
    Iy[1:-1, 1:-1] = cv2.subtract(frame1[2:, 1:-1], frame1[:-2, 1:-1]) / 2
    It[1:-1, 1:-1] = cv2.subtract(frame1[1:-1, 1:-1], frame2[1:-1, 1:-1])

    fvalue = np.zeros(frame1.shape + (5,))
    fvalue[..., 0] = Ix ** 2
    fvalue[..., 1] = Iy ** 2
    fvalue[..., 2] = Ix * Iy
    fvalue[..., 3] = Ix * It
    fvalue[..., 4] = Iy * It
    sum_fvalue = np.cumsum(np.cumsum(fvalue, axis=0), axis=1)
    sum_fvalue_without_border = (sum_fvalue[2 * border + 1:, 2 * border + 1:] -
                  sum_fvalue[2 * border + 1:, :-1 - 2 * border] -
                  sum_fvalue[:-1 - 2 * border, 2 * border + 1:] +
                  sum_fvalue[:-1 - 2 * border, :-1 - 2 * border])

    # print(sum_fvalue_without_border.shape)

    '''
    In note, here:
    sum_fvalue_without_border[...,0]  is Ix ** 2
    sum_fvalue_without_border[...,1]  is Iy ** 2
    sum_fvalue_without_border[...,2]  is Ix ** Iy
    sum_fvalue_without_border[...,3]  is Ix ** It
    sum_fvalue_without_border[...,4]  is Iy ** It
    
    '''
    opticalflow = np.zeros(frame1.shape + (2,))

    ## please calculate matrix A determinant det
    ## Please calcualte Velocity_x
    ## Please calcualte Velocity_y

    det =  1/(sum_fvalue_without_border[..., 0]*sum_fvalue_without_border[..., 1] - (sum_fvalue_without_border[..., 2])**2) ## fill blanks here
    velocity_x = np.where(det != 0,
                          1/ det * -1 * (sum_fvalue_without_border[..., 1]*sum_fvalue_without_border[..., 3] + sum_fvalue_without_border[..., 2] * sum_fvalue_without_border[..., 4]), ## fill blanks here
                          0)
    velocity_y = np.where(det != 0,
                          1/ det * (sum_fvalue_without_border[..., 3]*sum_fvalue_without_border[..., 2] - sum_fvalue_without_border[..., 0] * sum_fvalue_without_border[..., 4]), ## fill the blanks here
                          0)
    opticalflow[border + 1: -1 - border, border + 1: -1 - border, 0] = velocity_x[:-1, :-1]
    opticalflow[border + 1: -1 - border, border + 1: -1 - border, 1] = velocity_y[:-1, :-1]

    opticalflow = opticalflow.astype(np.float32)
    return opticalflow



## read image
fr1 = cv2.imread("59.jpg", cv2.IMREAD_GRAYSCALE)
fr2 = cv2.imread("132.jpg", cv2.IMREAD_GRAYSCALE)
fr3 = cv2.imread("569.jpg", cv2.IMREAD_GRAYSCALE)

u = opticalflow(fr2,fr3,8)

# Compare to Farnback optical flow method
flow12 = cv2.calcOpticalFlowFarneback(fr1, fr2, None,
                                      0.5, 3, 15, 3, 5, 1.2, 0)

#plot optical flow arrows
plot_optic_flow_arrows(fr1, u, 'bacteriaflowarrows.png', show=False)

#plot optical flow
cv2.imwrite('LKbacterialflow.jpg', plot_optic_flow(u))

#plot optical flow
cv2.imwrite('FBbacterialflow.jpg', plot_optic_flow(flow12))







# MobileNetV3 discussion but not finished implementation and how it connects to MTCCNN (Multi-Task Cascaded CNN)



