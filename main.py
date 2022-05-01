def skimage_open_image():
    from skimage import io
    file_name = 'Images/0-colors-image.jpg'

    im = io.imread(file_name)
    io.imshow(im)
    io.show()

def skimage_save_image():
    from skimage import io
    import os
    file_name = 'Images/lena.jpg'

    im = io.imread(file_name)
    io.imshow(im)
    io.show()
    cwd = os.getcwd()
    new_image_file = os.path.join(cwd, 'new-lena.jpg')
    io.imsave(new_image_file, im)

def pillow_open_image():
    from PIL import Image
    file_name = 'Images/1-castle.png'
    im = Image.open(file_name)
    im.show()
def pillow_save_image():
    from PIL import Image
    file_name = 'Images/1-castle.png'
    im = Image.open(file_name)
    im.save('new-castle-image.png')

def opencv_open_image():
    import cv2
    file_name = 'Images/bunny.png'
    im = cv2.imread(file_name)
    cv2.imsow(file_name, im)
    cv2.waitKey(10000)   #10초

def matplotlib_open_image():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    file_name = 'Images/cheetah.png'
    im = mpimg.imread(file_name)
    plt.imshow(im)
    plt.show()

def matplotlib_save_image():
    import matplotlib.image as mping

    file_name = 'Images/cheetah.png'
    im = mping.imread(file_name)
    mping.imsave('new-cheetah.png', im)

def matplotlib_show_image():
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    file_name1 = 'Images/cheetah.png'
    im = mpimg.imread(file_name1)
    im1 = mpimg.imread(file_name1)

    file_name2 = 'Images/duck_rgb.png'
    im2 = mpimg.imread(file_name2)

    fig, axs = plt.subplots(1,2)   #1 row , 2 cols
    axs[0].imshow(im1)
    axs[1].imshow(im2)
    plt.show()

def numpy_create_image():
    import numpy as np
    import matplotlib.pyplot as plt
    shape = (8, 8)    # 8*8의 64pix
    im = np.zeros(shape)
    im1 = np.zeros(shape)
    im1.fill(255)

    fig, axs = plt.subplots(1, 3)

    img_r = np.zeros((255, 50))
    for i in range(255) :
        img_r[i] = i

    my_cmap = 'inferno_r' # Default: viridis
    # 참고 color maps: https://gallantlab.github.io/colormaps.html

    axs[0].imshow(im, cmap=my_cmap, vmin=0, vmax=255)
    axs[1].imshow(im1, cmap=my_cmap, vmin=0, vmax=255)
    axs[2].imshow(img_r, cmap=my_cmap, vmin=0, vmax=255)

    axs[0].title.set_text('0 values'), axs[1].title.set_text('255 values')
    plt.show()


def view_image_by_channels():
    from skimage import io
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt
    file_name = 'Images/1-castle.png'

    im = io.imread(file_name)
    print(type(im), im.shape) #<class 'numpy.ndarray'>

    im_red = im[:, :, 0]      # first channel
    im_green = im[:, :, 1]    # second channel
    im_blue = im[:, :, 2]
    im_gray = rgb2gray(im)

    figure, axs = plt.subplots(nrows=2, ncols=3)
    axs[0, 0].imshow(im_red, cmap='Reds'), axs[0, 0].title.set_text('Reds')
    axs[0, 1].imshow(im_green, cmap='Greens'), axs[0, 1].title.set_text('Greens')
    axs[0, 2].imshow(im_blue, cmap='Blues'), axs[0, 2].title.set_text('Blues')
    axs[1, 0].imshow(im_gray, cmap='gray'), axs[1, 0].title.set_text('Gray')
    axs[1, 1].imshow(im), axs[1, 1].title.set_text('Original image')

    plt.tight_layout()
    plt.show()

def ex1() :  #랜덤한 그림 생성
    import numpy as np
    import matplotlib.pyplot as plt
    new = np.random.randint(0,255,(8,8))
    figure = plt.figure(figsize=(8,8))
    ax1 = figure.add_subplot(121)
    ax1.title.set_text('randum')
    plt.imshow(new, cmap='gray')
    plt.show()


def ex2() :    #체스판 생성
    import numpy as np
    import matplotlib.pyplot as plt
    check = np.zeros((8,8))
    check[1::2,::2] = 1    #체스판 행에 대해 홀수번에 1
    check[::2, 1::2] =1    #체스판 열에 대해 짝수번에 1
    plt.imshow(check, cmap='gray')
    plt.show()

def point_operation_process_addition():    #이미지 밝기
    from skimage import io
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt
    file_name = 'Images/einstein.jpg'

    im = io.imread(file_name)
    im_gray = rgb2gray(im)

    ks_add_bright = [0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 1]
    ks_reduce_bright = [-i for i in ks_add_bright]

    ks = ks_add_bright
    figure, axs = plt.subplots(nrows=1, ncols=len(ks_add_bright))
    for idx,k in enumerate(ks):
        im_k = im_gray + k
        axs[idx].imshow(im_k, cmap= 'gray', vmin=0, vmax=1)
        axs[idx].title.set_text(f'k={k}'), axs[idx].axis('off')

    plt.tight_layout()
    plt.show()



def point_operation_process_multiplication():     #밝기 곱
    from skimage import io
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt
    file_name = 'Images/player.png'
    im = io.imread(file_name)
    im_gray = rgb2gray(im)

    ks = [0.6, 0.8, 1, 1.2, 1.4, 1.6, 1.8, 2]
    rows = 2
    cols = int(len(ks)/rows)
    figure, axs = plt.subplots(nrows=rows, ncols=cols)
    row, col = 0, -1
    for k in ks:
        im_k = im_gray * k
        col += 1
        if col >= cols:
            row, col = 1, 0
        axs[row, col].imshow(im_k, cmap='gray', vmin=0, vmax=1)
        axs[row, col].title.set_text(f'k={k}'), axs[row, col].axis('off')
    plt.tight_layout()
    plt.show()



def inverting_image() :     #이미지 반전
    from skimage import io
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt
    file_name = 'Images/cameraman.jpg'
    im = io.imread(file_name)
    im_gray = rgb2gray(im)
    # -a+max(a)과 max(a)-a 로 화면전환함 a 값 변경 가능
    a_max = 1
    im_invert = a_max - im_gray
    figure, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(im_gray, cmap='gray', vmin=0,vmax=1)
    axs[1].imshow(im_invert,cmap='gray', vmin=0,vmax=1)

    plt.tight_layout()
    plt.show()

def detect_new_object() :     #틀린 픽셀 검색
    from skimage import io
    import matplotlib.pyplot as plt
    file1, file2 = 'images/goal1.png', 'images/goal2.png'

    im1, im2 = io.imread(file1), io.imread(file2)
    im = im2 - im1
    a_max = 125
    im = im - a_max

    fig, axs =plt.subplots(nrows=1, ncols=3)
    [ax.set_axis_off() for ax in axs.ravel()]
    axs[0].imshow(im1), axs[1].imshow(im2), axs[2].imshow(im)

    plt.tight_layout(), plt.show()
    plt.imshow(im), plt.axis('off'), plt.show()



def calculate_hist_of_image():
    from skimage import io
    import matplotlib.pyplot as plt
    import numpy as np
    file = 'images/room.jpg'
    im = io.imread(file)
    im_flatten = im.flatten()
    hist = np.zeros(256)
    for i in im_flatten:
        hist[i] += 1
    x = [i for i in range(0, 256)]
    fig, axs = plt.subplots(nrows=2,ncols=1)
    axs[0].imshow(im)
    axs[1].bar(x, hist)
    plt.show()

def hist_skimage() :
    from skimage import io
    from matplotlib import pyplot as plt
    file = 'Images/cameraman.jpg'
    im = io.imread(file)
    fig, axs = plt.subplots(nrows=2, ncols=1)
    axs[0].imshow(im)
    axs[1].hist(im.ravel(), bins=256)
    plt.show()

def hist_skimage_bychannels() :
    from skimage import io
    import matplotlib.pyplot as plt
    file = 'images/1-castle.png'
    im = io.imread(file)

    fig, axs = plt.subplots(nrows=2,ncols=1)
    axs[0].imshow(im)
    axs[1].hist(im.ravel(), bins=256, color='orange',)
    axs[1].hist(im[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
    axs[1].hist(im[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
    axs[1].hist(im[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
    plt.xlabel('Intensity Value')
    plt.ylabel('Count')
    plt.legend(['Total', 'Red_Channel', 'Green_channel','Blue_Channel'])
    plt.show()

def hist_equalization():
    from skimage import io, exposure
    import matplotlib.pyplot as plt

    file = 'Images/flowers.png'
    im = io.imread(file)
    im_equalization = exposure.equalize_hist(im)
    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[1, 0].hist(im.ravel(), bins=256), axs[0,0].imshow(im)
    axs[0, 1].imshow(im_equalization), axs[1,1].hist(im_equalization.ravel(), bins=256)
    plt.tight_layout()
    plt.show()

def get_5x5_neighbors(x, y):
    neighbors = [(x - 1, y - 1),
                 (x, y - 1),
                 (x + 1, y - 1),
                 (x - 1, y),
                 (x, y),
                 (x + 1, y),
                 (x - 1, y + 1),
                 (x, y + 1),
                 (x + 1, y + 1),
                 (x - 2, y - 2),
                 (x - 2, y - 1),
                 (x - 2, y),
                 (x - 2, y + 1),
                 (x - 2, y + 2),
                 (x + 2, y - 2),
                 (x + 2, y - 1),
                 (x + 2, y),
                 (x + 2, y + 1),
                 (x + 2, y + 2),
                 (x + 1, y + 2),
                 (x, y + 2),
                 (x - 1, y + 2),
                 (x - 1, y - 2),
                 (x, y - 2),
                 (x + 1, y - 2)
                 ]
    return neighbors

def get_average_neighbors(im, list_neighbors):
    total = 0
    for point in list_neighbors:
        total += im[point[0], point[1]]
    average = total / len(list_neighbors)
    return average

def get_neighbor_values(im, list_neighbors):
    list_value = []
    for point in list_neighbors:
        list_value.append(im[point[0], point[1]])
    return list_value

def get_matrix_33_filtering(matrix1, matrix2):
    import numpy as np
    matrix1, matrix2 = np.asarray(matrix1), np.asarray(matrix2)
    total = 0
    for i in range(3):
        for j in range(3):
            total += matrix1[i, j] * matrix2[i, j]
    return total/9

def gaussian_filter():
    import skimage.io
    import matplotlib.pyplot as plt
    import skimage.filters

    file_name = 'Images/cheetah.png'
    image = skimage.io.imread(file_name)
    plt.title("original image"), plt.imshow(image), plt.show()
    sigmoids = [3,5,7,9]   #각각 3x3,5x5,7x7,9x9

    for sigma in sigmoids:
        blurred = skimage.filters.gaussian(image, sigma=sigma, multichannel = True)
        plt.imshow(blurred), plt.title(f"gaussian filter, sigma = {sigma}")
        plt.show()



def sobel_filter_manual():
    from skimage import io
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt
    import numpy as np
    import math
    file_name = 'Images/1-castle.png'
    im = rgb2gray(io.imread(file_name))
    im_gray_pad = np.pad(im, pad_width=1)
    n, m = im_gray_pad.shape[0], im_gray_pad.shape[1]
    im_filter_x, im_filter_y = np.zeros(im_gray_pad.shape), np.zeros(im_gray_pad. shape)
    sobel_image = np.zeros(im_gray_pad. shape)

    Gx = [[-1, 0, 1],
          [-2, 0, 2],
          [-1, 0, 1]]
    Gy = [[1, 2, 1],
          [0, 0, 0],
          [-1, -2, -1]]

    for x in range(1, n - 1):
        for y in range(1, m - 1):
            neighbors = get_3x3_neighbors(x, y)
            neighbors_value = get_neighbor_values(im_gray_pad, neighbors)
            neighbors_33 = np.asarray(neighbors_value).reshape((3, 3))
            im_filter_x[x, y] = get_matrix_33_filtering(neighbors_33, Gx)
            im_filter_y[x, y] = get_matrix_33_filtering(neighbors_33, Gy)

    for i in range(n):
        for j in range(m):
            sobel_image[i, j] = math.sqrt(im_filter_x[i, j] ** 2 + im_filter_y[i, j] ** 2)

    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(im, cmap='gray'), axs[0].set_title('Original image')
    axs[1].imshow(sobel_image, cmap='gray'), axs[1].set_title('Sobel filter image')
    plt.tight_layout(), plt.show()

def sobel_filter_skimage():
    import matplotlib.pyplot as plt
    from skimage import filters
    from skimage.color import rgb2gray
    from skimage import io

    file_name = 'Images/1-castle.png'
    image = rgb2gray(io.imread(file_name))
    edge_sobel= filters.sobel(image)

    fig, axes = plt.subplots(ncols=2)
    axes[0].imshow(image, cmap='gray'), axes[0].set_title('Original image')
    axes[1].imshow(edge_sobel, cmap='gray'), axes[1].set_title('Sobel Edge Detection')
    plt.tight_layout(), plt.show()

def canny_edge_detection():
    import matplotlib.pyplot as plt
    from skimage import feature
    import skimage.io
    from skimage.color import rgb2gray

    file_name = 'Images/goal1.png'
    im = rgb2gray(skimage.io.imread(file_name))

    sigmas = [1, 3, 5]
    for sigma in sigmas:
        edge_image = feature.canny(im, sigma=sigma)
        plt.imshow(edge_image, cmap='gray')
        plt.title(f"Sigma = {sigma}")
        plt.show()


def region_processing():
    import numpy as np
    import matplotlib.pyplot as plt
    from skimage import io
    from skimage.color import rgb2gray
    from skimage.segmentation import watershed
    from skimage.filters import sobel


    file_name = 'Images/cameraman.jpg'
    img =rgb2gray(io. imread(file_name))
    sobel_img = sobel(img)
    markers = np.zeros_like(img)
    markers[img < 30/255] = 1
    markers[img > 150/255] = 2
    region_img = watershed(sobel_img, markers)

    fig, axs = plt.subplots(nrows=2, ncols=2)
    axs[0, 0].imshow(img, cmap='gray'), axs[0, 0].title.set_text('Original image')
    axs[0, 1].imshow(sobel_img, cmap='gray'), axs[0, 1].title.set_text('Sobel filter image')
    axs[1, 0].imshow(markers, cmap='gray'), axs[1, 0].title.set_text('marker image')
    axs[1, 1].imshow(region_img, cmap='gray'), axs[1, 1].title.set_text('region image')
    [ax.axis('off') for ax in axs.ravel()]

    plt.show()
    plt.tight_layout()

def average_smoothing():
    from skimage import io
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt
    import numpy as np
    file_name = 'Images/coins.jpg'

    im = io.imread(file_name)
    im = rgb2gray(im)
    im_gray_pad = np.pad(im, pad_width=1)

    n,m = im_gray_pad.shape[0], im_gray_pad.shape[1]
    im_filter = np.zeros(im_gray_pad.shape)

    for x in range(2,n-2):
        for y in range(2, m-2):
            neighbors = get_5x5_neighbors(x, y)
            im_filter[x,y] = get_average_neighbors(im_gray_pad,neighbors)

    fig, axs = plt.subplots(nrows=1, ncols=2)
    axs[0].imshow(im_gray_pad, cmap='gray'), axs[1].imshow(im_filter, cmap='gray')
    plt.tight_layout()
    plt.show()



