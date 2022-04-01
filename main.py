
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
    file_name = 'Images/cameraman.jpg'
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
    file = 'images/cameraman.jpg'
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
