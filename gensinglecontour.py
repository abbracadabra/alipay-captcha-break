import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import PIL.ImageFilter
import PIL.Image as Image
import PIL
import PIL.ImageOps
import os
import uuid
import PIL.ImageFilter as ImageFilter
import colorsys
import sys
import traceback
import cv2


dir = os.path.dirname(os.path.realpath(__file__))
savepath = os.path.join(dir,'singlesss')
nfontpath = os.path.join(dir,'fonts')
spfontpath = os.path.join(dir,'spfont')

def perspective(im, widthscale, heightscale, horizontaloblique, verticaloblique, shiftright, shiftdown, MagnifyX,
                MagnifyY, size=None):
    """

    :param im:
    :param widthscale: by which width is multiplied
    :param heightscale: by which height is multiplied
    :param horizontaloblique: Oblique distortion.Think about transformation between normal font to oblique font
    :param verticaloblique: Oblique distortion vertically.
    :param shiftright: Normal shift,rightward.No distortion applied
    :param shiftdown: Normal shift,downward
    :param MagnifyX: Magnify image.Regions further along X axis will be magnified more.
    :param MagnifyY: Magnify image.Regions further along Y axis will be magnified more.
    :return:
    """
    if size == None:
        size = im.size
    # Data is a 8-tuple (a, b, c, d, e, f, g, h)
    # For each pixel (x, y) in the output image, the new value is taken from a position (a x + b y + c)/(g x + h y + 1), (d x + e y + f)/(g x + h y + 1) in the input image
    im = im.transform(size, Image.PERSPECTIVE, (
    1 / widthscale, -horizontaloblique, -shiftright, -verticaloblique, 1 / heightscale, -shiftdown, MagnifyX, MagnifyY),
                      fillcolor=(0))
    im = im.crop(im.getbbox())
    return im

def affine(im, widthscale, heightscale, horizontaloblique, verticaloblique, shiftright, shiftdown, size=None):
    """

    :param im:
    :param widthscale:
    :param heightscale:
    :param horizontaloblique:
    :param verticaloblique:
    :param shiftright:
    :param shiftdown:
    :param size:
    :return:
    """

    if size == None:
        size = im.size
    else:
        im_resize = Image.new('L', size, (0))
        im_resize.paste(im, ((size[0] - im.size[0]) // 2, (size[1] - im.size[1]) // 2, (size[0] + im.size[0]) // 2,
                             (size[1] + im.size[1]) // 2))
        im = im_resize
    # Data is a 6-tuple (a, b, c, d, e, f)
    # For each pixel (x, y) in the output image, the new value is taken from a position (a x + b y + c, d x + e y + f) in the input image
    im = im.transform(size, Image.AFFINE,
                      (1 / widthscale, -horizontaloblique, -shiftright, -verticaloblique, 1 / heightscale, -shiftdown),
                      fillcolor=(0))
    im = im.crop(im.getbbox())
    return im

def zoom(im, scale, size=None):
    if size == None:
        size = im.size
    of = (1 - 1 / scale) / 2
    # Data is 4-tuple box coord
    im = im.transform(size, Image.EXTENT,
                      (im.size[0] * of, im.size[1] * of, im.size[0] * (1 - of), im.size[1] * (1 - of)),
                      fillcolor=(0))
    return im

def radial(im_pil, alpha, centre=(0.5, 0.5), fill=(0), size=None):
    """

    Fitzgibbon, 2001  https://stackoverflow.com/questions/6199636/formulas-for-barrel-pincushion-distortion/6227310#6227310
    barrel distortion: rd = ru(1- alpha * ru^2)  ru = rd/(1- alpha * rd^2)
    """
    if size == None:
        size = im_pil.size
    else:
        im_pil_resize = Image.new('L', size, (0))
        im_pil_resize.paste(im_pil, (
        (size[0] - im_pil.size[0]) // 2, (size[1] - im_pil.size[1]) // 2, (size[0] + im_pil.size[0]) // 2,
        (size[1] + im_pil.size[1]) // 2))
        im_pil = im_pil_resize
        centre = ((im_pil.size[0] * centre[0] + (size[0] - im_pil.size[0]) // 2) / size[0],
                  (im_pil.size[1] * centre[1] + (size[1] - im_pil.size[1]) // 2) / size[1])

    im = np.array(im_pil)
    im_pil.close()
    w = im.shape[1]
    h = im.shape[0]
    index_centre = ((h - 1) * centre[1], (w - 1) * centre[0])
    indices_x, indices_y = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dst_indices = np.stack((indices_x, indices_y), axis=-1)
    ru_inverse_rd = 1 / (1 - np.sum((dst_indices - index_centre) ** 2, axis=-1, keepdims=True) * alpha)
    map_indices = np.int16(index_centre + (dst_indices - index_centre) * ru_inverse_rd)

    def interpolate(i_index, j_index):
        if (i_index >= 0 and i_index < h and j_index >= 0 and j_index < w):
            mm = [im[i_index, j_index]]
            try:
                mm.append(im[i_index + 1, j_index])
            except:
                pass
            try:
                mm.append(im[i_index, j_index + 1])
            except:
                pass
            return np.max(mm, axis=0)
        else:
            return fill

    out = np.array([interpolate(i_index, j_index) for i_index, j_index in np.reshape(map_indices, (-1, 2))])
    out = np.reshape(out, im.shape)
    im = Image.fromarray(np.uint8(out))
    im = im.crop(im.getbbox())
    return im

def randomdistort(im):
    im = im.crop(im.getbbox())
    im = im.rotate(np.random.randn() * 10)
    im = im.crop(im.getbbox())

    # affine
    im = affine(im, widthscale=np.random.uniform(0.7,1.3),
                heightscale=np.random.uniform(0.7,1.3),
                horizontaloblique=np.random.uniform(-0.7,0),
                verticaloblique=np.random.uniform(-0.4,0.4),
                shiftright=0,
                shiftdown=0,size=(50,50))

    # flip
    lr = np.random.choice([0, 1])
    tb = np.random.choice([0, 1])
    if lr == 1:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if tb == 1:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)

    # perspective
    im = perspective(im,
                     widthscale=1,
                     heightscale=1,
                     horizontaloblique=0,
                     verticaloblique=0,
                     # shiftright=np.random.randint(0,int(w*0.5))*np.random.choice([1,-1]),
                     shiftright=0,
                     shiftdown=0,
                     MagnifyX=np.random.uniform(-0.003, 0.003),
                     MagnifyY=np.random.uniform(-0.003, 0.003),
                     size=(50, 50))

    # radial
    im = radial(im,np.random.uniform(-0.000005,0.000005), centre=(np.random.rand(),np.random.uniform(0.3,0.7)), fill=(0),size=(50, 50))
    if lr == 1:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    if tb == 1:
        im = im.transpose(Image.FLIP_TOP_BOTTOM)
    return im

chars = []
for i in range(10):
    chars.append(chr(i+48))
for i in range(26):
    chars.append(chr(i + 65))




def tocountour(im):#pil image
    im = np.array(im)
    #thresh, im = cv2.threshold(im, 230, 255, cv2.THRESH_BINARY)
    im = cv2.Canny(im, 250, 255)
    #_, contours, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #im = np.zeros(im.shape).astype(dtype='uint8')
    #cv2.drawContours(im, contours, -1, (255), 1)

    # x_list = []
    # valid_mask = []
    # for i, cnt in enumerate(contours):
    #     [x, y, w, h] = cv2.boundingRect(cnt)
    #     x_list.append(x)
    #     if w * h > 100:
    #         valid_mask.append(True)
    #     else:
    #         valid_mask.append(False)
    #
    # order = np.argsort(x_list)
    # valid_mask = np.array(valid_mask)[order]
    # order = order[valid_mask]
    # order_contours = [contours[ix] for ix in order]
    # if len(order_contours)>1:
    #     return
    #
    # im = np.zeros(im.shape)
    # cv2.drawContours(im, order_contours, 0, (255), 1)#contour index,color,line width
    return Image.fromarray(np.array(im,dtype='uint8'))


nfonts = [f for f in os.listdir(nfontpath)]
spfonts = [f for f in os.listdir(spfontpath)]

dele = PIL.Image.new('L', (32, 32), (0))
deledraw = PIL.ImageDraw.Draw(dele)
def makeone():
    c = np.random.choice(chars)
    fonts = nfonts
    fontpath = nfontpath
    if c=='Z':
        fonts = spfonts
        fontpath = spfontpath
    if c=='I' or c=='1' or c=='0' or c=='O':
        return
    fn = np.random.choice(fonts)
    fp = os.path.join(fontpath, fn)
    font = PIL.ImageFont.truetype(font=fp, size=np.random.randint(26, 27))

    cw, ch = deledraw.textsize(c, font=font)
    imc = PIL.Image.new('L', (cw, ch), (0))
    cdraw = PIL.ImageDraw.Draw(imc)
    cdraw.text((0, 0), c, font=font, fill=(255))
    im = randomdistort(imc)
    # left,up = (16-imc.size[0]//2,16-imc.size[1]//2)
    # # im.paste(imc, (left, up),
    # #          mask=imc.point(lambda px: (px == 255) * 255))
    # im.paste(imc, (left, up))
    im = tocountour(im)
    im = im.resize((32, 32))
    if not os.path.isdir(savepath):
        os.mkdir(savepath)
    im.save(os.path.join(savepath,
                         c + "_" + "".join(x for x in fn if x.isalnum()) + "_" + str(uuid.uuid4()) + ".jpg"),'JPEG')

for i in range(200000):
    try:
        makeone()
    except:
        print(sys.exc_info()[0])
