from os.path import exists
from glob import glob
import cv2
import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import scipy.io as sio
from mba import *
from skimage.feature import structure_tensor, structure_tensor_eigvals
from skimage import feature
from numpy import linalg as LA
import imutils
import scipy
from PIL import Image
from sklearn.cluster import MiniBatchKMeans
import time
import os


class get_arguments:
    data_folder = "../Inputs_Paper/Content/Input/"
    max_dim = 200
    ns = 2
    KERNEL = 30
    K = 3
    d = 6
    r = 6
    maxite = 8
    dirName = "../Inputs_Paper/Content/Grid/"


MODEL = "model.yml.gz"
# data_folder = './Results/aldreay'


def inRange(x, y, w, h):
    return x > 0 and x <= (h - 1) and y > 0 and y <= (w - 1)


def XDoG(inputIm):
    #    inputIm = cv.imread(input_img,0)
    Tao = 0.981
    Phi = 200
    Epsilon = -0.005
    k = 1.6
    Sigma = 1.8
    size = int(2 * np.ceil(2 * Sigma) + 1)
    size1 = int(2 * np.ceil(2 * k * Sigma) + 1)
    gFilteredIm1 = cv2.GaussianBlur(inputIm, (size, size), Sigma)
    gFilteredIm2 = cv2.GaussianBlur(inputIm, (size1, size1), Sigma * k, Sigma * k)

    differencedIm2 = gFilteredIm1 - (Tao * gFilteredIm2)

    x = differencedIm2.shape[0]
    y = differencedIm2.shape[1]

    # Extended difference of gaussians
    for i in range(x):
        for j in range(y):
            if differencedIm2[i, j] >= Epsilon:
                differencedIm2[i, j] = 1
            else:
                differencedIm2[i, j] = 1 + np.tanh(
                    Phi * ((differencedIm2[i, j] - Epsilon))
                )

    differencedIm2 = np.double(differencedIm2)
    out = np.zeros(differencedIm2.shape, np.double)
    normalized = cv2.normalize(differencedIm2, out, 1.0, 0.0, cv2.NORM_MINMAX)
    mean = np.mean(normalized)
    ret, img_in = cv.threshold(normalized, mean, 1.0, cv.THRESH_BINARY)

    return img_in


def draw_arrowline(path, input_img_path, max_dim):
    global flowField
    #    dis = cv2.imread(input_img,0)
    dis = np.expand_dims(cv2.imread(input_img_path, 0), axis=2)
    if dis.shape[2] == 1:
        imgt = dis
        imgt = np.append(imgt, dis, axis=2)
        imgt = np.append(imgt, dis, axis=2)
        dis = imgt
    long = max(dis.shape)
    scale = max_dim / long
    dis = scipy.misc.imresize(dis, scale)

    resolution = 50
    h, w = dis.shape[0], dis.shape[1]
    for i in range(0, h, resolution):
        for j in range(0, w, resolution):

            v = flowField[i][j]
            p = (j, i)
            p2 = (int(j + v[1] * 30), int(i + v[0] * 30))
            dis = cv2.arrowedLine(dis, p, p2, (0, 0, 255), 1, 8, 0, 0.3)
    cv2.imwrite(path + "etf_kernel" + "_" + ".png", dis)
    np.save(path + "np_etf_kernel" + "_" + ".npy", flowField)


def JFA(img2, nsamples):
    h = img2.shape[0]
    w = img2.shape[1]
    rawsites = np.transpose(np.nonzero(255 - img2))
    offsets = np.zeros((nsamples, 2))
    sqrtNSamples = np.round(np.sqrt(nsamples)).astype("int")
    jstep = 1.0 / sqrtNSamples
    for i in range(sqrtNSamples):
        for j in range(sqrtNSamples):
            offsets[i * sqrtNSamples + j, :] = [
                (i + np.random.rand()) * jstep,
                (j + np.random.rand()) * jstep,
            ]

    B = np.expand_dims(np.zeros((h, w)), 2)
    D = np.expand_dims(np.zeros((h, w)), 2)
    for oidx in range(nsamples):
        A = np.zeros((h, w)).astype("int")
        A_d = np.ones((h, w))
        for i in range(rawsites.shape[0]):
            x = rawsites[i, 0]
            y = rawsites[i, 1]
            A[x, y] = i + 1
            A_d[x, y] = 0

        offset = offsets[oidx, :]
        sites = rawsites.copy() + np.matlib.repmat(offset, rawsites.shape[0], 1)
        # Repeat k steps
        k = np.ceil(np.log2(np.maximum(h, w))).astype("int")
        for i in range(1, k):
            j = 2**i

            for y in range(w):
                for x in range(h):
                    if A[x, y] != 0:
                        for xx in range(-1, 1):
                            for yy in range(-1, 1):
                                x1 = np.round(x + w / j * xx).astype("int")
                                y1 = np.round(y + h / j * yy).astype("int")
                                if inRange(x1, y1, w, h):
                                    if A[x1, y1] == 0:
                                        A[x1, y1] = A[x, y]
                                        idx0 = A[x, y].astype("int") - 1
                                        dx0 = x1 - sites[idx0, 0]
                                        dy0 = y1 - sites[idx0, 1]
                                        dist0 = np.sqrt(dx0**2 + dy0**2)
                                        A_d[x1, y1] = dist0
                                    else:
                                        # Consider two possible closest feature points
                                        idx0 = A[x, y].astype("int") - 1
                                        idx1 = A[x1, y1].astype("int") - 1
                                        # Distance between the point and first feature point
                                        dx0 = x1 - sites[idx0, 0]
                                        dy0 = y1 - sites[idx0, 1]
                                        dist0 = dx0**2 + dy0**2
                                        # Distance between the point and sencond feature point
                                        dx1 = x1 - sites[idx1, 0]
                                        dy1 = y1 - sites[idx1, 1]
                                        dist1 = dx1**2 + dy1**2

                                        if dist0 < dist1:
                                            A[x1, y1] = A[x, y]
                                            A_d[x1, y1] = np.sqrt(dist0)

        A = np.expand_dims(A, 2)
        B = np.append(B, A, 2)
        A_d = np.expand_dims(A_d, 2)
        D = np.append(D, A_d, 2)

    return D[:, :, nsamples], B[:, :, nsamples]


def imagesc(img, title=None):
    cmin = [0, 0]
    cmax = [img.shape[0], img.shape[1]]
    C = np.mgrid[0 : cmax[0], 0 : cmax[1]]
    plt.figure()
    plt.pcolormesh(C[0], C[1], img)
    plt.colorbar()
    plt.xlim([cmin[0], cmax[0]])
    plt.ylim([cmin[1], cmax[1]])
    plt.tight_layout()
    if title is not None:
        plt.title(title)


def FVI(img):
    global flowField

    """
    This fuction determines the Feature Vector Interpolation applying scattered Data Interpolation.
    
    Inputs:
        - im_in: edges binary image scaled 0-255 (Feature linemap)
        
    Output:
        - Tangential Field vectors
        - Normal Field vectors
        - Angles Tangential
        - Magnitudes Tangetial
        - Angles Normal
        - Magnitudes Normal
        
    
    """

    def interpolation(imgO, nx, ny, imF):
        cmin = [0, 0]
        cmax = [imgO.shape[1], imgO.shape[0]]
        C = np.mgrid[0 : cmax[0], 0 : cmax[1]]
        DataIndf = np.transpose(np.nonzero(np.round(255 - imF)))
        DataIndx = DataIndf[:, 1]
        DataIndy = DataIndf[:, 0]
        DataInd = DataIndf.copy()
        DataInd[:, 0] = DataIndx
        DataInd[:, 1] = DataIndy
        DataVal = imgO[np.nonzero(np.round(255 - imF))]

        interp = mba2(cmin, cmax, [nx, ny], DataInd, DataVal)
        print(interp)
        G = interp(C.transpose((1, 2, 0)).copy())
        # plt.figure(figsize=(12,4))
        return G.transpose()

    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

    # Change Direciton
    aux = np.ones(sobelx.shape)
    aux[sobelx < 0] = -1
    sobelx = aux * sobelx
    sobely = aux * sobely

    Norm = np.sqrt(sobelx**2 + sobely**2)
    Grad_Mag = Norm.copy()
    Norm[Norm == 0] = 1
    ImF = np.zeros(sobelx.shape)
    ImF[Grad_Mag <= 0.1] = 255
    sobelx_N = np.divide(sobelx, Norm)
    sobely_N = np.divide(sobely, Norm)

    ty = -np.ones(sobelx.shape)
    ty[sobelx == 0] = 0
    ty_int = -np.multiply(ty, sobely)
    sobelx_in = sobelx.copy()
    sobelx_in[sobelx == 0] = 1
    tx = -np.divide(ty_int, sobelx_in)
    tx[np.logical_and((sobelx == 0), (sobely != 0))] = 1
    Norm = np.sqrt(tx**2 + ty**2)
    Mag_v2 = Norm
    Norm[Norm == 0] = 1
    tx_N = np.divide(tx, Norm)
    ty_N = np.divide(ty, Norm)

    Axx = sobelx * sobelx
    Ayy = sobely * sobely
    Axy = sobelx * sobely

    nxx = KERNEL
    nyy = KERNEL
    # interp = mba2(cmin, cmax, [nx,ny], coo, val)
    Axx_filled = interpolation(Axx, nxx, nyy, ImF)
    Ayy_filled = interpolation(Ayy, nxx, nyy, ImF)
    Axy_filled = interpolation(Axy, nxx, nyy, ImF)

    lambda1_filled, lambda2_filled = structure_tensor_eigvals(
        Axx_filled, Axy_filled, Ayy_filled
    )

    v2_x_filled = lambda2_filled - Ayy_filled
    v2_y_filled = Axy_filled

    Norm = np.sqrt(v2_x_filled**2 + v2_y_filled**2)
    Mag_v2 = Norm.copy()
    Norm[Norm == 0] = 1
    v2_x_filled_N = np.divide(v2_x_filled, Norm)
    v2_y_filled_N = np.divide(v2_y_filled, Norm)
    v2_x_filled_N[ImF == 0] = tx_N[ImF == 0]
    v2_y_filled_N[ImF == 0] = ty_N[ImF == 0]
    Ang_v2 = np.zeros(sobelx.shape)
    Ang_v2[v2_x_filled_N != 0] = np.arctan(
        np.divide(v2_y_filled_N[v2_x_filled_N != 0], v2_x_filled_N[v2_x_filled_N != 0])
    )

    v1_x_filled = Axy_filled
    v1_y_filled = lambda1_filled - Axx_filled
    Norm = np.sqrt(v1_x_filled**2 + v1_y_filled**2)
    Mag_v1 = Norm.copy()
    Norm[Norm == 0] = 1
    v1_x_filled_N = np.divide(v1_x_filled, Norm)
    v1_y_filled_N = np.divide(v1_y_filled, Norm)
    v1_x_filled_N[ImF == 0] = sobelx_N[ImF == 0]
    v1_y_filled_N[ImF == 0] = sobely_N[ImF == 0]
    Ang_v1 = np.zeros(sobelx.shape)
    Ang_v1[v1_x_filled_N != 0] = np.arctan(
        np.divide(v1_y_filled_N[v1_x_filled_N != 0], v1_x_filled_N[v1_x_filled_N != 0])
    )

    flowField[:, :, 0] = v1_x_filled_N
    flowField[:, :, 1] = v1_y_filled_N

    ETF_Mag = v2_x_filled_N**2 + v2_y_filled_N**2

    ETF = v2_x_filled_N
    ETF = np.expand_dims(ETF, 2)
    v2_y_filled2 = np.expand_dims(v2_y_filled_N, 2)
    ETF = np.append(ETF, v2_y_filled2, 2)

    ENF = v1_x_filled_N
    ENF = np.expand_dims(ENF, 2)
    v1_y_filled_N2 = np.expand_dims(v1_y_filled_N, 2)
    ENF = np.append(ENF, v1_y_filled_N2, 2)

    return ETF, ENF, Mag_v2, Ang_v2, Mag_v1, Ang_v1, ETF_Mag


def rotate_bound(image, angle, px):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))  # X
    nH = int((h * cos) + (w * sin))  # Y

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # compute the new bounding dimensions of the image
    # pxW = int((px[0] * sin) + (px[1] * cos) + M[0, 2])
    # pxH = int((px[0] * cos) + (px[1] * sin) + M[1, 2])

    # px_N = [pxW,pxH]

    px_N = np.matmul(M, px)

    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH)), px_N.astype("int")


def S_Reflection(k, lim):
    k2 = k % lim[0]
    return np.minimum(k2, lim - k2)


def StructureGrid(args):
    global flowField
    global KERNEL
    data_folder = args.content
    max_dim = args.max_dim
    ns = 2
    KERNEL = 30
    K = 3
    d = 6
    r = 6
    maxite = args.max_ite
    image_files = sorted(glob("{}/*.pn*g".format(data_folder)))
    # image_file=image_files[N_im]

    for N_im, image_file in enumerate(image_files):
        # for kkk in range(0,1):
        image_file = image_files[N_im]
        current_directory = os.getcwd()
        path = "GridCheckpoints/%04d/" % N_im
        final_directory = os.path.join(current_directory, path)
        if not os.path.exists(final_directory):
            os.makedirs(final_directory)
        Filenameout = image_file.replace("Image", "Grid")

        idx = Filenameout.find("\\")
        dirName = Filenameout[:idx]
        if exists(Filenameout):
            continue
        COLOUR_OR_GRAY = 0
        model = "model.yml.gz"
        img = cv.imread(image_file)
        long = max(img.shape)
        scale = max_dim / long
        img = scipy.misc.imresize(img, scale)
        img_shape = img.shape

        COLOUR_OR_GRAY = 0
        edge_detection = cv.ximgproc.createStructuredEdgeDetection(model)
        rgb_im = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)
        orimap = edge_detection.computeOrientation(edges)
        edges_in = edge_detection.edgesNms(edges, orimap)

        edges_in = (255 * (1 - edges_in)).astype("uint8")

        edgesImp = XDoG(edges_in)

        image = img
        (h, w) = image.shape[:2]

        # convert the image from the RGB color space to the L*a*b*
        # color space -- since we will be clustering using k-means
        # which is based on the euclidean distance, we'll use the
        # L*a*b* color space where the euclidean distance implies
        # perceptual meaning
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # reshape the image into a feature vector so that k-means
        # can be applied
        image = image.reshape((image.shape[0] * image.shape[1], 3))

        # apply k-means using the specified number of clusters and
        # then create the quantized image based on the predictions
        clt = MiniBatchKMeans(n_clusters=K)
        labels = clt.fit_predict(image)
        quant = clt.cluster_centers_.astype("uint8")[labels]

        # reshape the feature vectors to images
        quant = quant.reshape((h, w, 3))
        image = image.reshape((h, w, 3))

        # convert from L*a*b* to RGB
        quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)

        rgb_im = cv.cvtColor(quant, cv.COLOR_BGR2RGB)
        edges = edge_detection.detectEdges(np.float32(rgb_im) / 255.0)

        orimap = edge_detection.computeOrientation(edges)

        edges = edge_detection.edgesNms(edges, orimap)

        edges = (255 * (1 - edges)).astype("uint8")

        edgesImp = XDoG(edges)

        edgesKmeans = XDoG(edges)

        edgesF = 0.5 * edgesKmeans + 0.5 * edgesImp

        #
        ret, img_in = cv.threshold(edgesF, 0.5, 255, cv.THRESH_BINARY)
        img_in = img_in.astype("uint8")

        SIZE = img_in.shape
        flowField = np.zeros((SIZE[0], SIZE[1], 2), dtype=np.float32)

        start = time.time()
        ETF, ENF, Mag_v2, Ang_v2, Mag_v1, Ang_v1, ETF_Mag = FVI(img_in)
        draw_arrowline(path, image_file, max_dim)
        end = time.time()
        print("FVI Time:", end - start)

        start = time.time()
        Dist, Regions = JFA(img_in, ns)
        end = time.time()
        print("JFA Time:", end - start)

        # Distance values are convertes into real numbers in [0,d/2]
        x = np.mod(Dist, d)
        S = np.minimum(x, d - x)

        In_N = d / 2 * np.random.random_sample(S.shape)
        In_T = S

        In_T_F = In_T.copy()

        In_N_F = In_N.copy()

        # plt.close('all')
        start = time.time()

        ite = 1
        FinalGrid = In_N_F + In_T_F
        FinalGrid = (255 / FinalGrid.max() * FinalGrid).astype("uint8")
        FinalGrid_Im = Image.fromarray(FinalGrid, mode="L")
        Filename = path + "Iteration%04d.png" % (ite)
        FinalGrid_Im.save(Filename)

        GridFinal = np.zeros((SIZE[0], SIZE[1], 2), dtype=np.float32)
        GridFinal[:, :, 0] = In_T_F
        GridFinal[:, :, 1] = In_N_F

        np.save(path + "Grid" + str(KERNEL) + "_" + str(ite) + ".npy", GridFinal)

        while ite < maxite:
            for y in range(In_N.shape[0]):
                for x in range(In_N.shape[1]):
                    rotated, pxN = rotate_bound(
                        In_T_F, -Ang_v2[y, x] * 360 / (2 * np.pi), np.array([x, y, 1])
                    )

                    infX_L = np.maximum(pxN[0] - r - 1, 0)
                    infX_H = np.minimum(pxN[0] + r, rotated.shape[1])
                    infY_L = np.maximum(pxN[1] - r - 1, 0)
                    infY_H = np.minimum(pxN[1] + r, In_N.shape[0])
                    Win = rotated[infY_L:infY_H, infX_L:infX_H]
                    w_v = np.mean(Win, axis=1)
                    if w_v.size < 2 * r + 1:
                        continue
                    i = np.arange(-r, r + 0.1, 1)
                    Err = np.zeros((2 * r + 1, 1))
                    record = []
                    for m in range(2 * d + 1):
                        x_m = m / 2
                        ip = i + x_m
                        S_x = S_Reflection(
                            np.reshape(ip, (i.size, 1)), d * np.ones((i.size, 1))
                        )
                        Err[m] = np.mean((S_x - np.reshape(w_v, (w_v.size, 1))) ** 2)
                        record = np.append(record, S_x)
                    minInd = np.argmin(Err)
                    # x_m=np.arange(minInd/2, minInd/2+0.5, 0.5/i.size)
                    x_mf = minInd / 2
                    ipf = i + x_mf
                    S_x = S_Reflection(
                        np.reshape(ipf, (ipf.size, 1)), d * np.ones((ipf.size, 1))
                    )
                    aux = S_x[int(S_x.size / 2)]
                    In_T_F[y, x] = S_x[int(S_x.size / 2)]
            # plt.figure()
            # plt.pcolormesh(C[0], C[1], In_T_F)
            # plt.colorbar()

            for y in range(In_N.shape[0]):
                for x in range(In_N.shape[1]):
                    rotated, pxN = rotate_bound(
                        In_N_F, -Ang_v1[y, x] * 360 / (2 * np.pi), np.array([x, y, 1])
                    )
                    infX_L = np.maximum(pxN[0] - r - 1, 0)
                    infX_H = np.minimum(pxN[0] + r, rotated.shape[1])
                    infY_L = np.maximum(pxN[1] - r - 1, 0)
                    infY_H = np.minimum(pxN[1] + r, In_N.shape[0])
                    Win = rotated[infY_L:infY_H, infX_L:infX_H]
                    w_v = np.mean(Win, axis=1)
                    if w_v.size < 2 * r + 1:
                        continue
                    i = np.arange(-r, r + 0.1, 1)
                    Err = np.zeros((2 * r + 1, 1))
                    record = []
                    for m in range(2 * d + 1):
                        x_m = m / 2
                        ip = i + x_m
                        S_x = S_Reflection(
                            np.reshape(ip, (i.size, 1)), d * np.ones((i.size, 1))
                        )
                        Err[m] = np.mean((S_x - np.reshape(w_v, (w_v.size, 1))) ** 2)
                        record = np.append(record, S_x)
                    minInd = np.argmin(Err)
                    # x_m=np.arange(minInd/2, minInd/2+0.5, 0.5/i.size)
                    x_mf = minInd / 2
                    ipf = i + x_mf
                    S_x = S_Reflection(
                        np.reshape(ipf, (ipf.size, 1)), d * np.ones((ipf.size, 1))
                    )
                    aux = S_x[int(S_x.size / 2)]
                    In_N_F[y, x] = S_x[int(S_x.size / 2)]
            ite += 1
            FinalGrid = In_N_F + In_T_F
            FinalGrid = (255 / FinalGrid.max() * FinalGrid).astype("uint8")
            FinalGrid_Im = Image.fromarray(FinalGrid, mode="L")

            Filename = path + "Iteration%04d.png" % (ite)
            FinalGrid_Im.save(Filename)
            GridFinal[:, :, 0] = In_T_F
            GridFinal[:, :, 1] = In_N_F
            np.save(path + "Grid" + str(KERNEL) + "_" + str(ite) + ".npy", GridFinal)
        if not os.path.exists(dirName):
            os.makedirs(dirName)
        # Filenameout=dirName+'%04d.png' % (N_im+1)
        FinalGrid_Im.save(Filenameout)


if __name__ == "__main__":
    args = get_arguments()
    StructureGrid(args)
