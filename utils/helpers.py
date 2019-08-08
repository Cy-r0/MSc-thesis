


def adjust_lr(optimizer, i, max_i):
    """
    Gradually decrease learning rate as iterations increase.
    """
    lr = TRAIN_LR * (1 - i/(max_i + 1)) ** TRAIN_POWER
    optimizer.param_groups[0]['lr'] = lr
    optimizer.param_groups[1]['lr'] = 10 * lr
    return lr

def colormap(batch, cmap="viridis"):
    """
    Convert grayscale images to matplotlib colormapped image.

    Args:
        - batch (3D tensor): images to convert.
        - cmap (string): name of colormap to use.
    """
    cmap = plt.cm.get_cmap(name=cmap)

    # Get rid of singleton dimension (n. channels)
    batch = batch.squeeze()

    # Apply colormap and get rid of alpha channel
    batch = torch.tensor(cmap(batch))[..., 0:3]

    # Swap dimensions to match NCHW format
    batch = batch.permute(0, 3, 1, 2)

    return batch

def get_params(model, key):
    """
    Get model parameters.
    NB: backbone is trained 10 times slower than the rest of the network.
    Also, only conv2d layers are trained, batchnormalisation layers are kept
        the same because backbone was already pretrained on Imagenet.
    """
    for m in model.named_modules():
        if key == "1x":
            if "backbone" in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == "10x":
            if "backbone" not in m[0] and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p

def log_confusion_mat(confusion_mat, figsize, title, fmt, epoch):
    """
    Log confusion matrix to tensorboard as matplotlib figure.
    """
    fig = plt.figure(figsize=figsize)
    sn.heatmap(pd.DataFrame(confusion_mat), annot=True, fmt=fmt)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    confusion_img = figure_to_image(fig, close=True)
    tbX_logger.add_image(title, confusion_img, epoch)

def normalise_confusion_mat(confusion_mat):
    normalised = np.zeros(confusion_mat.shape)

    for c_i in range(len(confusion_mat)):
        normalised[c_i] = confusion_mat[c_i] / confusion_mat[c_i].sum()

    return normalised

def postprocess(segs, dists, energy_cut, min_area):
    """
    Extract object instances from neural network outputs (seg and dist).
    Current pipeline:
        Binarise dist image at chosen energy level (lower=black, higher=white);
        Find contours on dist image;
        Discard contours with small area;
        Calculate overlap of each contour with seg image and assign class to blobs;
        Discard contours whose class is background;
        Grow remaining contours (How? e.g. reintegrating lower level into them).

    Args:
        - segs (4D ndarray of uint8).
        - dists (3D ndarray of uint8).
        - energy_cut (int): energy level to binarize image at.
        - min_area (int): minimum area of a contour in pixels.
    """

    for seg, dist in zip(segs, dists):

        # This block is super fast (0.5 ms)
        _, thres = cv2.threshold(np.copy(dist), energy_cut, 255, cv2.THRESH_BINARY)
        _, contours, _ = cv2.findContours(np.copy(thres), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contoured = np.copy(dist)

        # This list comprehension is quite expensive (8 ms)
        contours = [c for c in contours if cv2.contourArea(c) >= min_area]

        for contour in contours:

            for seg_class in seg:

                # Create binary mask from contour

                # Mask && seg class to select only pixels inside mask

                # 


        cv2.drawContours(contoured, contours, -1, 255, 1)
        cv2.imshow("img", contoured)
        cv2.waitKey(0)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation="nearest")

