from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms, models

import numpy as np
import cv2


####################
# Helper Functions #
####################
def load_image(img_path=None, img=None, max_size=400, shape=None):
    '''
    Load in and transform an image, making sure the image is <= 400 pixels in x-y dims.
    :param img_path:
    :type img_path:
    :param max_size:
    :type max_size:
    :param shape:
    :type shape:
    :return:
    :rtype:
    '''

    if img_path is not None:
        image = Image.open(img_path).convert('RGB')

    if img_path is None and img is not None:
        image = Image.fromarray(img)

    # large images heavily slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225))
    ])

    # discard the transparent, alpha channel (that's the :3) and add batch size
    image = in_transform(image)[:3, :, :].unsqueeze(0)

    return image


def im_convert(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1, 2, 0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image


def get_features(image, model, layers=None):
    """
    Run an image forward through a model and get the features for a set
    of layers. Default layers are for VGGNet macthing Gatys et al (2016)
    :param image:
    :type image:
    :param model:
    :type model:
    :param layers:
    :type layers:
    :return:
    :rtype:
    """

    if layers is None:
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # content representation
            '28': 'conv5_1'
        }

    features = {}
    x = image

    # model._modules is dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x

    return features


def gram_matrix(tensor):
    '''
    Calculate the Gram Matrix of a given tensor
    Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    :param tensor:
    :type tensor:
    :return:
    :rtype:
    '''

    _, d, h, w = tensor.size()

    # reshape, so we're multuplying the features for each channel
    tensor = tensor.view(d, h * w)

    # calculate the gram matrix
    gram = torch.mm(tensor, tensor.t())

    return gram


def style_transfer(content=None, style=None):
    # get the 'features' portion of VGG19, we do not neet the 'classifier' portion
    vgg = models.vgg19(pretrained=True).features
    # freeze all VGG parameters since we're only optimizing target image
    for param in vgg.parameters():
        param.requires_grad_(False)
    # let's have a look at the structure
    print(vgg)

    ################
    # Run the code #
    ################
    # load the content and style images (if None passed)
    if content is None:
        content = load_image('imgs/octopus.jpg')

    if style is None:
        style = load_image('imgs/hockney.jpg', shape=content.shape[-2:])

    # display both images
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(im_convert(content))
    ax2.imshow(im_convert(style))
    plt.show()
    # get content and style features only once before training
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    # calculate the gram matrices for each layer of our style representation
    style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}
    # create a third 'target' image and prep it for change
    # better start with the target as a copy of the content image,
    # then iteratively change its style
    target = content.clone().requires_grad_(True)
    # weights for each style layer
    # weighting earlyer layers more will result in larger style artifacts
    # notice we are excluding 'conv4_2' - the content representation
    style_weights = {
        'conv1_1': 1.,
        'conv2_1': .75,
        'conv3_1': .2,
        'conv4_1': .2,
        'conv5_1': .2
    }
    content_weight = 1  # alpha
    style_weight = 1e6  # beta
    # for displaying the target image, intermittently
    show_every = 400
    loss_every = 10
    # iteration hyperparameters
    optimizer = optim.Adam([target], lr=.003)
    steps = 2000
    for ii in range(1, steps + 1):

        # get the features from target image
        target_features = get_features(target, vgg)

        # content loss
        content_loss = torch.mean((target_features['conv4_2'] \
                                   - content_features['conv4_2']) ** 2)

        # style loss
        # initialize the style loss to 0
        style_loss = 0
        for layer in style_weights:
            # get the target style representation for the layer
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            _, d, h, w = target_feature.shape

            # get the 'style' style representation
            style_gram = style_grams[layer]
            # style loss for one layer, weighted approprately
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram) ** 2)
            # add to the style loss
            style_loss += layer_style_loss / (d * h * w)

        # calculate total loss
        total_loss = content_weight * content_loss + style_weight * style_loss

        # update your target image
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # displayer intermediate images and print the loss
        if ii % show_every == 0:
            print('Total loss: ', total_loss.item())
            plt.imshow(im_convert(target))
            plt.show()

        if ii % loss_every == 0:
            print('Total loss: {:d}'.format(int(total_loss.item())))
    # display content and final, target image
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(im_convert(content))
    ax2.imshow(im_convert(target))


def getWebCamPic():
    cap = cv2.VideoCapture(0)
    while (True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Display the resulting frame
        # cv2.imshow('frame',gray)
        cv2.imshow('🦄', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return frame


img = getWebCamPic()
content = load_image(img=img)

# in order to use the default pictures, just don't pass any params
style_transfer(content=content)
