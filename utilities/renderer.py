

"""
RENDERER CLASS - save, print, display functionality
"""

import numpy as np
import os
import sys
import ntpath
import time
from . import utilities
from subprocess import Popen, PIPE

def save_images(samples, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    """
    image_dir = samples.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    #samples.add_header(name)
    ims, txts, links = [], [], []

    for label, im_data in visuals.items():
        im = utilities.tensor2im(im_data)
        image_name = '%s_%s.png' % (name, label)
        save_path = os.path.join(image_dir, image_name)
        utilities.save_image(im, save_path, aspect_ratio=aspect_ratio)
        ims.append(image_name)
        txts.append(label)
        links.append(image_name)
    #samples.add_images(ims, txts, links, width=width)


class Render():
    """
    This class includes several functions that can display/save images and print/save logging information.
    """

    def __init__(self, options):
        """Initialize the Render class

        Parameters:
            options -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: create a logging file to store training losses
        """
        self.options = options  # cache the option
        self.name = options.name
        self.saved = False


        self.img_dir = os.path.join(options.checkpoints_dir, options.name, 'samples')
        utilities.mkdirs([self.img_dir])
        
        # create a logging file to store training losses
        self.log_name = os.path.join(options.checkpoints_dir, options.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def display_current_results(self, visuals, epoch, save_result):
        """
        Parameters:
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results 
        """
        
        self.saved = True
        # save images to the disk
        for label, image in visuals.items():
            image_numpy = utilities.tensor2im(image)
            img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
            utilities.save_image(image_numpy, img_path)


    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data, options):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        
        message = '(epoch: %d,  iters: %d,  time: %.3f,  data loading: %.3f)    ' % (epoch, iters, t_comp, t_data)
        
        for k, v in losses.items():
            message += ' %s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message

    def print_current_losses_nodata(self, epoch, iters, losses, t_comp, options):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
        """
        
        message = '(epoch: %d,  iters: %d,  time: %.3f)    ' % (epoch, iters, t_comp)
        
        for k, v in losses.items():
            message += ' %s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message