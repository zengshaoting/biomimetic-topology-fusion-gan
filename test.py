import os
from options.test_options import TestOptions
from datasetTools import create_dataset
from models import create_model
from utilities.renderer import save_images
from utilities import samples

if __name__ == '__main__':
    options = TestOptions().parse()  # get test optionsions
    # hard-code some parameters for test
    options.num_threads = 1   # test code only supports num_threads = 1
    options.batch_size = 1    # test code only supports batch_size = 1
    options.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
  
    dataset = create_dataset(options)  # create a dataset given options.dataset_mode and other optionsions
    model = create_model(options)      # create a model given options.model and other optionsions
    model.setup(options)               # regular setup: load and print networks; create schedulers
    
    # create a sample
    samples_dir = os.path.join(options.results_dir, options.name, '{}_{}'.format(options.phase, options.load_epoch))  # define the website directory
    if options.load_iter > 0:  # load_iter is 0 by default
        samples_dir = '{:s}_iter{:d}'.format(samples_dir, options.load_iter)
    print('creating test samples directory', samples_dir)
    samples = samples.Samples(samples_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (options.name, options.phase, options.load_epoch))
    
    
    # test with eval mode. This only affects layers like batchnorm and dropout.
    if options.eval:
        model.eval()
    for i, datasetTools in enumerate(dataset):
        if i >= options.num_test:  # only apply our model to options.num_test images.
            break
        model.set_input(datasetTools)  # unpack data from data loader
        model.test()           # run inference
        
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        
        print('processing (%04d)-th image... %s' % (i, img_path))
            
        save_images(samples, visuals, img_path, aspect_ratio=1)
    
