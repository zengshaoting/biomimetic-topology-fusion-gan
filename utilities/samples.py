import os


class Samples:
    """
    """

    def __init__(self, sample_dir, title, refresh=0):
        """
        Parameters:
            sample_dir (str) -- a directory that stores the testing samples
            title (str)   -- the webpage name
            refresh (int) -- how often the website refresh itself; if 0; no refreshing
        """
        self.title = title
        self.sample_dir = sample_dir
        self.img_dir = os.path.join(self.sample_dir, 'samples')
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)
        if not os.path.exists(self.img_dir):
            os.makedirs(self.img_dir)


    def get_image_dir(self):
        """Return the directory that stores images"""
        return self.img_dir