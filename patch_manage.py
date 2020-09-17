from torch import optim


class BasePatch(object):
    """
    Default parameters for all config files.
    """

    def __init__(self):
        """
        Set the defaults.
        """
        self.img_dir = "./FDDB/Train/mtcnn_train_images_filtered/"
        self.lab_dir = "./FDDB/Train/mtcnn_train_labels_filtered/"

        self.printfile = "./non_printability/30values.txt"

        self.start_learning_rate = 0.03
        self.scheduler_factory = lambda x: optim.lr_scheduler.ReduceLROnPlateau(x, 'min', patience=50)
        self.patch_size_gray = (250, 450)
        self.patch_size_glasses = (175,500)
        self.patch_name = 'base'

        self.batch_size = 8

patch_configs = {
    "base": BasePatch

}
