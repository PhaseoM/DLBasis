from . import conf
from . import resnet
from . import resnet_110
from . import plaincnn
from . import plaincnn_110
from . import dataprocess
from . import visualization_2


def main():
    # lab2.resnet.model_load_test(
    #     r"D:\PyProjects\nnLabs\lab_2_cnn\models\resnet_20-5151dc8"
    # )
    conf.epochs = 2
    conf.block_n = 3
    resnet.main()
    plaincnn.main()
    conf.block_n = 5
    resnet.main()
    plaincnn.main()
    conf.block_n = 9
    resnet.main()
    plaincnn.main()
    conf.block_n = 18
    resnet_110.main()
    plaincnn_110.main()
    dataprocess.info_save()
    visualization_2.main(33)
