from nnLabs import lab2
# from plyer import notification

if __name__ == "__main__":
    # lab2.resnet.model_load_test(
    #     r"D:\PyProjects\nnLabs\lab_2_cnn\models\resnet_20-5151dc8"
    # )
    # lab2.conf.epochs = 2
    # lab2.conf.block_n = 3
    # lab2.resnet.main()
    # lab2.plaincnn.main()
    # lab2.conf.block_n = 5
    # lab2.resnet.main()
    # lab2.plaincnn.main()
    lab2.conf.block_n = 9
    lab2.resnet.main()
    # lab2.plaincnn.main()
    # lab2.conf.block_n = 18
    # lab2.resnet_110.main()
    # lab2.plaincnn_110.main()
    # lab2.dataprocess.info_save()
    # lab2.visualization.main()
