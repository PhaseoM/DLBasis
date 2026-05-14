from nnLabs.lab_1_bpnn import experiments
from plyer import notification

if __name__ == "__main__":
    # experiments.exp_reg()
    # experiments.exp_cls()
    experiments.exp()
    notification.notify(
        title="运行完毕",
        message="参数对比测试完成",
        timeout=1,
    )
