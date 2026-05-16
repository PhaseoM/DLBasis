from nnLabs.lab_1_bpnn import experiments, torch_bpnn_reg
from plyer import notification

if __name__ == "__main__":
    # experiments.exp_reg()
    # experiments.exp_cls()
    experiments.exp()
    # experiments.regular_exp_reg()
    # experiments.regular_exp_cls()
    # torch_bpnn_reg.main()
    notification.notify(
        title="运行完毕",
        message="参数对比测试完成",
        timeout=1,
    )
