class Config:
    def __init__(
        self,
        batchsize=64,
        step=1e-4,
        lamb=1e-4,
        eps=1e-12,
        is_regular=False,
    ):
        self.batchsize = batchsize
        self.step = step
        self.lamb = lamb
        self.eps = eps
        self.is_regular = is_regular
