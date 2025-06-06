from pytorch_lightning.callbacks import EarlyStopping, Callback

class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("##### TRAINING STARTED #####")

    def on_train_end(self, trainer, pl_module):
        print("##### TRAINING ENDED #####")
