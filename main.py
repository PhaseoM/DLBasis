from nnLabs import lab1, lab2, lab3
from pathlib import Path
# from plyer import notification

if __name__ == "__main__":
    # lab3.vit_normal.run()
    # lab3.vit_finetuning.run()
    # lab3.vit_adamw.run()
    # lab3.data_process.run()
    dump_path = Path("D:\\PyProjects\\data_analysis\\lab_3_vit_m\\pkldump\\")
    lab3.data_process_2.run(dump_path)
