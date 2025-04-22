# Scattering Denoising
Calculating_Notebook.ipynb is the main notebook that you can use for your own purposes. main_script.py is essentially the same if you want to run through sbatch on a gpu on the cluster. 

Of course I am still working on its user friendliness, and there's a lot of room of improvement. If you need help with anything, please do not hesitate to contact me. 

Perhaps the most important remark I can give you is that the loss is computed in the BR_loss function in __init__.py and functions therein. (This will eventually turn to a user-defined function passed through __main__).

I would also keep thresholding = False for now. While thresholding works, I still want to run some checks on the thresholding on cross statistics. 

Have fun! 
