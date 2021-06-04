## Instructions
The program runs with the command `python3 predict_on_test.py 
-t=PATH_FOR_TEST_SET -m=PATH_FOR_THE_MODEL -b=PATH_FOR_NORBERT 
-d=CHOSEN_DEVICE`.
This creates the file `predictions.conllu`, in the **outputs** folder, 
replacing 
BIO's labels with our predictions. Besides, a F1-score between y_true and 
y_pred will be printed in the terminal. 

Attention: flags -m, -b and -d are 
not required because these are set as default with our files. However, -d 
is set by default as 'cpu', but it can also be changed to 'gpu' if preferred.
