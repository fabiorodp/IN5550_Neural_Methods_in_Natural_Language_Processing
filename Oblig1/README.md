# Obligatory Assignment 1:
* The tasks can be found in [2021_IN5550_1.pdf](2021_in5550_1.pdf).
* The database can be found in [data/signal_20_obligatory1_train.tsv.gz](data/signal_20_obligatory1_train.tsv.gz).
* The report in pdf can be found in [report/final_report.pdf](report/final_report.pdf).

## Dependencies:
```
- Python        3.7.4
- torch         1.6.0  
- pandas        0.23.1
- numpy         1.18.1
- scikit-learn  0.22.1
- scipy         1.4.1 
```

## Structure of the code:

```
├── IN5550
|  └── Oblig1
|  |  └── data
|  |  |  └── sample.tsv
|  |  |  └── signal_20_obligatory1_train.tsv.gz
|  |  |  └── valid.tsv
|  |  └── packages
|  |  |  └── ann_models.py
|  |  |  └── preprocessing.py
|  |  |  └── studies.py
|  |  |  
|  |  └── eval_on_test.py
|  |  └── evaluation.py
|  |  └── final.py
|  |  └── plot.py
```

## How to run the program:
The program is run from the file **eval_on_test.py**. The script loads our best performing pretrained model and uses it to evaluate the provided test set (arg[1]), and print the metrics accuracy, recall, precision and F1-score.

To run the program, use the command `python3 eval_on_test.py PATH_TO_TEST_SET`. If no argument is provided, the model will evaluate data/valid.tsv. 
