try:
    from packages.ner_eval import Evaluator
except:
    from .ner_eval import Evaluator


def f1(precision, recall, eps=1e-7):
        score = 2 * (precision * recall) / (precision + recall + eps)
        return score

def evaluate(y_pred, y_true, indexer):
    gold_labels = []
    predicted_labels = []
    for seq_true, seq_pred in zip(y_true, y_pred):
        lab_true = [
            val2key(indexer, tok.item()) 
            if val2key(indexer, tok.item()) != '[MASK]' else 'O'
            for tok in seq_true
        ]
        lab_pred = [
            val2key(indexer, tok.item()) 
            if val2key(indexer, tok.item()) != '[MASK]' else 'O'
            for tok in seq_pred
        ]
        gold_labels.append(lab_true)
        predicted_labels.append(lab_pred)

    entities = ["PER", "ORG", "LOC", "GPE_LOC", "GPE_ORG", "PROD", "EVT", "DRV"]

    print('true[0]:', gold_labels[0])
    print('pred[0]:', predicted_labels[0])
    evaluator = Evaluator(gold_labels, predicted_labels, entities)
    results, results_agg = evaluator.evaluate()

    prec = results["strict"]["precision"]
    rec = results["strict"]["recall"]
    return f1(prec, rec)

def val2key(d, v):
    return list(d.keys())[
        list(d.values()).index(v)
    ]
