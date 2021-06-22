import sklearn
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import nltk
import argparse


def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'postag': postag,
        'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]


def preprocess(iob_train="esp.train", iob_test="esp.testb"):
    train_sents = list(nltk.corpus.conll2002.iob_sents(iob_train))
    test_sents = list(nltk.corpus.conll2002.iob_sents(iob_test))

    X_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    X_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    return X_train, y_train, X_test, y_test


def train(args):
    X_train, y_train, _, _ = preprocess(iob_train=args.iob_train, iob_test=args.iob_test)

    crf = sklearn_crfsuite.CRF(
        algorithm='l2sgd',  # algorithm='lbfgs'
        # c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    crf.fit(X_train, y_train)

    labels = list(crf.classes_)
    labels.remove("O")

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )

    return crf, labels, sorted_labels


def eval(args):
    _, _, X_test, y_test = preprocess(iob_train=args.iob_train, iob_test=args.iob_test)
    crf, labels, sorted_labels = train(args)
    
    y_pred = crf.predict(X_test)

    f1_metrics = metrics.flat_f1_score(y_test, y_pred,
                        average='weighted', labels=labels)

    cls_metrics = metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    )

    return f1_metrics, cls_metrics

def main(args):
    f1_metrics, cls_metrics = eval(args)
    # return f1_metrics, cls_metrics
    print("F1 Metrics:")
    print(f1_metrics)
    print()
    print("CLS Metrics:")
    print(cls_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iob_train", type=str, default="esp.train")
    parser.add_argument("--iob_test", type=str, default="esp.testb")
    parser.add_argument("--algorithm", type=str, default="l2sgd")
    parser.add_argument("--max_iter", type=str, default="100")
    args = parser.parse_args()

    main(args)


# # l2sgd
# F1 Metrics:
# 0.7946124551770353

# CLS Metrics:
#               precision    recall  f1-score   support

#        B-LOC      0.790     0.804     0.797      1084
#        I-LOC      0.691     0.652     0.671       325
#       B-MISC      0.740     0.537     0.622       339
#       I-MISC      0.744     0.578     0.651       557
#        B-ORG      0.817     0.816     0.817      1400
#        I-ORG      0.824     0.816     0.820      1104
#        B-PER      0.831     0.882     0.855       735
#        I-PER      0.882     0.940     0.910       634

#    micro avg      0.808     0.789     0.798      6178
#    macro avg      0.790     0.753     0.768      6178
# weighted avg      0.804     0.789     0.795      6178


# # lbfgs
# F1 Metrics:
# 0.7964686316443963

# CLS Metrics:
#               precision    recall  f1-score   support

#        B-LOC      0.810     0.784     0.797      1084
#        I-LOC      0.690     0.637     0.662       325
#       B-MISC      0.731     0.569     0.640       339
#       I-MISC      0.699     0.589     0.639       557
#        B-ORG      0.807     0.832     0.820      1400
#        I-ORG      0.852     0.786     0.818      1104
#        B-PER      0.850     0.884     0.867       735
#        I-PER      0.893     0.943     0.917       634

#    micro avg      0.813     0.787     0.799      6178
#    macro avg      0.791     0.753     0.770      6178
# weighted avg      0.809     0.787     0.796      6178
