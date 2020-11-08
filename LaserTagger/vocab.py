# vocab object from harvardnlp/opennmt-py
from copy import deepcopy

class Vocab(object):
    def __init__(self, filename=None, data=None, lower=False, vocab=None):
        self.idxToLabel = {}
        self.labelToIdx = {}
        self.lower = lower

        # start from predefined vocabulary
        if vocab is not None:
            self.idxToLabel = deepcopy(vocab.idxToLabel)
            self.labelToIdx = deepcopy(vocab.labelToIdx)

        # Special entries will not be pruned.
        self.special = []
        if data is not None:
            self.addSpecials(data)
        if filename is not None:
            self.loadFile(filename)

    def size(self):
        return len(self.idxToLabel)

    # Load entries from a file.
    def loadFile(self, filename):
        with open(filename, 'r', encoding='utf8', errors='ignore') as f:
            for line in f.readlines():
                token = line.rstrip('\n')
                if token.startswith("//"):
                    continue
                self.add(token)

    def getIndex(self, key, default=0):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return self.labelToIdx[default]

    def getLabel(self, idx, default=0):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return self.idxToLabel[default]

    # Mark this `label` and `idx` as special
    def addSpecial(self, label, idx=None):
        idx = self.add(label)
        self.special += [idx]

    # Mark all labels in `labels` as specials
    def addSpecials(self, labels):
        for label in labels:
            self.addSpecial(label)

    # Add `label` in the dictionary. Use `idx` as its index if given.
    def add(self, label):
        label = label.lower() if self.lower else label
        if label in self.labelToIdx:
            idx = self.labelToIdx[label]
        else:
            idx = len(self.idxToLabel)
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        return idx

    # Convert `labels` to indices. Use `unkWord` if not found.
    # Optionally insert `bosWord` at the beginning and `eosWord` at the .
    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):
        vec = []

        if bosWord is not None:
            vec += [self.getIndex(bosWord)]

        unk = self.getIndex(unkWord)
        vec += [self.getIndex(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.getIndex(eosWord)]

        return vec

    # Convert `idx` to labels. If index `stop` is reached, convert it and return.
    def convertToLabels(self, idx, stop):
        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels
