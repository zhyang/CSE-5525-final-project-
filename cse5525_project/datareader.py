import spacy
from torchtext.data import Field
from torchtext.data import TabularDataset
from torchtext.data import BucketIterator


spacy_en = spacy.load('en')


class DataReader(object):
    def __init__(self, data_path, batch_size, device, embedding=None):
        self.TEXT = Field(sequential=True, tokenize=self.tokenize, lower=True, include_lengths = True)
        self.LABEL = Field(sequential=False, use_vocab=False)
        self.datafield = [("id", None), ("sentence", self.TEXT), ("label", self.LABEL)]
        train_data, dev_data_a, test_data_a = TabularDataset.splits(path=data_path, train='train.csv',
                                                                    validation="dev_a.csv", test="test_a.csv",
                                                                    format='csv', skip_header=True, fields=self.datafield)
        dev_data_b, test_data_b = TabularDataset.splits(path=data_path, validation="dev_b.csv", test="test_b.csv",
                                                        format='csv', skip_header=True, fields=self.datafield)
        self.TEXT.build_vocab(train_data)
        if embedding:
            self.TEXT.vocab.load_vectors(embedding)
            self.embedding = self.TEXT.vocab.vectors.to(device)
        else:
            self.embedding = None
        # self.train_iter, self.val_iter = BucketIterator(train_data, batch_size=batch_size, device=device,
        #                                                 sort_key=lambda x: len(x.sentence), sort_within_batch=True)
        self.train_iter, self.dev_a_iter, self.test_a_iter, self.dev_b_iter, self.test_b_iter = \
            BucketIterator.splits((train_data, dev_data_a, test_data_a, dev_data_b, test_data_b), batch_size=batch_size,
                                  sort_within_batch=True, sort_key=lambda x: len(x.sentence), device=device)

    def tokenize(self, text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
