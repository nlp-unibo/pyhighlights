import csv
import json
from pathlib import Path
from statistics import mode

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset


class ToyLoader:

    def __init__(
            self,
            data_dir: Path
    ):
        self.data_dir = data_dir
        self.df_dir = data_dir.joinpath('toy', 'toy_sequential_dataset_easy.pkl')

    def data(
            self
    ) -> pd.DataFrame:
        df = pd.read_pickle(self.df_dir)
        df['mask'] = self.build_attention_mask(df=df)
        df['highlight'] = self.parse_highlights(df=df)
        df['sample_id'] = np.arange(df.shape[0])
        return df

    def build_attention_mask(
            self,
            df
    ):
        masks = []
        for text in df.text.values:
            masks.append([1] * len(list(text)))
        return masks


    def parse_highlights(
            self,
            df
    ):
        highlights = []
        for indexes, text in zip(df['structure_indexes'].values, df.text.values):
            highlight = np.zeros((len(list(text)),))
            highlight[indexes] = 1
            highlights.append(highlight.tolist())

        return highlights

    def get_splits(
            self
    ):
        train_count = int(self.data.shape[0] * 0.80)
        val_count = int(train_count * 0.20)

        train_df = self.data[:train_count]
        test_df = self.data[train_count:]
        val_df = train_df.sample(n=val_count)
        train_df = train_df[~train_df.index.isin(val_df.index.values)]

        return train_df, val_df, test_df



class HatexplainLoader:

    def __init__(
            self,
            data_dir: Path
    ):
        self.data_dir = data_dir.joinpath('hatexplain')

    @cached_property
    def data(
            self
    ) -> pd.DataFrame:
        train_df = pd.read_pickle(self.data_dir.joinpath('train.pkl'))
        train_df['split'] = ['train'] * train_df.shape[0]

        val_df = pd.read_pickle(self.data_dir.joinpath('val.pkl'))
        val_df['split'] = ['val'] * val_df.shape[0]

        test_df = pd.read_pickle(self.data_dir.joinpath('test.pkl'))
        test_df['split'] = ['test'] * test_df.shape[0]

        df = pd.concat((train_df, val_df, test_df))
        df = df[df['post_tokens'].map(len) <= 30]
        df.rename(columns={'post_tokens': 'text'}, inplace=True)

        df['mask'] = self.build_attention_mask(df=df)
        df['highlight'] = self.parse_highlights_majority(df=df)
        df['label'] = self.parse_labels(df=df)
        df['sample_id'] = np.arange(df.shape[0])
        return df

    def build_attention_mask(
            self,
            df
    ):
        masks = []
        for text in df.text.values:
            masks.append([1] * len(text))
        return masks

    def parse_labels(
            self,
            df
    ):
        labels = []
        for annotation in df['annotators'].values:
            votes = annotation["label"]
            votes = [x if x != 2 else 0 for x in votes]  # replace labels 2 with labels 0
            majority = mode(votes)
            labels.append(majority)

        return labels

    def parse_highlights_majority(
            self,
            df
    ):
        highlights = []
        for highlight_group, text in zip(df['rationales'].values, df['text'].values):
            total_tokens = len(text)

            if len(highlight_group) == 1:
                highlight = highlight_group[0][:total_tokens]
            elif len(highlight_group) > 1:
                n_annotators = len(highlight_group)
                min_vote = n_annotators / 2.0
                highlight_group = np.array([seq[:total_tokens] for seq in highlight_group])
                votes = np.sum(highlight_group, axis=0)
                highlight = np.where(votes > min_vote, 1.0, 0.0)
            else:
                highlight = [0] * len(text)

            highlights.append(highlight)

        return highlights

    def get_splits(
            self
    ):
        train_df = self.data[self.data.split == 'train']
        val_df = self.data[self.data.split == 'val']
        test_df = self.data[self.data.split == 'test']

        return train_df, val_df, test_df


class HotelData(Dataset):

    def __init__(
            self,
            data_dir: Path,
            aspect: str,
            mode,
            word2idx,
            max_length=256
    ):
        super(HotelData, self).__init__()
        self.inputs = []
        self.masks = []
        self.labels = []

        self.path = data_dir.joinpath('hotel', f'hotel_{aspect.capitalize()}.{mode}')

        examples = self._create_examples(self._read_csv(self.path), mode=mode)
        self._convert_examples_to_arrays(examples, max_length, word2idx)

    def __len__(
            self
    ):
        return len(self.labels)

    def __getitem__(
            self,
            item
    ):
        return self.inputs[item], self.masks[item], th.tensor(item), self.labels[item], th.zeros_like(self.inputs[item])

    def _read_csv(
            self,
            file_path: Path,
            quotechar=None
    ):
        with file_path.open('rt', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(
            self,
            lines,
            mode
    ):
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            label = int(line[1])
            text = line[2]
            examples.append({'text': text, "label": label})

        print('Dataset: Hotel Review')
        print('{} samples has {}'.format(mode, len(examples)))

        pos_examples = [example for example in examples if example['label'] == 1]
        neg_examples = [example for example in examples if example['label'] == 0]

        print('%s data: %d positive examples, %d negative examples.' %
              (mode, len(pos_examples), len(neg_examples)))

        return examples

    def _convert_single_text(
            self,
            text,
            max_length,
            word2idx
    ):
        """
        Converts a single text into a list of ids with mask.
        """
        input_ids = []

        text_ = text.strip().split(" ")

        if len(text_) > max_length:
            text_ = text_[0:max_length]

        for word in text_:
            word = word.strip()
            try:
                input_ids.append(word2idx[word])
            except:
                # if the word is not exist in word2idx, use <unknown> token
                input_ids.append(0)

        # The mask has 1 for real tokens and 0 for padding tokens.
        input_mask = [1] * len(input_ids)

        # zero-pad up to the max_seq_length.
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length

        return input_ids, input_mask

    def _convert_examples_to_arrays(
            self,
            examples,
            max_length,
            word2idx
    ):
        """
        Convert a set of train/dev examples numpy arrays.
        Outputs:
            data -- (num_examples, max_seq_length).
            masks -- (num_examples, max_seq_length).
            labels -- (num_examples, num_classes) in a one-hot format.
        """

        data = []
        labels = []
        masks = []
        for idx, example in enumerate(examples):
            if idx == 484:
                print()
            input_ids, input_mask = self._convert_single_text(example["text"],
                                                              max_length, word2idx)

            data.append(input_ids)
            masks.append(input_mask)
            labels.append(example["label"])

        self.inputs = th.from_numpy(np.array(data))
        self.masks = th.from_numpy(np.array(masks))
        self.labels = th.from_numpy(np.array(labels))


class HotelAnnotation(Dataset):

    def __init__(
            self,
            data_dir: Path,
            aspect: str,
            word2idx,
            max_length=256
    ):
        super(HotelAnnotation, self).__init__()
        self.input_ids = []
        self.masks = []
        self.labels = []
        self.rationales = []
        self._create_examples(
            self._read_tsv(data_dir.joinpath('hotel', f'hotel_{aspect.capitalize()}.test')),
            word2idx,
            max_length)

    def __len__(
            self
    ):
        return len(self.labels)

    def __getitem__(
            self,
            item
    ):
        return self.input_ids[item], self.masks[item], th.tensor(item), self.labels[item], self.rationales[item]

    def _read_tsv(
            self,
            annotation_path: Path,
            quotechar=None
    ):
        """Reads a tab separated value file."""
        with annotation_path.open('rt') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines

    def _create_examples(
            self,
            lines,
            word2idx,
            max_length
    ):
        data = []
        labels = []
        masks = []
        rationales = []

        print('Dataset: Hotel Review')

        for i, line in enumerate(lines):
            if i == 0:
                continue
            text_ = line[2].split(" ")
            label_ = int(line[1])
            rationale = [int(x) for x in line[3].split(" ")]

            assert len(text_) == len(rationale)

            # process the text
            input_ids = []
            if len(text_) > max_length:
                text_ = text_[0:max_length]

            for word in text_:
                word = word.strip()
                try:
                    input_ids.append(word2idx[word])
                except:
                    # word is not exist in word2idx, use <unknown> token
                    input_ids.append(0)
            # process mask
            # The mask has 1 for real word and 0 for padding tokens.
            input_mask = [1] * len(input_ids)

            # construct rationale
            binary_rationale = [0] * len(input_ids)
            for k in range(len(binary_rationale)):
                # print(k)
                if k < len(rationale):
                    binary_rationale[k] = rationale[k]

            # zero-pad up to the max_seq_length.
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_mask.append(0)
                binary_rationale.append(-1)

            assert (len(input_ids) == max_length)
            assert (len(input_mask) == max_length)

            data.append(input_ids)
            labels.append(label_)
            masks.append(input_mask)
            rationales.append(binary_rationale)

        self.input_ids = th.from_numpy(np.array(data))
        self.masks = th.from_numpy(np.array(masks))
        self.labels = th.from_numpy(np.array(labels))
        self.rationales = th.from_numpy(np.array(rationales))
        tot = self.labels.shape[0]
        print('annotation samples has {}'.format(tot))
        pos = th.sum(self.labels)
        neg = tot - pos
        print('annotation data: %d positive examples, %d negative examples.' % (pos, neg))


class HotelLoader:

    def __init__(
            self,
            aspect: str,
            data_dir: Path,
            max_length: int = 256
    ):
        self.aspect = aspect
        self.data_dir = data_dir
        self.max_length = max_length

        self.df_dir = data_dir.joinpath('hotel')
        self.filename_template = 'hotel_{0}.{1}'

    def read_file(
            self,
            filepath: Path,
            has_highlights: bool = False
    ):

        df_dict = {}
        with filepath.open("rt", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=None)
            for line_idx, line in enumerate(reader):
                if line_idx == 0:
                    continue

                label = int(line[1])
                text = line[2]

                df_dict.setdefault('text', []).append([w.strip() for w in text.strip().split(" ")])
                df_dict.setdefault('label', []).append(label)

                if has_highlights:
                    highlight = [int(x) for x in line[3].split(" ")]
                    df_dict.setdefault('highlight', []).append(highlight)

        df = pd.DataFrame.from_dict(df_dict)
        return df

    @cached_property
    def data(
            self
    ) -> pd.DataFrame:
        train_path = self.df_dir.joinpath(self.filename_template.format(self.aspect.capitalize(), 'train'))
        train_df = self.read_file(filepath=train_path)
        train_df['split'] = ['train'] * train_df.shape[0]
        train_texts, train_masks = self.trim_texts_and_mask(df=train_df)
        train_df['text'] = train_texts
        train_df['mask'] = train_masks
        train_df['highlight'] = self.parse_highlights(df=train_df)

        val_path = self.df_dir.joinpath(self.filename_template.format(self.aspect.capitalize(), 'dev'))
        val_df = self.read_file(filepath=val_path)
        val_df['split'] = ['val'] * val_df.shape[0]
        val_texts, val_masks = self.trim_texts_and_mask(df=val_df)
        val_df['text'] = val_texts
        val_df['mask'] = val_masks
        val_df['highlight'] = self.parse_highlights(df=val_df)

        test_path = self.df_dir.joinpath(self.filename_template.format(self.aspect.capitalize(), 'test'))
        test_df = self.read_file(filepath=test_path, has_highlights=True)
        test_df['split'] = ['test'] * test_df.shape[0]
        test_texts, test_masks = self.trim_texts_and_mask(df=test_df)
        test_df['text'] = test_texts
        test_df['mask'] = test_masks
        test_df['highlight'] = self.parse_highlights(df=test_df)

        df = pd.concat((train_df, val_df, test_df))
        df['sample_id'] = np.arange(df.shape[0])
        return df

    def trim_texts_and_mask(
            self,
            df
    ):
        texts = []
        masks = []
        for text in df.text.values:
            pad_text = text[:self.max_length]
            mask = [1] * len(pad_text)

            diff = self.max_length - len(pad_text)
            assert diff >= 0
            if diff > 0:
                pad_text += ['<PAD>'] * diff
                mask += [0] * diff

            assert len(mask) == len(pad_text) == self.max_length

            texts.append(pad_text)
            masks.append(mask)

        return texts, masks

    def parse_highlights(
            self,
            df
    ):
        if 'highlight' not in df.columns:
            return [[0] * len(text) for text in df.text.values]

        highlights = []
        for text, rationale in zip(df.text.values, df.highlight.values):
            highlight = rationale[:self.max_length]
            diff = self.max_length - len(highlight)
            assert diff >= 0
            if diff > 0:
                highlight += [-1] * diff

            assert len(highlight) == len(text)

            highlights.append(highlight)

        return highlights

    def get_splits(
            self
    ):
        train_df = self.data[self.data.split == 'train']
        val_df = self.data[self.data.split == 'val']
        test_df = self.data[self.data.split == 'test']

        return train_df, val_df, test_df


class BeerLoader:

    def __init__(
            self,
            aspect: int,
            data_dir: Path,
            max_length: int = 256,
            neg_threshold=0.4,
            pos_threshold=0.6
    ):
        self.aspect = aspect
        self.data_dir = data_dir
        self.max_length = max_length
        self.neg_threshold = neg_threshold
        self.pos_threshold = pos_threshold

        self.df_dir = data_dir.joinpath('beer', 'reviews.aspect{0}.{1}.txt')

    def data(
            self
    ) -> pd.DataFrame:
        train_df = pd.read_csv(self.df_dir.as_posix().format(self.aspect, 'train'), sep='\t', header=None,
                               names=['label', 'text'])
        train_df['split'] = ['train'] * train_df.shape[0]
        train_df['text'] = self.trim_texts(df=train_df)
        train_df['label'] = self.parse_labels(df=train_df)
        train_df = train_df[train_df.label != -1]
        train_df['highlight'] = self.add_highlights(df=train_df)

        val_df = pd.read_csv(self.df_dir.as_posix().format(self.aspect, 'heldout'), sep='\t', header=None,
                             names=['label', 'text'])
        val_df['split'] = ['val'] * val_df.shape[0]
        val_df['text'] = self.trim_texts(df=val_df)
        val_df['label'] = self.parse_labels(df=val_df)
        val_df = val_df[val_df.label != -1]
        val_df['highlight'] = self.add_highlights(df=val_df)

        test_df = self.read_annotations()
        test_df['split'] = ['test'] * test_df.shape[0]

        df = pd.concat((train_df, val_df, test_df))
        df['sample_id'] = np.arange(df.shape[0])
        return df

    def parse_labels(
            self,
            df
    ):
        labels = []
        for label in df.label.values:
            parsed = float(label.split()[self.aspect])
            if parsed <= self.neg_threshold:
                labels.append(0)
            elif parsed >= self.pos_threshold:
                labels.append(1)
            else:
                labels.append(-1)

        return labels

    def trim_texts(
            self,
            df
    ):
        return [np.array(item.strip().split(" ")[:self.max_length]) for item in df.text.values]

    def add_highlights(
            self,
            df
    ):
        return [[0] * len(text) for text in df.text.values]

    def read_annotations(
            self
    ):
        annotation_path = self.data_dir.joinpath('beer', 'annotations.json')

        df_dict = {}
        with annotation_path.open('rt', encoding='utf-8') as f:
            for json_line in f:
                data = json.loads(json_line)

                text = data['x'][:self.max_length]
                label = float(data['y'][self.aspect])
                rationale = data[str(self.aspect)]

                if len(rationale) == 0:
                    continue

                if label <= self.neg_threshold:
                    label = 0
                elif label >= self.pos_threshold:
                    label = 1
                else:
                    continue

                highlight = np.zeros(len(text))
                for start, end in rationale:
                    if start >= self.max_length:
                        continue
                    if end >= self.max_length:
                        end = self.max_length

                    highlight[start:end] = 1

                df_dict.setdefault('text', []).append(text)
                df_dict.setdefault('label', []).append(label)
                df_dict.setdefault('highlight', []).append(highlight)

        df = pd.DataFrame.from_dict(df_dict)
        return df

    def get_splits(
            self
    ):
        train_df = self.data[self.data.split == 'train']
        val_df = self.data[self.data.split == 'val']
        test_df = self.data[self.data.split == 'test']

        return train_df, val_df, test_df