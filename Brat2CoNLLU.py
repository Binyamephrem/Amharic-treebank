#!/usr/bin/python
# -*- coding: utf-8 -*-

import codecs
from collections import defaultdict
import os

ud_pos_filename = 'ud_pos.txt'
root_dir = "data_UD"
exception_list = []


# 1. Load a dictionary of correspondences Amharic POS tags -> Universal Dependencies tags
universal_dependencies_file = codecs.open(ud_pos_filename, 'r', 'utf-8')
ud_to_pos_list = [line.strip('\n').split() for line in universal_dependencies_file.readlines()]
ud_pos_classes = {}
for ud_poss_pair in ud_to_pos_list:
    ud_tag = ud_poss_pair[0]
    pos_list = ud_poss_pair[1:]
    for pos in pos_list:
        ud_pos_classes[pos.upper()] = ud_tag.upper()


class UDAnnotationParser:

    def __init__(self, text_filename, anno_filename):
        # Maps  sentence_number -> (beginning of sentence, end_of_sentence)
        self.sentence_offset_map = {}
        # Maps  sentence_number -> token_number -> (token, begining, end)
        self.tokens_offset_map = {}
        self.sentences = []
        self.original_sentences = {}
        self.clitics_host_words_dict = {}
        # Store token information in a dictionary.
        self.tokens_dict = defaultdict(dict)
        self.parse_text_file(text_filename)
        self.parse_annotation_file(anno_filename)
        self.file_name_string = text_filename.replace('/', '-').split('.')[0]

    def parse_text_file(self, text_filename):
        # Build a map with all the bracketed words and their positions.
        text_in_file = codecs.open(text_filename, 'r', 'utf-8')
        raw_text = text_in_file.readlines()
        sentence_offset = 0
        for sentence_number, line in enumerate(raw_text):
            clean_line = line.strip('\n')
            self.sentence_offset_map.update(
                {sentence_number + 1: (clean_line, sentence_offset, sentence_offset + len(line) + 1)})
            original_sentence = ' '.join([word.strip(u'[]') for word in clean_line.split() if '_' not in word])
            self.original_sentences.update({sentence_number + 1: original_sentence})
            sentence = clean_line.split()
            tokens_offset = 0
            token_number = 0
            for token in sentence:
                if sentence_number + 1 not in self.tokens_offset_map.keys():
                    self.tokens_offset_map.update({sentence_number + 1: {}})
                if token.startswith(u'[') and token.endswith(u']'):
                    tokens_offset += len(token) + 1
                    previous_bracketed_word = token
                    continue
                if '_' in token:
                    subtokens = token.split('_')
                    clitics_number = token.count('_')
                    position = (sentence_number + 1, token_number + 1)
                    clitics_host_word = (previous_bracketed_word.strip(u'[]'), clitics_number)
                    previous_bracketed_word = ''
                    self.clitics_host_words_dict[position] = clitics_host_word
                    for subtoken_number, subtoken in enumerate(subtokens):
                        self.tokens_offset_map[sentence_number + 1].update(
                            {token_number + 1: (subtoken, tokens_offset, tokens_offset + len(subtoken))})
                        tokens_offset += len(subtoken) + 1
                        token_number += 1
                else:
                    self.tokens_offset_map[sentence_number + 1].update(
                        {token_number + 1: (token, tokens_offset, tokens_offset + len(token) + 1)})
                    tokens_offset += len(token) + 1
                    token_number += 1
            self.sentences.append(sentence)
            sentence_offset += len(line)

    def parse_annotation_file(self, anno_filename):
        # 1. Extract all relevant annotations for every word and clitic.
        annotation_file = codecs.open(anno_filename, 'r', 'utf-8')
        annotations_info = [line.strip('\n').split('\t') for line in annotation_file.readlines()]

        # annotations have up to 5 fields, usually:
        # For Tag information:
        #   (1) Annotation ID [tab] (2) POS [space] Beginning of Span [space] End of Span + 1 [tab] (3) Token
        for fields in annotations_info:
            annotation_id = fields[0].strip(' \n\t')
            # Check if it is an Attribute or a Relation and ignore it.
            if not annotation_id.startswith('T'):
                continue
            pos, span_start, span_end = fields[1].strip(' \n\t').split()
            token = fields[2]
            # Should have a format like:
            # T18     SUBJC 94 95     ይ
            assert len(fields) == 3
            sentence_number, token_number = self.get_sentence_token_number_pair((int(span_start), int(span_end)))
            if sentence_number and token_number:
                self.tokens_dict[annotation_id] = {'id': (sentence_number, token_number),
                                                   'pos': pos, 'surf': token, 'morpho': []}
        # Store morphological information.
        # For Attributes:
        #   (1) Attribute ID [tab] (2) Attribute Name [space] Annotation Id [Optional: [space] Value]
        for fields in annotations_info:
            annotation_id = fields[0].strip(' \n\t')
            if not annotation_id.startswith('A'):
                continue
            # Should have a format like:
            # A2      Agreement T19 Subj
            assert len(fields) == 2
            annotation = fields[1].strip(' \n\t').split()
            if annotation[0] == 'Poss':
                annotation.append('Yes')
            # Should have a format like:
            # Agreement T19 Subj
            assert len(annotation) == 3
            feature, token_id, value = annotation
            self.tokens_dict[token_id]['morpho'].append((feature, value))

        # Store dependency relation information.
        # For Relations:
        #   (1) Relation ID [tab] (2) Relation Name [space] Argument1 (Annotation ID) [space] Argument2 (Annotation ID)
        for fields in annotations_info:
            annotation_id = fields[0].strip(' \n\t')
            if not annotation_id.startswith('R'):
                continue
            assert len(fields) >= 2
            relation_name, argument1, argument2 = fields[1].strip(' \n\t').split()
            arg_token_id = argument1.split(':')[1].strip(' \n\t')
            head_token_id = argument2.split(':')[1].strip(' \n\t')
            if arg_token_id in self.tokens_dict:
                arg_id = self.tokens_dict[arg_token_id]['id'][1]
                self.tokens_dict[head_token_id]['arg_id'] = arg_id
                self.tokens_dict[head_token_id]['relation'] = relation_name
                

    def get_sentence_token_number_pair(self, token_position_pair):
        token_begin, token_end = token_position_pair
        possible_sentence = "Unknown"
        for sentence in self.sentence_offset_map.keys():
            string, sentence_begin, sentence_end = self.sentence_offset_map[sentence]
            if token_begin >= sentence_begin and token_end <= sentence_end:
                possible_sentence = sentence, sentence_begin, sentence_end, string
                adjusted_begin = token_begin - sentence_begin
                adjusted_end = token_end - sentence_begin
                for token in self.tokens_offset_map[sentence].keys():
                    word, token_index_start, token_end_start = self.tokens_offset_map[sentence][token]
                    if adjusted_begin >= token_index_start and adjusted_end <= token_end_start:
                        return sentence, token
        print("TOKEN NOT FOUND: ", token_position_pair, possible_sentence)
        return None, None

    # 2. Retrieve the full word (with clitics collapsed) and print them.
    def get_clitics_host_word_if_any(self, annotation):
        sentence_number, word_num = annotation['id']
        clitics_host_word, clitics_number = self.clitics_host_words_dict.get((sentence_number, word_num), (None, None))
        if clitics_host_word is None:
            return None
        clitics_word_line = u'{0}-{1}\t{2}\t{3}'.format(word_num, word_num + clitics_number, clitics_host_word, '\t'.join(['_'] * 8))
        return clitics_word_line

    def print_conll(self, conll_file_name, verbose=False):
        with codecs.open(conll_file_name, 'w', 'utf-8') as conll_file:
            sorted_tokens = sorted(self.tokens_dict.items(), key=lambda x: x[1]['id'])
            last_id = 0
            first_line = True
            for token_id, annotation in sorted_tokens:
                # If the first token of the sentence, add an empty line.
                output_string = ''
                if annotation['id'][1] < last_id or first_line:
                    sentence_number = annotation['id'][0]
                    if not first_line:
                        output_string += "\n"
                    output_string += "# sent_id = {}-s{}\n# text = {}\n".format(self.file_name_string,
                                                                                   sentence_number,
                                                                                   self.original_sentences[sentence_number])
                    first_line = False
                clitics_word_line = self.get_clitics_host_word_if_any(annotation)
                if clitics_word_line:
                    output_string += '{}\n'.format(clitics_word_line)
                if annotation['morpho']:
                    morpho_str = '|'.join(['='.join(feat_val) for feat_val in sorted(annotation['morpho'], key=lambda x: x[0])])
                else:
                    morpho_str = '_'
                id_str = str(annotation['id'][1])
                assert annotation['pos'] == annotation['pos'].upper()
                # print(annotation)
                ud_pos = ud_pos_classes[annotation['pos']]
                arg_id = annotation.get('arg_id', 0)
                relation = annotation.get('relation', 'root')
                if relation == 'advmod' and ud_pos == 'PART' and annotation['pos'] == 'NEG':
                    morpho_str = 'Polarity=Neg'
                output_line = u'{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}'.format(id_str,
                                                                                         annotation['surf'],
                                                                                         annotation['surf'],
                                                                                         ud_pos,
                                                                                         annotation['pos'],
                                                                                         morpho_str,
                                                                                         arg_id,
                                                                                         relation,
                                                                                         '_', '_')
                output_string += '{}\n'.format(output_line)
                if verbose:
                    print(output_string, end='')
                conll_file.write(output_string)
                last_id = annotation['id'][1]
            conll_file.write('\n')  # adding an empty line at the end of the last tree
for root, dirs, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".txt"):
            filename = os.path.join(root, file).split('.')[0]
            if filename not in exception_list:
                text_file = '{}.txt'.format(filename)
                annotation_file = '{}.ann'.format(filename)
                conll_file = '{}.conllu'.format(filename)
                print(text_file)
                UDAnnotationParser(text_file, annotation_file).print_conll(conll_file, verbose=False)
