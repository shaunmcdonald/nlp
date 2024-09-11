#!/local-data/python_ve/bin/python3.9
# coding=utf-8
import argparse
import os
import nltk
import json

import utils
from nltk_nlp import NltkPOSStats, NltkPOSMethods

__author__ = 'smcdonald'
"""
.. module:: read_pdf
    :synopsis: A script to read lines in one or more PDF files and write to CSV.

Arguments
------------------
Arguments must be passed in the order listed.

    dir_in
        String: The absolute filepath to the input directory containing the PDF file(s)
        Short form -i

    dir_out
        String: The absolute filepath to the output directory for CSV file(s) 
        Short form -o

"""


def get_level_from_filename(search_str):
    """
    Searches CEFR CBA filename for the level name as substring.
    :param search_str: String, CBA filename.
    :return: String, CEFR Level or None if no match is found
    """
    levels = ['A1', 'A2', 'B1', 'B1Plus', 'B2', 'C1', 'C2']
    for level in levels:
        if '_'+level+'_' in search_str:
            return level

    return None


def get_words_by_level(dict_list_of_lists):
    """
    Converts Dict of list of lists to Dict of word lists
    :param dict_list_of_lists: {'A1': [['w1', 'w2'], ['w3', 'w4']],...}
    :return: Dictionary of lists, {'A1': ['w1', 'w2', 'w3', 'w4'],...}
    """
    return_dict = {}
    for k, l_of_ls in dict_list_of_lists.items():
        new_list = []
        for word_list in l_of_ls:
            new_list.append(word for word in word_list)

        return_dict[k] = new_list

    return return_dict


def write_to_file(output_data, dir_out, filename, ftype):
    """
    Write data to file based on ftype
    :param output_data: Content to write to file, structure is based on file type
    :param dir_out: Output directory
    :param filename: Name of the output file
    :param ftype:
    :return:
    """
    with open(os.path.join(dir_out, filename), "w") as file_obj:
        if ftype == 'json':
            json.dump(output_data, file_obj, indent=1)
        if ftype == 'matrix':
            for key, list_of_lists in output_data.items():
                if key == 'CLASSES':  # process header {key: [simple list]}
                    this_string = key + ',' + ','.join(item for item in list_of_lists)
                    file_obj.write("{0}\n".format(this_string))
                    continue
                for pos_list in list_of_lists:
                    this_string = key + ',' + ','.join(str(item) for item in pos_list)
                    file_obj.write("{0}\n".format(this_string))


def nltk_handler(file_list, pos_by_level_dict, dir_out):
    """

    :param file_list:
    :param pos_by_level_dict:
    :param dir_out:
    :return:
    """
    level_text_dict = {}
    level_pos_dict = {}
    nltk_methods = NltkPOSMethods()
    for file_name in file_list:
        file_with_ext = os.path.split(file_name)[1]
        base_file = os.path.splitext(file_with_ext)[0]
        output_filename = base_file + '.json'
        print('Getting text from: ', file_with_ext)
        file_text = utils.file_to_string(file_name)

        current_key = get_level_from_filename(base_file)

        sentences = NltkPOSMethods.get_sentences(file_text)
        # Creating the sentence dictionary here in case a new requirement calls for
        # processing sentences before they become dict(keys).
        sent_dict = {}
        for sentence in sentences:
            # sub_conj_list = get_sub_conjunction(sentence)
            pos_tuple_list, word_count, verb_count = nltk_methods.tag_with_upenn(
                sentence)

            if _pos_tags:
                sent_dict[sentence] = {'wordcount': word_count, 'verb_count': verb_count,
                                       'pos': pos_tuple_list}
            if _pos_stats:
                # convert tuple list to list of code lists
                pos_by_level_dict[current_key].append(NltkPOSStats.get_list_from_tuples(
                    pos_tuple_list, 'pos'))
            if _freq_dist == 'pos':
                # Add the current key to the dict
                if current_key not in level_pos_dict:
                    level_pos_dict[current_key] = []
                # Freq Dist needs a list of POS, not a list of lists of POS.
                for pos in NltkPOSStats.get_list_from_tuples(pos_tuple_list, 'pos'):
                    level_pos_dict[current_key].append(pos)
            if _freq_dist == 'words':
                # Add the current key to the dict
                if current_key not in level_text_dict:
                    level_text_dict[current_key] = []
                # Freq Dist needs a list of words, not a list of lists of words.
                for word in NltkPOSStats.get_list_from_tuples(pos_tuple_list, 'words'):
                    level_text_dict[current_key].append(word)

        # print(sent_dict)
        # print(json.dumps(sent_dict, indent=1))
        # exit(1)
    if _freq_dist:
        for level, text in level_text_dict.items():
            # To get the level vocab v = set(text) or sorted(set(text))
            # get the % of unique words per total v = len(set(text))/len(text
            # count a word c = text.count("the")
            # a word's % of total 100*text.count("the")/len(text)
            # TODO finish this. I may want to finish all distribution work here or
            #  pass to a
            #  separate method. freq_dist = FreqDist(word_list),
            #  freq_dist.plot(50, cumulative=True), to plot the top 50 words.
            freq_dist_obj = nltk.probability.FreqDist(text)
            print(freq_dist_obj.plot(50, title='CEFR Level ' + level, cumulative=True))
            exit(1)

    if _pos_stats:
        pos_matrix = nltk_methods.get_pos_level_matrix(pos_by_level_dict)
        write_to_file(pos_matrix, dir_out, 'pos_matrix.csv', 'matrix')


def main(dir_in, dir_out):
    """
    Open and read lines in one or more PDF files and write to CSV.
    :param dir_in:
    :param dir_out:
    :return:
    """

    # Get the path to NLTK_DATA from env, otherwise use Linux path
    # nltk_data = os.getenv('NLTK_DATA', '/local_data/python_ve/nltk_data')

    file_list = utils.get_files(dir_in)
    to_process = len(file_list)
    processed = 0
    # TODO ideas: if POS needs improving consider 1) adding a training step on a
    #  corpus, 2) try finding a domain specific corpus, 3) using different taggers (
    #  PerceptronTagger or CRFTagger) 4) using morphological cues to ID prefixes and
    #  suffixes, 5) post-processing to fix common errors.

    pos_by_level_dict = {'A1': [], 'A2': [], 'B1': [], 'B1Plus': [],
                         'B2': [], 'C1': [], 'C2': []}
    # dictionary of list of lists of words, usually by CEFR level
    if _which_lib == 'nltk':
        nltk_handler(file_list, pos_by_level_dict, dir_out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Part of Speech tagging operations.')
    parser.add_argument('-i', '--dir_in', help='The path to the directory containing '
                                               'input files. Short form -i.')
    parser.add_argument('-o', '--dir_out', help='The path to the directory where output '
                                                'files will be written. Short form -o.')
    parser.add_argument('-l', '--library', type=int, default="spacy",
                        help='Pass "spacy" or "nltk". Short form -l.')
    parser.add_argument('-pt', '--pos_tags', type=int, default=1,
                        help='Pass 0 or 1 switch of pos tagging. Short form -pt.')
    parser.add_argument('-ps', '--pos_stats', type=bool, default=False,
                        help='Pass any value to only retrieve POS codes per level. '
                             'Short form -ps.')
    parser.add_argument('-fd', '--freq_dist', type=str, default=False,
                        help='Frequency Distribution for "words" or "pos". '
                             'Short form -fd.')

    # ALL FILES "C:\Users\mcdonalds\Documents\Content_Engineering\TCS\OTGA\content_leveling\elt\cba_cefr\pos_tagging\all"
    opts = parser.parse_args()
    _which_lib = opts.library
    _pos_tags = opts.pos_tags
    _pos_stats = opts.pos_stats
    _freq_dist = opts.freq_dist
    main(opts.dir_in, opts.dir_out)
