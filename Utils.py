import os, glob
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compile_files_in_folder(folder_path, type="*txt", encoding="utf-8-sig", join_type="/n"):
    compiled_array = []

    for file in glob.glob(os.path.join(folder_path, type)):
        with open(file, 'r', encoding=encoding) as f:
            compiled_array.append(f.read())

    return join_type.join(compiled_array)


def create_encoding_dict(set_to_be_indexed):
    return {str:index for index, str in enumerate(set_to_be_indexed)}

def create_indexed_array(string_or_list_to_be_indexed):
    return np.array(string_or_list_to_be_indexed)

def encode_contents(content, encoding_dict):
    return np.array([encoding_dict[each] for each in content])

def decode_contents(index,decoding_dict):
    return decoding_dict[index]
    
def char_shift(seq, shift_amt=1):
    current_seq = seq[:-1*shift_amt]
    next_seq = seq[shift_amt:]
    return current_seq, next_seq

def train_test_split_by_index(dataset, test_percent=10):
    data_length = len(dataset)
    test_length = data_length * (test_percent/100)
    test_index = int(data_length-test_length)
    trainset = dataset[:test_index]
    testset = dataset[test_index:]
    return trainset, testset

def save_as_text_file(content, filename_with_ext, full_folder_path=None):
    if (full_folder_path): current_file_loc = full_folder_path
    else: current_file_loc = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_file_loc, filename_with_ext)
    new_file = open(file_path, "a")
    current_date_time = datetime.now()
    new_file.write("\n" + str(current_date_time) + "\n")
    new_file.write("\n" + content)
    new_file.write("\n\n--End--\n\n")
    print("\nFile saved successfully to {a}.".format(a=file_path))

def save_show_loss_plot(model, filename_with_ext, full_folder_path=None, show=False, figsize=(15,6)):
    if (full_folder_path): current_file_loc = full_folder_path
    else: current_file_loc = os.path.dirname(os.path.realpath(__file__))
    file_path = os.path.join(current_file_loc, filename_with_ext)
    pd.DataFrame(model.history.history)[['loss','val_loss']].plot()
    plt.savefig(file_path)
    if (show): plt.show()

def get_set_current_path(filename=None):
    current_file_loc = os.path.dirname(os.path.realpath(__file__))
    if (filename):
        return os.path.join(current_file_loc, filename)
    return current_file_loc