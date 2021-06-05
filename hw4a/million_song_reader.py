import csv
import os

import boto3
import h5py
import numpy as np

"""A Complete list of features we are interested in.

'artist_familiarity',
'artist_hotttnesss',
'artist_id',
'artist_latitude',
'artist_location',
'artist_longitude',
'artist_name',
'title',
"artist_terms",
"artist_terms_freq",
"artist_terms_weight",
'danceability',
'duration',
'end_of_fade_in',
'energy',
'key',
'key_confidence',
'loudness',
'mode',
'mode_confidence',
'start_of_fade_out',
'tempo',
'time_signature',
'time_signature_confidence'
'year',
"""


def process_h5_file(h5_file):
    """Process a single h5 file to extract features listed above from the raw MSD.

     For example, to get `artist_familiarity`, refer to:

     https://github.com/tbertinmahieux/MSongsDB/blob/master/PythonSrc/hdf5_getters.py

     So we see that it does h5.root.metadata.songs.cols.artist_familiarity[songidx]
     and it would translate to:

       num_songs = len(file['metadata']['songs'])
       file['metadata']['songs'][:num_songs]['artist_familiarity']

     Since there is one song per file, it simplifies to:

       file['metadata']['songs'][:1]['artist_familiarity']

     We recommend downloading one file, opening it with h5py, and explore/practice

     To see the datatype and shape:

     http://millionsongdataset.com/pages/field-list/
     http://millionsongdataset.com/pages/example-track-description/
     """

    # return the row as a list of values
    row = []

    """
    You should include all fields mentioned at the top of this file.
    You may store the feature names as lists of strings and process the
    features by groups with loops.
    """

    # Example group name
    metadata = [
        'artist_familiarity',  # metadata/songs
        'artist_hotttnesss',  # metadata/songs
        'artist_id',  # metadata/songs
        'artist_latitude',  # metadata/songs
        'artist_location',  # metadata/songs
        'artist_longitude',  # metadata/songs
        'artist_name',  # metadata/songs
        'title',  # metadata/songs
        'song_hotttnesss', #metadata/songs
        "artist_terms",  # metadata
        "artist_terms_freq",  # metadata
        "artist_terms_weight",  # metadata
        'danceability',  # analysis/songs
        'duration',  # analysis/songs
        'end_of_fade_in',  # analysis/songs
        'energy',  # analysis/songs
        'key',  # analysis/songs
        'key_confidence',  # analysis/songs
        'loudness',  # analysis/songs
        'mode',  # analysis/songs
        'mode_confidence',  # analysis/songs
        'start_of_fade_out',  # analysis/songs
        'tempo',  # analysis/songs
        'time_signature',  # analysis/songs
        'time_signature_confidence',  # analysis/songs
        'year',  # musicbrainz/songs
    ]

    """
    Extract field values

    The values of some fields need to be decoded as 'utf-8'.

    You should inspect each feature to make sure that the values
    make sense.

    If song_hotttnesss is NaN, return [] immediately.

    HINT: use the `.decode('utf-8')` function
    """

    def parse_val_to_row(val, field, row):
        tmp_row = []
        shape = val[field].shape[0]
        # if it can be decoded
        if str(val[field].dtype).find('S') != -1:
            # when the shape of data is 0, put a space char
            if shape == 0:
                tmp_row.append('')
                return row.append(tmp_row)
            for j in range(shape):
                tmp_row.append(val[field][j].decode('utf-8'))
        else:
            if shape == 0:
                tmp_row.append('')
                return row.append(tmp_row)
            for j in range(shape):
                tmp_row.append(val[field][j])
        return row.append(tmp_row)

    f = h5py.File(h5_file, 'r')
    val1 = f['metadata']['songs'][:1]
    val2 = f['metadata']
    val3 = f['analysis']['songs'][:1]
    val4 = f['musicbrainz']['songs'][:1]
    if np.isnan(val1['song_hotttnesss']):
        return row

    for i in range(len(metadata)):
        field = metadata[i]
        if i < 9:
            parse_val_to_row(val1, field, row)
            continue
        if i < 12:
            parse_val_to_row(val2, field, row)
            continue
        if i < 25:
            parse_val_to_row(val3, field, row)
            continue
        parse_val_to_row(val4, field, row)
    return row


def process_h5_file_wrapper(path):
    """
    Wrapper function that processes a local h5 file.

    Note that we are treating the h5 file as local
    because we are mounting the MSD snapshot on our instance.

    Do defensive programming by wrapping your call to `process_h5_file`
    in try/except. Think about why this is useful.
    """
    row = process_h5_file(path)
    '''
    try:
        row = process_h5_file(path)
    except:
        print("can not parse %s" %(path))
    '''
    return row


def save_rows(chunk_id, rows, save_local=False):
    """
    Save a list of rows into a temporary local CSV and optionally upload to S3.

    - chunk_id: Chunk id, also the name of our csv file
    - rows: A list of rows which are results of `transform_local`
    - save_local: False if upload to S3. True if save to local (for testing).

    HINT: You may use the `csv` module.
    (You can use other libs like pandas if you like)
    """

    path = f'processed/{chunk_id}.csv'
    with open(path, 'w') as f:
        w = csv.writer(f)
        w.writerows(rows)
    # Write a csv file to path.
    # The file is temporary if it is going to be uploaded to S3.

    # YOUR CODE HERE

    if save_local:
        print(f'csv saved to: {path}')
        return
    """
    If not `save_local`, save the csv file to S3, remove the temp csv file.
    HINT: Use `boto3`. You will need your S3 bucket name.

    You may find the "Bucket Instance Version" section of the
    following tutorial helpful:
    https://realpython.com/python-boto3-aws-s3/
    """

    # YOUR CODE HERE
    s3 = boto3.resource('s3')
    s3.meta.client.upload_file(f'processed/{chunk_id}.csv','10605hw4',f'{chunk_id}.csv')
    # Remove the tempory csv file after we upload it to S3
    os.remove(path)
    print(f'csv upload to: {path}')

"""Convert all files

In this step, we will divide the h5 data points to chunks, where each chunk
will produce a `csv` file that gets stored into your S3 bucket.

We will use `argparse` to parse our command line arguments. The two arguments
are the number of workers and the worker's ID.

For example, a sample run may be:
    `python million_song_reader.py 4 0`

Note: If you have 4 workers, you are expected to run the scripts 4 times with
worker ids 0 to 3, either on a single machine (multiple thread)
or on multiple machines.

Your job is to implement scripts that can partition the conversion task
into `num_workers` parts so that they can run in parallel.
This will speed up the conversion procedure with the same budget by taking
full use of resources.

You will use `process_h5_file_wrapper` and `save_rows`.
You may find `os.walk('YOUR_PATH')` helpful.

You should accumulate `CHUNK_SIZE` rows before calling `save_rows` to write
a chunk to disk. Do not change `CHUNK_SIZE` for grading purposes.
"""
if __name__ == "__main__":
    CHUNK_SIZE = 10000
    save_to_local = True

    import argparse

    parser = argparse.ArgumentParser(description='null')

    parser.add_argument('num_workers', metavar='N',
                        type=int, help='num_workers')
    parser.add_argument('worker_id', metavar='i', type=int, help='worker_id')
    args = parser.parse_args()

    # YOUR CODE HERE
    chunk_id = 0
    current_chunk_size = 0
    rows = []

    data_path = '/home/ec2-user/songs/data/'
    base_root = data_path
    partition = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    delta = (len(partition) + args.num_workers - 1) // args.num_workers
    start = delta * args.worker_id
    end = delta * (args.worker_id + 1)

    for prefix in partition[start:end]:
        for root, dirs, files in os.walk(data_path + prefix):
            print("current dir: %s\t\t accumulate #rows: %d"%(root, len(rows)))
            for name in files:
                if name.endswith('.h5'):
                    row = process_h5_file_wrapper(os.path.join(root, name))
                    if len(row) > 0 and len(row[1]) != 0 and row[1][0] > 1.0:
                        print(name)
                        exit()
                    if len(row) == 0:
                        continue
                    current_chunk_size += 1
                    rows.append(row)
                    if current_chunk_size == CHUNK_SIZE:
                        save_rows(f'{args.worker_id}_{chunk_id}', rows, save_to_local)
                        chunk_id += 1
                        current_chunk_size = 0
                        rows = []

    if len(rows) != 0:
        save_rows(f'{args.worker_id}_{chunk_id}', rows, save_to_local)
