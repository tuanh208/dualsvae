import tensorflow as tf
from opennmt import constants
from opennmt.utils.misc import count_lines
from utils.utils_ import create_embeddings, random_replace

def load_vocab(vocab_path, vocab_size):
    if not vocab_size:
        vocab_size = count_lines(vocab_path) + 1 #for UNK
        print("vocab size of",vocab_path,":",vocab_size)
    vocab = tf.contrib.lookup.index_table_from_file(vocab_path, vocab_size = vocab_size - 1, num_oov_buckets = 1)
    return vocab, vocab_size

def get_dataset_size(data_file):
    return count_lines(data_file)

def get_padded_shapes(dataset):    
    return tf.contrib.framework.nest.map_structure(
    lambda shape: shape.as_list(), dataset.output_shapes)

def filter_irregular_batches(multiple):    
    if multiple == 1:
        return lambda dataset: dataset

    def _predicate(*x):
        flat = tf.contrib.framework.nest.flatten(x)
        batch_size = tf.shape(flat[0])[0]
        return tf.equal(tf.mod(batch_size, multiple), 0)

    return lambda dataset: dataset.filter(_predicate)

def prefetch_element(buffer_size=None):  
    support_auto_tuning = hasattr(tf.data, "experimental") or hasattr(tf.contrib.data, "AUTOTUNE")
    if not support_auto_tuning and buffer_size is None:
        buffer_size = 1
    return lambda dataset: dataset.prefetch(buffer_size)

def load_data(src_path, src_vocab, batch_size=32, batch_type ="examples", batch_multiplier = 1, tgt_path=None, tgt_vocab=None, 
              max_len=50, bucket_width = 1, mode="Training", padded_shapes = None, 
              shuffle_buffer_size = None, prefetch_buffer_size = 100000, num_threads = 4, version=None, distribution=None, tf_idf_table=None,
              decoder_input_word_dropout_rate = 0.1):

    batch_size = batch_size * batch_multiplier
    print("batch_size", batch_size)
    
    def _make_dataset(text_path):
        dataset = tf.data.TextLineDataset(text_path)
        dataset = dataset.map(lambda x: tf.string_split([x]).values) #split by spaces
        return dataset    
       
    def _batch_func(dataset):
        return dataset.padded_batch(batch_size,
                                    padded_shapes=padded_shapes or get_padded_shapes(dataset))

    def _key_func(dataset):                
        #bucket_id = tf.squeeze(dataset["domain"])
        features_length = dataset["src_length"] #features_length_fn(features) if features_length_fn is not None else None
        labels_length = dataset["tgt_length"] #labels_length_fn(labels) if labels_length_fn is not None else None        
        bucket_id = tf.constant(0, dtype=tf.int32)
        if features_length is not None:
            bucket_id = tf.maximum(bucket_id, features_length // bucket_width)
        if labels_length is not None:
            bucket_id = tf.maximum(bucket_id, labels_length // bucket_width)
        return tf.cast(bucket_id, tf.int64)
        #return tf.to_int64(bucket_id)

    def _reduce_func(unused_key, dataset):
        return _batch_func(dataset)

    def _window_size_func(key):
        if bucket_width > 1:
            key += 1  # For bucket_width == 1, key 0 is unassigned.
        size = batch_size // (key * bucket_width)
        if batch_multiplier > 1:
            # Make the window size a multiple of batch_multiplier.
            size = size + batch_multiplier - size % batch_multiplier
        return tf.to_int64(tf.maximum(size, batch_multiplier))             
    
    bos = tf.constant([constants.START_OF_SENTENCE_ID], dtype=tf.int64)
    eos = tf.constant([constants.END_OF_SENTENCE_ID], dtype=tf.int64)
    
    if version==None:
        print("old dataprocessing version")
        src_dataset = _make_dataset(src_path)            
        if mode=="Training" or mode=="Inference":
            tgt_dataset = _make_dataset(tgt_path)
            dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))
        elif mode == "Predict":
            dataset = src_dataset

        if mode=="Training":    
            unk_id_src = src_vocab.lookup(tf.constant(constants.UNKNOWN_TOKEN))
            unk_id_tgt = tgt_vocab.lookup(tf.constant(constants.UNKNOWN_TOKEN))
    
            dataset = dataset.map(lambda x,y:{                      
                    "src_raw": x,
                    "tgt_raw": y,
                    "src_ids": src_vocab.lookup(x),
                    "tgt_ids": tgt_vocab.lookup(y),
                    "src_ids_in": tf.concat([bos, src_vocab.lookup(x)], axis=0),
                    "src_ids_in_dropped": tf.concat([bos, random_replace(src_vocab.lookup(x), 
                                                                         new_value = unk_id_src, 
                                                                         p = decoder_input_word_dropout_rate )], axis=0),
                    "tgt_ids_in": tf.concat([bos, tgt_vocab.lookup(y)], axis=0),
                    "tgt_ids_in_dropped": tf.concat([bos, random_replace(tgt_vocab.lookup(y), 
                                                                         new_value = unk_id_tgt, 
                                                                         p = decoder_input_word_dropout_rate )], axis=0),
                    "src_ids_out": tf.concat([src_vocab.lookup(x), eos], axis=0),
                    "tgt_ids_out": tf.concat([tgt_vocab.lookup(y), eos], axis=0),
                    "src_length": tf.shape(src_vocab.lookup(x))[0],
                    "tgt_length": tf.shape(tgt_vocab.lookup(y))[0],                
                    }, num_parallel_calls=num_threads)    
                       
        elif mode == "Inference":            
            dataset = dataset.map(lambda x, y:{                    
                    "src_raw": x,                
                    "src_ids": src_vocab.lookup(x),                
                    "src_length": tf.shape(src_vocab.lookup(x))[0], 
                    "tgt_raw": y,                
                    "tgt_ids": tgt_vocab.lookup(y),                
                    "tgt_length": tf.shape(tgt_vocab.lookup(y))[0], 
                    }, num_parallel_calls=num_threads) 
            
        elif mode == "Predict":            
            dataset = dataset.map(lambda x:{
                    "src_raw": x,                
                    "src_ids": src_vocab.lookup(x),                
                    "src_length": tf.shape(src_vocab.lookup(x))[0],                
                    }, num_parallel_calls=num_threads)
            
        if mode=="Training":            
            if shuffle_buffer_size is not None and shuffle_buffer_size != 0:            
                dataset_size = get_dataset_size(src_path) 
                if dataset_size is not None:
                    if shuffle_buffer_size < 0:
                        shuffle_buffer_size = dataset_size
                elif shuffle_buffer_size < dataset_size:        
                    dataset = dataset.apply(random_shard(shuffle_buffer_size, dataset_size))        
                dataset = dataset.shuffle(shuffle_buffer_size)

            dataset = dataset.filter(lambda x: tf.logical_and(tf.logical_and(tf.greater(x["src_length"],0), tf.greater(x["tgt_length"], 0)), tf.logical_and(tf.less_equal(x["src_length"], max_len), tf.less_equal(x["tgt_length"], max_len))))
            
            if bucket_width is None:
                dataset = dataset.apply(_batch_func)
            else:
                if hasattr(tf.data, "experimental"):
                    group_by_window_fn = tf.data.experimental.group_by_window
                else:
                    group_by_window_fn = tf.contrib.data.group_by_window
                print("batch type: ", batch_type)
                if batch_type == "examples":
                    dataset = dataset.apply(group_by_window_fn(_key_func, _reduce_func, window_size = batch_size))
                elif batch_type == "tokens":
                    dataset = dataset.apply(group_by_window_fn(_key_func, _reduce_func, window_size_func = _window_size_func))   
                else:
                    raise ValueError(
                            "Invalid batch type: '{}'; should be 'examples' or 'tokens'".format(batch_type))
            dataset = dataset.apply(filter_irregular_batches(batch_multiplier))             
            dataset = dataset.repeat()
            dataset = dataset.apply(prefetch_element(buffer_size=prefetch_buffer_size))                        
        else:
            dataset = dataset.apply(_batch_func)                      
        
    return dataset.make_initializable_iterator()