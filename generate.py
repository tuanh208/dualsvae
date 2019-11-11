import tensorflow as tf
import yaml
from models.Dual_SVAE import Dual_SVAE

def generate_now(config_file, checkpoint_path=None):
    
    with open(config_file, "r") as stream:
        config = yaml.load(stream)
    
    def print_bytes_sentence(byted_list) :
        sen = []
        for b in byted_list:
            sb = b.decode("utf-8") 
            if sb not in ['<blank>','<s>','</s>'] :
                sen.append(sb)
        print(" ".join(sen))
        return None
    
    graph = tf.Graph()
    
    with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess_:
     
        eval_model = Dual_SVAE(config_file, "Generation")
        #emb_src_batch = eval_model.emb_src_batch_()
        saver = tf.train.Saver()
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        if checkpoint_path==None:
            checkpoint_dir = config["model_dir"]
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

        print(("Evaluating model %s"%checkpoint_path))
        saver.restore(sess_, checkpoint_path)        

        predictions_src, predictions_tgt \
                    = eval_model.prediction_()
        
        tokens_src = predictions_src["tokens"]
        length_src = predictions_src["length"] 
        
        tokens_tgt = predictions_tgt["tokens"]
        length_tgt = predictions_tgt["length"] 

        preds_src, preds_tgt = sess_.run([tokens_src, tokens_tgt])

        for i in range(len(preds_src)):
            print("============================================================")
            print("Generated source sentences: ")
            print("============================================================")
            for tokenized_line in preds_src[i]:
                print_bytes_sentence(tokenized_line)
            print("============================================================")
            print("Generated target sentences:")
            print("============================================================")
            for tokenized_line in preds_tgt[i]:
                print_bytes_sentence(tokenized_line)
        
    return None

config_file = 'configs/Dual_SVAE_TRANSFORMER_config_tuanh.yml'
generate_now(config_file)