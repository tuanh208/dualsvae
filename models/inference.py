from models.Dual_SVAE import Dual_SVAE
import yaml
import os
import tensorflow as tf

def inference_Dual_SVAE(config_file, src_eval_file=None, tgt_eval_file=None, checkpoint_path=None):
    
    with open(config_file, "r") as stream:
        config = yaml.load(stream)
        
    if src_eval_file == None :
        src_eval_file = config["eval_feature_file"]
        
    if tgt_eval_file == None :
        tgt_eval_file = config["eval_label_file"]
    
    from opennmt.utils.misc import print_bytes
    
    graph = tf.Graph()
    
    with tf.Session(graph=graph,config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True))) as sess_:
     
        eval_model = Dual_SVAE(config_file, "Inference", src_eval_file, tgt_eval_file)
        #emb_src_batch = eval_model.emb_src_batch_()
        saver = tf.train.Saver()
        tf.tables_initializer().run()
        tf.global_variables_initializer().run()

        if checkpoint_path==None:
            checkpoint_dir = config["model_dir"]
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

        print(("Evaluating model %s"%checkpoint_path))
        saver.restore(sess_, checkpoint_path)        

        predictions_src_from_src, predictions_src_from_tgt, predictions_tgt_from_src, predictions_tgt_from_tgt \
                    = eval_model.prediction_()
        
        tokens_src_from_src = predictions_src_from_src["tokens"]
        length_src_from_src = predictions_src_from_src["length"] 
        
        tokens_src_from_tgt = predictions_src_from_tgt["tokens"]
        length_src_from_tgt = predictions_src_from_tgt["length"] 
        
        tokens_tgt_from_src = predictions_tgt_from_src["tokens"]
        length_tgt_from_src = predictions_tgt_from_src["length"] 
        
        tokens_tgt_from_tgt = predictions_tgt_from_tgt["tokens"]
        length_tgt_from_tgt = predictions_tgt_from_tgt["length"] 
        
        sess_.run(eval_model.iterator_initializers())
        
        # pred_dict = sess_.run([predictions])
        pred_dict = None
        
        print("write to :%s"%os.path.join(config["model_dir"],"eval","*/*"+os.path.basename(checkpoint_path)))
        
        source_from_source_path = os.path.join(config["model_dir"],
                                             "eval",
                                             "source_from_source",
                                             os.path.basename(src_eval_file) + ".s-s." + os.path.basename(checkpoint_path))
        
        source_from_target_path = os.path.join(config["model_dir"],
                                             "eval",
                                             "source_from_target",
                                             os.path.basename(tgt_eval_file) + ".s-t." + os.path.basename(checkpoint_path))
        
        target_from_source_path = os.path.join(config["model_dir"],
                                             "eval",
                                             "target_from_source",
                                             os.path.basename(src_eval_file) + ".t-s." + os.path.basename(checkpoint_path))
        
        target_from_target_path = os.path.join(config["model_dir"],
                                             "eval",
                                             "target_from_target",
                                             os.path.basename(tgt_eval_file) + ".t-t." + os.path.basename(checkpoint_path))
        
        with open(source_from_source_path,"w") as output_0_, \
                open(source_from_target_path,"w") as output_1_, \
                    open(target_from_source_path,"w") as output_2_, \
                        open(target_from_target_path,"w") as output_3_ :
            while True:                 
                try:                
                    _tokens_src_from_src, _length_src_from_src, \
                    _tokens_src_from_tgt, _length_src_from_tgt, \
                    _tokens_tgt_from_src, _length_tgt_from_src, \
                    _tokens_tgt_from_tgt, _length_tgt_from_tgt = sess_.run([tokens_src_from_src, length_src_from_src,
                                                                            tokens_src_from_tgt, length_src_from_tgt,
                                                                            tokens_tgt_from_src, length_tgt_from_src,
                                                                            tokens_tgt_from_tgt, length_tgt_from_tgt])
                  
                    for b in range(_tokens_src_from_src.shape[0]):                        
                        pred_toks = _tokens_src_from_src[b][0][:_length_src_from_src[b][0] - 1]                                                
                        pred_sent = b" ".join(pred_toks)                        
                        print_bytes(pred_sent, output_0_)    
                        
                    for b in range(_tokens_src_from_tgt.shape[0]):                        
                        pred_toks = _tokens_src_from_tgt[b][0][:_length_src_from_tgt[b][0] - 1]                                                
                        pred_sent = b" ".join(pred_toks)                        
                        print_bytes(pred_sent, output_1_) 
                        
                    for b in range(_tokens_tgt_from_src.shape[0]):                        
                        pred_toks = _tokens_tgt_from_src[b][0][:_length_tgt_from_src[b][0] - 1]                                                
                        pred_sent = b" ".join(pred_toks)                        
                        print_bytes(pred_sent, output_2_) 
                        
                    for b in range(_tokens_tgt_from_tgt.shape[0]):                        
                        pred_toks = _tokens_tgt_from_tgt[b][0][:_length_tgt_from_tgt[b][0] - 1]                                                
                        pred_sent = b" ".join(pred_toks)                        
                        print_bytes(pred_sent, output_3_) 
                        
                except tf.errors.OutOfRangeError:
                    break
        
        print("Finish inference !")
        
    return source_from_source_path, source_from_target_path, target_from_source_path, target_from_target_path, pred_dict

def eval_Dual_SVAE(config_file, src_evaluator, tgt_evaluator, checkpoint_path=None) :
    
    with open(config_file, "r") as stream:
        config = yaml.load(stream)

    path_sfs, path_sft, path_tfs, path_tft, prediction_dict = inference_Dual_SVAE(config_file, checkpoint_path=checkpoint_path)
    score_sfs = src_evaluator.score(config["eval_feature_file"], path_sfs)
    score_sft = src_evaluator.score(config["eval_feature_file"], path_sft)
    score_tfs = tgt_evaluator.score(config["eval_label_file"], path_tfs)
    score_tft = tgt_evaluator.score(config["eval_label_file"], path_tft)

    print("=========================================EVALUATION SCORE==========================================")
    print("src_from_src BLEU at checkpoint %s: %f"%(checkpoint_path, score_sfs))
    print("src_from_tgt BLEU at checkpoint %s: %f"%(checkpoint_path, score_sft))
    print("tgt_from_src BLEU at checkpoint %s: %f"%(checkpoint_path, score_tfs))
    print("tgt_from_tgt BLEU at checkpoint %s: %f"%(checkpoint_path, score_tft))
    print("===================================================================================================")

    return score_sfs, score_sft, score_tfs, score_tft