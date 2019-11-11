import tensorflow as tf
from opennmt.utils.optim import *
from opennmt.utils.evaluator import *
from models.Dual_SVAE import Dual_SVAE
from models.inference import eval_Dual_SVAE
from utils.utils_ import kl_coeff
import numpy as np
import os
import yaml
import time
import datetime

# Load config file
config_file = 'configs/Dual_SVAE_TRANSFORMER_config_tuanh.yml'
with open(config_file, "r") as stream:
    config = yaml.load(stream)

# Make necessary directories if not exist
if not os.path.exists(os.path.join(config["model_dir"],"eval")):
    os.makedirs(os.path.join(config["model_dir"],"eval"))
if not os.path.exists(os.path.join(config["model_dir"],"eval","source_from_source")):
    os.makedirs(os.path.join(config["model_dir"],"eval","source_from_source"))
if not os.path.exists(os.path.join(config["model_dir"],"eval","source_from_target")):
    os.makedirs(os.path.join(config["model_dir"],"eval","source_from_target"))
if not os.path.exists(os.path.join(config["model_dir"],"eval","target_from_source")):
    os.makedirs(os.path.join(config["model_dir"],"eval","target_from_source"))
if not os.path.exists(os.path.join(config["model_dir"],"eval","target_from_target")):
    os.makedirs(os.path.join(config["model_dir"],"eval","target_from_target"))
if not os.path.exists(os.path.join(config["model_dir"],"important ckpts")):
    os.makedirs(os.path.join(config["model_dir"],"important ckpts"))

# Create the model
print("Create the model ...")
training_model = Dual_SVAE(config_file, "Training")

# Create global step
global_step = tf.train.create_global_step()

# Set the losses
if config.get("Loss_Function","Cross_Entropy")=="Cross_Entropy":
    CElosses, kl_loss_src, kl_loss_tgt = training_model.loss_()
    loss_src_from_src, loss_src_from_tgt, loss_tgt_from_src, loss_tgt_from_tgt = CElosses
    
    use_kl_annealing = config.get("use_kl_annealing", True)
    if use_kl_annealing : 
        kl_weight = tf.cond(tf.reduce_max(CElosses) > 2.5, lambda: tf.minimum(kl_coeff(global_step),0.001), lambda: tf.minimum(kl_coeff(global_step), 1.))
    else:
        kl_weight = tf.constant(1.)

    tf.summary.scalar("kl_weight", kl_weight)
    
    VAE_loss_src = loss_src_from_src + kl_loss_src * kl_weight
    VAE_loss_tgt = loss_tgt_from_tgt + kl_loss_tgt * kl_weight
    
    model_total_loss = VAE_loss_src + VAE_loss_tgt + loss_src_from_tgt + loss_tgt_from_src

    tf.summary.scalar("model_total_loss", model_total_loss)

inputs = training_model.inputs_()

print("Create training op ...")
if config["mode"] == "Training":
    optimizer_params = config["optimizer_parameters"]
    with tf.variable_scope("main_training", reuse = tf.AUTO_REUSE):
        train_op, _ = optimize_loss(model_total_loss, config["optimizer_parameters"])
        
print("Create the writers and evaluators ...")
src_evaluator = BLEUEvaluator(config["eval_feature_file"], config["model_dir"])
tgt_evaluator = BLEUEvaluator(config["eval_label_file"], config["model_dir"])
writer_bleu = [tf.summary.FileWriter(os.path.join(config["model_dir"],"BLEU","src_from_src")),
               tf.summary.FileWriter(os.path.join(config["model_dir"],"BLEU","src_from_tgt")),
               tf.summary.FileWriter(os.path.join(config["model_dir"],"BLEU","tgt_from_src")),
               tf.summary.FileWriter(os.path.join(config["model_dir"],"BLEU","tgt_from_tgt"))]
writer = tf.summary.FileWriter(config["model_dir"])

var_list_ = tf.global_variables()

saver = tf.train.Saver(var_list_, max_to_keep=config["max_to_keep"])
saver_max_sft = tf.train.Saver(var_list_, max_to_keep = 1)
saver_max_tfs = tf.train.Saver(var_list_, max_to_keep = 1)

print("Start the training session ...")
with tf.Session(config=tf.ConfigProto(log_device_placement=False, 
                                        allow_soft_placement=True, 
                                        gpu_options=tf.GPUOptions(allow_growth=True))) as sess :
    print("Initialize the parameters ...")
    sess.run(tf.tables_initializer())
    sess.run(tf.global_variables_initializer())

    training_summary = tf.summary.merge_all()
    global_step_ = sess.run(global_step)

    checkpoint_path = tf.train.latest_checkpoint(config["model_dir"])
    if checkpoint_path:
        try :
            print("Continue training:...")
            print("Load parameters from %s"%checkpoint_path)
            saver.restore(sess, checkpoint_path)        
            global_step_ = sess.run(global_step)
            print("global_step: ", global_step_)

            eval_Dual_SVAE(config_file, src_evaluator, tgt_evaluator)
            
        except TypeError:
            print("There is a TypeError, the output maybe empty !")
            pass

    else:
        print("Training from scratch")
        
    sess.run(training_model.iterator_initializers())

    total_loss = []
    best_bleu_sft = 0.
    best_bleu_tfs = 0.

    run_time = 0.

    n_iterations = config["iteration_number"]

    print("Start training from step {:d}...".format(global_step_))

    while global_step_ <= n_iterations:                       
        #=================== 1 iteration=======================
        start_time = time.time()
        
        loss_src_from_src_, loss_src_from_tgt_, loss_tgt_from_src_, loss_tgt_from_tgt_, \
        kl_loss_src_, kl_loss_tgt_, model_total_loss_, \
        global_step_, _, \
        kl_weight_ = sess.run([loss_src_from_src, loss_src_from_tgt, loss_tgt_from_src, loss_tgt_from_tgt,
                            kl_loss_src, kl_loss_tgt, model_total_loss,
                                global_step, 
                                train_op ,
                                kl_weight])
        
        run_time += time.time() - start_time
        
        total_loss.append(model_total_loss_)

        #==================printing things======================
        if (np.mod(global_step_, config["printing_freq"])) == 0:            
            print((datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            print("*********************Step: {:d} - RTime: {:f}**********************".format(global_step_,run_time))
            print("CE Loss sfs: {:4f} - CE Loss sft: {:4f} - CE Loss tfs: {:4f} - CE Loss tft: {:4f}".format(loss_src_from_src_, loss_src_from_tgt_, loss_tgt_from_src_, loss_tgt_from_tgt_))
            print("KL Loss Source: {:4f} - KL Loss Target: {:4f} - KL Weight: {:4f}".format(kl_loss_src_, kl_loss_tgt_,kl_weight_))
            print("TotalLoss at step {:d}: {:4f}".format(global_step_, np.mean(total_loss)))
            print("********************************************************************************")
            run_time = 0.             

        if (np.mod(global_step_, config["summary_freq"])) == 0:
            training_summary_ = sess.run(training_summary)
            writer.add_summary(training_summary_, global_step=global_step_)
            writer.flush()
            total_loss = []

        if (np.mod(global_step_, config["save_freq"])) == 0 and global_step_ > 0:    
            print((datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            checkpoint_path = os.path.join(config["model_dir"], 'model.ckpt')
            print(("save to %s"%(checkpoint_path)))
            saver.save(sess, checkpoint_path, global_step = global_step_)

        if (np.mod(global_step_, config["eval_freq"])) == 0 and global_step_ >0: 
            try :
                checkpoint_path = tf.train.latest_checkpoint(config["model_dir"])
                
                score_sfs, score_sft, score_tfs, score_tft = eval_Dual_SVAE(config_file, src_evaluator, tgt_evaluator, checkpoint_path=checkpoint_path)

                if score_sft > best_bleu_sft :
                    print((datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    checkpoint_path = os.path.join(config["model_dir"],"important ckpts", 'bestBLEU_sft.model.ckpt')
                    print(("save to %s"%(checkpoint_path)))
                    saver_max_sft.save(sess, checkpoint_path, global_step = global_step_)
                    best_bleu_sft = score_sft

                if score_tfs > best_bleu_tfs :
                    print((datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                    checkpoint_path = os.path.join(config["model_dir"],"important ckpts", 'bestBLEU_tfs.model.ckpt')
                    print(("save to %s"%(checkpoint_path)))
                    saver_max_tfs.save(sess, checkpoint_path, global_step = global_step_)
                    best_bleu_tfs = score_tfs

                score_summary = tf.Summary(value=[tf.Summary.Value(tag="source_BLEU", simple_value=score_sft)])
                writer_bleu[0].add_summary(score_summary, global_step_)
                writer_bleu[0].flush()
                
                score_summary = tf.Summary(value=[tf.Summary.Value(tag="source_BLEU", simple_value=score_sfs)])
                writer_bleu[1].add_summary(score_summary, global_step_)
                writer_bleu[1].flush()

                score_summary = tf.Summary(value=[tf.Summary.Value(tag="target_BLEU", simple_value=score_tfs)])
                writer_bleu[2].add_summary(score_summary, global_step_)
                writer_bleu[2].flush()
                
                score_summary = tf.Summary(value=[tf.Summary.Value(tag="target_BLEU", simple_value=score_tft)])
                writer_bleu[3].add_summary(score_summary, global_step_)
                writer_bleu[3].flush()
                    
            except TypeError:
                print("There is a TypeError, the output maybe empty !")
                pass
