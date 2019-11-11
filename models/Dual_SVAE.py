import tensorflow as tf
import opennmt as onmt
from opennmt.utils.parallel import GraphDispatcher
from opennmt.utils.losses import cross_entropy_sequence_loss
from utils.dataprocess import *
import yaml

class Dual_SVAE:
    def _compute_loss(self, outputs, src_ids_out, src_length, tgt_ids_out, tgt_length, 
                                        mu_src, logvar_src, mu_tgt, logvar_tgt, params, mode):
        
        if mode == "Training":
            mode = tf.estimator.ModeKeys.TRAIN            
        else:
            mode = tf.estimator.ModeKeys.EVAL            
          
        if self.Loss_type == "Cross_Entropy":
            if isinstance(outputs, dict):
                logits_src_from_src = outputs["logits_src_from_src"]
                logits_src_from_tgt = outputs["logits_src_from_tgt"]
                logits_tgt_from_src = outputs["logits_tgt_from_src"]
                logits_tgt_from_tgt = outputs["logits_tgt_from_tgt"]
                            
            loss_src_from_src, loss_normalizer_src_from_src, loss_token_normalizer_src_from_src = \
                    cross_entropy_sequence_loss(logits_src_from_src,
                                                src_ids_out, 
                                                src_length + 1,                                                         
                                                label_smoothing = params.get("label_smoothing", 0.0),
                                                average_in_time = params.get("average_loss_in_time", True),
                                                mode = mode)
            
            loss_src_from_tgt, loss_normalizer_src_from_tgt, loss_token_normalizer_src_from_tgt = \
                    cross_entropy_sequence_loss(logits_src_from_tgt,
                                                src_ids_out, 
                                                src_length + 1,                                                         
                                                label_smoothing = params.get("label_smoothing", 0.0),
                                                average_in_time = params.get("average_loss_in_time", True),
                                                mode = mode)
            
            loss_tgt_from_src, loss_normalizer_tgt_from_src, loss_token_normalizer_tgt_from_src = \
                    cross_entropy_sequence_loss(logits_tgt_from_src,
                                                tgt_ids_out, 
                                                tgt_length + 1,                                                         
                                                label_smoothing = params.get("label_smoothing", 0.0),
                                                average_in_time = params.get("average_loss_in_time", True),
                                                mode = mode)
            
            loss_tgt_from_tgt, loss_normalizer_tgt_from_tgt, loss_token_normalizer_tgt_from_tgt = \
                    cross_entropy_sequence_loss(logits_tgt_from_tgt,
                                                tgt_ids_out, 
                                                tgt_length + 1,                                                         
                                                label_smoothing = params.get("label_smoothing", 0.0),
                                                average_in_time = params.get("average_loss_in_time", True),
                                                mode = mode)
            
            #----- Calculating kl divergence --------

            kld_loss_src = -0.5 * tf.reduce_sum(logvar_src - tf.pow(mu_src, 2) - tf.exp(logvar_src) + 1, 1)
            kld_loss_tgt = -0.5 * tf.reduce_sum(logvar_tgt - tf.pow(mu_tgt, 2) - tf.exp(logvar_tgt) + 1, 1)

            return loss_src_from_src, loss_normalizer_src_from_src, loss_token_normalizer_src_from_src, \
                   loss_src_from_tgt, loss_normalizer_src_from_tgt, loss_token_normalizer_src_from_tgt, \
                   loss_tgt_from_src, loss_normalizer_tgt_from_src, loss_token_normalizer_tgt_from_src, \
                   loss_tgt_from_tgt, loss_normalizer_tgt_from_tgt, loss_token_normalizer_tgt_from_tgt, \
                   kld_loss_src, kld_loss_tgt
        
    
    def _initializer(self, params):
        
        if params["Architecture"] == "Transformer":
            print("tf.variance_scaling_initializer")
            return tf.variance_scaling_initializer(
        mode="fan_avg", distribution="uniform", dtype=self.dtype)
        else:            
            param_init = params.get("param_init")
            if param_init is not None:
                print("tf.random_uniform_initializer")
                return tf.random_uniform_initializer(
              minval=-param_init, maxval=param_init, dtype=self.dtype)
        return None
        
    def __init__(self, config_file, mode, test_feature_file=None, test_label_file=None):

        def _normalize_loss(num, den=None):
            """Normalizes the loss."""
            if isinstance(num, list):  # Sharded mode.
                if den is not None:
                    assert isinstance(den, list)
                    return tf.add_n(num) / tf.add_n(den) #tf.reduce_mean([num_/den_ for num_,den_ in zip(num, den)]) #tf.add_n(num) / tf.add_n(den)
                else:
                    return tf.reduce_mean(num)
            elif den is not None:
                return num / den
            else:
                return num

        def _extract_loss(all_loss, Loss_type="Cross_Entropy"):
            """Extracts and summarizes the loss."""
            print("loss numb:", len(all_loss))
            if Loss_type=="Cross_Entropy":  
                CElosses = []
                tboard_losses = []
                
                # print(all_loss)
                
                for i in range(4) :
                    actual_loss = _normalize_loss(all_loss[3*i+0], den=all_loss[3*i+1])
                    tboard_loss = _normalize_loss(all_loss[3*i+0], den=all_loss[3*i+2])
                    CElosses.append(actual_loss)
                    tboard_losses.append(tboard_loss)
                    
                    
                loss_kd_src = _normalize_loss(all_loss[-2])
                loss_kd_tgt = _normalize_loss(all_loss[-1])
                
                tf.summary.scalar("CEloss_src_from_src", tboard_losses[0])
                tf.summary.scalar("CEloss_src_from_tgt", tboard_losses[1])
                tf.summary.scalar("CEloss_tgt_from_src", tboard_losses[2])
                tf.summary.scalar("CEloss_tgt_from_tgt", tboard_losses[3])
                tf.summary.scalar("loss_kd_src", loss_kd_src)
                tf.summary.scalar("loss_kd_tgt", loss_kd_tgt)

            return CElosses,loss_kd_src, loss_kd_tgt                  

        def _loss_op(inputs, params, mode):
            """Single callable to compute the loss."""
            logits, _, src_ids_out, src_length, tgt_ids_out, tgt_length, mu_src, logvar_src, mu_tgt, logvar_tgt \
                                                                                = self._build(inputs, params, mode)
            losses = self._compute_loss(logits, src_ids_out, src_length, tgt_ids_out, tgt_length, 
                                        mu_src, logvar_src, mu_tgt, logvar_tgt, params, mode)
            
            return losses

        with open(config_file, "r") as stream:
            config = yaml.load(stream)
        
        Loss_type = config.get("Loss_Function","Cross_Entropy")
        
        self.Loss_type = Loss_type
        self.config = config 
        self.using_tf_idf = config.get("using_tf_idf", False)
        
        train_batch_size = config["training_batch_size"]   
        eval_batch_size = config["eval_batch_size"]
        
        self.word_dropout = config.get("word_dropout",0.1)
        
        self.latent_variable_size = config.get("latent_variable_size",128)
        self.latent_variable_size_for_output = config.get("latent_variable_size_for_output", 128)
        
        max_len = config["max_len"]
        
        example_sampling_distribution = config.get("example_sampling_distribution",None)
        self.dtype = tf.float32
        
        # Input pipeline:
        # Return lookup table of type index_table_from_file
        src_vocab, src_vocab_size = load_vocab(config["src_vocab_path"], config.get("src_vocab_size", None))
        tgt_vocab, tgt_vocab_size = load_vocab(config["tgt_vocab_path"], config.get("tgt_vocab_size", None))
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        
        load_data_version = config.get("dataprocess_version",None)
        
        if mode == "Training":    
            print("num_devices", config.get("num_devices",1))
            
            dispatcher = GraphDispatcher(
                config.get("num_devices",1), 
                daisy_chain_variables=config.get("daisy_chain_variables",False), 
                devices= config.get("devices",None)
            ) 
            
            batch_multiplier = config.get("num_devices", 1)
            num_threads = config.get("num_threads", 4)
            
            if Loss_type == "Wasserstein":
                self.using_tf_idf = True
                
            if self.using_tf_idf:
                tf_idf_table = build_tf_idf_table(
                    config["tgt_vocab_path"], 
                    self.src_vocab_size, 
                    config["domain_numb"], 
                    config["training_feature_file"])           
                self.tf_idf_table = tf_idf_table
                
            iterator = load_data(
                config["training_label_file"], 
                src_vocab, 
                batch_size = train_batch_size, 
                batch_type=config["training_batch_type"], 
                batch_multiplier = batch_multiplier, 
                tgt_path=config["training_feature_file"], 
                tgt_vocab=tgt_vocab, 
                max_len = max_len, 
                mode=mode, 
                shuffle_buffer_size = config["sample_buffer_size"], 
                num_threads = num_threads, 
                version = load_data_version, 
                distribution = example_sampling_distribution,
                decoder_input_word_dropout_rate = config["decoder_input_word_dropout_rate"]
            )
            
            inputs = iterator.get_next()
            data_shards = dispatcher.shard(inputs)

            with tf.variable_scope(config["Architecture"], initializer=self._initializer(config)):
                losses_shards = dispatcher(_loss_op, data_shards, config, mode)

            self.loss = _extract_loss(losses_shards, Loss_type=Loss_type) 

        elif mode == "Inference": 
            assert test_feature_file != None
            
            iterator = load_data(
                test_feature_file, 
                src_vocab, 
                batch_size = eval_batch_size, 
                batch_type = "examples", 
                batch_multiplier = 1, 
                tgt_path=test_label_file, 
                tgt_vocab=tgt_vocab, 
                max_len = max_len, 
                mode = mode, 
                version = load_data_version
            )
            
            inputs = iterator.get_next() 
            
            with tf.variable_scope(config["Architecture"]):
                _ , self.predictions, _, _, _, _, _, _, _, _ = self._build(inputs, config, mode)
                
        elif mode == "Generation":
            iterator = None
            inputs = None
            
            with tf.variable_scope(config["Architecture"]):
                _ , self.predictions, _, _, _, _, _, _, _, _ = self._build(inputs, config, mode)
        else :
            print("ERROR: mode must be in : [\"Training\",\"Inference\",\"Generation\"] .")
            return
            
        self.iterator = iterator
        self.inputs = inputs
        
    def loss_(self):
        return self.loss
    
    def prediction_(self):
        return self.predictions
   
    def inputs_(self):
        return self.inputs
    
    def iterator_initializers(self):
        if isinstance(self.iterator,list):
            return [iterator.initializer for iterator in self.iterator]
        else:
            return [self.iterator.initializer]        
           
    def _build(self, inputs, config, mode):     
        
        assert config["Architecture"] == "Transformer", "This is only for Transformer"

        debugging = config.get("debugging", False)
        Loss_type = self.Loss_type       
        print("Loss_type: ", Loss_type)           

        hidden_size = config["hidden_size"]       
        print("hidden size: ", hidden_size)
                
        tgt_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(config["tgt_vocab_path"], vocab_size= int(self.tgt_vocab_size) - 1, default_value=constants.UNKNOWN_TOKEN)
        src_vocab_rev = tf.contrib.lookup.index_to_string_table_from_file(config["src_vocab_path"], vocab_size= int(self.src_vocab_size) - 1, default_value=constants.UNKNOWN_TOKEN)
        
        end_token = constants.END_OF_SENTENCE_ID
        
        # Embedding        
        size_src = config.get("src_embedding_size",512)
        size_tgt = config.get("tgt_embedding_size",512)
        latent_variable_size = self.latent_variable_size
        latent_variable_size_for_output = self.latent_variable_size_for_output
        
        with tf.variable_scope("src_embedding"):
            src_emb = create_embeddings(self.src_vocab_size, depth=size_src)

        with tf.variable_scope("tgt_embedding"):
            tgt_emb = create_embeddings(self.tgt_vocab_size, depth=size_tgt)

        self.tgt_emb = tgt_emb
        self.src_emb = src_emb

        # Build encoder, decoder
#---------------------------------------------TRANSFORMER-----------------------------------------#
        if config["Architecture"] == "Transformer":
            nlayers = config.get("nlayers",4)

#==============================ENCODER==================================
# Requires 2 encoder for SVAE
            encoder_src = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
                nlayers, 
                num_units=hidden_size, 
                num_heads=8, 
                ffn_inner_dim=2048, 
                dropout=0.1, 
                attention_dropout=0.1, 
                relu_dropout=0.1
            )  
    
            encoder_tgt = onmt.encoders.self_attention_encoder.SelfAttentionEncoder(
                nlayers, 
                num_units=hidden_size, 
                num_heads=8, 
                ffn_inner_dim=2048, 
                dropout=0.1, 
                attention_dropout=0.1, 
                relu_dropout=0.1
            )  
            
#==============================DECODER==================================
            decoder_src = onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
                nlayers, 
                num_units=hidden_size, 
                num_heads=8, 
                ffn_inner_dim=2048, 
                dropout=0.1, 
                attention_dropout=0.1, 
                relu_dropout=0.1
            )
    
            decoder_tgt = onmt.decoders.self_attention_decoder.SelfAttentionDecoder(
                nlayers, 
                num_units=hidden_size, 
                num_heads=8, 
                ffn_inner_dim=2048, 
                dropout=0.1, 
                attention_dropout=0.1, 
                relu_dropout=0.1
            )

        print("Model type: ", config["Architecture"])
        
        output_layer = None
        
        def outputs_to_z(outputs, W_out_to_mu, b_out_to_mu, W_out_to_logvar, b_out_to_logvar):
            """
            Map the outputs vector of encoder to latent random variable z.

            Args:
                - outputs: shape [batch_size, max_length, hidden_size]
                - W_out_to_mu, W_out_to_logvar : shape [hidden_size, latent_variable_size_for_output]
                - b_out_to_mu, b_out_to_logvar: shape [latent_variable_size_for_output]
            Return: 
                - z, mu, logvar: shape [batch_size, max_length, latent_variable_size_for_output] 
            """

            # important infos
            batch_size = tf.shape(outputs)[0]
            max_length = tf.shape(outputs)[1]
            latent_variable_size_for_output = tf.shape(W_out_to_mu)[1]

            mu = tf.map_fn(lambda x: tf.add(tf.matmul(x, W_out_to_mu) , b_out_to_mu), outputs)
            logvar = tf.map_fn(lambda x: tf.add(tf.matmul(x, W_out_to_logvar) , b_out_to_logvar), outputs)
            std = tf.exp(0.5*logvar)

            z = tf.random_normal([batch_size, max_length, latent_variable_size_for_output])
#                 # Can be done other way :
#                 z = tf.random_normal([batch_size, latent_variable_size_for_output])
#                 z = tf.reshape(z, [batch_size,1,-1]) 
#                 z = tf.tile(z,[1, max_length, 1])

            z = z * std + mu

            return z, mu, logvar

        def z_to_inputs(z, W_z_to_input, b_z_to_input):
            """
            Map the latent random variable z to the inputs vector for decoder

            Args:
                - z: shape [batch_size, max_length, latent_variable_size_for_output]
                - W_z_to_input : shape [latent_variable_size_for_output, hidden_size]
                - b_z_to_output : shape [hidden_size]

            Return: 
                - z, mu, logvar: shape [batch_size, max_length, latent_variable_size_for_output] 
            """
            return tf.map_fn(lambda x: tf.matmul(x, W_z_to_input) + b_z_to_input, z)
        
#         with tf.variable_scope("z_weights_in"):
        
#             W_out_to_mu = tf.get_variable('output_to_mu_weight', shape = [hidden_size, latent_variable_size_for_output])
#             b_out_to_mu = tf.get_variable('output_to_mu_bias', shape = [latent_variable_size_for_output])
#             W_out_to_logvar = tf.get_variable('output_to_logvar_weight', shape = [hidden_size, latent_variable_size_for_output])
#             b_out_to_logvar = tf.get_variable('output_to_logvar_bias', shape = [latent_variable_size_for_output])
        
#         with tf.variable_scope("z_weights_out"):
#             W_z_to_input = tf.get_variable('z_to_input_weight', shape = [latent_variable_size_for_output, hidden_size])
#             b_z_to_input = tf.get_variable('z_to_input_bias', shape = [hidden_size])

#-----------------------------------------------------TRAINING MODE-----------------------------------------------#
        if mode =="Training":            
            print("Building model in Training mode")
            
            src_length = inputs["src_length"]
            tgt_length = inputs["tgt_length"]
            
            emb_src_batch = tf.nn.embedding_lookup(src_emb, inputs["src_ids"]) # dim = [batch, length, depth]
            emb_tgt_batch = tf.nn.embedding_lookup(tgt_emb, inputs["tgt_ids"])  
            
            emb_src_batch_in = tf.nn.embedding_lookup(src_emb, inputs["src_ids_in"])   
            emb_tgt_batch_in = tf.nn.embedding_lookup(tgt_emb, inputs["tgt_ids_in"])  
            
            emb_src_batch_in_dropped = tf.nn.embedding_lookup(src_emb, inputs["src_ids_in_dropped"])
            emb_tgt_batch_in_dropped = tf.nn.embedding_lookup(tgt_emb, inputs["tgt_ids_in_dropped"])  
            
            tgt_ids_out_batch = inputs["tgt_ids_out"]
            src_ids_out_batch = inputs["src_ids_out"]
            
            self.emb_tgt_batch = emb_tgt_batch
            self.emb_src_batch = emb_src_batch
            
            print("emb_src_batch: ", emb_src_batch)
            print("emb_tgt_batch: ", emb_tgt_batch)
                        
            #========ENCODER_PROCESS======================
            with tf.variable_scope("source_encoder", reuse=tf.AUTO_REUSE):
                        
                encoder_outputs_src, encoder_states_src, encoder_seq_length_src = encoder_src.encode(
                        emb_src_batch, 
                        sequence_length = src_length, 
                        mode=tf.estimator.ModeKeys.TRAIN
                )
                    #encoder_outputs_src: [batch_size, max_length, hidden_size]
                
            with tf.variable_scope("target_encoder", reuse=tf.AUTO_REUSE):
                        
                encoder_outputs_tgt, encoder_states_tgt, encoder_seq_length_tgt = encoder_tgt.encode(
                        emb_tgt_batch, 
                        sequence_length = tgt_length, 
                        mode=tf.estimator.ModeKeys.TRAIN
                )
                    
            #======GENERATIVE_PROCESS====================
            
            with tf.variable_scope("source_output_to_z"):
                
                W_out_to_mu = tf.get_variable('output_to_mu_weight', shape = [hidden_size, latent_variable_size_for_output])
                b_out_to_mu = tf.get_variable('output_to_mu_bias', shape = [latent_variable_size_for_output])
                W_out_to_logvar = tf.get_variable('output_to_logvar_weight', shape = [hidden_size, latent_variable_size_for_output])
                b_out_to_logvar = tf.get_variable('output_to_logvar_bias', shape = [latent_variable_size_for_output])
                
                z_from_src, mu_src, logvar_src = outputs_to_z(encoder_outputs_src, W_out_to_mu, b_out_to_mu,
                                                                                 W_out_to_logvar, b_out_to_logvar)
                
            with tf.variable_scope("target_output_to_z"):
                
                W_out_to_mu = tf.get_variable('output_to_mu_weight', shape = [hidden_size, latent_variable_size_for_output])
                b_out_to_mu = tf.get_variable('output_to_mu_bias', shape = [latent_variable_size_for_output])
                W_out_to_logvar = tf.get_variable('output_to_logvar_weight', shape = [hidden_size, latent_variable_size_for_output])
                b_out_to_logvar = tf.get_variable('output_to_logvar_bias', shape = [latent_variable_size_for_output])
                
                z_from_tgt, mu_tgt, logvar_tgt = outputs_to_z(encoder_outputs_tgt, W_out_to_mu, b_out_to_mu,
                                                                                 W_out_to_logvar, b_out_to_logvar)
                
            with tf.variable_scope("source_z_to_input", reuse = tf.AUTO_REUSE):
                
                W_z_to_input = tf.get_variable('z_to_input_weight', shape = [latent_variable_size_for_output, hidden_size])
                b_z_to_input = tf.get_variable('z_to_input_bias', shape = [hidden_size])
                
                inputs_src_from_src = z_to_inputs(z_from_src, W_z_to_input, b_z_to_input)
                inputs_src_from_tgt = z_to_inputs(z_from_tgt, W_z_to_input, b_z_to_input)
                
            with tf.variable_scope("target_z_to_input", reuse = tf.AUTO_REUSE):
                
                W_z_to_input = tf.get_variable('z_to_input_weight', shape = [latent_variable_size_for_output, hidden_size])
                b_z_to_input = tf.get_variable('z_to_input_bias', shape = [hidden_size])
                
                inputs_tgt_from_src = z_to_inputs(z_from_src, W_z_to_input, b_z_to_input)
                inputs_tgt_from_tgt = z_to_inputs(z_from_tgt, W_z_to_input, b_z_to_input)
               
            #======DECODER_PROCESS====================
            with tf.variable_scope("source_decoder"): 
                logits_src_from_src, dec_states_src_from_src, dec_length_src_from_src, attention_src_from_src\
                                        = decoder_src.decode(
                                          emb_src_batch_in_dropped, 
                                          src_length + 1,
                                          vocab_size = int(self.src_vocab_size),
                                          initial_state = None,
                                          output_layer = output_layer,                                              
                                          mode = tf.estimator.ModeKeys.TRAIN,
                                          memory = inputs_src_from_src,
                                          memory_sequence_length = encoder_seq_length_src,
                                          return_alignment_history = True)
            
            with tf.variable_scope("source_decoder", reuse = True): 
                logits_src_from_tgt, dec_states_src_from_tgt, dec_length_src_from_tgt, attention_src_from_tgt \
                                        = decoder_src.decode(
                                          emb_src_batch_in_dropped, 
                                          src_length + 1,
                                          vocab_size = int(self.src_vocab_size),
                                          initial_state = None,
                                          output_layer = output_layer,                                              
                                          mode = tf.estimator.ModeKeys.TRAIN,
                                          memory = inputs_src_from_tgt,
                                          memory_sequence_length = encoder_seq_length_tgt,
                                          return_alignment_history = True)
                    
            with tf.variable_scope("target_decoder"): 
                logits_tgt_from_tgt, dec_states_tgt_from_tgt, dec_length_tgt_from_tgt, attention_tgt_from_tgt \
                                        = decoder_tgt.decode(
                                          emb_tgt_batch_in_dropped, 
                                          tgt_length + 1,
                                          vocab_size = int(self.tgt_vocab_size),
                                          initial_state = None,
                                          output_layer = output_layer,                                              
                                          mode = tf.estimator.ModeKeys.TRAIN,
                                          memory = inputs_tgt_from_tgt,
                                          memory_sequence_length = encoder_seq_length_tgt,
                                          return_alignment_history = True)
                
            with tf.variable_scope("target_decoder", reuse = True): 
                logits_tgt_from_src, dec_states_tgt_from_src, dec_length_tgt_from_src, attention_tgt_from_src \
                                        = decoder_tgt.decode(
                                          emb_tgt_batch_in_dropped, 
                                          tgt_length + 1,
                                          vocab_size = int(self.tgt_vocab_size),
                                          initial_state = None,
                                          output_layer = output_layer,                                              
                                          mode = tf.estimator.ModeKeys.TRAIN,
                                          memory = inputs_tgt_from_src,
                                          memory_sequence_length = encoder_seq_length_src,
                                          return_alignment_history = True)
                
            outputs = {
                    "logits_src_from_src": logits_src_from_src,
                    "logits_src_from_tgt": logits_src_from_tgt,
                    "logits_tgt_from_tgt": logits_tgt_from_tgt,
                    "logits_tgt_from_src": logits_tgt_from_src
            }
                
            predictions = None

#-----------------------------------------------------INFERENCE MODE-----------------------------------------------#
        elif mode == "Inference":
            
            print("Build model in Inference mode")
            
            beam_width = config.get("beam_width", 5)
            
            src_length = inputs["src_length"]
            tgt_length = inputs["tgt_length"]
            
            emb_src_batch = tf.nn.embedding_lookup(src_emb, inputs["src_ids"])
            emb_tgt_batch = tf.nn.embedding_lookup(tgt_emb, inputs["tgt_ids"])
                        
            batch_size = tf.shape(inputs["src_ids"])[0]
            start_tokens = tf.fill([batch_size], constants.START_OF_SENTENCE_ID)
                                   
           #========ENCODER_PROCESS======================
            with tf.variable_scope("source_encoder", reuse=tf.AUTO_REUSE):
                encoder_outputs_src, encoder_states_src, encoder_seq_length_src = encoder_src.encode(
                    emb_src_batch, 
                    sequence_length = src_length, 
                    mode=tf.estimator.ModeKeys.TRAIN
                )

            with tf.variable_scope("target_encoder", reuse=tf.AUTO_REUSE):
                encoder_outputs_tgt, encoder_states_tgt, encoder_seq_length_tgt = encoder_tgt.encode(
                    emb_tgt_batch, 
                    sequence_length = tgt_length, 
                    mode=tf.estimator.ModeKeys.TRAIN
                )
                
            #======GENERATIVE_PROCESS====================
            with tf.variable_scope("source_output_to_z"):
                
                W_out_to_mu = tf.get_variable('output_to_mu_weight', shape = [hidden_size, latent_variable_size_for_output])
                b_out_to_mu = tf.get_variable('output_to_mu_bias', shape = [latent_variable_size_for_output])
                W_out_to_logvar = tf.get_variable('output_to_logvar_weight', shape = [hidden_size, latent_variable_size_for_output])
                b_out_to_logvar = tf.get_variable('output_to_logvar_bias', shape = [latent_variable_size_for_output])
                
                z_from_src, mu_src, logvar_src = outputs_to_z(encoder_outputs_src, W_out_to_mu, b_out_to_mu,
                                                                                 W_out_to_logvar, b_out_to_logvar)
                
            with tf.variable_scope("target_output_to_z"):
                
                W_out_to_mu = tf.get_variable('output_to_mu_weight', shape = [hidden_size, latent_variable_size_for_output])
                b_out_to_mu = tf.get_variable('output_to_mu_bias', shape = [latent_variable_size_for_output])
                W_out_to_logvar = tf.get_variable('output_to_logvar_weight', shape = [hidden_size, latent_variable_size_for_output])
                b_out_to_logvar = tf.get_variable('output_to_logvar_bias', shape = [latent_variable_size_for_output])
                
                z_from_tgt, mu_tgt, logvar_tgt = outputs_to_z(encoder_outputs_tgt, W_out_to_mu, b_out_to_mu,
                                                                                 W_out_to_logvar, b_out_to_logvar)
                
            with tf.variable_scope("source_z_to_input", reuse = tf.AUTO_REUSE):
                
                W_z_to_input = tf.get_variable('z_to_input_weight', shape = [latent_variable_size_for_output, hidden_size])
                b_z_to_input = tf.get_variable('z_to_input_bias', shape = [hidden_size])
                
                inputs_src_from_src = z_to_inputs(z_from_src, W_z_to_input, b_z_to_input)
                inputs_src_from_tgt = z_to_inputs(z_from_tgt, W_z_to_input, b_z_to_input)
                
            with tf.variable_scope("target_z_to_input", reuse = tf.AUTO_REUSE):
                
                W_z_to_input = tf.get_variable('z_to_input_weight', shape = [latent_variable_size_for_output, hidden_size])
                b_z_to_input = tf.get_variable('z_to_input_bias', shape = [hidden_size])
                
                inputs_tgt_from_src = z_to_inputs(z_from_src, W_z_to_input, b_z_to_input)
                inputs_tgt_from_tgt = z_to_inputs(z_from_tgt, W_z_to_input, b_z_to_input)
            
            #======DECODER_PROCESS====================
            print("Inference with beam width %d"%(beam_width))
            max_length = tf.shape(encoder_outputs_src)[1]
            maximum_iterations = config.get("maximum_iterations", tf.round(2 * max_length))

            if beam_width <= 1:  
                with tf.variable_scope("source_decoder"): 
                    sampled_ids_src_from_src, _, sampled_length_src_from_src, log_probs_src_from_src, alignment_src_from_src\
                            = decoder_src.dynamic_decode(
                                src_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.src_vocab_size),
                                initial_state=None,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_src_from_src,
                                memory_sequence_length=encoder_seq_length_src,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
                with tf.variable_scope("source_decoder", reuse = True): 
                    sampled_ids_src_from_tgt, _, sampled_length_src_from_tgt, log_probs_src_from_tgt, alignment_src_from_tgt\
                            = decoder_src.dynamic_decode(
                                src_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.src_vocab_size),
                                initial_state=None,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_src_from_tgt,
                                memory_sequence_length=encoder_seq_length_tgt,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
                with tf.variable_scope("target_decoder"): 
                    sampled_ids_tgt_from_src, _, sampled_length_tgt_from_src, log_probs_tgt_from_src, alignment_tgt_from_src\
                            = decoder_tgt.dynamic_decode(
                                tgt_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.tgt_vocab_size),
                                initial_state=None,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_tgt_from_src,
                                memory_sequence_length=encoder_seq_length_src,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
                with tf.variable_scope("target_decoder", reuse = True): 
                    sampled_ids_tgt_from_tgt, _, sampled_length_tgt_from_tgt, log_probs_tgt_from_tgt, alignment_tgt_from_tgt\
                            = decoder_tgt.dynamic_decode(
                                tgt_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.tgt_vocab_size),
                                initial_state=None,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_tgt_from_tgt,
                                memory_sequence_length=encoder_seq_length_tgt,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
            else:
                length_penalty = config.get("length_penalty", 0)
                
                with tf.variable_scope("source_decoder"): 
                    sampled_ids_src_from_src, _, sampled_length_src_from_src, log_probs_src_from_src, alignment_src_from_src\
                            = decoder_src.dynamic_decode_and_search(
                                src_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.src_vocab_size),
                                initial_state=None,
                                beam_width = beam_width,
                                length_penalty = length_penalty,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_src_from_src,
                                memory_sequence_length=encoder_seq_length_src,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
                
                with tf.variable_scope("source_decoder", reuse = True): 
                    sampled_ids_src_from_tgt, _, sampled_length_src_from_tgt, log_probs_src_from_tgt, alignment_src_from_tgt\
                            = decoder_src.dynamic_decode_and_search(
                                src_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.src_vocab_size),
                                initial_state=None,
                                beam_width = beam_width,
                                length_penalty = length_penalty,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_src_from_tgt,
                                memory_sequence_length=encoder_seq_length_tgt,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
                with tf.variable_scope("target_decoder"): 
                    sampled_ids_tgt_from_src, _, sampled_length_tgt_from_src, log_probs_tgt_from_src, alignment_tgt_from_src\
                            = decoder_tgt.dynamic_decode_and_search(
                                tgt_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.tgt_vocab_size),
                                initial_state=None,
                                beam_width = beam_width,
                                length_penalty = length_penalty,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_tgt_from_src,
                                memory_sequence_length=encoder_seq_length_src,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
                with tf.variable_scope("target_decoder", reuse = True): 
                    sampled_ids_tgt_from_tgt, _, sampled_length_tgt_from_tgt, log_probs_tgt_from_tgt, alignment_tgt_from_tgt\
                            = decoder_tgt.dynamic_decode_and_search(
                                tgt_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.tgt_vocab_size),
                                initial_state=None,
                                beam_width = beam_width,
                                length_penalty = length_penalty,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_tgt_from_tgt,
                                memory_sequence_length=encoder_seq_length_tgt,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )

            tokens_src_from_src = src_vocab_rev.lookup(tf.cast(sampled_ids_src_from_src, tf.int64))
            tokens_src_from_tgt = src_vocab_rev.lookup(tf.cast(sampled_ids_src_from_tgt, tf.int64))
            tokens_tgt_from_src = tgt_vocab_rev.lookup(tf.cast(sampled_ids_tgt_from_src, tf.int64))
            tokens_tgt_from_tgt = tgt_vocab_rev.lookup(tf.cast(sampled_ids_tgt_from_tgt, tf.int64))
            
            predictions = [
                {
              "tokens": tokens_src_from_src,
              "length": sampled_length_src_from_src,
              "log_probs": log_probs_src_from_src,
              "alignment": alignment_src_from_src,
                            },
                {
              "tokens": tokens_src_from_tgt,
              "length": sampled_length_src_from_tgt,
              "log_probs": log_probs_src_from_tgt,
              "alignment": alignment_src_from_tgt,
                            },
                {
              "tokens": tokens_tgt_from_src,
              "length": sampled_length_tgt_from_src,
              "log_probs": log_probs_tgt_from_src,
              "alignment": alignment_tgt_from_src,
                            },
                {
              "tokens": tokens_tgt_from_tgt,
              "length": sampled_length_tgt_from_tgt,
              "log_probs": log_probs_tgt_from_tgt,
              "alignment": alignment_tgt_from_tgt,
                            },
                ]
            outputs = None
            
            tgt_ids_out_batch = None
            src_ids_out_batch = None
            
            mu_src = None
            logvar_src = None
            
            mu_tgt = None
            logvar_tgt = None
            
            self.outputs = outputs
            
#-----------------------------------------------------GENERATION MODE-----------------------------------------------#
        elif mode == "Generation":
            
            print("Build model in Generation mode")
            
            gen_beam_width = config.get("beam_width_for_generation", 5)
            # gen_beam_width = 2
            
            max_length = config.get("max_length_for_generation", 20)
            min_length = config.get("min_length_for_generation", 10)
            gen_batch_size = config.get("batch_size_for_generation", 5)
            start_tokens = tf.fill([gen_batch_size], constants.START_OF_SENTENCE_ID)
            
            random_memory_sequence = tf.random_uniform([gen_batch_size], minval=min_length, maxval=max_length+1, 
                                                       dtype=tf.int64)
            max_memory_sequence = tf.fill([gen_batch_size], max_length)
            
            input_memory_sequence = random_memory_sequence
            
            gen_max_length = tf.reduce_max(input_memory_sequence)
                
            #======GENERATIVE_PROCESS====================
            
            z = tf.random_normal([gen_batch_size, gen_max_length, latent_variable_size_for_output])
            
            with tf.variable_scope("source_z_to_input"):
                
                W_z_to_input = tf.get_variable('z_to_input_weight', shape = [latent_variable_size_for_output, hidden_size])
                b_z_to_input = tf.get_variable('z_to_input_bias', shape = [hidden_size])
                
                inputs_src = z_to_inputs(z, W_z_to_input, b_z_to_input)
                
            with tf.variable_scope("target_z_to_input"):
                
                W_z_to_input = tf.get_variable('z_to_input_weight', shape = [latent_variable_size_for_output, hidden_size])
                b_z_to_input = tf.get_variable('z_to_input_bias', shape = [hidden_size])
                
                inputs_tgt = z_to_inputs(z, W_z_to_input, b_z_to_input)
            
            print(inputs_src)
            print(inputs_tgt)
            print(input_memory_sequence)
            
            #======DECODER_PROCESS====================
            print("Inference with beam width %d"%(gen_beam_width))
            
            maximum_iterations = config.get("maximum_iterations", tf.round(2 * gen_max_length))
            maximum_iterations = 80

            if gen_beam_width <= 1:  
                with tf.variable_scope("source_decoder"): 
                    sampled_ids_src, _, sampled_length_src, log_probs_src, alignment_src\
                            = decoder_src.dynamic_decode(
                                src_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.src_vocab_size),
                                initial_state=None,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_src,
                                memory_sequence_length=input_memory_sequence,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
                with tf.variable_scope("target_decoder"): 
                    sampled_ids_tgt, _, sampled_length_tgt, log_probs_tgt, alignment_tgt\
                            = decoder_tgt.dynamic_decode(
                                tgt_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.tgt_vocab_size),
                                initial_state=None,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_tgt,
                                memory_sequence_length=input_memory_sequence,
                                dtype=tf.float32,
                                return_alignment_history=True
                            )
            else:
                length_penalty = config.get("length_penalty", 0)
                                       
                with tf.variable_scope("target_decoder"): 
                    sampled_ids_tgt, _, sampled_length_tgt, log_probs_tgt\
                            = decoder_tgt.dynamic_decode_and_search(
                                tgt_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.tgt_vocab_size),
                                initial_state=None,
                                beam_width = gen_beam_width,
                                length_penalty = length_penalty,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_tgt,
                                memory_sequence_length=input_memory_sequence,
                                dtype=tf.float32,
                                return_alignment_history=False,
                            )
                    
                with tf.variable_scope("source_decoder"): 
                    sampled_ids_src, _, sampled_length_src, log_probs_src\
                            = decoder_src.dynamic_decode_and_search(
                                src_emb,
                                start_tokens,
                                end_token,
                                vocab_size=int(self.src_vocab_size),
                                initial_state=None,
                                beam_width = gen_beam_width,
                                length_penalty = length_penalty,
                                maximum_iterations=maximum_iterations,
                                output_layer = output_layer,
                                mode=tf.estimator.ModeKeys.PREDICT,
                                memory=inputs_src,
                                memory_sequence_length=input_memory_sequence,
                                dtype=tf.float32,
                                return_alignment_history=False
                            )

            tokens_src = src_vocab_rev.lookup(tf.cast(sampled_ids_src, tf.int64))
            tokens_tgt = tgt_vocab_rev.lookup(tf.cast(sampled_ids_tgt, tf.int64))
            alignment_src = None
            alignment_tgt = None
            
            predictions = [
                {
              "tokens": tokens_src,
              "length": sampled_length_src,
              "log_probs": log_probs_src,
              "alignment": alignment_src,
                            },
                {
              "tokens": tokens_tgt,
              "length": sampled_length_tgt,
              "log_probs": log_probs_tgt,
              "alignment": alignment_tgt,
                            },
                
                ]
            outputs = None
            
            tgt_ids_out_batch = None
            src_ids_out_batch = None
            src_length = None
            tgt_length = None
            
            mu_src = None
            logvar_src = None
            
            mu_tgt = None
            logvar_tgt = None
            
            self.outputs = outputs
        
        return outputs, predictions, src_ids_out_batch, src_length, tgt_ids_out_batch, tgt_length,\
                            mu_src, logvar_src, mu_tgt, logvar_tgt