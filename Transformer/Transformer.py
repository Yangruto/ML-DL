import numpy as np
import tensorflow as tf
from typing import Union
from tensorflow.keras import layers
import pickle
import joblib
from tqdm import tqdm
import re
import pandas as pd
from ckiptagger import WS

NUM_LAYERS = 6
D_MODEL = 512
DFF = 2048
NUM_HEAD = 8
SENTENCE_LENGTH = 100
EPOCH = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 32

def create_padding_mask(input_sentence:tf.Tensor) -> tf.Tensor:
    mask = tf.cast(tf.math.equal(input_sentence, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :] # (batch_size, 1, 1, seq_len)

def create_look_ahead_mask(sentence_length:int) -> tf.Tensor:
    """
    Create a mask to blind the future data.
        target_tensor: the tensor would be blinded by the mask
    """
    mask = 1 - tf.linalg.band_part(tf.ones((sentence_length, sentence_length)), -1, 0) # create a matrix its upper triangle values are 1 (input, num_lower, num_upper)
    return mask  # (seq_len, seq_len)

class PositionLayer(tf.keras.layers.Layer):
    """
    Build a position layer to embed the text position.
        position: the maximum sentence length
        d_model: the dimension of hidden layer
    """
    def __init__(self, position:int, d_model:int) -> object:
        super(PositionLayer, self).__init__()
        self.d_model = d_model
        self.position = position

    def encoding(self, inputs:tf.Tensor) -> tf.Tensor:
        """
        Encode the inputs according to their positions and dimensions. 
            inputs: the inputs for position encoding
        """
        position = np.arange(self.position)
        position = np.reshape(position, [1, -1, 1])
        position = np.tile(position, [inputs.shape[0], 1, self.d_model])
        for i in range(0, int(self.d_model), 2):
            position[:, :, i] = np.sin(position[:, :, i] / 10000 ** (i / self.d_model))
            position[:, :, i + 1] = np.cos(position[:, :, i + 1] / 10000 ** (i / self.d_model))                                                              
        return tf.cast(position, dtype=tf.float32)
    
    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        """
        Execute position layer encoding.
            inputs: the inputs for position encoding
        """
        position_encoding = self.encoding(inputs)
        return inputs + position_encoding

class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    """
    Build a multi-head attention layer consists of 4 linear transformations: Q, K, V, O.\ 
    Split data into n heads and go through Q, K, V transformations and do scaled dot-product attention then concat.\ 
    Finally, transform by O linear transformation.
        head_size: how many sub head you want to split
        d_model: the dimension of hidden layer
        masked: weather to mask the future data 
    """
    def __init__(self, head_size:int, d_model:int) -> object:
        super(MultiHeadAttentionLayer, self).__init__()
        self.head_size = head_size
        self.d_model = d_model
        self.depth = tf.cast(d_model / head_size, tf.int32) # the new dimension for multihead attention
        self.wQ = layers.Dense(self.d_model)
        self.wV = layers.Dense(self.d_model)
        self.wK = layers.Dense(self.d_model)
        self.wO = layers.Dense(self.d_model)

    def split_input(self, inputs:tf.Tensor) -> tf.Tensor:
        """
        Split the dimension of input to many pieces according to the head_size.
            inputs: the inputs for multi-head attention layer
        """
        splited_input = tf.reshape(inputs, (inputs.shape[0], -1, self.head_size, self.depth))
        return tf.transpose(splited_input, perm=[0, 2, 1, 3]) # (batch_size, num_heads, seq_len_q, depth)

    def multi_head_self_attention(self, q:tf.Tensor, k:tf.Tensor, v:tf.Tensor, mask:tf.Tensor=None) -> Union[tf.Tensor, tf.Tensor]:
        """
        Calculate the multi-head self attention values.
            q: query token
            k: key token 
            v: value token 
        """
        Q = self.split_input(self.wQ(q)) # (batch_size, num_heads, seq_len_q, depth)
        K = self.split_input(self.wK(k))
        V = self.split_input(self.wV(v))

        K_dim = tf.cast(K.shape[3], tf.float32)
    
        attention = tf.matmul(Q, K, transpose_b=True) / tf.sqrt(K_dim) # (batch_size, num_heads, seq_len_q, seq_len_k)
        if mask is not None:
            attention += (mask * -1e9)
        
        attention_weights = tf.nn.softmax(attention, axis=-1)
        attention = tf.matmul(attention_weights, V) # (batch_size, num_heads, seq_len_attention, depth)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3]) # (batch_size, seq_len_attention, num_heads, depth)
        concat_attention = tf.reshape(attention, (q.shape[0], -1, self.d_model)) # (batch_size, seq_len_q, d_model)
        multi_head_attention_value = self.wO(concat_attention)
        return multi_head_attention_value, attention_weights

    def call(self, q:tf.Tensor, k:tf.Tensor, v:tf.Tensor, mask:tf.Tensor) -> tf.Tensor:
        """
        Execute the multi-head self attention layer.
            q: query token
            k: key token
            v: value token
        """
        mutli_haed_attention_value = self.multi_head_self_attention(q, k, v, mask)
        return mutli_haed_attention_value

class FeedForwardLayer(tf.keras.layers.Layer):
    """
    Build a feed-forward layer.\ 
    Feed-forward layer is a layer consists of two linear transformations with a ReLU activation.
        d_model: the dimension of layer output
        dff: the dimension of hidden layer
    """
    def __init__(self, d_model:int, dff:int) -> object:
        super(FeedForwardLayer, self).__init__()
        self.d_model = d_model
        self.dff = dff
        self.dense_1 = layers.Dense(self.dff, activation='relu')
        self.dense_2 = layers.Dense(self.d_model)

    def call(self, inputs:tf.Tensor) -> tf.Tensor:
        """
        Execute the feed-forward layer.
            inputs: the input of feed-forward layer
        """
        x = self.dense_1(inputs)
        x = self.dense_2(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    """
    A Encoder layer consists of one multi-head attention layer and one feed-forward layer, both end with normalization layers. 
        head_size: how many sub heads you want to split
        d_model: the dimension of hidden layer
        dff: the dimension of layer output
    """
    def __init__(self, head_size:int, d_model:int, dff:int) -> object:
        super(EncoderLayer, self).__init__()

        self.multi_head = MultiHeadAttentionLayer(head_size, d_model)
        self.feed_forward = FeedForwardLayer(d_model, dff)

        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)

        self.drop_out_1 = layers.Dropout(0.1)
        self.drop_out_2 = layers.Dropout(0.1)

    def call(self, inputs:tf.Tensor, padding_mask:tf.Tensor, training:bool) -> tf.Tensor:
        """
        Execute the encoder layer.
            inputs: the inputs of encoder layer
            training: this execute will update weights or not
        """
        multi_head_attention_value, _ = self.multi_head(inputs, inputs, inputs, padding_mask)
        multi_head_attention_value = self.drop_out_1(multi_head_attention_value, training=training)
        multi_head_attention_result = self.layer_norm_1(multi_head_attention_value + inputs)

        feed_forward_value = self.feed_forward(multi_head_attention_result)
        feed_forward_value = self.drop_out_2(feed_forward_value, training=training)
        feed_forward_result = self.layer_norm_2(feed_forward_value + multi_head_attention_result)
        return feed_forward_result

class DecoderLayer(tf.keras.layers.Layer):
    """
    A Decoder layer consists of one masked multi-head attention layer, one multi-head attention layer and one feed-forward layer, all end with normalization layers. 
        head_size: how many sub head you want to split
        d_model: the dimension of hidden layer
        dff: the dimension of layer output
    """
    def __init__(self, head_size:int, d_model:int, dff:int) -> object:
        super(DecoderLayer, self).__init__()
        
        self.multi_head_1 = MultiHeadAttentionLayer(head_size, d_model)
        self.multi_head_2 = MultiHeadAttentionLayer(head_size, d_model)
        self.feed_forward = FeedForwardLayer(d_model, dff)

        self.layer_norm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.layer_norm_3 = layers.LayerNormalization(epsilon=1e-6)

        self.drop_out_1 = layers.Dropout(0.1)
        self.drop_out_2 = layers.Dropout(0.1)
        self.drop_out_3 = layers.Dropout(0.1)

    def call(self, inputs:tf.Tensor, encoder_output:tf.Tensor, padding_mask:tf.Tensor, lookahead_mask:tf.Tensor, training:bool) -> Union[tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Execute the decoder layer.
            inputs: the inputs of decoder layer
            encoder_output: the outputs of encoder 
            training: the execute will update wieghts or not
        """
        multi_head_attention_value_1, attention_weights_1 = self.multi_head_1(inputs, inputs, inputs, padding_mask)
        multi_head_attention_value_1 = self.drop_out_1(multi_head_attention_value_1, training=training)
        multi_head_attention_result_1 = self.layer_norm_1(multi_head_attention_value_1 + inputs)

        multi_head_attention_value_2, attention_weights_2 = self.multi_head_2(multi_head_attention_result_1, encoder_output, encoder_output, lookahead_mask) # q is from the previous decoder layer output, k & v are from encoder output
        multi_head_attention_value_2 = self.drop_out_2(multi_head_attention_value_2, training=training)
        multi_head_attention_result_2 = self.layer_norm_2(multi_head_attention_value_2 + multi_head_attention_result_1)

        feed_forward_value = self.feed_forward(multi_head_attention_result_2)
        feed_forward_value = self.drop_out_3(feed_forward_value, training=training)
        feed_forward_result = self.layer_norm_3(feed_forward_value + multi_head_attention_result_2)
        return feed_forward_result, attention_weights_1, attention_weights_2

class Encoder(tf.keras.layers.Layer):
    """
    Build a Encoder with N encoder layers.
        num_layer: how many encoder layers would be in the encoder
        head_size: how many sub heads you want to split
        d_model: the dimension of model output
        dff: the dimension of hidden layer
        vocab_size: the number of vocabularies
        sequence_length: the length of sequence
    """
    def __init__(self, num_layer:int, head_size:int, d_model:int, dff:int, vocab_size:int, sequence_length:int) -> object:
        super(Encoder, self).__init__()
        
        self.num_layer = num_layer
        self.head_size = head_size
        self.d_model = d_model
        self.dff = dff
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.embedding = layers.Embedding(self.vocab_size + 1 , self.d_model, input_length=self.sequence_length)
        self.encoder_layer = [EncoderLayer(self.head_size, self.d_model, dff) for _ in range(self.num_layer)]
        self.position_layer = PositionLayer(self.sequence_length, self.d_model)
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs:tf.Tensor, padding_mask:tf.Tensor, training:bool) -> tf.Tensor:
        """
        Execute a encoder.
            inputs: the inputs of encoder
            training: this execute will update wieghts or not
        """
        x = self.embedding(inputs)
        x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.position_layer(x)
        x = self.dropout(x, training=training)
        for i in range(self.num_layer):
            x = self.encoder_layer[i](x, padding_mask, training=training)
        return x

class Decoder(tf.keras.layers.Layer):
    """
    Build a decoder.
        num_layer: how many encoder layers would be in the encoder
        head_size: how many sub heads you want to split
        d_model: the dimension of model output
        dff: the dimension of hidden layer
        vocab_size: the number of vocabularies
        sequence_length: the length of sequence
    """
    def __init__(self, num_layer:int, head_size:int, d_model:int, dff:int, vocab_size:int, sequence_length:int) -> object:
        super(Decoder, self).__init__()
        
        self.num_layer = num_layer
        self.head_size = head_size
        self.d_model = d_model
        self.dff = dff
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.embedding = layers.Embedding(self.vocab_size + 1 , self.d_model, input_length=self.sequence_length)
        self.decoder_layer = [DecoderLayer(self.head_size, self.d_model, self.dff) for _ in range(self.num_layer)]
        self.position_layer = PositionLayer(self.sequence_length, self.d_model)
        self.dropout = layers.Dropout(0.1)

    def call(self, inputs:int, encoder_output:int, padding_mask:tf.Tensor, lookahead_mask:tf.Tensor, training:int) -> Union[tf.Tensor, tf.Tensor]:
        """
        Execute a decoder with N decoder layers.
            inputs: the inputs of decoder
            encoder_output: the outputs of encoder 
            training: this execute will update wieghts or not
        """
        x = self.embedding(inputs)
        x *= tf.sqrt(tf.cast(self.d_model, tf.float32))
        x = self.position_layer(x)
        x = self.dropout(x, training=training)

        attention_weight = {}

        for i in range(self.num_layer):
            x, attention_weight_1, attention_weight_2 = self.decoder_layer[i](x, encoder_output, padding_mask, lookahead_mask, training)
            attention_weight[f'layer_{i + 1}_attention_1'] = attention_weight_1
            attention_weight[f'layer_{i + 1}_attention_2'] = attention_weight_2
        return x, attention_weight

class Transformer(tf.keras.Model):
    """
    Build a transformer model.
        num_layer: how many encoder and decoder layers would be in the encoder and decoder
        head_size: how many sub heads you want to split
        d_model: the dimension of model output
        dff: the dimension of hidden layer
        vocab_size: the number of vocabularies
        sequence_length: the length of sequence
    """
    def __init__(self, num_layer:int, head_size:int, d_model:int, dff:int, vocab_size:int, sequence_length:int) -> object:
        super().__init__()

        self.num_layer = num_layer
        self.head_size = head_size
        self.d_model = d_model
        self.dff = dff
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        self.tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            ngrams=None,
            output_mode='int',
            output_sequence_length=self.sequence_length,
            pad_to_max_tokens=None,
            vocabulary=None)
        self.encoder = Encoder(self.num_layer, self.head_size, self.d_model, self.dff, self.vocab_size, self.sequence_length)
        self.decoder = Decoder(self.num_layer, self.head_size, self.d_model, self.dff, self.vocab_size, self.sequence_length)
        self.dense = layers.Dense(self.vocab_size, activation='softmax')
        
        self.checkpoint = self.checkpoint(self)
    
    def set_dataset(self, question:tf.Tensor, ground_truth:tf.Tensor, decoder_input:tf.Tensor, batch_size:int) -> None:
        """
        Set the training dataset of the transformer.
        """
        self.batch_size = batch_size
        self.question = self.tokenizer(question)
        self.ground_truth = self.tokenizer(ground_truth)
        self.decoder_input = self.tokenizer(decoder_input)

        self.dataset = tf.data.Dataset.from_tensor_slices(tf.reshape(tf.concat([self.question, self.decoder_input, self.ground_truth], 1), (-1, 3, self.sequence_length)))
        self.dataset = self.dataset.batch(self.batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        self.dataset = self.dataset.shuffle(len(self.dataset), reshuffle_each_iteration=True)

    def build_tokenizer(self, words:tf.Tensor) -> tf.keras.layers.TextVectorization:
        """
        Build a tokenizer.
        """
        self.tokenizer = tf.keras.layers.TextVectorization(
            max_tokens=self.vocab_size,
            standardize='lower_and_strip_punctuation',
            split='whitespace',
            ngrams=None,
            output_mode='int',
            output_sequence_length=self.sequence_length,
            pad_to_max_tokens=None,
            vocabulary=None)
        self.tokenizer.adapt(words)
        self.vocab = self.tokenizer.get_vocabulary()

    def save_tokenizer(self, path:str) -> None:
        """
        Save the tokenizer to a specific path.
            path: tokenizer destination path
        """
        pickle.dump({'config': self.tokenizer.get_config(),
            'weights': self.tokenizer.get_weights()},
            open(path, 'wb'))

    def load_tokenizer(self, path:str) -> None:
        """
        Load the tokenizer.
            path: tokenizer destination path
        """
        tokenizer_pickle = joblib.load(path)
        self.tokenizer = tf.keras.layers.TextVectorization.from_config(tokenizer_pickle['config'])
        self.tokenizer.set_weights(tokenizer_pickle['weights'])
        self.vocab = self.tokenizer.get_vocabulary()

    def initiate_optimizer(self, learning_rate:float) -> None:
        """
        Initiate the optimizer with the input learning rate.
            learning_rate: the learning rate of the optimizer
        """
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

    def initiate_loss_function(self):
        """
        Initiare the loss function of the transformer model.
        """
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    def create_masks(self, input, target):
        padding_mask = create_padding_mask(input)
        look_ahead_mask = create_look_ahead_mask(self.sequence_length)
        dec_target_padding_mask = create_padding_mask(target)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return padding_mask, look_ahead_mask

    def call(self, inputs_x:tf.Tensor, inputs_y:tf.Tensor, training:bool) -> Union[tf.Tensor, tf.Tensor]:
        """
        Execute a transformer model.
            inputs_x: the training data 
            inputs_y: the target data
            training: this execute will update wieghts or not
        """
        padding_mask, look_ahead_mask = self.create_masks(inputs_x, inputs_y)
        x = self.encoder(inputs_x, padding_mask, training)
        x, attention_weight = self.decoder(inputs_y, x, padding_mask, look_ahead_mask, training)
        x = self.dense(x) # (batch_size, tar_seq_len, target_vocab_size)
        return x, attention_weight
        
    def predict(self, sentence:str) -> Union[tf.Tensor, tf.Tensor]:
        """
        Predict the result by the sentence. (one sample at a time) 
            sentence: the sentence you want to predict
        """
        end = self.vocab.index('eos')

        decoder_input = tf.constant(['sos'])
        decoder_input = self.tokenizer(decoder_input)

        sentence = [' '.join(i) for i in ws(sentence)]
        sentence = self.tokenizer(sentence)
        
        for i in range(self.sequence_length):
            pred, _ = self.call(sentence, decoder_input, False)
            pred_id = pred[:, i, tf.argmax(pred[:, i, :], axis=2)]
            if pred_id == end:
                break
            decoder_input[0, i + 1] = pred_id
        pred = [self.vocab[i] for i in pred]
        decoder_input = [self.vocab[i] for i in decoder_input]
        return pred, decoder_input

    def batch_loss(self, pred:tf.Tensor, target:tf.Tensor) -> tf.Tensor:
        """
        Calculate the model loss batchly. (Only keep the loss of ground gruth words)
            pred: the result of model predict
            target: the ground truth
        """
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        loss_ = self.loss_function(target, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask
        return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

    def accuracy_function(self, pred:tf.Tensor, target:tf.Tensor) -> tf.Tensor:
        """
        Calculate the model accuracy.
            pred: the result of model predict
            target: the ground truth
        """
        mask = tf.math.logical_not(tf.math.equal(target, 0))
        accuracy = tf.equal(target, tf.argmax(pred, axis=2))
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=accuracy.dtype)
        accuracy *= mask
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    
    @tf.function()
    def fit(self, inputs_x:tf.Tensor, decoder_input:tf.Tensor, ground_truth:tf.Tensor) -> None:
        """
        Train a transformer model.
            inputs_x: the training data
            decoder_input: the sentences with sos ahead
            ground_truth: the sentences end with eos
        """
        with tf.GradientTape() as tape:
            prediction, attention_weight = self(inputs_x, decoder_input, training=True)
            loss = self.batch_loss(prediction, ground_truth)
        gradient = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradient, self.trainable_variables))
        self.total_loss(loss)
        self.total_accuracy(self.accuracy_function(prediction, ground_truth))
    
    def train(self, EPOCH):
        self.total_loss = tf.keras.metrics.Mean(name='train_loss')
        self.total_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
        for i in range(1, EPOCH + 1):
            self.total_loss.reset_state()
            self.total_accuracy.reset_state()
            for k in tqdm(self.dataset):
                self.fit(k[:, 0], k[:, 1], k[:, 2])
            print(f'Step: {i}, Accuracy: {self.total_accuracy.result():.4f}, Loss: {self.total_loss.result():.4f}')
            self.checkpoint.save()
        
    class checkpoint:
        def __init__(self, model:object) -> None:
            """
            Create checkpoint for transformer model.
                model: the model checkpoint tracking
            """
            self.model = model
            self.__create()
            self.ckpt_path = './'
        
        def set_path(self, ckpt_path:str) -> None:
            """
            Set checkpoint saving path.
                ckpt_path: checkpoint path
            """
            self.ckpt_path = ckpt_path

        def __create(self) -> None:
            """
            Create checkpoint.
            """
            checkpoint = tf.train.Checkpoint(
                transformer = self.model)
            self.checkpoint = checkpoint

        def save(self) -> None:
            """
            Save checkpoint.
            """
            self.__create()
            self.checkpoint.save(self.ckpt_path)

        def load(self) -> None:
            """
            Load checkpoint.
            """
            self.checkpoint.restore(tf.train.latest_checkpoint(self.ckpt_path))

def data_prepare(data_path:str) -> Union[tf.Tensor, tf.Tensor, tf.Tensor]:
    """
    Read QA training data, then convert them into tensors.
    There are four outputs: decoder_input, question, ground_truth, words (for tokenizer)
        data_path: training data path
    """
    training_data = pd.read_csv(data_path)
    question = training_data.question
    answer = training_data.answer
    question = tf.constant(question)
    answer = tf.constant(answer)
    decoder_input = tf.constant(['sos ']) + answer
    ground_truth = answer + tf.constant([' eos']) 
    answer = tf.constant(['sos ']) + answer + tf.constant([' eos'])
    words, _ = tf.unique(tf.concat([tf.reshape(question, [-1, ]), tf.reshape(answer, [-1, ])], axis=0))
    return decoder_input, question, ground_truth, words

if __name__ == "__main__":
    # data preparing
    decoder_input, question, ground_truth, words = data_prepare('./subtitle.csv')
    # model preparing
    transformer = Transformer(num_layer=NUM_LAYERS, head_size=NUM_HEAD, d_model=D_MODEL, dff=DFF, vocab_size=words.shape[0], sequence_length=SENTENCE_LENGTH)
    # tokenizer    
    transformer.build_tokenizer(words)
    transformer.save_tokenizer('./tokenizer.pkl')
    # optimizer
    transformer.initiate_optimizer(LEARNING_RATE)
    # loss function
    transformer.initiate_loss_function()
    # data setting
    transformer.set_dataset(question, ground_truth, decoder_input, BATCH_SIZE)
    # checkpoint setting
    transformer.checkpoint.set_path('./')
    # training
    transformer.train(EPOCH)

# # Retrain
# decoder_input, question, ground_truth, words = data_prepare('./subtitle.csv')
# transformer = Transformer(num_layer=NUM_LAYERS, head_size=NUM_HEAD, d_model=D_MODEL, dff=DFF, vocab_size=words.shape[0], sequence_length=SENTENCE_LENGTH)
# transformer.load_tokenizer('./tokenizer.pkl')
# transformer.initiate_optimizer(LEARNING_RATE)
# transformer.initiate_loss_function()
# transformer.set_dataset(question, ground_truth, decoder_input, BATCH_SIZE)
# transformer.checkpoint.set_path('./')
# transformer.checkpoint.load()
# transformer.train(EPOCH)

# predict
# transformer.predict('嗨，你好')

"""
Target
sentence: SOS A lion in the jungle is sleeping EOS
decoder_inputs = SOS A lion in the jungle is sleeping (input of decoder) 
ground_truth = A lion in the jungle is sleeping EOS (the next token should be predicted)

During training: Teacher forcing is passing the true output to the next time step regardless of what the model predicts at the current time step.
"""