import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import keras.backend as K


class masked_softmax(tf.keras.layers.Layer):
	def __init__(self, value=-1e6):
		super().__init__()
		self.value = value

	def call(self, X, valid_lens, **kwargs):
		"""
		Parameters:
		-----------
		X: 2D tensor specifying the attention matrix [query_seq_length, key_seq_length]
		"""
		if valid_lens == None:
			return tf.nn.softmax(X, axis=-1)
			print("Im here")
		else:
			shape_X = tf.shape(X)
			X = tf.reshape(X, shape=(-1, X.shape[-1]))
			reshape_X = tf.shape(X)
			maxlen = X.shape[1]
			mask = tf.range(start=0, limit=shape_X[-1], dtype=tf.float32)[None,:]
			mask = tf.broadcast_to(mask, shape=reshape_X)

			valid_lens = tf.repeat(valid_lens, repeats = shape_X[1])
			mask = mask < tf.cast(valid_lens[:, None], dtype=tf.float32)

			X = tf.where(mask, X, self.value)

			return tf.nn.softmax(tf.reshape(X, shape=shape_X), axis=-1)

class positionalEncoding(tf.keras.layers.Layer):  
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        # Create a long enough P
        self.P = np.zeros((1, max_len, num_hiddens))
        X = np.arange(max_len, dtype=np.float32).reshape(
            -1,1)/np.power(10000, np.arange(
            0, num_hiddens, 2, dtype=np.float32) / num_hiddens)
        self.P[:, :, 0::2] = np.sin(X)
        self.P[:, :, 1::2] = np.cos(X)

    def call(self, X, **kwargs):
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X, **kwargs)


class position_wise_embedding(tf.keras.layers.Layer):
	"""
	Defome position wise embedding for sequence input embedding

	Parameters:
	-----------
	kernel: the embedding matrix
	"""
	def __init__(self, output_dim):
		super().__init__()
		self.output_dim = output_dim

	def build(self, input_shape, **kwargs):
		self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], self.output_dim),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

	def call(self, input_data, **kwargs):
		output_embedding = tf.matmul(input_data, self.kernel)

		return output_embedding


class dotproductattention(tf.keras.layers.Layer):  #@save
	"""
	Define scaled dot product layer

	Parameters:
	-----------
	kernel_key: embedding matrix for key
	kernel_value: embedding matrix for value
	kernel_query: embedding matrix for query

	Returns:
	--------
	attention_score: the scale dot product score
	"""
	def __init__(self, output_dim):
		super().__init__()
		self.output_dim = output_dim
		self.masked_softmax = masked_softmax()

	def build(self, input_shape):
		self.kernel_key = self.add_weight(name = 'kernel_key', shape = (input_shape[-1], self.output_dim),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

		self.kernel_query  = self.add_weight(name = 'kernel_quary', shape = (input_shape[-1], self.output_dim),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

		self.kernel_value = self.add_weight(name='kernel_value', shape=(input_shape[-1], self.output_dim),
			initializer=tf.keras.initializers.he_normal(seed=None), trainable=True)

	def call(self, queries, keys, values, valid_lens=None, **kwargs):
		d = queries.shape[-1]
		queries = tf.matmul(queries, self.kernel_query)
		keys = tf.matmul(keys, self.kernel_key)
		values = tf.matmul(values, self.kernel_value)

		scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
		tf.cast(d, dtype=tf.float32))

		self.attention_weights = self.masked_softmax(scores, valid_lens)
		return self.attention_weights, values

class attention_embedding(tf.keras.layers.Layer):
	"""
	Define attention embedding layer, perform dummy
	attention multiplication with value input

	Parameters:
	-----------
	att_weights: the attention score matrix
	input_value: input value matrix

	Returns:
	--------
	the attention embedding matrix
	"""
	def __init__(self):
		super().__init__()

	def call(self, att_weights, input_value, **kwargs):

		return tf.matmul(att_weights, input_value)


class residual_connection(tf.keras.layers.Layer):
	"""
	Define reidual connection layer
	"""
	def __init__(self):
		super().__init__()

	def call(self, X, Y, **kwargs):
		X = tf.math.l2_normalization(X, axis=-1)
		Y = tf.math.l2_normalization(Y, axis=-1)
		return tf.math.add(X,Y)


class feed_forward_layer(tf.keras.layers.Layer):
	"""
	Define a dummy feed forward layer for intigrating 
	gene expression data

	Parameters:
	-----------
	kernel: the trainable kernel matrix

	Returns:
	--------
	the embedding vector
	"""
	def __init__(self, output_dim):
		super().__init__()
		self.output_dim = output_dim

	def build(self, input_shape, **kwargs):
		self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], self.output_dim),
		initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

	def call(self, input_data, **kwargs):
		output_embedding = tf.matmul(input_data, self.kernel)


class concatenation_layer(tf.keras.layers.Layer):
	"""
	Define concatenation layer
	"""
	def __init__(self):
		super().__init__()

	def call(self, X, Y, **kwargs):
		X = tf.math.l2_normalization(X, axis=-1)
		Y = tf.math.l2_normalization(Y, axis=-1)

		return tf.concat([X,Y],axis=1)


class drug_transformer():
	"""
	Implement the drug transformer model architecture
	"""
	def __init__(self, num_hiddens, num_hiddens_fc):

		self.masked_softmax = masked_softmax()
		self.position_wise_embedding = position_wise_embedding(100)
		self.dotproductattention = dotproductattention(100)
		self.attention_embedding = attention_embedding()
		self.residual_connection = residual_connection()
		self.feed_forward_layer = feed_forward_layer(100)
		self.flattern = tf.keras.layers.Flatten()
		self.projection = tf.keras.layers.Dense(1)
		self.concatenation_layer = concatenation_layer()

		self.feed_forward_encoder_layer = feed_forward_layer(500)

	def model_construction(self, doc_valid_lens=None):
		"""
		construct the transformer model
		"""
		X_input = Input((130, 56))
		Y_input = Input((5842))
		enc_valid_lens = Input(())

		X = self.position_wise_embedding(X_input)

		att_score, value = self.dotproductattention(X, X, X)

		att_embedding = self.attention_embedding(att_score, value)

		encoder_embedding = self.residual_connection(att_embedding, value)

		encoder_flattern = self.flattern(encoder_embedding)

		encoder_flattern_ = self.feed_forward_encoder_layer(encoder_flattern)

		decoder_embedding = self.feed_forward_layer(Y_input)

		final_embedding = self.concatenation_layer(encoder_flattern_, decoder_embedding)

		prediction = self.projection(final_embedding)


		self.model = Model(inputs=(X_input, Y_input, enc_valid_lens), outputs=prediction)

	#return model

	def model_compile(self):
		"""
		model compiling for training
		"""
		self.model.compile(loss= "mean_squared_error" , 
			optimizer="adam", metrics=["mean_squared_error"])












