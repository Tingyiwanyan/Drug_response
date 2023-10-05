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

	def call(self, X, valid_lens=None, **kwargs):
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

class positionalencoding(tf.keras.layers.Layer):  
	"""Positional encoding."""
	def __init__(self, num_hiddens, num_length, max_len=1000):
		super().__init__()
		#self.dropout = tf.keras.layers.Dropout(dropout)
		# Create a long enough P
		self.num_length = num_length
		self.max_len = max_len
		self.num_hiddens = num_hiddens
	def call(self, X, **kwargs):
		#X = X + self.P[:, :X.shape[1], :]
		#return self.dropout(X, **kwargs)
		self.P = np.zeros((1, self.max_len, self.num_hiddens))
		XX = np.arange(self.max_len, dtype=np.float32).reshape(
			-1,1)/np.power(10000, np.arange(
				0, self.num_hiddens, 2, dtype=np.float32) / self.num_hiddens)
		self.P[:, :, 0::2] = np.sin(XX)
		self.P[:, :, 1::2] = np.cos(XX)
		shape_X = tf.shape(X)
		X = tf.math.l2_normalize(X, axis=-1)
		self.P = tf.cast(tf.math.l2_normalize(self.P[:, :self.num_length,:], axis=-1), 
			dtype=tf.float32)

		return tf.cast(tf.math.l2_normalize(tf.math.add(X, self.P), axis=-1), dtype=tf.float32)


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
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(
			initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

	def call(self, input_data, **kwargs):
		output_embedding = tf.keras.activations.relu(tf.matmul(input_data, self.kernel) + self.b)

		return tf.cast(tf.math.l2_normalize(output_embedding, axis=-1), dtype=tf.float32)


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
		#self.masked_softmax = masked_softmax()

		self.kernel_key = tf.keras.layers.Dense(output_dim, activation='sigmoid', 
			kernel_regularizer=regularizers.L2(1e-4))

		self.kernel_query = tf.keras.layers.Dense(output_dim, activation='sigmoid', 
			kernel_regularizer=regularizers.L2(1e-4))

		self.kernel_value = tf.keras.layers.Dense(output_dim, activation='relu', 
			kernel_regularizer=regularizers.L2(1e-4))

	"""
	def build(self, input_shape):
		self.kernel_key = self.add_weight(name = 'kernel_key', shape = (input_shape[-1], self.output_dim),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

		b_init = tf.zeros_initializer()
		self.b_key = tf.Variable(
			initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

		self.kernel_query  = self.add_weight(name = 'kernel_quary', shape = (input_shape[-1], self.output_dim),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

		self.b_query = tf.Variable(
			initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

		#self.kernel_value = self.add_weight(name='kernel_value', shape=(input_shape[-1], self.output_dim),
		#	initializer=tf.keras.initializers.he_normal(seed=None), trainable=True)

		#self.b_value = tf.Variable(
		#	initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)
	"""

	def call(self, queries, keys, values, valid_lens=None, **kwargs):
		d = queries.shape[-1]
		#queries = tf.matmul(queries, self.kernel_query) + self.b_query
		queries = self.kernel_query(queries)
		#keys = tf.matmul(keys, self.kernel_key) + self.b_key
		keys = self.kernel_key(keys)
		#values = tf.matmul(values, self.kernel_value) + self.b_value
		values = self.kernel_value(values)

		scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
			tf.cast(d, dtype=tf.float32))

		#self.attention_weights = self.masked_softmax(scores, valid_lens)
		return scores, values

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

		return tf.cast(tf.math.l2_normalize(tf.matmul(att_weights, input_value), axis=-1), dtype=tf.float32)


class residual_connection(tf.keras.layers.Layer):
	"""
	Define reidual connection layer
	"""
	def __init__(self):  
		super().__init__()

	def call(self, X, Y, **kwargs):
		#X = tf.math.l2_normalize(X, axis=-1)
		#Y = tf.math.l2_normalize(Y, axis=-1)
		return tf.cast(tf.math.l2_normalize(tf.math.add(X,Y), axis=-1), dtype=tf.float32)


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
		self.dropout = tf.keras.layers.Dropout(0.1)

	def build(self, input_shape, **kwargs):
		self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], self.output_dim),
		initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)
		b_init = tf.zeros_initializer()
		self.b = tf.Variable(
			initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

	def call(self, input_data, **kwargs):
		output_embedding = tf.keras.activations.relu(tf.matmul(input_data, self.kernel) + self.b)

		#return self.dropout(tf.cast(tf.math.l2_normalize(output_embedding, axis=-1), dtype=tf.float32),**kwargs)
		return tf.cast(tf.math.l2_normalize(output_embedding, axis=-1), dtype=tf.float32)


class concatenation_layer(tf.keras.layers.Layer):
	"""
	Define concatenation layer
	"""
	def __init__(self):
		super().__init__()

	def call(self, X, Y, **kwargs):
		X = tf.math.l2_normalize(X, axis=-1)
		Y = tf.math.l2_normalize(Y, axis=-1)

		return tf.cast(tf.math.l2_normalize(tf.concat([X,Y],axis=1), axis=-1), dtype=tf.float32)


class encoder_block(tf.keras.layers.Layer):
	"""
	Define self attention encoder block, add the position encoding
	since drug smile sequence has the order information

	Parameters:
	-----------
	num_hiddens: the hidden dimension for embedding matrix
	seq_length: the length for input sequence

	Returns:
	-------
	encoder_embedding: the encoder embedding output
	att_score: the self attention score
	"""
	def __init__(self, num_hiddens, seq_length):
		super().__init__()
		self.masked_softmax = masked_softmax()
		self.pos_encoding = positionalencoding(num_hiddens,seq_length)
		self.position_wise_embedding = position_wise_embedding(num_hiddens)
		self.dotproductattention = dotproductattention(num_hiddens)
		self.attention_embedding = attention_embedding()
		self.residual_connection = residual_connection()

	def call(self, X, enc_valid_lens, **kwargs):
		X = self.position_wise_embedding(X)
		X = self.pos_encoding(X)
		score, value = self.dotproductattention(X, X, X)
		att_score = self.masked_softmax(score, enc_valid_lens)
		att_embedding = self.attention_embedding(att_score, value)
		encoder_embedding = self.residual_connection(att_embedding, value)

		return encoder_embedding, att_score


class decoder_block(tf.keras.layers.Layer):
	"""
	Define self-attention & cross attention decoder block, since 
	gene expression doesn't contain sequencing information, so 
	we don't add position encoding in the embedding layer. 

	Parameters:
	-----------
	num_hiddens_self: hidden embedding dimension for the self att block
	num_hiddens_cross: hidden embedding dimension for the cross att block

	Returns:
	--------
	cross_decoder_embedding: the decoder embedding output
	self_att_score: the self att score in decoder self att block
	cross_att_score: the cross att score in the decoder cross att block
	"""
	def __init__(self, num_hiddens, num_hiddens_output):
		super().__init__()
		self.masked_softmax = masked_softmax()
		self.position_wise_embedding = position_wise_embedding(num_hiddens)
		self.self_dotproductattention = dotproductattention(num_hiddens)
		self.self_att_embedding = attention_embedding()
		self.self_residual_connection = residual_connection()

		self.cross_att_dotproduct = dotproductattention(num_hiddens)
		self.cross_att_embedding = attention_embedding()
		self.cross_residual_connection = residual_connection()
		self.cross_position_wise_embedding = position_wise_embedding(num_hiddens_output)

	def call(self, X, encoder_output, enc_valid_lens, **kwargs):
		X = self.position_wise_embedding(X)
		score, value = self.self_dotproductattention(X, X, X)
		self_att_score = self.masked_softmax(score)
		self_att_embedding = self.self_att_embedding(self_att_score, value)
		self_encoder_embedding = self.self_residual_connection(self_att_embedding, value)

		cross_score, cross_value = self.cross_att_dotproduct(self_encoder_embedding,encoder_output, 
			encoder_output)
		cross_att_score = self.masked_softmax(cross_score, enc_valid_lens)
		cross_att_embedding = self.cross_att_embedding(cross_att_score, cross_value)
		cross_decoder_embedding = self.cross_residual_connection(cross_att_embedding, self_encoder_embedding)
		cross_decoder_embedding = self.cross_position_wise_embedding(cross_decoder_embedding)

		return cross_decoder_embedding, self_att_score, cross_att_score


class drug_transformer():
	"""
	Implement the drug transformer model architecture
	"""
	def __init__(self):

		"""
		encoder block 1
		"""
		self.encoder_1 = encoder_block(20, 130)

		"""
		decoder block 1
		"""
		self.decoder_1 = decoder_block(20, 1)

		"""
		flattern layer, fully connected layer and final projection layer
		"""
		self.flattern = tf.keras.layers.Flatten()
		self.fc_layer = feed_forward_layer(20)
		self.projection = tf.keras.layers.Dense(1)
		self.concatenation_layer = concatenation_layer()

		#self.feed_forward_encoder_layer = feed_forward_layer(50)

		"""
		trying out simple position-wise embedding
		"""
		self.pw_encoder_1 = position_wise_embedding(50)
		self.pw_decoder_1 = position_wise_embedding(50)

		self.pw_encoder_1_2 = position_wise_embedding(1)
		self.pw_decoder_1_2 = position_wise_embedding(1)

		self.fc_layer_encoder_1 = feed_forward_layer(10)
		self.fc_layer_docoder_1 = feed_forward_layer(10)

		self.pw_encoder_2 = position_wise_embedding(50)
		self.pw_decoder_2 = position_wise_embedding(50)

		self.pw_encoder_2_2 = position_wise_embedding(1)
		self.pw_decoder_2_2 = position_wise_embedding(1)

		self.fc_layer_encoder_2 = feed_forward_layer(10)
		self.fc_layer_docoder_2 = feed_forward_layer(10)

		self.pw_encoder_3 = position_wise_embedding(50)
		self.pw_decoder_3 = position_wise_embedding(50)

		self.pw_encoder_3_2 = position_wise_embedding(1)
		self.pw_decoder_3_2 = position_wise_embedding(1)

		self.fc_layer_encoder_3 = feed_forward_layer(10)
		self.fc_layer_docoder_3 = feed_forward_layer(10)

		self.pw_encoder_4 = position_wise_embedding(50)
		self.pw_decoder_4 = position_wise_embedding(50)

		self.pw_encoder_4_2 = position_wise_embedding(1)
		self.pw_decoder_4_2 = position_wise_embedding(1)

		self.fc_layer_encoder_4 = feed_forward_layer(10)
		self.fc_layer_docoder_4 = feed_forward_layer(10)

	def model_construction(self):
		"""
		construct the transformer model
		"""
		X_input = Input((130, 56))
		Y_input = Input((5842, 1))
		enc_valid_lens = Input(())

		"""
		X, att_encoder_1 = self.encoder_1(X_input, enc_valid_lens)

		Y, att_self_decoder_1, att_cross_decoder_1 = self.decoder_1(Y_input, X, enc_valid_lens)

		X = self.flattern(X)

		Y = self.flattern(Y)
		#Y = self.fc_layer(Y)

		Y = self.concatenation_layer(X,Y)
		Y = self.fc_layer(Y)
		prediction = self.projection(Y)


		self.model = Model(inputs=(X_input, Y_input, enc_valid_lens), outputs=prediction)
		"""

		X1 = self.pw_encoder_1(X_input)
		X1 = self.pw_encoder_1_2(X1)
		Y1 = self.pw_decoder_1(Y_input)
		Y1 = self.pw_decoder_1_2(Y1)

		X2 = self.pw_encoder_2(X_input)
		X2 = self.pw_encoder_2_2(X2)
		Y2 = self.pw_decoder_2(Y_input)
		Y2 = self.pw_decoder_2_2(Y2)

		X3 = self.pw_encoder_3(X_input)
		X3 = self.pw_encoder_3_2(X3)
		Y3 = self.pw_decoder_3(Y_input)
		Y3 = self.pw_decoder_3_2(Y3)

		X3 = self.flattern(X3)
		#X2 = self.fc_layer_encoder_2(X2)
		Y3 = self.flattern(Y3)
		Y3 = self.fc_layer_docoder_3(Y3)

		X2 = self.flattern(X2)
		#X2 = self.fc_layer_encoder_2(X2)
		Y2 = self.flattern(Y2)
		Y2 = self.fc_layer_docoder_2(Y2)

		X1 = self.flattern(X1)
		#X1 = self.fc_layer_encoder_1(X1)
		Y1 = self.flattern(Y1)
		Y1 = self.fc_layer_docoder_1(Y1)

		Y1 = self.concatenation_layer(X1,Y1)
		Y2 = self.concatenation_layer(X2,Y2)
		Y3 = self.concatenation_layer(X3,Y3)

		Y = self.concatenation_layer(Y1,Y2)
		Y = self.concatenation_layer(Y,Y3)

		Y = self.fc_layer(Y)
		prediction = self.projection(Y)

		self.model = Model(inputs=(X_input, Y_input), outputs=prediction)

	#return model

	def model_compile(self):
		"""
		model compiling for training
		"""
		self.model.compile(loss= "mean_squared_error" , 
			optimizer="adam", metrics=["mean_squared_error"])












