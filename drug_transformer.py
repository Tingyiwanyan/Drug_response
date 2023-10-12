import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import keras.backend as K


class masked_softmax(tf.keras.layers.Layer):
	def __init__(self, value=-1e7):
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
			#print("Im here")
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

class masked_softmax_selected(tf.keras.layers.Layer):
	"""
	Define selected maksed for softmax layer.
	Algorithm: for the calculated dot product scores, first select
	the customized number of attention slots for picking features
	"""
	def __init__(self, top_k = 10, value=-1e7):
		super().__init__()
		self.value = value
		self.top_k = top_k

	def call(self, X, **kwargs):
		"""
		Parameters:
		-----------
		X: 2D tensor specifying the attention score matrix (before softmax) [query_seq_length, key_seq_length]
		"""
		shape_X = tf.shape(X)
		X = tf.reshape(X, shape=(-1, X.shape[-1]))
		X_top_index = tf.math.top_k(X, k=self.top_k)[0]
		X_top_index = tf.reduce_min(X_top_index, axis=-1)
		X_top_index = tf.expand_dims(X_top_index, axis=1)
		reshape_X = tf.shape(X)
		X_top_index = tf.broadcast_to(X_top_index, shape=reshape_X)

		mask = X >= X_top_index
		#maxlen = X.shape[1]
		#mask = tf.range(start=0, limit=shape_X[-1], dtype=tf.float32)[None,:]
		#mask = tf.broadcast_to(mask, shape=reshape_X)

		#valid_lens = tf.repeat(valid_lens, repeats = shape_X[1])
		#mask = mask < tf.cast(valid_lens[:, None], dtype=tf.float32)

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

		#self.kernel_key = tf.keras.layers.Dense(output_dim, activation='sigmoid', 
		#	kernel_regularizer=regularizers.L2(1e-4))

		#self.kernel_query = tf.keras.layers.Dense(output_dim, activation='sigmoid', 
		#	kernel_regularizer=regularizers.L2(1e-4))

		self.kernel_value = tf.keras.layers.Dense(output_dim, activation='relu', 
			kernel_regularizer=regularizers.L2(1e-4))

	
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
	

	def call(self, queries, keys, values, valid_lens=None, **kwargs):
		d = queries.shape[-1]
		queries = tf.matmul(queries, self.kernel_query) + self.b_query
		#queries = self.kernel_query(queries)
		keys = tf.matmul(keys, self.kernel_key) + self.b_key
		#keys = self.kernel_key(keys)
		#values = tf.matmul(values, self.kernel_value) + self.b_value
		values = self.kernel_value(values)

		scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
			tf.cast(d, dtype=tf.float32))

		#self.attention_weights = self.masked_softmax(scores, valid_lens)
		return scores, values, queries

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

		self.masked_softmax_ = masked_softmax()
		self.masked_softmax_deco_self = masked_softmax()
		self.masked_softmax_deco_cross =masked_softmax()

		"""
		1st head attention
		"""
		self.dotproductattention1 = dotproductattention(10)

		self.dotproductattention_deco = dotproductattention(10)

		self.dotproductattention_deco_cross = dotproductattention(10)

		"""
		2nd head attention
		"""
		self.dotproductattention2 = dotproductattention(10)

		self.dotproductattention_deco2 = dotproductattention(10)

		self.dotproductattention_deco_cross2 = dotproductattention(10)

		"""
		3rd head attention
		"""
		self.dotproductattention3 = dotproductattention(10)

		self.dotproductattention_deco3 = dotproductattention(10)

		self.dotproductattention_deco_cross3 = dotproductattention(10)



		self.att_embedding = attention_embedding()
		self.r_connection = residual_connection()

		self.dense_1 = tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_2 = tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_3 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_4 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_5 = tf.keras.layers.Dense(1)

		self.kernel_key = tf.keras.layers.Dense(50, activation='sigmoid', 
			kernel_regularizer=regularizers.L2(1e-4))

		self.kernel_query = tf.keras.layers.Dense(50, activation='sigmoid', 
			kernel_regularizer=regularizers.L2(1e-4))

		self.pos_encoding = positionalencoding(10,130)

		self.flattern_enco = tf.keras.layers.Flatten()
		self.flattern_deco = tf.keras.layers.Flatten()

	def model_construction(self):
		"""
		construct the transformer model
		"""
		X_input = Input((130, 56))
		Y_input = Input((5842, 1))
		enc_valid_lens = Input(())

		X = self.dense_1(X_input)

		X = self.pos_encoding(X)

		Y = self.dense_2(Y_input)


		"""
		self attention for the encoder
		"""
		score, value, query = self.dotproductattention1(X,X,X, enc_valid_lens)
		att_score = self.masked_softmax_(score, enc_valid_lens)
		att_embedding_ = self.att_embedding(att_score, value)

		score2, value2, query2 = self.dotproductattention1(X,X,X, enc_valid_lens)
		att_score2 = self.masked_softmax_(score2, enc_valid_lens)
		att_embedding_2 = self.att_embedding(att_score2, value2)

		att_embedding_ = tf.concat([att_embedding_, att_embedding_2], axis=-1)
		value = tf.concat([value,value2],axis=-1)


		X = self.r_connection(value, att_embedding_)


		"""
		self attention for the deocoder
		"""
		Y = self.dense_2(Y_input)
		score_deco, value_deco, query_deco = self.dotproductattention_deco(Y,Y,Y)
		att_score_deco = self.masked_softmax_deco_self(score_deco)
		att_embedding_deco = self.att_embedding(att_score_deco, value_deco)

		score_deco2, value_deco2, query_deco2 = self.dotproductattention_deco(Y,Y,Y)
		att_score_deco2 = self.masked_softmax_deco_self(score_deco2)
		att_embedding_deco2 = self.att_embedding(att_score_deco2, value_deco2)

		att_embedding_deco = tf.concat([att_embedding_deco, att_embedding_deco2],axis=-1)
		value_deco = tf.concat([value_deco, value_deco2],axis=-1)


		Y = self.r_connection(value_deco, att_embedding_deco)

		"""
		cross attention for the deocoder
		"""
		score_deco_cross, value_deco_cross, query_deco_cross = self.dotproductattention_deco_cross(Y,X,X, enc_valid_lens)
		att_score_deco_cross = self.masked_softmax_deco_cross(score_deco_cross)
		att_embedding_deco_cross = self.att_embedding(att_score_deco_cross, value_deco_cross)

		score_deco_cross2, value_deco_cross2, query_deco_cross2 = self.dotproductattention_deco_cross(Y,X,X, enc_valid_lens)
		att_score_deco_cross2 = self.masked_softmax_deco_cross(score_deco_cross2)
		att_embedding_deco_cross2 = self.att_embedding(att_score_deco_cross2, value_deco_cross2)

		att_embedding_deco_cross = tf.concat([att_embedding_deco_cross, att_embedding_deco_cross2],axis=-1)
		query_deco_cross = tf.concat([query_deco_cross, query_deco_cross2],axis=-1)

		Y = self.r_connection(query_deco_cross, att_embedding_deco_cross)


		#X = self.flattern_enco(X)
		Y = self.flattern_deco(Y)

		#Y = tf.concat([X,Y],axis=1)

		Y = self.dense_3(Y)
		Y = self.dense_4(Y)
		Y = self.dense_5(Y)

		self.model = Model(inputs=(X_input, Y_input, enc_valid_lens), outputs=Y)

		self.model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

		return self.model













