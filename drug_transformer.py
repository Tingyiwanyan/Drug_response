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


class masked_softmax_sliding_window(tf.keras.layers.Layer):
	"""
	Assign sliding window mask

	Parameters:
	-----------
	top_k: sliding window size
	step_size: sliding window step size
	"""
	def __init__(self, top_k = 200, value=-1e7, maxval=5843, step_size=30):
		super().__init__()
		self.value = value
		self.top_k = top_k
		self.maxval = maxval
		self.step_size = step_size

	def call(self, X, **kwargs):
		"""
		Parameters:
		-----------
		X: 2D tensor specifying the attention score matrix (before softmax) [query_seq_length, key_seq_length]
		"""
		shape_X = tf.shape(X)
		mask = tf.zeros((shape_X[1],shape_X[-1]), dtype=bool)

		select_mask = tf.ones((shape_X[1]*self.top_k), dtype=bool)

		gene_indices = tf.range(start=0, limit=shape_X[1], dtype=tf.dtypes.int32)
		gene_indices = tf.repeat(gene_indices, repeats = self.top_k)

		slide_single = tf.range(start=0, limit=self.top_k, dtype=tf.dtypes.int32)
		slide_window = tf.expand_dims(slide_single, axis=0)
		slide_window = tf.broadcast_to(slide_window, shape=(shape_X[1], self.top_k))

		max_value_step = self.maxval - self.top_k
		step_ = tf.cast(tf.range(start=0, limit=max_value_step, delta=self.step_size), dtype=tf.float32)
		pad_size = tf.cast(self.maxval - tf.shape(step_)[0], dtype=tf.float32)

		#pad_window = tf.constant(step_[-1],shape=(pad_size,))

		pad_window = tf.ones(shape=(pad_size,))*step_[-1]

		step_ = tf.concat([step_, pad_window], axis=-1)

		step_ = tf.expand_dims(step_, axis=1)
		step_ = tf.cast(tf.broadcast_to(step_, shape=(shape_X[1], self.top_k)),tf.int32)

		slide_window_indices = tf.math.add(slide_window, step_)

		slide_window_indices = tf.reshape(slide_window_indices, shape=(shape_X[1]*self.top_k,))


		#random_indices = tf.random.uniform(shape=[shape_X[1]*self.top_k], minval=0, maxval=self.maxval, dtype=tf.dtypes.int32)

		gene_indices = tf.stack([gene_indices, slide_window_indices],axis=-1)

		mask = tf.tensor_scatter_nd_update(mask, gene_indices, select_mask)
		mask = tf.expand_dims(mask, axis=0) 
		mask = tf.broadcast_to(mask, shape=shape_X)

		X = tf.reshape(X, shape=(-1, X.shape[-1]))
		mask = tf.reshape(mask, shape=(-1, X.shape[-1]))
		reshape_X = tf.shape(X)


		X_slide_window = tf.where(mask, X, self.value)

		mask = tf.reshape(mask, shape=shape_X)


		return tf.nn.softmax(tf.reshape(X_slide_window, shape=shape_X), axis=-1)

class masked_softmax_random(tf.keras.layers.Layer):
	"""
	Assign random mask to attentions(query->key)
	Algorithm: for each query token, select random corresponded key tokens,
	then assign mask to those tokens.

	Parameters:
	-----------
	top_k: customized variable indicating how many random tokens to be selected
	value: the negative infinity value in order to make softmax probability zero
	"""
	def __init__(self, top_k = 100, value=-1e7, maxval=5843):
		super().__init__()
		self.value = value
		self.top_k = top_k
		self.maxval = maxval

	def call(self, X, **kwargs):
		"""
		Parameters:
		-----------
		X: 2D tensor specifying the attention score matrix (before softmax) [query_seq_length, key_seq_length]
		"""
		shape_X = tf.shape(X)
		mask = tf.zeros((shape_X[1],shape_X[-1]), dtype=bool)

		select_mask = tf.ones((shape_X[1]*self.top_k), dtype=bool)

		gene_indices = tf.range(start=0, limit=shape_X[1], dtype=tf.dtypes.int32)
		gene_indices = tf.repeat(gene_indices, repeats = self.top_k)

		random_indices = tf.random.uniform(shape=[shape_X[1]*self.top_k], minval=0, maxval=self.maxval, dtype=tf.dtypes.int32)

		gene_indices = tf.stack([gene_indices, random_indices],axis=-1)

		mask = tf.tensor_scatter_nd_update(mask, gene_indices, select_mask)
		mask = tf.expand_dims(mask, axis=0)
		mask = tf.broadcast_to(mask, shape=shape_X)

		X = tf.reshape(X, shape=(-1, X.shape[-1]))
		mask = tf.reshape(mask, shape=(-1, X.shape[-1]))
		reshape_X = tf.shape(X)


		X_random = tf.where(mask, X, self.value)

		mask = tf.reshape(mask, shape=shape_X)


		return tf.nn.softmax(tf.reshape(X_random, shape=shape_X), axis=-1), 
		tf.nn.softmax(tf.reshape(X, shape=shape_X), axis=-1), mask

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
	Define position wise embedding for sequence input embedding

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

class feature_selection_layer(tf.keras.layers.Layer):
	"""
	Define feature selection layer using greedy trainable 
	feature selection:
	https://openreview.net/pdf?id=TTLLGx3eet
	the output dimension is one, and input through a softmax layer 
	for representing the selection probability.

	Parameters:
	-----------
	select_dim: the feature dimensions to be selected.

	Returns:
	--------
	select_indices: the selected feature indices.
	"""
	def __init__(self, select_dim=200):
		super().__init__()
		self.select_dim = select_dim
		#self.output_dim = 1

	def build(self, input_shape, **kwargs):
		self.kernel = self.add_weight(name = 'kernel', shape = (input_shape[-1], 1),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)
		#b_init = tf.zeros_initializer()
		#self.b = tf.Variable(
		#	initial_value=b_init(shape=(self.output_dim,), dtype="float32"), trainable=True)

	def call(self, input_data, **kwargs):
		output_score = tf.matmul(input_data, self.kernel)
		shape_score = tf.shape(output_score)
		#print(output_score.shape)
		#output_score = tf.reduce_mean(output_score, axis=0)
		output_score = tf.nn.softmax(output_score,axis=1)
		output_score__ = tf.reshape(output_score,shape=[shape_score[0],shape_score[1]])
		#print(output_score.shape)
		top_indices = tf.math.top_k(output_score__, k=self.select_dim).indices
		#print(top_indices.shape)
		output_embedding = tf.gather(input_data, indices=top_indices, batch_dims=1)
		output_score_ = tf.gather(output_score, indices=top_indices, batch_dims=1)
		output_score_ = tf.nn.softmax(output_score_,axis=1)
		shape_score_ = tf.shape(output_score_)
		#output_score_ = tf.reshape(output_score_, shape = [shape_score_[0],shape_score_[1]])
		output_embedding = tf.math.multiply(output_embedding, output_score_)
		output_embedding = tf.cast(tf.math.l2_normalize(tf.math.reduce_sum(output_embedding, axis=1)), dtype=tf.float32)

		return tf.cast(top_indices, tf.int32), output_score, output_embedding


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

class dotproductattention_column(tf.keras.layers.Layer):  #@save
	"""
	Define scaled dot product layer
	for column-wise specific gene self-attention

	Parameters:
	-----------
	kernel_key: embedding matrix for key
	kernel_value: embedding matrix for value
	kernel_query: embedding matrix for query

	Returns:
	--------
	attention_score: the scale dot product score
	"""
	def __init__(self, output_dim):#, column_limit=200):
		super().__init__()
		self.output_dim = output_dim
		#self.column_limit = column_limit
		#self.masked_softmax = masked_softmax()

		#self.kernel_key = tf.keras.layers.D ense(output_dim, activation='sigmoid', 
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

	def call(self, queries, keys, values, indices_, **kwargs):
		#indices_ = tf.range(start=start_, limit=self.column_limit)
		keys = tf.gather(keys, indices=indices_, batch_dims=1)
		queries = tf.gather(queries, indices=indices_, batch_dims=1)
		values = tf.gather(values, indices=indices_, batch_dims=1)
		d = queries.shape[-1]
		queries = tf.matmul(queries, self.kernel_query) + self.b_query
		#queries = self.kernel_query(queries)
		keys = tf.matmul(keys, self.kernel_key) + self.b_key
		#keys = self.kernel_key(keys)
		#values = tf.matmul(values, self.kernel_value) + self.b_value
		values = self.kernel_value(values)
		#values_ = tf.gather(values, indices=indices_, axis=1)
		scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
			tf.cast(d, dtype=tf.float32))

		#self.attention_weights = self.masked_softmax(scores, valid_lens)
		return scores, values, queries#, values_

class dotproductattention_linformer(tf.keras.layers.Layer):  #@save
	"""
	Define scaled dot product layer
	Adding Linformer algorithms for reduce the gene-gene
	self-attention computation to linear time:
	https://arxiv.org/abs/2006.04768

	Parameters:
	-----------
	kernel_key: embedding matrix for key
	kernel_value: embedding matrix for value
	kernel_query: embedding matrix for query
	kernel_projection_e: the linear projection matrix for keys
	kernel_projection_f: the linear projection matrix for values

	Returns:
	--------
	attention_score: the scale dot product score
	"""
	def __init__(self, output_dim, project_dim=200):
		super().__init__()
		self.output_dim = output_dim
		self.project_dim = project_dim
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

		self.kernel_projection_e = self.add_weight(name = 'kernel_quary', shape = (self.project_dim, input_shape[1]),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

		self.kernel_projection_f = self.add_weight(name = 'kernel_quary', shape = (self.project_dim, input_shape[1]),
			initializer = tf.keras.initializers.he_normal(seed=None), trainable = True)

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

		values_linformer = tf.matmul(self.kernel_projection_f, values)
		
		projected_keys = tf.matmul(self.kernel_projection_e, keys)
		scores = tf.matmul(queries, projected_keys, transpose_b=True)/tf.math.sqrt(tf.cast(d, dtype=tf.float32))

		#self.attention_weights = self.masked_softmax(scores, valid_lens)
		return scores, values, queries, values_linformer, self.kernel_projection_f

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
		#return tf.cast(tf.math.add(X,Y), dtype=tf.float32)


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
		self.dotproductattention = dotproductattention(num_hiddens)
		self.att_embedding = attention_embedding()
		self.r_connection = residual_connection()

	def call(self, X, enc_valid_lens=None, **kwargs):
		X = self.pos_encoding(X)
		score, value, query = self.dotproductattention(X,X,X)
		att_score = self.masked_softmax(score, enc_valid_lens)
		att_embedding_ = self.att_embedding(att_score, value)

		encoder_embedding = self.r_connection(value, att_embedding_)
		#encoder_embedding = value

		return encoder_embedding, att_score


class decoder_self_block(tf.keras.layers.Layer):
	"""
	Define self-attention decoder block, since 
	gene expression doesn't contain sequencing information, so 
	we don't add position encoding in the embedding layer. 
	Use Linformer for the attention operation

	Parameters:
	-----------
	num_hiddens_self: hidden embedding dimension for the self att block

	Returns:
	--------
	cross_decoder_embedding: the decoder embedding output
	self_att_score: the self att score in decoder self att block
	cross_att_score: the cross att score in the decoder cross att block
	"""
	def __init__(self, num_hiddens):
		super().__init__()
		self.masked_softmax_deco_self = masked_softmax()
		self.dotproductattention_deco = dotproductattention_column(num_hiddens)
		#self.dotproductattention_deco = dotproductattention(num_hiddens)
		self.att_embedding = attention_embedding()
		self.r_connection = residual_connection()

	def call(self, Y, enc_valid_lens=None, **kwargs):
		#score_deco, value_deco, query_deco, value_linformer_deco, kernel_projection_f = self.dotproductattention_deco(Y,Y,Y)
		score_deco, value_deco, query_deco,value_deco_ = self.dotproductattention_deco(Y,Y,Y)
		att_score_deco = self.masked_softmax_deco_self(score_deco, enc_valid_lens)
		#att_embedding_deco = self.att_embedding(att_score_deco, value_linformer_deco)
		att_embedding_deco = self.att_embedding(att_score_deco, value_deco_)

		self_deco_embedding = self.r_connection(value_deco, att_embedding_deco)
		#self_deco_embedding = value_deco

		#return self_deco_embedding, att_score_deco, kernel_projection_f
		return self_deco_embedding, att_score_deco

class decoder_cross_block(tf.keras.layers.Layer):
	"""
	Define decoder cross attention 
	"""
	def __init__(self, num_hiddens):
		super().__init__()
		self.masked_softmax_deco_cross = masked_softmax()
		self.dotproductattention_deco_cross = dotproductattention(num_hiddens)
		self.att_embedding = attention_embedding()
		self.r_connection = residual_connection()

	def call(self, Y, X, enc_valid_lens, **kwargs):
		score_deco_cross, value_deco_cross, query_deco_cross = self.dotproductattention_deco_cross(Y,X,X)
		att_score_deco_cross = self.masked_softmax_deco_cross(score_deco_cross, enc_valid_lens)
		att_embedding_deco_cross = self.att_embedding(att_score_deco_cross, value_deco_cross)


		#att_embedding_deco_cross = tf.concat([att_embedding_deco_cross, att_embedding_deco_cross2],axis=-1)
		#query_deco_cross = tf.concat([query_deco_cross, query_deco_cross2],axis=-1)

		cross_embedding = self.r_connection(query_deco_cross, att_embedding_deco_cross)

		return cross_embedding, att_score_deco_cross


class drug_transformer():
	"""
	Implimentation of drug transformer
	"""
	def __init__(self):

		self.dense_1 = tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_2 = tf.keras.layers.Dense(10, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_3 = tf.keras.layers.Dense(500, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_4 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_5 = tf.keras.layers.Dense(1)

		self.flattern_enco = tf.keras.layers.Flatten()
		self.flattern_deco = tf.keras.layers.Flatten()

		"""
		1st head attention
		"""
		self.encoder_1 = encoder_block(10,130)
		#self.decoder_self_1 = encoder_block(20,130)
		self.decoder_self_1 = decoder_self_block(10)
		self.decoder_cross_1 = decoder_cross_block(10)

		"""
		2nd head attention
		"""
		self.encoder_2 = encoder_block(10,130)
		self.decoder_self_2 = decoder_self_block(10)
		self.decoder_cross_2 = decoder_cross_block(10)

	def model_construction(self):
		"""
		construct the transformer model
		"""
		X_input = Input((130, 56))
		Y_input = Input((5843, 1))
		enc_valid_lens = Input(())

		X = self.dense_1(X_input)
		Y = self.dense_2(Y_input)

		"""
		multi head transformer
		"""
		X, encoder_att_score = self.encoder_1(X, enc_valid_lens)
		#X, encoder_att_score_2 = self.encoder_2(X, enc_valid_lens)

		#X = tf.concat([X_,X_2],axis=-1)

		#Y, att_score_deco, kernel_projection_f = self.decoder_self_1(Y)
		Y, att_score_deco = self.decoder_self_1(Y, enc_valid_lens)
		#Y, att_score_deco_2, kernel_projection_f_2 = self.decoder_self_2(Y)

		#Y = tf.concat([Y_,Y_2],axis=-1)

		Y, att_score_deco_cross = self.decoder_cross_1(Y, X, enc_valid_lens)

		Y = self.flattern_deco(Y)



		#Y = tf.concat([X,Y],axis=1)

		Y = self.dense_3(Y)
		Y = self.dense_4(Y)
		Y = self.dense_5(Y)

		self.model = Model(inputs=(X_input, Y_input, enc_valid_lens), outputs=Y)

		self.model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

		return self.model




class drug_transformer_():
	"""
	Implement the drug transformer model architecture
	"""
	def __init__(self):

		self.masked_softmax_ = masked_softmax()
		self.masked_softmax_2 = masked_softmax()
		self.masked_softmax_deco_self = masked_softmax()
		self.masked_softmax_deco_self2 = masked_softmax()
		self.masked_softmax_deco_cross = masked_softmax()
		self.masked_softmax_deco_cross2 = masked_softmax()

		self.feature_selction = feature_selection_layer()

		"""
		1st head attention
		"""
		self.dotproductattention1 = dotproductattention(50)

		self.dotproductattention_deco = dotproductattention_column(50)

		self.dotproductattention_deco_cross = dotproductattention(50)

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

		self.dense_1 = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_2 = tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_3 = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_4 = tf.keras.layers.Dense(50, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.dense_5 = tf.keras.layers.Dense(1)

		self.dense_6 = tf.keras.layers.Dense(20, activation='relu', kernel_regularizer=regularizers.L2(1e-4))

		self.kernel_key = tf.keras.layers.Dense(50, activation='sigmoid', 
			kernel_regularizer=regularizers.L2(1e-4))

		self.kernel_query = tf.keras.layers.Dense(50, activation='sigmoid', 
			kernel_regularizer=regularizers.L2(1e-4))

		self.pos_encoding = positionalencoding(100,130)

		self.flattern_enco = tf.keras.layers.Flatten()
		self.flattern_deco = tf.keras.layers.Flatten()

	def model_construction(self):
		"""
		construct the transformer model
		"""
		X_input = Input((130, 56))
		Y_input = Input((5843, 1))
		enc_valid_lens = Input(())

		X = self.dense_1(X_input)

		X = self.pos_encoding(X)

		"""
		self attention for the encoder
		"""
		score, value, query = self.dotproductattention1(X,X,X)
		att_score = self.masked_softmax_(score, enc_valid_lens)
		att_embedding_ = self.att_embedding(att_score, value)

		#score2, value2, query2 = self.dotproductattention2(X,X,X)
		#att_score2 = self.masked_softmax_2(score2, enc_valid_lens)
		#att_embedding_2 = self.att_embedding(att_score2, value2)

		#att_embedding_ = tf.concat([att_embedding_, att_embedding_2], axis=-1)
		#value = tf.concat([value,value2],axis=-1)


		X = self.r_connection(value, att_embedding_)


		"""
		self attention for the deocoder
		"""
		Y = self.dense_2(Y_input)
		#top_indices, output_score, Y = self.feature_selction(Y)
		#print(top_indices.shape)
		#score_deco, value_deco, query_deco, value_linformer_deco = self.dotproductattention_deco(Y,Y,Y)
		#score_deco, value_deco, query_deco = self.dotproductattention_deco(Y,Y,Y,top_indices)
		#att_score_deco = self.masked_softmax_deco_self(score_deco)
		#att_embedding_deco = self.att_embedding(att_score_deco, value_deco)

		#score_deco2, value_deco2, query_deco2 = self.dotproductattention_deco2(Y,Y,Y)
		#att_score_deco2 = self.masked_softmax_deco_self2(score_deco2)
		#att_embedding_deco2 = self.att_embedding(att_score_deco2, value_deco2)

		#att_embedding_deco = tf.concat([att_embedding_deco, att_embedding_deco2],axis=-1)
		#value_deco = tf.concat([value_deco, value_deco2],axis=-1)


		#Y = self.r_connection(value_deco, att_embedding_deco)
		#Y = value_deco

		"""
		cross attention for the deocoder
		"""
		score_deco_cross, value_deco_cross, query_deco_cross = self.dotproductattention_deco_cross(Y,X,X)
		att_score_deco_cross = self.masked_softmax_deco_cross(score_deco_cross, enc_valid_lens)
		att_embedding_deco_cross = self.att_embedding(att_score_deco_cross, value_deco_cross)

		#score_deco_cross2, value_deco_cross2, query_deco_cross2 = self.dotproductattention_deco_cross2(Y,X,X)
		#att_score_deco_cross2 = self.masked_softmax_deco_cross2(score_deco_cross2, enc_valid_lens)
		#att_embedding_deco_cross2 = self.att_embedding(att_score_deco_cross2, value_deco_cross2)

		#att_embedding_deco_cross = tf.concat([att_embedding_deco_cross, att_embedding_deco_cross2],axis=-1)
		#query_deco_cross = tf.concat([query_deco_cross, query_deco_cross2],axis=-1)

		Y = self.r_connection(query_deco_cross, att_embedding_deco_cross)


		#X = self.flattern_enco(X)
		#Y = self.flattern_deco(Y)
		#Y = self.dense_3(Y)
		#top_indices, output_score, Y = self.feature_selction(Y)

		#Y = tf.concat([X,Y],axis=1)

		#Y = self.dense_3(Y)
		#Y = self.dense_4(Y)
		Y = self.dense_6(Y)
		Y = self.flattern_deco(Y)
		Y = self.dense_5(Y)

		self.model = Model(inputs=(X_input, Y_input, enc_valid_lens), outputs=Y)

		self.model.compile(loss= "mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])

		return self.model













