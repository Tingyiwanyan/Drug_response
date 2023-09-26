import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


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
		else:
			shape_X = tf.shape(X)
			X = tf.reshape(X, shape=(-1, X.shape[-1]))
			reshape_X = tf.shape(X)
			print("x shape")
			print(X.shape)
			maxlen = X.shape[1]
			mask = tf.range(start=0, limit=shape_X[-1], dtype=tf.float32)[None,:]
			mask = tf.broadcast_to(mask, shape=reshape_X)
			#mask = tf.expand_dims(mask, 1)
			#mask = tf.broadcast_to(mask, shape=shape_X)

			valid_lens = tf.repeat(valid_lens, repeats = shape_X[1])
			mask = mask < tf.cast(valid_lens[:, None], dtype=tf.float32)

			print("mask shape")
			print(mask.shape)

			print("valid_len shape")
			print(valid_lens.shape)

			X = tf.where(mask, X, self.value)

			return tf.nn.softmax(tf.reshape(X, shape=shape_X), axis=-1)
			#return tf.nn.softmax(X, axis=-1)


#def masked_softmax(X, valid_lens, value=-1e6):
	"""
	Apply masked softmax to attention matrix, note, this function adapts
	only to encoder

	Parameters:
	-----------
	X: input attention matrix, 3D tensor:[batch_size, query_seq_length, key_seq_length]
	valid_lens: 1D tensor specifying the valid length for each attention matrix: [batch_size, ]

	Returns:
	--------
	masked softmax attention score matrix
	"""
	"""
	if valid_lens == None:
		return tf.nn.softmax(X, axis=-1)
	else:
		shape_X = X.shape
		X = tf.reshape(X, shape=(-1, X.shape[-1]))
		maxlen = X.shape[1]
		mask = tf.range(start=0, limit=shape_X[-1], dtype=tf.float32)[None,:]
		#mask = tf.broadcast_to(mask, shape=(shape_X[0], shape_X[-1]))

		valid_lens = tf.repeat(valid_lens, repeats = shape_X[1])
		mask = mask < tf.cast(valid_lens[:, None], dtype=tf.float32)

		X = tf.where(mask, X, value)

		return tf.nn.softmax(tf.reshape(X, shape=shape_X), axis=-1)#, X, mask
	"""

class PositionalEncoding(tf.keras.layers.Layer):  
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

class PositionWiseFFN(tf.keras.layers.Layer):  #@save
    """The positionwise feed-forward network."""
    def __init__(self, ffn_num_hiddens, ffn_num_outputs):
        super().__init__()
        self.dense1 = tf.keras.layers.Dense(ffn_num_hiddens)
        self.relu = tf.keras.layers.ReLU()
        self.dense2 = tf.keras.layers.Dense(ffn_num_outputs)

    def call(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class AddNorm(tf.keras.layers.Layer):  #@save
    """The residual connection followed by layer normalization."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.ln = tf.keras.layers.LayerNormalization()

    def call(self, X, Y, **kwargs):
        return self.ln(self.dropout(Y, **kwargs) + X)


class DotProductAttention(tf.keras.layers.Layer):  #@save
    """Scaled dot product attention."""
    def __init__(self, dropout):
        super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)
        self.masked_softmax = masked_softmax()

    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = self.masked_softmax(scores, valid_lens)
        return self.attention_weights
        #return tf.matmul(self.attention_weights, values)

class MultiHeadAttention(tf.keras.layers.Layer):  
	def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
		super().__init__()
		self.attention = DotProductAttention(dropout)
		#self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
		self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))
		self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))
		self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))

	def call(self, queries, keys, values, valid_lens, **kwargs):

	    queries = self.W_q(queries)
	    #queries = self.pos_encoding(queries)
	    keys = self.W_k(keys)
	    #keys = self.pos_encoding(keys)
	    values = self.W_v(values)
	    #values = self.pos_encoding(values)

	    #if valid_lens is not None:
	        # On axis 0, copy the first item (scalar or vector) for num_heads
	        # times, then copy the next item, and so on
	        #valid_lens = tf.repeat(valid_lens, repeats=self.num_heads, axis=0)

	    # Shape of output: (batch_size * num_heads, no. of queries,
	    # num_hiddens / num_heads)
	    output = self.attention(queries, keys, values, valid_lens, **kwargs)

	    # Shape of output_concat: (batch_size, no. of queries, num_hiddens)
	    #output_concat = self.transpose_output(output)
	    return output


class TransformerEncoderBlock(tf.keras.layers.Layer):  #@save
    """The Transformer encoder block."""
    def __init__(self, num_hiddens, num_heads, dropout, bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(num_hiddens, num_heads, dropout,bias)
        self.addnorm1 = AddNorm(dropout)        
        self.ffn = PositionWiseFFN(num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs),
                          **kwargs)
        #return self.addnorm2(Y, self.ffn(Y), **kwargs)
        return Y


class TransformerDecoderBlock(tf.keras.layers.Layer):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, num_heads, dropout):
        super().__init__()
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(dropout)
        self.attention2 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(num_hiddens, num_hiddens)
        self.addnorm3 = AddNorm(dropout)

    def call(self, X, enc_outputs, enc_valid_lens, dec_valid_lens=None,**kwargs):
        """
        Parameters:
        -----------
        X: the decoder input sequence (query)
        enc_outputs: the encoder output embedding (key & value)

        Returns:
		--------
		decoder embedding output
        """
        X2 = self.attention1(X, X, X, dec_valid_lens, **kwargs)
        Y = self.addnorm1(X, X2, **kwargs)
        # Encoder-decoder attention. Shape of enc_outputs:
        # (batch_size, num_steps, num_hiddens)
        Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens,
                             **kwargs)
        Z = self.addnorm2(Y, Y2, **kwargs)
        return self.addnorm3(Z, self.ffn(Z), **kwargs)


class Drug_transformer():
	"""
	Implement the drug transformer model architecture
	"""
	def __init__(self, num_hiddens, num_hiddens_fc, num_head=1,drop_out=0.1):
		
		self.trans_encoder = TransformerEncoderBlock(num_hiddens,num_heads=num_head,dropout=drop_out)
		self.trans_decoder = TransformerDecoderBlock(num_hiddens,num_heads=num_head,dropout=drop_out)

		self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))
		self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))
		self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))

		self.attention = DotProductAttention(dropout)

		self.num_hiddens = num_hiddens
		self.embedding_encoder = tf.keras.layers.Dense(num_hiddens, use_bias=False, 
        	activation= "relu",kernel_regularizer=regularizers.L2(1e-4))
		self.embedding_decoder = tf.keras.layers.Dense(num_hiddens, use_bias=False, 
        	activation= "relu",kernel_regularizer=regularizers.L2(1e-4))

		self.fc_decoder = tf.keras.layers.Dense(num_hiddens_fc, use_bias=False, 
        	activation= "relu",kernel_regularizer=regularizers.L2(1e-4))

		self.flattern = tf.keras.layers.Flatten()

		self.projection = tf.keras.layers.Dense(1)

		self.pos_encoding = PositionalEncoding(num_hiddens, drop_out)

	def model_construction(self, doc_valid_lens=None):
		"""
		construct the transformer model
		"""
		X_input = Input((130, 56))
		Y_input = Input((5842))
		enc_valid_lens = Input(())

		queries = self.W_q(X_input)
	    #queries = self.pos_encoding(queries)
	    keys = self.W_k(X_input)
	    #keys = self.pos_encoding(keys)
	    values = self.W_v(X_input)



		"""
		drug smile sequence with position encoding
		"""
		#X = self.embedding_encoder(X_input)
		#X = self.pos_encoding(X)
		#X = self.trans_encoder(X, enc_valid_lens)

		attention = self.attention(queries, keys, values, enc_valid_lens)

		X = tf.matmul(attention, values)

		X = tf.keras.layers.Add()([X, values])


		intermediate_shape = tf.shape(X)
		X = tf.reshape(X, shape=(intermediate_shape[0], intermediate_shape[1]*intermediate_shape[2]))

		"""
		Gene expression without position encoding
		"""
		Y = self.embedding_decoder(Y_input)

		Y = tf.concat([X,Y],axis=1)
		#Y = self.trans_decoder(Y, X, enc_valid_lens)

		Y = self.fc_decoder(Y)
		#Y = self.flattern(Y)
		Y = self.projection(Y)

		self.model = Model(inputs=(X_input, Y_input, enc_valid_lens), outputs=Y)

		#return model

	def model_compile(self):
		"""
		model compiling for training
		"""
		self.model.compile(loss= "mean_squared_error" , 
			optimizer="adam", metrics=["mean_squared_error"])



		



