import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

"""
def masked_softmax(X, valid_lens):  #@save
    # X: 3D tensor, valid_lens: 1D or 2D tensor
    def _sequence_mask(X, valid_len, value=0):
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen, dtype=tf.float32)[
            None, :] < tf.cast(valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)

    if valid_lens is None:
        return tf.nn.softmax(X, axis=-1)
    else:
        shape = X.shape
        if len(valid_lens.shape) == 1:
            valid_lens = tf.repeat(valid_lens, repeats=shape[1])

        else:
            valid_lens = tf.reshape(valid_lens, shape=-1)
        # On the last axis, replace masked elements with a very large negative
        # value, whose exponentiation outputs 0
        X = _sequence_mask(tf.reshape(X, shape=(-1, shape[-1])), valid_lens,
                           value=-1e6)
        return tf.nn.softmax(tf.reshape(X, shape=shape), axis=-1)
"""

def masked_softmax(X, valid_lens, value=-1e6):
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
	if valid_lens == None:
		return tf.nn.softmax(X, axis=-1)
	else:
		shape_X = X.shape
		X = tf.reshape(X, shape=(-1, X.shape[-1]))
		maxlen = X.shape[1]
		mask = tf.range(start=0, limit=shape_X[-1], dtype=tf.float32)[None,:]
		mask = tf.broadcast_to(mask, shape=(X.shape[0], shape_X[-1]))

		valid_lens = tf.repeat(valid_lens, repeats = shape_X[1])
		mask = mask < tf.cast(valid_lens[:, None], dtype=tf.float32)

		X = tf.where(mask, X, value)

		return tf.nn.softmax(tf.reshape(X, shape=shape_X), axis=-1)#, X, mask


class PositionalEncoding(tf.keras.layers.Layer):  
    """Positional encoding."""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        #super().__init__()
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
        #super().__init__()
        self.dropout = tf.keras.layers.Dropout(dropout)

    def call(self, queries, keys, values, valid_lens=None, **kwargs):
        d = queries.shape[-1]
        scores = tf.matmul(queries, keys, transpose_b=True)/tf.math.sqrt(
            tf.cast(d, dtype=tf.float32))
        self.attention_weights = masked_softmax(scores, valid_lens)
        return tf.matmul(self.dropout(self.attention_weights, **kwargs), values)


class MultiHeadAttention(tf.keras.layers.Layer):  
	def __init__(self, num_hiddens, num_heads, dropout, bias=False, **kwargs):
		self.attention = DotProductAttention(dropout)
		self.W_q = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))
		self.W_k = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))
		self.W_v = tf.keras.layers.Dense(num_hiddens, use_bias=bias, 
			activation= "relu",kernel_regularizer=regularizers.L2(1e-4))

	def call(self, queries, keys, values, valid_lens, **kwargs):

	    queries = self.W_q(queries)
	    keys = self.W_k(keys)
	    values = self.W_v(values)

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
        #super().__init__()
        self.attention = MultiHeadAttention(num_hidden, num_heads, dropout,bias)
        self.addnorm1 = AddNorm(dropout)
        self.ffn = PositionWiseFFN(num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(dropout)

    def call(self, X, valid_lens, **kwargs):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens, **kwargs),
                          **kwargs)
        return self.addnorm2(Y, self.ffn(Y), **kwargs)


class TransformerDecoderBlock(tf.keras.layers.Layer):
    # The i-th block in the Transformer decoder
    def __init__(self, num_hiddens, num_heads, dropout):
        #super().__init__()
        self.attention1 = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
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
	def __init__(self, num_hiddens, num_head=1,drop_out=0.1):
		
		self.trans_encoder = TransformerEncoderBlock(num_hiddens,num_head=num_head,drop_out=drop_out)
		self.trans_decoder = TransformerDecoderBlock(num_hiddens,num_head=num_head,drop_out=drop_out)

		self.num_hiddens = num_hiddens
		self.embedding_encoder = tf.keras.layers.Dense(num_hiddens, use_bias=False, 
        	activation= "relu",kernel_regularizer=regularizers.L2(1e-4))
		self.embedding_decoder = tf.keras.layers.Dense(num_hiddens, use_bias=False, 
        	activation= "relu",kernel_regularizer=regularizers.L2(1e-4))

		self.pos_encoding = PositionalEncoding(num_hiddens, drop_out)

	def model_construction(self, enc_valid_lens, doc_valid_lens=None):
		"""
		construct the transformer model
		"""
		X_input = Input((None, 130, 56))
		Y_input = Input((None, 130, 1))
		enc_valid_lens = Input((None, ))

		"""
		drug smile sequence with position encoding
		"""
		X = self.embedding_encoder(X_input)
		X = self.pos_encoding(X)
		X = self.trans_encoder(X, enc_valid_lens)

		"""
		Gene expression without position encoding
		"""
		Y = self.embedding_decoder(Y_input)
		Y = self.trans_decoder(Y, X, enc_valid_lens)

		model = Model(inputs=(X_input, Y_input), outputs=Y)

		return model


		



