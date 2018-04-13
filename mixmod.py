import keras.backend as K
from keras.layers import Layer, RNN
import keras.activations as activations
import keras.initializers as initializers
import keras.regularizers as regularizers
import keras.constraints as constraints

def _generate_dropout_mask(ones, rate, training=None, count=1):
	def dropped_inputs():
		return K.dropout(ones, rate)
	if count > 1:
		return [K.in_train_phase(
			dropped_inputs,
			ones,
			training=training) for _ in range(count)]
	return K.in_train_phase(
		dropped_inputs,
		ones,
		training=training)

'''
MixMod is a variation on the LSTM and GRU models, going for maximum
simplicity while still preserving the desirable qualities. Additionally,
it has separate parameters for hidden and output dimensionalities.
'''
class MixModCell(Layer):
	def __init__(self, mem, out,
		activation='tanh',
		mix_activation='tanh',
		mod_activation='sigmoid',
		
		use_bias=True,
		
		kernel_initializer='glorot_uniform',
		recurrent_initializer='orthogonal',
		bias_initializer='zeros',
		
		kernel_regularizer=None,
		recurrent_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		
		kernel_constraint=None,
		recurrent_constraint=None,
		bias_constraint=None,
		
		dropout=0.,
		recurrent_dropout=0.,
		**kw
	):
		super(MixModCell, self).__init__(**kw)
		self.mem_dim = mem
		self.out_dim = out
		
		self.state_size = mem
		
		self.activation = activations.get(activation)
		self.mix_activation = activations.get(mix_activation)
		self.mod_activation = activations.get(mod_activation)
		
		self.use_bias = use_bias
		
		self.kernel_initializer = initializers.get(kernel_initializer)
		self.recurrent_initializer = initializers.get(recurrent_initializer)
		self.bias_initializer = initializers.get(bias_initializer)
		
		self.kernel_regularizer = regularizers.get(kernel_regularizer)
		self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
		self.bias_regularizer = regularizers.get(bias_regularizer)
		
		self.kernel_constraint = constraints.get(kernel_constraint)
		self.recurrent_constraint = constraints.get(recurrent_constraint)
		self.bias_constraint = constraints.get(bias_constraint)
		
		self.dropout = min(1., max(0., dropout))
		self.recurrent_dropout = min(1., max(0., recurrent_dropout))

		self._dropout_mask = None
		self._recurrent_dropout_mask = None
	
	def build(self, inp):
		self.kernel = self.add_weight(
			shape=(inp[-1] + self.mem_dim, self.out_dim),
			name="kernel",
			initializer=self.kernel_initializer,
			regularizer=self.kernel_regularizer,
			constraint=self.kernel_constraint
		)
		self.recurrent_kernel = self.add_weight(
			shape=(inp[-1] + self.mem_dim, self.mem_dim*2 + 2),
			name="recurrent_kernel",
			initializer=self.recurrent_initializer,
			regularizer=self.recurrent_regularizer,
			constraint=self.recurrent_constraint
		)
		
		
		if self.use_bias:
			self.bias = self.add_weight(
				shape=(self.out_dim,),
				name="bias",
				initializer=self.bias_initializer,
				regularizer=self.bias_regularizer,
				constraint=self.bias_constraint
			)
		else:
			self.bias = None
		
		self.kernel_x = self.recurrent_kernel[:, :self.mem_dim]
		self.kernel_m = self.recurrent_kernel[
			:, self.mem_dim:self.mem_dim*2
		]
		self.bias_x = K.flatten(self.recurrent_kernel[
			:inp[-1], self.mem_dim*2:self.mem_dim*2 + 1
		])
		self.bias_m = K.flatten(self.recurrent_kernel[
			:inp[-1], self.mem_dim*2 + 1:
		])
		self.built = True
	
	def call(self, x, states, training=None):
		if 0 < self.dropout < 1 and self._dropout_mask is None:
			self._dropout_mask = _generate_dropout_mask(
				K.ones_like(x),
				self.dropout,
				training=training,
				count=1
			)
		if (0 < self.recurrent_dropout < 1 and
			self._recurrent_dropout_mask is None
		):
			self._recurrent_dropout_mask = _generate_dropout_mask(
				K.ones_like(states[0]),
				self.recurrent_dropout,
				training=training,
				count=1
			)
		# dropout matrices for input units
		dp_mask = self._dropout_mask
		# dropout matrices for recurrent units
		rec_dp_mask = self._recurrent_dropout_mask
		
		if 0 < self.dropout < 1:
			x *= self._dropout_mask[0]
		
		h = states[0]
		if 0 < self.recurrent_dropout < 1:
			h *= self._recurrent_dropout_mask[0]
		
		hx = K.concatenate([h, x])
		
		xo = K.bias_add(K.dot(hx, self.kernel_x), self.bias_x)
		xo = self.mix_activation(xo)
		
		mo = K.bias_add(K.dot(hx, self.kernel_m), self.bias_m)
		mo = self.mod_activation(mo)
		
		nh = mo*h + (1 - mo)*xo
		yo = K.dot(hx, self.kernel)
		
		if self.use_bias:
			yo = K.bias_add(yo, self.bias)
		
		yo = self.activation(yo)
		
		return yo, [nh]
	
	def get_config(self):
		config = {
			'mem_dim': self.mem_dim, 'out_dim': self.out_dim,
			
			'activation': activations.serialize(self.activation),
			'recurrent_activation':
				activations.serialize(self.recurrent_activation),
			
			'use_bias': self.use_bias,
			
			'kernel_initializer':
				initializers.serialize(self.kernel_initializer),
			'recurrent_initializer':
				initializers.serialize(self.recurrent_initializer),
			'bias_initializer': initializers.serialize(self.bias_initializer),
			
			'kernel_regularizer':
				regularizers.serialize(self.kernel_regularizer),
			'recurrent_regularizer':
				regularizers.serialize(self.recurrent_regularizer),
			'bias_regularizer': regularizers.serialize(self.bias_regularizer),
			
			'kernel_constraint': constraints.serialize(self.kernel_constraint),
			'recurrent_constraint':
				constraints.serialize(self.recurrent_constraint),
			'bias_constraint': constraints.serialize(self.bias_constraint),
			
			'dropout': self.dropout,
			'recurrent_dropout': self.recurrent_dropout,
		}
		base_config = super(MixModCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class MixMod(RNN):
	def __init__(self, mem, out,
		activation='tanh',
		mix_activation='tanh',
		mod_activation='sigmoid',
		
		use_bias=True,
		
		kernel_initializer='glorot_uniform',
		recurrent_initializer='orthogonal',
		bias_initializer='zeros',
		
		kernel_regularizer=None,
		recurrent_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		
		kernel_constraint=None,
		recurrent_constraint=None,
		bias_constraint=None,
		
		dropout=0.,
		recurrent_dropout=0.,
		**kw
	):
		self.cell = MixModCell(mem, out,
			activation,
			mix_activation,
			mod_activation,
			use_bias,
			kernel_initializer,
			recurrent_initializer,
			bias_initializer,
			kernel_regularizer,
			recurrent_regularizer,
			bias_regularizer,
			activity_regularizer,
			kernel_constraint,
			recurrent_constraint,
			bias_constraint,
			dropout,
			recurrent_dropout
		)
		
		super(MixMod, self).__init__(self.cell, **kw)
	
	@property
	def units(self):
		return self.cell.units
	@property
	def activation(self):
		return self.cell.activation
	@property
	def recurrent_activation(self):
		return self.cell.recurrent_activation
	
	'''
	@property
	def use_bias(self):
		return self.cell.use_bias
	'''
	
	@property
	def kernel_initializer(self):
		return self.cell.kernel_initializer
	@property
	def recurrent_initializer(self):
		return self.cell.recurrent_initializer
	@property
	def bias_initializer(self):
		return self.cell.bias_initializer
	@property
	def kernel_regularizer(self):
		return self.cell.kernel_regularizer
	@property
	def recurrent_regularizer(self):
		return self.cell.recurrent_regularizer
	@property
	def bias_regularizer(self):
		return self.cell.bias_regularizer
	@property
	def kernel_constraint(self):
		return self.cell.kernel_constraint
	@property
	def recurrent_constraint(self):
		return self.cell.recurrent_constraint
	@property
	def bias_constraint(self):
		return self.cell.bias_constraint
		
	def get_config(self):
		config = {}
		base_config = super(MixModCell, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))
	
	@classmethod
	def from_config(cls, config):
		return cls(**config)
