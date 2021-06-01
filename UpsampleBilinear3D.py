import numpy as np
import tensorflow as tf

class UpsampleBilinear3D( tf.keras.layers.Layer ) :  # implements bilinear upsampling layer

  def __init__( self, scale = 2 ) :
      super( UpsampleBilinear3D, self).__init__()
      self.scale_factor = scale


  def get_config( self ) :  # this function is needed to save the layer when saving model or checkpoints
    config = super( UpsampleBilinear3D, self ).get_config()
    config.update( { 'scale' : self.scale_factor } )
    return config


  def build( self, input_shape ) :
      # Create bilinear upsampling weights matrix
      filter_size = 2 * self.scale_factor - self.scale_factor % 2
      upsample_filter = np.zeros( ( filter_size,
                                    filter_size,
                                    filter_size,
                                    input_shape[ -1 ],
                                    input_shape[ -1 ] ), dtype = np.float32 )
      if filter_size % 2 == 1 :
          center = self.scale_factor - 1
      else :
          center = self.scale_factor - 0.5
      og = np.ogrid[ : filter_size, : filter_size, : filter_size ]  # makes grid vectors along DHW dimensions
      upsample_kernel =  ( 1 - abs( og[ 0 ] - center ) / self.scale_factor ) * \
                         ( 1 - abs( og[ 1 ] - center ) / self.scale_factor ) * \
                         ( 1 - abs( og[ 2 ] - center ) / self.scale_factor )  # tensor product of the 3 centered grid vector
      for i in range( input_shape[ -1 ] ) :
          upsample_filter[ :, :, :, i, i ] = upsample_kernel
      self.upsample_filter = tf.constant( upsample_filter, dtype = 'float32', name = 'upsample_filter' )  # store as tf constant
      self.dims = input_shape


  def call( self, x ) :
      # up-sample the input
      if self.scale_factor > 1 :  # otherwise nothing to be done, just return input
          # up-sample the input tensor using the bilinear upsampling kernel build in build() and the tf.conv3d_transpose()
          batches = tf.shape( x )[ 0 ]
          x = tf.nn.conv3d_transpose( x, self.upsample_filter,
                                    output_shape = [ batches,
                                                     self.dims[ 1 ] * self.scale_factor,
                                                     self.dims[ 2 ] * self.scale_factor,
                                                     self.dims[ 3 ] * self.scale_factor,
                                                     self.dims[ 4 ] ],
                                    strides = [ 1, self.scale_factor, self.scale_factor, self.scale_factor, 1 ],
                                    name = 'UpsampleKernel' + str( self.scale_factor ) )
      return x