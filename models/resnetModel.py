def resnet(input_shape=(128,512,1), num_classes=4):
  resnet = tf.keras.applications.ResNet50(include_top=False, input_shape=input_shape)
  resnet.trainable = False

  avg_pool = layers.GlobalAveragePooling2D()(resnet.output)
  dropout = layers.Dropout(0.5)(avg_pool)
  dense = layers.Dense(128, 'relu')(dropout)
  dropout = layers.Dropout(0.5)(dense)
  dense = layers.Dense(128, 'relu')(dropout)
  dropout = layers.Dropout(0.5)(dense)
  output = layers.Dense(num_classes, 'softmax')(dropout)

  model = tf.keras.Model(inputs=resnet.input, outputs=output)
  return model
