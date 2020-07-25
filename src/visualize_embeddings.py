import os
import keras
import tensorflow

ROOT_DIR = '/tmp/tfboard'

os.makedirs(ROOT_DIR, exist_ok=True)

OUTPUT_MODEL_FILE_NAME = os.path.join(ROOT_DIR, 'tf.ckpt')


def get_model():
    model = keras.models.load_model('/home/gus/workspaces/wpy/IACapstone/resources/output')
    return model

# get the keras model
model = get_model()
# get the tensor name from the embedding layer
tensor_name = next(filter(lambda x: x.name == 'embedding', model.layers)).W.name

# the vocabulary
metadata_file_name = os.path.join(ROOT_DIR, tensor_name)

# embedding_df = get_embedding()
# embedding_df.to_csv(metadata_file_name, header=False, columns=[])

saver = tensorflow.train.Saver()
saver.save(keras.backend.get_session(), OUTPUT_MODEL_FILE_NAME)

summary_writer = tensorflow.train.SummaryWriter(ROOT_DIR)

config = tensorflow.contrib.tensorboard.plugins.projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = tensor_name
embedding.metadata_path = metadata_file_name
tensorflow.contrib.tensorboard.plugins.projector.visualize_embeddings(summary_writer, config)
