import ml_collections
from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import run_lib
import pickle
from configs.utils import read_config, fix_config
import uuid
import datetime

FLAGS = flags.FLAGS

#config_flags.DEFINE_config_file(
#  "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("config", None, "Training configuration path.")
flags.DEFINE_string("checkpoint_path", None, "Checkpoint directory.")
flags.DEFINE_string("data_path", None, "Checkpoint directory.")
flags.DEFINE_string("log_path", "./", "Checkpoint directory.")
flags.DEFINE_enum("mode", "train", ["train", "test", "multi_scale_test", "compute_dataset_statistics", 'manifold_dimension', 'conditional_manifold_dimension', 'inspect_VAE', 'inspect_corrected_VAE', 'scoreVAE_fidelity'], "Running mode: train or test")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_boolean("debug", False, "Use GPU?")
flags.DEFINE_string("log_name", None, "Log name")
flags.mark_flags_as_required(["config", "mode", "log_path"])


def main(argv):

  #check if config is a script or a binary and load accordingly
  if FLAGS.config[-3:] == 'pkl':
    with open(FLAGS.config, 'rb') as file:
      config = pickle.load(file)
      #config = fix_config(config)
  elif FLAGS.config[-2:] == 'py':
    config = read_config(FLAGS.config)
    #config = fix_config(config)
  else:
    raise RuntimeError('Unknown config extension. Provide a path to .py or .pkl file.')
  
  if FLAGS.checkpoint_path is not None:
        config.model.checkpoint_path =  FLAGS.checkpoint_path

  if FLAGS.debug:
    print('-----Debug mode-----')
    # config.training.gpus = 0
    config.logging.log_path = 'test_logs/'
    # time as string
    time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")
    config.logging.log_name = str(f"{time}__{uuid.uuid4()}")
    config.training.batch_size = 16
    config.validation.batch_size = 16
    config.eval.batch_size = 16
    config.training.visualisation_freq = 1
    config.data.percentage_use = 5
    
    # config.debug = ml_collections.ConfigDict()
    # config.debug.skip_training = True
    # config.debug.skip_validation = False
    
  if FLAGS.mode == 'train':
    run_lib.train(config, FLAGS.log_path, FLAGS.checkpoint_path, FLAGS.log_name)
  elif FLAGS.mode == 'test':
    run_lib.test(config, FLAGS.log_path, FLAGS.checkpoint_path)
  elif FLAGS.mode == 'multi_scale_test':
    run_lib.multi_scale_test(config, FLAGS.log_path)
  elif FLAGS.mode == 'compute_dataset_statistics':
    run_lib.compute_data_stats(config)
  elif FLAGS.mode == 'manifold_dimension':
    run_lib.get_manifold_dimension(config)
  elif FLAGS.mode == 'conditional_manifold_dimension':
    run_lib.get_conditional_manifold_dimension(config)
  elif FLAGS.mode == 'inspect_VAE':
    run_lib.inspect_VAE(config)
  elif FLAGS.mode == 'inspect_corrected_VAE':
    run_lib.inspect_corrected_VAE(config)
  elif FLAGS.mode == 'scoreVAE_fidelity':
    run_lib.scoreVAE_fidelity(config)

if __name__ == "__main__":
  app.run(main)