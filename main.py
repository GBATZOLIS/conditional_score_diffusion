from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import run_lib

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
  "config", None, "Training configuration.", lock_config=False)
flags.DEFINE_string("checkpoint_path", None, "Checkpoint directory.")
flags.DEFINE_string("data_path", None, "Checkpoint directory.")
flags.DEFINE_string("log_path", "./", "Checkpoint directory.")
flags.DEFINE_enum("mode", "train", ["train", "test", "multi_scale_test", "compute_dataset_statistics"], "Running mode: train or test")
flags.DEFINE_string("eval_folder", "eval",
                    "The folder name for storing evaluation results")
flags.DEFINE_boolean("debug", False, "Use GPU?")
flags.DEFINE_string("log_name", None, "Log name")
flags.mark_flags_as_required(["config", "mode", "log_path"])


def main(argv):
  if FLAGS.debug:
    FLAGS.config.training.gpus = 0
  if FLAGS.mode == 'train':
    run_lib.train(FLAGS.config, FLAGS.log_path, FLAGS.checkpoint_path, FLAGS.log_name)
  elif FLAGS.mode == 'test':
    run_lib.test(FLAGS.config, FLAGS.log_path, FLAGS.checkpoint_path)
  elif FLAGS.mode == 'multi_scale_test':
    run_lib.multi_scale_test(FLAGS.config, FLAGS.log_path)
  elif FLAGS.mode == 'compute_dataset_statistics':
    run_lib.compute_data_stats(FLAGS.config)

if __name__ == "__main__":
  app.run(main)


