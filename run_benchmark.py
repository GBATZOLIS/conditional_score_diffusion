import argparse
from benchmark import Benchmark
from configs.utils import read_config

def main(args):

    mammoth_config = read_config('configs/mammoth/vesde.py')
    uniform_10_config = read_config('configs/ksphere/N_1/uniform_10.py')
    uniform_50_config = read_config('configs/ksphere/N_1/uniform_50.py')
    line_config = read_config('configs/line/vesde.py')
    non_uniform_10_1_config = read_config('configs/ksphere/N_1/non_uniform_1.py')
    non_uniform_10_075_config = read_config('configs/ksphere/N_1/non_uniform_075.py')
    non_uniform_10_05_config = read_config('configs/ksphere/N_1/non_uniform_05.py')
    squares_10_3_5_config = read_config('configs/fixedsquaresmanifold/10_3_5.py')
    squares_20_3_5_config = read_config('configs/fixedsquaresmanifold/20_3_5.py')
    squares_100_3_5_config = read_config('configs/fixedsquaresmanifold/100_3_5.py')
    gaussian_manifold_10 = read_config('configs/fixedgaussiansmanifold/10.py')
    gaussian_manifold_20 = read_config('configs/fixedgaussiansmanifold/20.py')
    gaussian_manifold_100 = read_config('configs/fixedgaussiansmanifold/100.py')
    mnist = read_config('configs/mnist/unconditional.py')
    stylegan_2d = read_config('configs/dimension_estimation/styleGAN/style_gan_2d_BeatGAN.py')

    configs_dict = {
        'mammoth': mammoth_config,
        'uniform_10': uniform_10_config,
        'unifrom_50': uniform_50_config,
        'line': line_config,
        'non_uniform_10_1': non_uniform_10_1_config,
        'non_uniform_10_075': non_uniform_10_075_config,
        'non_uniform_10_05': non_uniform_10_05_config,
        'squares_10': squares_10_3_5_config,
        'squares_20': squares_20_3_5_config,
        'squares_100': squares_100_3_5_config,
        'gaussian_manifold_10': gaussian_manifold_10,
        'gaussian_manifold_20': gaussian_manifold_20,
        'gaussian_manifold_100': gaussian_manifold_100,
        'mnist': mnist,
        'stylegan_2d': stylegan_2d

    }

    if args.max_samples > -1:
        for _, config in configs_dict.items():
            config.data.data_samples = args.max_samples

    benchmark = Benchmark(file_name=args.file, configs_dict=configs_dict)
    benchmark.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='benchmark.csv')
    parser.add_argument('--max_samples', type=int, default=-1)
    args = parser.parse_args()
    main(args)