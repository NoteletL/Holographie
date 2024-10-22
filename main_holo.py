import time
import os
import torch
from holodncnn.utils import *
from holodncnn.model import DnCNN
from holodncnn.nntools import Experiment, DenoisingStatsManager
from holodncnn.holosets import TrainHoloset, EvalHoloset
import yaml
import argparse

def config():
    '''Add arguments
    '''
    parser = argparse.ArgumentParser(
        description='DnCNN')
    parser.add_argument('--config', dest='config', type=str, default='./dataset.yaml',
                        help='configuration file')
    parser.add_argument('--test_mode', dest='test_mode', action='store_true',
                        help='testing phase')
    return parser.parse_args()


def save_clean_pred_rad(output_dir, exp, clean_pred_rad, noisy, clean, nom_img="NoisyPhasePred"):
    """This method is used to save the result of a de-noising operation

    Arguments:
        args (ArgumentParser) :         The different info used to do and save the de-noising operation
        exp (Experiment) :              The de-noising model
        clean_pred_rad (numpy.array) :  The de-noised image
        noisy (numpy.array) :           The noised image
        clean (numpy.array) :           The noise free image
        nom_img (str, optional) :       The saving name for the result
    """

    #save_name = os.path.join(args.save_test_dir, args.input_dir, "Test")
    save_name = os.path.join(output_dir, "Test")

    if not os.path.exists(save_name):
        os.makedirs(save_name)

    save_images(os.path.join(save_name, '%s-noisy.tiff' % (nom_img)), noisy)
    save_images(os.path.join(save_name, '%s-clean.tiff' % (nom_img)), clean)

    save_images(os.path.join(save_name, '%s-%d.tiff' % (nom_img, exp.epoch)), clean_pred_rad)
    save_MAT_images(os.path.join(save_name, '%s-%d.mat' % (nom_img, exp.epoch)), clean_pred_rad)

    epoch = exp.epoch
    psnr = cal_psnr(rad_to_flat(clean_pred_rad), rad_to_flat(clean))
    std = cal_std_phase(clean_pred_rad, clean)

    print("\n")
    print("image : ", nom_img)
    print("epoch : ", epoch)
    print("psnr : ", psnr)
    print("std : ", std)
    print("\n")

    with open(os.path.join(save_name, '%s-%d.res' % (nom_img, exp.epoch)), 'w') as f:
        print("image : ", nom_img, file=f)
        print("epoch : ", epoch, file=f)
        print("psnr : ", psnr, file=f)
        print("std : ", std, file=f)





def evaluate_with_ref(param, data, exp):
    """Denoise a liste of images and compute metrics when the clean reference is known
        - clean is the clean reference of size ()
        - noisy is the noisy image to be processed
    """
    running_std = 0
    assert (len(data) > 0), "No data to evaluate"

    for i in range(len(data)):
        noisy, clean = data[i]
        clean_pred_rad = noisy
        for j in range(param['test']['nb_iteration']):
            clean_pred_rad = denoising_single_image(clean_pred_rad, exp)

        save_clean_pred_rad(param['test']['save_test_dir'],
                            exp,
                            clean_pred_rad.numpy().squeeze(),
                            noisy.numpy().squeeze(),
                            clean.numpy().squeeze(),
                            nom_img="test-{:0>2}".format(i))

        std = cal_std_phase(clean_pred_rad, clean)
        running_std += std
    print("average_std : ", running_std / noisy.shape[0])

def evaluate_without_ref(param, data, exp):
    """Denoise a liste of images and compute metrics when the clean reference is known
        - clean is the clean reference of size ()
        - noisy is the noisy image to be processed
    """

    running_std = 0
    for i in range(len(data)):
        noisy, _ = data[i]
        clean_pred_rad = noisy

        for j in range(param['test']['nb_iteration']):
            clean_pred_rad = denoising_single_image(clean_pred_rad, exp)

        save_clean_pred_rad(param['test']['save_test_dir'],
                            exp,
                            clean_pred_rad.numpy().squeeze(),
                            noisy.numpy().squeeze(),
                            noisy.numpy().squeeze(),
                            nom_img="test-{:0>2}".format(i))
        std = cal_std_phase(clean_pred_rad, noisy)
        running_std += std

    print("average_std : ", running_std / noisy.shape[0])



def denoising_single_image(noisy, exp):
    """This method is used to do a de-noising operation on a given image

    Arguments:
        args (ArgumentParser) : The different info used to do the de-noising operation
        noisy (numpy.array) :   The image to denoise
        exp (Experiment) :      The model used to denoise
    """

    #noisyPy = noisy.reshape(1, args.image_mode, args.test_image_size[0], args.test_image_size[1])

    # noisyPy_cos = torch.Tensor(normalize_data(noisyPy, 'cos', None))
    # noisyPy_sin = torch.Tensor(normalize_data(noisyPy, 'sin', None))

    noisyPy_cos = torch.cos(noisy).unsqueeze(0)
    noisyPy_sin = torch.sin(noisy).unsqueeze(0)

    # clean_pred_cos = exp.test(noisyPy_cos).detach().cpu().numpy()
    # clean_pred_sin = exp.test(noisyPy_sin).detach().cpu().numpy()

    clean_pred_cos = exp.test(noisyPy_cos)
    clean_pred_sin = exp.test(noisyPy_sin)

    clean_pred_cos = clean_pred_cos.detach()
    clean_pred_sin = clean_pred_sin.detach()

    clean_pred_cos = clean_pred_cos.cpu()
    clean_pred_sin = clean_pred_sin.cpu()

    clean_pred_cos = clean_pred_cos.numpy()
    clean_pred_sin = clean_pred_sin.numpy()

    clean_pred_rad = torch.Tensor(np.angle(clean_pred_cos + clean_pred_sin * 1J)).squeeze(0)
    #clean_pred_rad = clean_pred_rad.reshape(1, noisyPy_cos.size(2), noisyPy_cos.size(2), 1)

    return clean_pred_rad


def run(param, test_mode):
    """This method is the main method
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    net = DnCNN(D=param['model']['D'],
                C=param['model']['C'],
                image_mode=1
                ).to(device)
    adam = torch.optim.Adam(net.parameters(), lr=param['model']['lr'])

    exp = Experiment(net,
                        adam,
                        DenoisingStatsManager(),
                        perform_validation_during_training=param['model']['perform_validation'],
                        input_dir=param['model']['input_dir'],
                        output_dir=param['model']['output_dir'],
                        startEpoch=param['model']['start_epoch'],
                        freq_save=param['model']['freq_save'])

    if not test_mode:
        print("\n=>Training until epoch :<===", param['model']['num_epochs'])
        print("\n\Model training")

        trainData = TrainHoloset(param['train']['path'],
                                 param['train']['csv'],
                                 param['train']['extension'],
                                 param['train']['matlab_key_clean'],
                                 param['train']['matlab_key_noisy'],
                                 param['train']['augmentation'],
                                 param['train']['patch']['nb_patch_per_image'],
                                 param['train']['patch']['size'],
                                 param['train']['patch']['stride'],
                                 param['train']['patch']['step'],
                                 )
        evalData = EvalHoloset(param['eval']['path'],
                               param['eval']['csv'],
                               param['eval']['extension'],
                               param['eval']['matlab_key_clean'],
                               param['eval']['matlab_key_noisy'],
                               )

        exp.initData(trainData, evalData, batch_size=param['model']['batch_size'])

        exp.run(num_epochs=param['model']['num_epochs'])

        if (param['model']['graph']):
            exp.trace()

    else:

        testData = EvalHoloset(param['test']['path'],
                               param['test']['csv'],
                               param['test']['extension'],
                               param['test']['matlab_key_clean'],
                               param['test']['matlab_key_noisy'],
                               )
        print("Nb of test noisy images : ", len(testData))
        print("This test is with reference : ", testData.ref)

        if testData.ref:
            evaluate_with_ref(param, testData, exp)
        else:
            evaluate_without_ref(param, testData, exp)


if __name__ == '__main__':
    args = config()
    with open(args.config, 'r') as f:
        param = yaml.load(f, Loader=yaml.FullLoader)

    print("\n")
    list(map(lambda p: print(p + " : ", param[p]), param))
    print("\n")

    torch.manual_seed(123456789)

    timeElapsed = time.time()
    run(param, args.test_mode)
    print("Time elapsed : ", time.time() - timeElapsed)
