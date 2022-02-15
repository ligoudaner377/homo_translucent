"""
The class name should be consistent with both the filename and its model option.
The filename should be <model>_dataset.py
The class name should be <Model>Dataset.py
It implements a simple image-to-image translation baseline based on regression loss.

You need to implement the following functions:
    <modify_commandline_options>:　Add model-specific options and rewrite default values for existing options.
    <__init__>: Initialize this model class.
    <set_input>: Unpack input data and perform data pre-processing.
    <forward>: Run forward pass. This will be called by both <optimize_parameters> and <test>.
    <optimize_parameters>: Update network weights; it will be called in every training iteration.
"""
import torch
import itertools
from .base_model import BaseModel
from . import networks
from . import renderer

class TwoshotrDirectRefineModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(batch_size=64, dataset_mode='translucent',
                            display_freq=12800, update_html_freq=12800, print_freq=12800,
                            n_epochs=10, n_epochs_decay=10, display_ncols=6, isTwoshot=True)  # You can rewrite default values for this model..
        # You can define new arguments for this model.
        parser.add_argument('--netPredictorInit', type=str, default='resnet_predictor', help='specify network architecture')
        parser.add_argument('--netLightDecInit', type=str, default='basic_decoder', help='specify network architecture')
        parser.add_argument('--netRendererInit', type=str, default='resnet_renderer', help='specify network architecture')
        parser.add_argument('--netPredictor', type=str, default='resnet_predictor_refine', help='specify network architecture')
        parser.add_argument('--netLightDec', type=str, default='basic_decoder', help='specify network architecture')
        parser.add_argument('--netRenderer', type=str, default='resnet_renderer', help='specify network architecture')
        parser.add_argument('--step', type=str, default='refine', help='specify the training step: init|refine')

        return parser

    def __init__(self, opt):
        """Initialize this model class.

        Parameters:
            opt -- training/test options

        A few things can be done here.
        - (required) call the initialization function of BaseModel
        - define loss function, visualization images, model names, and optimizers
        """
        BaseModel.__init__(self, opt)  # call the initialization method of BaseModel
        self.phase = opt.phase
        self.step = opt.step
        self.isEdit = opt.isEdit
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['Normal', 'Rough', 'Albedo', 'Depth', 'Scatter', 'Coeffs', 'Reconstruct', 'Direct']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.
        if self.isTrain:
            if self.step == 'init':
                self.visual_names = ['normal_predict_init', 'depth_predict_init_vis', 'rough_predict_init', 'albedo_predict_init', 'direct_predict_init', 'scene_predict_init',
                                     'normal_image', 'depth_image_vis', 'rough_image', 'albedo_image', 'direct_image', 'edited_image',
                                     'scene_image', 'non_flash_image']
            else:
                self.visual_names = ['normal_predict', 'depth_predict_vis', 'rough_predict', 'albedo_predict', 'direct_predict', 'scene_predict',
                                     'normal_image', 'depth_image_vis', 'rough_image', 'albedo_image', 'direct_image', 'edited_image',
                                     'scene_image', 'non_flash_image']
        else:
            self.visual_names = ['normal_predict', 'depth_predict_vis', 'rough_predict', 'albedo_predict', 'direct_predict', 'scene_predict',
                                 'normal_image', 'depth_image_vis', 'rough_image', 'albedo_image', 'direct_image', 'scene_image']
        # specify the models you want to save/load to the disk.
        if self.isTrain:
            if self.step == 'init':
                self.model_save_names = ['PredictorInit', 'LightDecInit', 'RendererInit']
                self.model_load_names = []
            else:
                self.model_save_names = ['Predictor', 'LightDec', 'Renderer']
                self.model_load_names = ['PredictorInit', 'LightDecInit', 'RendererInit']
        else:
            self.model_save_names = []
            self.model_load_names = ['Predictor', 'LightDec', 'Renderer', 'PredictorInit', 'LightDecInit', 'RendererInit']
        self.model_names = list(set(self.model_load_names + self.model_save_names))

        # define networks; you can use opt.isTrain to specify different behaviors for training and test.

        self.netPredictorInit = networks.define_G(7, 8, opt.ngf, opt.netPredictorInit, gpu_ids=self.gpu_ids)
        self.netLightDecInit = networks.define_De(75+9, opt.netLightDecInit, gpu_ids=self.gpu_ids)
        self.netRendererInit = networks.define_G(14, 3, opt.ngf, opt.netRendererInit, gpu_ids=self.gpu_ids)

        if self.step == 'refine' or self.phase == 'test':
            self.netPredictor = networks.define_G(21, 8, opt.ngf, opt.netPredictor, gpu_ids=self.gpu_ids)
            self.netLightDec = networks.define_De((75+9)*2, opt.netLightDec, gpu_ids=self.gpu_ids)
            self.netRenderer = networks.define_G(28, 3, opt.ngf, opt.netRenderer, gpu_ids=self.gpu_ids)
        self.direct_renderer = renderer.RenderLayerPointLightTorch()
        # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
        self.L2Loss = torch.nn.MSELoss()
        if self.isTrain:  # only defined during training time
            # define and initialize optimizers. You can define one optimizer for each network, or use itertools.chain to group them.
            if self.step == 'refine':
                self.optimizer = torch.optim.Adam(itertools.chain(self.netPredictor.parameters(),
                                                                  self.netLightDec.parameters(),
                                                                  self.netRenderer.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer = torch.optim.Adam(itertools.chain(self.netPredictorInit.parameters(),
                                                                  self.netLightDecInit.parameters(),
                                                                  self.netRendererInit.parameters()),
                                                  lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = [self.optimizer]
        # Our program will automatically call <model.setup> to define schedulers, load networks, and print networks

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.
        """

        self.binary_mask = input['mask'].to(self.device)
        self.scene_image = input['scene'].to(self.device)
        self.non_flash_image = input['non_flash'].to(self.device)
        self.num_pixel = torch.sum(self.binary_mask, axis=[-1, -2]).squeeze()
        self.background_image = self.normalize(self.inverse_normalize(self.scene_image) * ~self.binary_mask)
        if 'normal' in input:
            self.normal_image = input['normal'].to(self.device)
            self.albedo_image = input['albedo'].to(self.device)
            self.rough_image = input['rough'].to(self.device)
            self.depth_image = input['depth'].to(self.device)
            g_scatter = input['g_scatter'].unsqueeze(-1)
            sigma_t_scatter = input['sigma_t_scatter']
            albedo_scatter = input['albedo_scatter']
            surface_opacity = input['surface_opacity'].unsqueeze(-1)
            radiance = input['radiance'].unsqueeze(-1)
            self.scatter_para = self.normalize(
                torch.cat([g_scatter, sigma_t_scatter, albedo_scatter, surface_opacity, radiance], axis=-1).to(
                    self.device))
            self.coeffs_para = input['coeffs'].to(self.device)
            self.coeffs_para = self.coeffs_para.view(self.coeffs_para.shape[0], -1)

        if self.phase == 'train':
            self.edited_image = input['edited'].to(self.device)
            self.new_surface_opacity = self.normalize(input['new_surface_opacity'].unsqueeze(-1).to(self.device))
            self.new_radiance = self.normalize(input['new_radiance'].unsqueeze(-1).to(self.device))
        else:
            self.new_surface_opacity = torch.zeros((1, 1)).to(self.device)
            self.new_radiance = torch.zeros((1, 1)).to(self.device)
        if self.phase == 'test':
            self.image_paths = input['image_paths']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""
        inp = torch.cat([self.scene_image,
                         self.non_flash_image,
                         self.binary_mask], axis=1)
        if self.step == 'init':
            (self.normal_predict_init,
             self.depth_predict_init,
             self.albedo_predict_init,
             self.rough_predict_init,
             self.scatter_predict_init,
             self.coeffs_predict_init) = self.netPredictorInit(inp)
        else:
            with torch.no_grad():
                (self.normal_predict_init,
                 self.depth_predict_init,
                 self.albedo_predict_init,
                 self.rough_predict_init,
                 self.scatter_predict_init,
                 self.coeffs_predict_init) = self.netPredictorInit(inp)
                if self.phase == 'test' and not self.isEdit:
                    self.new_surface_opacity = self.scatter_predict_init[:, -2:-1]
                    self.new_radiance = self.scatter_predict_init[:, -1:]
        self.neural_rendering()

    def neural_rendering_init(self):
        # render direct image with predicted parameter
        self.direct_predict_init = self.direct_renderer.forward_batch(self.albedo_predict_init,
                                                                      torch.nn.functional.normalize(
                                                                          self.normal_predict_init,
                                                                          eps=1e-5),
                                                                      self.rough_predict_init,
                                                                      self.depth_predict_init,
                                                                      self.binary_mask,
                                                                      self.inverse_normalize(self.new_radiance))
        self.direct_predict_init = self.normalize(self.direct_predict_init)
        # replace the original radiance and surface opacity
        self.new_scatter_init = torch.cat([self.scatter_predict_init[:, :-2],
                                           self.new_surface_opacity,
                                           self.new_radiance], axis=1)
        # decode light and scattering parameter
        self.light_feature_init = self.netLightDecInit(torch.cat([self.new_scatter_init,
                                                                  self.coeffs_predict_init], axis=1))
        self.BRDF_predcit_init = torch.cat([self.albedo_predict_init,
                                            self.depth_predict_init,
                                            self.rough_predict_init,
                                            self.normal_predict_init,
                                            self.background_image,
                                            self.direct_predict_init], axis=1)
        # neural rendering
        self.scene_predict_init = self.netRendererInit((self.BRDF_predcit_init, self.light_feature_init))

    def neural_rendering_refine(self):
        # render direct image with predicted parameter
        self.direct_predict = self.direct_renderer.forward_batch(self.albedo_predict,
                                                                 torch.nn.functional.normalize(self.normal_predict,
                                                                                               eps=1e-5),
                                                                 self.rough_predict,
                                                                 self.depth_predict,
                                                                 self.binary_mask,
                                                                 self.inverse_normalize(self.new_radiance))
        self.direct_predict = self.normalize(self.direct_predict)
        # replace the original surface opacity and radiance
        self.new_scatter = torch.cat([self.scatter_predict[:, :-2],
                                      self.new_surface_opacity,
                                      self.new_radiance], axis=1)
        # decoder light
        self.light_feature = self.netLightDec(torch.cat([self.new_scatter,
                                                         self.coeffs_predict,
                                                         self.new_scatter_init,
                                                         self.coeffs_predict_init], axis=1))
        self.BRDF_predcit = torch.cat([self.albedo_predict,
                                       self.depth_predict,
                                       self.rough_predict,
                                       self.normal_predict,
                                       self.direct_predict,
                                       self.albedo_predict_init,
                                       self.depth_predict_init,
                                       self.rough_predict_init,
                                       self.normal_predict_init,
                                       self.direct_predict_init,
                                       self.scene_predict_init,
                                       self.background_image
                                       ], axis=1)
        self.scene_predict = self.netRenderer((self.BRDF_predcit, self.light_feature))

    def neural_rendering(self):
        # render GT direct image
        if hasattr(self, 'albedo_image'):
            self.direct_image = self.direct_renderer.forward_batch(self.albedo_image,
                                                                   torch.nn.functional.normalize(self.normal_image,
                                                                                                 eps=1e-5),
                                                                   self.rough_image,
                                                                   self.depth_image,
                                                                   self.binary_mask,
                                                                   self.inverse_normalize(self.new_radiance))
            self.direct_image = self.normalize(self.direct_image)
        if self.step == 'init':
            self.neural_rendering_init()
        else:
            with torch.no_grad():
                self.neural_rendering_init()

            inp_images = torch.cat([self.scene_image,
                                    self.non_flash_image,
                                    self.binary_mask,
                                    self.albedo_predict_init,
                                    self.depth_predict_init,
                                    self.rough_predict_init,
                                    self.normal_predict_init,
                                    self.direct_predict_init,
                                    self.scene_predict_init], axis=1)
            inp = (inp_images, self.light_feature_init)
            (self.normal_predict,
             self.depth_predict,
             self.albedo_predict,
             self.rough_predict,
             self.scatter_predict,
             self.coeffs_predict) = self.netPredictor(inp)
            if self.phase == 'test' and not self.isEdit:
                self.new_surface_opacity = self.scatter_predict[:, -2:-1]
                self.new_radiance = self.scatter_predict[:, -1:]
            self.neural_rendering_refine()

    def compute_loss(self):
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        if self.step == 'init':
            self.loss_Normal = self.imageLoss(self.normal_predict_init, self.normal_image)
            self.loss_Depth = self.imageLoss(self.depth_predict_init, self.depth_image)
            self.loss_Rough = self.imageLoss(self.rough_predict_init, self.rough_image)
            self.loss_Albedo = self.imageLoss(self.albedo_predict_init, self.albedo_image)
            self.loss_Scatter = self.L2Loss(self.scatter_predict_init, self.scatter_para)
            self.loss_Coeffs = self.L2Loss(self.coeffs_predict_init, self.coeffs_para)
            self.loss_Direct = self.imageLoss(self.direct_predict_init, self.direct_image)
            self.loss_Reconstruct = self.imageLoss(self.scene_predict_init, self.edited_image)
        else:
            self.loss_Normal = self.imageLoss(self.normal_predict, self.normal_image)
            self.loss_Depth = self.imageLoss(self.depth_predict, self.depth_image)
            self.loss_Rough = self.imageLoss(self.rough_predict, self.rough_image)
            self.loss_Albedo = self.imageLoss(self.albedo_predict, self.albedo_image)
            self.loss_Scatter = self.L2Loss(self.scatter_predict, self.scatter_para)
            self.loss_Coeffs = self.L2Loss(self.coeffs_predict, self.coeffs_para)
            self.loss_Direct = self.imageLoss(self.direct_predict, self.direct_image)
            if self.phase == 'train':
                self.loss_Reconstruct = self.imageLoss(self.scene_predict, self.edited_image)
            else:
                self.loss_Reconstruct = self.imageLoss(self.scene_predict, self.scene_image)

        self.loss = self.loss_Normal + \
                    5*self.loss_Depth + \
                    self.loss_Rough + \
                    self.loss_Albedo + \
                    2*self.loss_Scatter + \
                    self.loss_Coeffs + \
                    self.loss_Reconstruct + \
                    self.loss_Direct

    def backward(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""

        self.compute_loss()
        self.loss.backward()       # calculate gradients of network w.r.t. loss

    def optimize_parameters(self):
        """Update network weights; it will be called in every training iteration."""
        self.forward()               # first call forward to calculate intermediate results
        for optimizer in self.optimizers:
            optimizer.zero_grad()    # clear networks' existing gradients
        self.backward()              # calculate gradients for network s
        for optimizer in self.optimizers:
            optimizer.step()        # update gradients for networks

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        if hasattr(self, 'depth_image'):
            self.depth_image_vis = self.normalize_depth(self.depth_image)
            self.direct_image = self.normalize(torch.clamp(self.inverse_normalize(self.direct_image) ** (1 / 2.2), 0, 1))
        if hasattr(self, 'edited_image'):
            self.edited_image = self.normalize(self.inverse_normalize(self.edited_image) * self.binary_mask)
        if self.step == 'refine':
            self.depth_predict_vis = self.normalize_depth(self.depth_predict)
            self.normal_predict = self.normalize(self.inverse_normalize(self.normal_predict) * self.binary_mask)
            self.albedo_predict = self.normalize(self.inverse_normalize(self.albedo_predict) * self.binary_mask)
            self.rough_predict = self.normalize(self.inverse_normalize(self.rough_predict) * self.binary_mask)
            self.scene_predict = self.normalize(self.inverse_normalize(self.scene_predict) * self.binary_mask)
            self.direct_predict = self.normalize(torch.clamp(self.inverse_normalize(self.direct_predict) ** (1 / 2.2), 0, 1))
        else:
            self.depth_predict_init_vis = self.normalize_depth(self.depth_predict_init)
            self.normal_predict_init = self.normalize(self.inverse_normalize(self.normal_predict_init) * self.binary_mask)
            self.albedo_predict_init = self.normalize(self.inverse_normalize(self.albedo_predict_init) * self.binary_mask)
            self.rough_predict_init = self.normalize(self.inverse_normalize(self.rough_predict_init) * self.binary_mask)
            self.scene_predict_init = self.normalize(self.inverse_normalize(self.scene_predict_init) * self.binary_mask)
            self.direct_predict_init = self.normalize(torch.clamp(self.inverse_normalize(self.direct_predict_init) ** (1 / 2.2), 0, 1))

    def normalize_depth(self, depth_raw):
        """Normalize depth image to [-1, 1] for visualization"""
        depth_raw -= self.binary_mask * 100
        depth = (depth_raw - depth_raw.min()) * self.binary_mask
        depth = (depth/depth.max()) * 2 - 1
        return depth

    def normalize(self, inp):
        return inp * 2 - 1

    def inverse_normalize(self, image):
        return (image + 1) / 2

    def imageLoss(self, im1, im2):
        return torch.mean(torch.sum(torch.abs(im1 - im2) * self.binary_mask, axis=[-1, -2, -3]) / self.num_pixel)

