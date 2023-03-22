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

class EditTwoshotModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new model-specific options and rewrite default values for existing options.

        Parameters:
            parser -- the option parser
            is_train -- if it is training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(batch_size=32, dataset_mode='translucent',
                            display_freq=12800, update_html_freq=12800, print_freq=12800,
                            n_epochs=10, n_epochs_decay=10, display_ncols=8, isEdit=True, isTwoshot=True)  # You can rewrite default values for this model..
        # You can define new arguments for this model.
        parser.add_argument('--netPredictor', type=str, default='resnet_predictor', help='specify network architecture')
        parser.add_argument('--netLightDec', type=str, default='basic_decoder', help='specify network architecture')
        parser.add_argument('--netRenderer', type=str, default='resnet_renderer', help='specify network architecture')
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
        # specify the training losses you want to print out. The program will call base_model.get_current_losses to plot the losses to the console and save them to the disk.
        self.loss_names = ['Normal', 'Rough', 'Depth', 'Scatter', 'Coeffs', 'Reconstruct',
                           'Reconstruct_albedo', 'Reconstruct_sigma_t', 'Reconstruct_g']
        # specify the images you want to save and display. The program will call base_model.get_current_visuals to save and display these images.

        self.visual_names = ['normal_predict', 'depth_predict_vis', 'rough_predict', 'direct_predict', 'scene_predict', 'albedo_edit_predict', 'sigma_t_edit_predict', 'g_edit_predict',
                             'normal_image', 'depth_image_vis', 'rough_image', 'direct_image', 'scene_image', 'albedo_edit_image', 'sigma_t_edit_image', 'g_edit_image']

        # specify the models you want to save/load to the disk.
        if self.isTrain:
            self.model_save_names = ['Predictor', 'LightDec', 'Renderer']
            self.model_load_names = []
        else:
            self.model_save_names = []
            self.model_load_names = ['Predictor', 'LightDec', 'Renderer']
        self.model_names = list(set(self.model_load_names + self.model_save_names))

        # define networks; you can use opt.isTrain to specify different behaviors for training and test.
        self.netPredictor = networks.define_G(7, None, opt.ngf, opt.netPredictor, gpu_ids=self.gpu_ids)
        self.netLightDec = networks.define_De(27 + 8, opt.netLightDec, gpu_ids=self.gpu_ids)
        self.netRenderer = networks.define_G(6, 3, opt.ngf, opt.netRenderer, gpu_ids=self.gpu_ids)
        self.direct_renderer = renderer.RenderLayerPointLightEnvTorch()


        # define your loss functions. You can use losses provided by torch.nn such as torch.nn.L1Loss.
        self.L2Loss = torch.nn.MSELoss()
        if self.isTrain:  # only defined during training time
            # define and initialize optimizers. You can define one optimizer for each network, or use itertools.chain to group them.
            self.optimizer = torch.optim.Adam(itertools.chain(self.netPredictor.parameters(),
                                                              self.netLightDec.parameters(),
                                                              self.netRenderer.parameters()),
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
        self.num_pixel = torch.sum(self.binary_mask, dim=[-1, -2]).squeeze()
        self.background_image = self.normalize(self.inverse_normalize(self.scene_image) * ~self.binary_mask)
        if 'normal' in input:
            self.normal_image = input['normal'].to(self.device)
            self.rough_image = input['rough'].to(self.device)
            self.depth_image = input['depth'].to(self.device)
            g = self.normalize_g(input['g'].unsqueeze(-1)).to(self.device)
            sigma_t = self.normalize(input['sigma_t']).to(self.device)
            albedo = self.normalize_albedo(input['albedo']).to(self.device)
            radiance = self.normalize_radiance(input['radiance'].unsqueeze(-1)).to(self.device)
            self.scatter_para = torch.cat([g, sigma_t, albedo, radiance], dim=-1)
            self.coeffs_para = input['coeffs'].to(self.device)

            self.albedo_edit_image = input['albedo_edit'].to(self.device)
            self.g_edit_image = input['g_edit'].to(self.device)
            self.sigma_t_edit_image = input['sigma_t_edit'].to(self.device)

            self.new_g = self.normalize_g(input['new_g'].unsqueeze(-1)).to(self.device)
            self.new_sigma_t = self.normalize(input['new_sigma_t']).to(self.device)
            self.new_albedo = self.normalize_albedo(input['new_albedo']).to(self.device)

        if self.phase == 'test':
            self.image_paths = input['image_paths']

    def forward(self):
        """Run forward pass. This will be called by both functions <optimize_parameters> and <test>."""

        self.parameter_estimating()
        self.direct_rendering()
        self.neural_rendering()
        if hasattr(self, 'normal_image'):
            self.neural_rendering(mode='sigma_t')
            self.neural_rendering(mode='albedo')
            self.neural_rendering(mode='g')

        self.coeffs_predict = self.coeffs_predict.view(self.coeffs_predict.size(0), 3, 9)

    def parameter_estimating(self):
        inp = torch.cat([self.scene_image,
                         self.non_flash_image,
                         self.binary_mask], dim=1)
        (self.normal_predict,
         self.depth_predict,
         self.rough_predict,
         self.scatter_predict,
         self.coeffs_predict) = self.netPredictor(inp)


    def direct_rendering(self):
        self.coeffs_predict = self.coeffs_predict.view(self.coeffs_predict.size(0), 3, 9)

        if hasattr(self, 'normal_image'):
            self.direct_image = self.render_direct_image()
        self.direct_predict = self.render_direct_predict()

        self.coeffs_predict = self.coeffs_predict.view(self.coeffs_predict.shape[0], -1)

    def neural_rendering(self, mode=None):
        surface_predcit = torch.cat([self.background_image,
                                     self.direct_predict], dim=1)
        if mode == 'sigma_t':
            light_feature = self.netLightDec(torch.cat([self.scatter_predict[:, 0:1],
                                                        self.new_sigma_t,
                                                        self.scatter_predict[:, 4:],
                                                        self.coeffs_predict], dim=1))
            self.sigma_t_edit_predict = self.netRenderer((surface_predcit, light_feature))
        elif mode == 'albedo':
            light_feature = self.netLightDec(torch.cat([self.scatter_predict[:, 0:4],
                                                        self.new_albedo,
                                                        self.scatter_predict[:, 7:],
                                                        self.coeffs_predict], dim=1))
            self.albedo_edit_predict = self.netRenderer((surface_predcit, light_feature))
        elif mode == 'g':
            light_feature = self.netLightDec(torch.cat([self.new_g,
                                                        self.scatter_predict[:, 1:],
                                                        self.coeffs_predict], dim=1))
            self.g_edit_predict = self.netRenderer((surface_predcit, light_feature))
        else:
            light_feature = self.netLightDec(torch.cat([self.scatter_predict,
                                                        self.coeffs_predict], dim=1))
            self.scene_predict = self.netRenderer((surface_predcit, light_feature))

    def render_direct_image(self):
        point_image = self.direct_renderer.forward_batch(torch.zeros_like(self.normal_image),
                                                         torch.nn.functional.normalize(self.normal_image, eps=1e-5),
                                                         self.rough_image,
                                                         self.depth_image,
                                                         self.binary_mask,
                                                         self.inverse_normalize_radiance(self.scatter_para[:, -1]))

        env_image = self.direct_renderer.forward_env(torch.zeros_like(self.normal_image),
                                                     torch.nn.functional.normalize(self.normal_image, eps=1e-5),
                                                     self.rough_image,
                                                     self.binary_mask,
                                                     self.coeffs_para)
        image = torch.clamp(point_image + env_image, 0, 1)
        return self.normalize(image)

    def render_direct_predict(self):
        point_image = self.direct_renderer.forward_batch(torch.zeros_like(self.normal_predict),
                                                         torch.nn.functional.normalize(self.normal_predict, eps=1e-5),
                                                         self.rough_predict,
                                                         self.depth_predict,
                                                         self.binary_mask,
                                                         self.inverse_normalize_radiance(self.scatter_predict[:, -1]))

        env_image = self.direct_renderer.forward_env(torch.zeros_like(self.normal_predict),
                                                     torch.nn.functional.normalize(self.normal_predict, eps=1e-5),
                                                     self.rough_predict,
                                                     self.binary_mask,
                                                     self.coeffs_predict)
        image = torch.clamp(point_image + env_image, 0, 1)
        return self.normalize(image)

    def compute_loss(self):
        # caculate the intermediate results if necessary; here self.output has been computed during function <forward>
        # calculate loss given the input and intermediate results

        self.loss_Normal = self.imageLoss(self.normal_predict, self.normal_image)
        self.loss_Depth = self.imageLoss(self.depth_predict, self.depth_image)
        self.loss_Rough = self.imageLoss(self.rough_predict, self.rough_image)
        self.loss_Scatter = self.L2Loss(self.scatter_predict, self.scatter_para)
        self.loss_Coeffs = self.L2Loss(self.coeffs_predict, self.coeffs_para)
        self.loss_Reconstruct = self.imageLoss(self.scene_predict, self.scene_image)
        self.loss_Reconstruct_sigma_t = self.imageLoss(self.sigma_t_edit_predict, self.sigma_t_edit_image)
        self.loss_Reconstruct_albedo = self.imageLoss(self.albedo_edit_predict, self.albedo_edit_image)
        self.loss_Reconstruct_g = self.imageLoss(self.g_edit_predict, self.g_edit_image)


        self.loss = self.loss_Normal + \
                    5 * self.loss_Depth + \
                    self.loss_Rough + \
                    self.loss_Scatter + \
                    self.loss_Coeffs + \
                    self.loss_Reconstruct + \
                    self.loss_Reconstruct_albedo + \
                    self.loss_Reconstruct_g + \
                    self.loss_Reconstruct_sigma_t

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
            self.direct_image = self.normalize(
                torch.clamp(self.inverse_normalize(self.direct_image) ** (1 / 2.2), 0, 1))
        if hasattr(self, 'g_edit_predict'):
            self.g_edit_predict = self.normalize(self.inverse_normalize(self.g_edit_predict) * self.binary_mask)
        if hasattr(self, 'albedo_edit_predict'):
            self.albedo_edit_predict = self.normalize(self.inverse_normalize(self.albedo_edit_predict) * self.binary_mask)
        if hasattr(self, 'sigma_t_edit_predict'):
            self.sigma_t_edit_predict = self.normalize(self.inverse_normalize(self.sigma_t_edit_predict) * self.binary_mask)
        if hasattr(self, 'scene_predict'):
            self.scene_predict = self.normalize(self.inverse_normalize(self.scene_predict) * self.binary_mask)
        self.depth_predict_vis = self.normalize_depth(self.depth_predict)
        self.normal_predict = self.normalize(self.inverse_normalize(self.normal_predict) * self.binary_mask)
        self.rough_predict = self.normalize(self.inverse_normalize(self.rough_predict) * self.binary_mask)
        self.direct_predict = self.normalize(torch.clamp(self.inverse_normalize(self.direct_predict) ** (1 / 2.2), 0, 1))

    def normalize_depth(self, depth_raw):
        """Normalize depth image to [-1, 1] for visualization"""
        depth_raw -= self.binary_mask * 100
        depth = (depth_raw - depth_raw.min()) * self.binary_mask
        depth = (depth/depth.max()) * 2 - 1
        return depth

    @staticmethod
    def normalize_radiance(radiance):
        """raw radiance values are between [0.35, 0.75], normalize them to [-1, 1]"""
        radiance -= 0.35
        radiance /= 0.4
        return radiance * 2 - 1

    @staticmethod
    def inverse_normalize_radiance(radiance):
        """map radiance values back to [0.35, 0.75]"""
        radiance = (radiance + 1) / 2
        radiance *= 0.4
        radiance += 0.35
        return radiance

    @staticmethod
    def normalize_albedo(albedo):
        """raw albedo values are between [0.3, 0.95], normalize them to [-1, 1]"""
        albedo -= 0.3
        albedo /= 0.65
        return albedo * 2 - 1

    @staticmethod
    def inverse_normalize_albedo(albedo):
        """map albedo values back to [0.3, 0.95]"""
        albedo = (albedo + 1) / 2
        albedo *= 0.65
        albedo += 0.3
        return albedo

    @staticmethod
    def normalize_g(g):
        """raw g values are between [0.0, 0.9], normalize them to [-1, 1]"""
        g /= 0.9
        return g * 2 - 1

    @staticmethod
    def inverse_normalize_g(g):
        """map g values back to [0.0, 0.9]"""
        g = (g + 1) / 2
        g *= 0.9
        return g

    @staticmethod
    def normalize(inp):
        return inp * 2 - 1

    @staticmethod
    def inverse_normalize(image):
        return (image + 1) / 2

    def imageLoss(self, im1, im2):
        num_channel = im1.shape[1]
        return torch.mean(torch.sum(torch.abs(im1 - im2) * self.binary_mask, dim=[-1, -2, -3])
                          / (self.num_pixel * num_channel))

