'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore

from logger import Logger
logger = Logger.get_logger('logger')

class Model:
    '''
    Class for the based model
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.model_name = model_name
        self.bin_model_ = model_name + '.bin'
        self.xml_model_ = model_name + '.xml'
        self.device_ = device
        self.extensions_ = extensions
        self.infer_engine_ = None
        self.network_ = None
        self.exec_network_ = None
        self.input_blob_ = []
        self.output_blob_ = []
        self.load_model()
        self.check_model() # check the unsupported layer(s)

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''

        # Initialize the Inference Engine
        self.infer_engine_ = IECore()

        if self.extensions_ and "CPU" in self.device_:
            self.infer_engine_.add_extension(self.extensions_, self.device_)

        # Read the IR as a IENetwork
        self.network_ = self.infer_engine_.read_network(model=self.xml_model_, weights=self.bin_model_)
        # Return the loaded inference plugin ###
        self.exec_network_ = self.infer_engine_.load_network(self.network_, self.device_)

        self.input_blob_ = next(iter(self.network_.inputs))
        self.output_blob_ = next(iter(self.network_.outputs))

        logger.debug('* blob info *')
        logger.debug(' - input : %s' % self.input_blob_)
        logger.debug(' - output: %s' % self.output_blob_)

    def get_input_shape(self):
        """ Return the shape of the input layer """
        return self.network_.inputs[self.input_blob_].shape

    def wait(self):
        """ Wait for the request to be complete """
        infer_status = self.exec_network_.requests[0].wait(-1)
        return infer_status

    def get_output(self):
        """ Extract and return the output results """
        infer_output = self.exec_network_.requests[0].outputs[self.output_blob_]
        logger.debug('  - extracting DNN output from blob (%s)' % self.output_blob_)
        logger.debug('  - output shape: {}'.format(infer_output.shape))
        return infer_output

    def check_model(self):
        """ Check for supported layers """
        layers_map = self.infer_engine_.query_network(self.network_, "CPU")

        logger.debug('* layers info *')
        layer_num = 0
        unsupport_layers = []

        for layer, hw in layers_map.items():
            if not hw in ['CPU']:
                logger.debug(' [U] #%d: %s [%s]' % (layer_num, layer, hw))
                unsupport_layers.append(layer)
            else:
                logger.debug(' [S] #%d: %s [%s]' % (layer_num, layer, hw))
                pass
            layer_num += 1

        logger.info(f'[ {self.model_name} ]')

        if len(unsupport_layers) == 0:
            logger.info('   All the layers are supported')
        else:
            assert False, f'   There is unsupported layer(s): {unsupport_layers}'
