'''
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
'''
from openvino.inference_engine import IENetwork, IECore

class Model_FaceDetection:
    '''
    Class for the Face Detection Model.
    '''
    def __init__(self, model_name, device='CPU', extensions=None):
        '''
        TODO: Use this to set your instance variables.
        '''
        self.bin_model_ = model_name + '.bin'
        self.xml_model_ = model_name + '.xml'
        self.device_ = device
        self.extensions_ = extensions

        # Initialize the Inference Engine
        self.infer_engine_ = IECore()

        if extensions and "CPU" in device:
            print(f'device: {device}')
            self.infer_engine_.add_extension(self.extensions_, self.device_)

        # Read the IR as a IENetwork
        self.network_ = self.infer_engine_.read_network(model=self.xml_model_, weights=self.bin_model_)

    def load_model(self):
        '''
        TODO: You will need to complete this method.
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        '''
        raise NotImplementedError

    def predict(self, image):
        '''
        TODO: You will need to complete this method.
        This method is meant for running predictions on the input image.
        '''
        raise NotImplementedError

    def check_model(self):
        raise NotImplementedError

    def preprocess_input(self, image):
        '''
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        '''
        raise NotImplementedError

    def preprocess_output(self, outputs):
        '''
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        '''
        raise NotImplementedError
