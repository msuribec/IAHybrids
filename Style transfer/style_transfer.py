
import os
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_hub
import numpy as np

from tqdm import tqdm

class StyleGan2:
    """Class that represents a stylegan2
    Parameters
    ----------
    path_video : str
        Path to the video to be stylized
    path_style_image : str
        Path to the style image
    
    """
    def __init__(self, path_video, path_style_image):

        self.TRAINED_MODEL_PATH = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
        self.hub_model = tensorflow_hub.load(self.TRAINED_MODEL_PATH)

        print("Creating necessary folders")

        self.Results_path = 'Results'
        self.frames_path  = f'{self.Results_path}/Frames'
        self.stylized_frames_path  = f'{self.Results_path}/StylizedFrames'
        self.output_video_path = f'{self.Results_path}/stylized_video.mp4'
        self.create_folders_if_not_exist([self.Results_path,self.frames_path, self.stylized_frames_path])

        self.path_video = path_video
        self.path_style_image = path_style_image
        
        self.video_frames = []
        self.imgs = []

        print("Capturing video")
        self.video_capture = cv2.VideoCapture(self.path_video)
        self.fps = int(self.video_capture.get(cv2.CAP_PROP_FPS))
        self.W = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        

        print("Saving frames from video")
        self.save_frames_from_video()

        self.C = self.video_frames[0].shape[-1]
        self.size = (self.W, self.H)

        print("Transferring style") 
        self.save_style_transfer()

    def tensor_to_image(self,tensor):
        """
        Function to convert a tensor to an image
        Code obtained from tensorflow tutorials page
        Source: https://www.tensorflow.org/tutorials/generative/style_transfer
        Parameters
        ----------
        tensor : tensor
            Tensor to be converted to image
        Returns
        -------
        np.ndarray
            Image converted from tensor
        """
        tensor = tensor*255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor)>3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return np.squeeze(tensor)


    def load_img(self,img_tensor):
        """ Function that loads an image and resizes it to 512x512
        Code adapted from tensorflow tutorials page
        Source: https://www.tensorflow.org/tutorials/generative/style_transfer
        Parameters:
        ----------
        img_tensor: tensor
            Tensor representing an image
        Returns:
        -------
        tensor
            Tensor representing an image
        """
        max_dim = 512
        img = tf.image.convert_image_dtype(img_tensor, tf.float32)
        shape = tf.cast(tf.shape(img)[:-1], tf.float32)
        long_dim = max(shape)
        scale = max_dim / long_dim
        new_shape = tf.cast(shape * scale, tf.int32)
        img = tf.image.resize(img, new_shape)
        img = img[tf.newaxis, :]
        return img

    def imshow(self,image, title=None):
        """Function to show an image
        Code obtained from tensorflow tutorials page
        Source: https://www.tensorflow.org/tutorials/generative/style_transfer
        Parameters
        ----------
        image : tensor
            Tensor representing an image
        title : str, optional
            Title of the image, by default None
        """
        image = tf.squeeze(image, axis=0) if len(image.shape) > 3 else image
        plt.imshow(image)
        if title: plt.title(title)
    

    def create_folders_if_not_exist(self,folder_paths):
        """Function to create folders if they don't exist
        Parameters
        ----------
        folder_paths : list
            List of paths to folders
        """
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def save_frames_from_video(self):
        """Function to save frames from video
        """

        style_image_tensor = self.to_tensor(path_to_img=self.path_style_image)
        self.style_image = self.load_img(style_image_tensor)
        self.tensor_style_img = tf.constant(self.style_image)

        if self.video_capture.isOpened():
            i = 0
            ret = True
            while ret:
                ret, frame = self.video_capture.read()
                if ret:
                    frame_path = f"{self.frames_path}/{i}.png"
                    self.video_frames.append(frame)
                    cv2.imwrite(frame_path, frame)
                    tensor_img = self.to_tensor(numpy_img= frame)
                    img = self.load_img(tensor_img)
                    self.imgs.append(img)
                i += 1
            self.video_capture.release()
        cv2.destroyAllWindows()


    def to_tensor(self,numpy_img= None, path_to_img=None):
        """Function to convert an image to a tensor, the argument can
        be the numpy array representing the image or the path to the image

        Parameters
        ----------
        numpy_img : np.ndarray, optional
            Numpy array representing an image, by default None
        path_to_img : str, optional
            Path to an image, by default None
        Returns
        -------
        tensor
            Tensor representing an image
        """

        got_args = numpy_img is not None or path_to_img is not None
        assert got_args, "Must pass in either numpy_img or path_to_img"
        
        if numpy_img is not None:
            flipped = (numpy_img[:, :, ::-1]).astype(int)
            img = tf.convert_to_tensor(flipped, dtype=tf.uint8)
        else:
            img = tf.io.read_file(path_to_img)
            img = tf.image.decode_image(img, channels=3)
        return img


    def flip_array(self,array):
        """Function to flip an array
        Parameters
        ----------
        array : np.ndarray
            Array to be flipped
        Returns
        -------
        np.ndarray
            Flipped array
        """

        flipped_array = array.copy()
        flipped_array[:, [0, 2]] = array[:, [2, 0]]
        return flipped_array

    def save_style_transfer(self):
        """Function to perform the style transfer and save the frames to a new video
        """
        out = cv2.VideoWriter(self.output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        for i, image in tqdm(enumerate(self.imgs), desc = "Stylizing frames", total= len(self.imgs)):
            stylized_img = self.hub_model(tf.constant(image), self.tensor_style_img)[0]
            frame_path = f"{self.stylized_frames_path}/{i}.png"
            image = self.tensor_to_image(stylized_img )
            image = cv2.resize(image, self.size)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(frame_path, image)
            out.write(image)
            
        out.release()
    
if __name__ == "__main__":    

    sg = StyleGan2('Data/video.mp4','Data/style_image.png')

